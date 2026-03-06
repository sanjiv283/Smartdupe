import json
import os
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from starlette.requests import Request
from sqlalchemy.orm import Session

import database
import models
from database import get_db
from helpers import save_file, validate_file, get_file_type
from hashing import partial_hash, full_hash
from dedupe import size_tolerance_group, find_exact_duplicate
from ai_similarity import get_embedding_for_file, cluster_files, compute_similarity_matrix
from auth import hash_password, verify_password, create_access_token, get_current_user

app = FastAPI(title="SmartDupe")

templates = Jinja2Templates(directory="templates")
os.makedirs("uploads", exist_ok=True)

# Create all DB tables on startup
models.Base.metadata.create_all(bind=database.engine)


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/register", status_code=201)
def register(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    existing = db.query(models.User).filter_by(username=form.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    user = models.User(username=form.username, password=hash_password(form.password))
    db.add(user)
    db.commit()
    return {"message": f"User '{form.username}' registered successfully"}


@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(username=form.username).first()
    if not user or not verify_password(form.password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


# ── Upload + full dedupe pipeline ─────────────────────────────────────────────

@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    user = db.query(models.User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    results = []

    for upload_file in files:

        # 1. Read into memory for size check
        contents = await upload_file.read()
        size = len(contents)
        await upload_file.seek(0)

        # 2. Validate type and size
        try:
            validate_file(upload_file.filename, size)
        except ValueError as e:
            results.append({"file": upload_file.filename, "error": str(e)})
            continue

        # 3. Save to disk with UUID filename
        stored_filename, path, _ = await save_file(upload_file)
        file_type = get_file_type(upload_file.filename)

        # 4. Two-stage hashing (text files are normalised inside hashing.py)
        try:
            p_hash = partial_hash(path)
            f_hash = full_hash(path)
        except RuntimeError as e:
            results.append({"file": upload_file.filename, "error": str(e)})
            continue

        # 5. Size-tolerance grouping to narrow candidates
        existing_files = db.query(models.File).filter_by(user_id=user.id).all()

        class _Temp:
            id = -1

        _Temp.size = size

        candidate_groups = size_tolerance_group(existing_files + [_Temp()])
        candidates = []
        for group in candidate_groups:
            if any(getattr(f, "id", None) == -1 for f in group):
                candidates = [f for f in group if getattr(f, "id", None) != -1]
                break

        # 6. Exact duplicate check via full hash
        class _HashTemp:
            pass

        ht = _HashTemp()
        ht.full_hash = f_hash

        exact_dup = find_exact_duplicate(ht, candidates)
        is_duplicate = exact_dup is not None
        duplicate_of_id = exact_dup.id if exact_dup else None

        # 7. Generate AI embedding
        embedding = get_embedding_for_file(path, file_type)
        embedding_json = json.dumps(embedding) if embedding else None

        # 8. AI similarity — compute cosine similarity against ALL user files
        #    Always store the best score so the UI can always display it.
        #    The "near_duplicate" flag is raised at >0.92.
        best_similarity_score = None
        best_similarity_filename = None
        ai_similarity_result = None

        if not is_duplicate and embedding:
            best_score = -1.0
            for candidate in existing_files:
                if candidate.embedding_vector:
                    try:
                        cand_vec = json.loads(candidate.embedding_vector)
                        sim = float(cos_sim([embedding], [cand_vec])[0][0])
                        if sim > best_score:
                            best_score = sim
                            best_similarity_score = round(sim, 4)
                            best_similarity_filename = candidate.original_filename
                    except Exception:
                        continue

            if best_score >= 0.92:
                ai_similarity_result = {
                    "near_duplicate_of": best_similarity_filename,
                    "similarity_score": best_similarity_score,
                    "similarity_percent": f"{best_similarity_score * 100:.1f}%",
                }

        # 9. Save to database
        file_record = models.File(
            original_filename=upload_file.filename,
            stored_filename=stored_filename,
            size=size,
            file_type=file_type,
            path=path,
            partial_hash=p_hash,
            full_hash=f_hash,
            is_duplicate=is_duplicate,
            duplicate_of_id=duplicate_of_id,
            similarity_score=best_similarity_score,
            similar_to_filename=best_similarity_filename,
            embedding_vector=embedding_json,
            user_id=user.id,
        )
        db.add(file_record)
        db.commit()
        db.refresh(file_record)

        # 10. Re-run clustering across all user files after every upload
        all_user_files = db.query(models.File).filter_by(user_id=user.id).all()
        cluster_map = cluster_files(all_user_files)
        for fid, cid in cluster_map.items():
            db.query(models.File).filter_by(id=fid).update({"cluster_id": cid})
        db.commit()

        result = {
            "file": upload_file.filename,
            "type": file_type,
            "size_bytes": size,
            "is_exact_duplicate": is_duplicate,
            "cluster_id": cluster_map.get(file_record.id),
            "similarity_score": best_similarity_score,
            "similarity_percent": (
                f"{best_similarity_score * 100:.1f}%" if best_similarity_score is not None else None
            ),
            "most_similar_to": best_similarity_filename,
        }
        if duplicate_of_id:
            result["duplicate_of_id"] = duplicate_of_id
        if ai_similarity_result:
            result["ai_near_duplicate"] = ai_similarity_result

        results.append(result)

    return {"uploaded": results}


# ── Files list ────────────────────────────────────────────────────────────────

@app.get("/files")
def list_files(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    user = db.query(models.User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found — please log in again")
    files = db.query(models.File).filter_by(user_id=user.id).all()
    return {
        "files": [
            {
                "id": f.id,
                "filename": f.original_filename,
                "size": f.size,
                "type": f.file_type,
                "is_duplicate": f.is_duplicate,
                "duplicate_of_id": f.duplicate_of_id,
                "similarity_score": f.similarity_score,
                "similarity_percent": (
                    f"{f.similarity_score * 100:.1f}%" if f.similarity_score is not None else None
                ),
                "similar_to_filename": f.similar_to_filename,
                "cluster_id": f.cluster_id,
                "uploaded_at": str(f.uploaded_at),
            }
            for f in files
        ]
    }


# ── Grouped files view ────────────────────────────────────────────────────────

@app.get("/files/groups")
def list_file_groups(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """
    Return files grouped by cluster_id.
    Files without embeddings (cluster_id = None) are listed separately.
    """
    user = db.query(models.User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    files = db.query(models.File).filter_by(user_id=user.id).all()

    clusters: dict[int, list] = {}
    ungrouped = []

    for f in files:
        entry = {
            "id": f.id,
            "filename": f.original_filename,
            "size": f.size,
            "type": f.file_type,
            "is_duplicate": f.is_duplicate,
            "duplicate_of_id": f.duplicate_of_id,
            "similarity_score": f.similarity_score,
            "similarity_percent": (
                f"{f.similarity_score * 100:.1f}%" if f.similarity_score is not None else None
            ),
            "similar_to_filename": f.similar_to_filename,
            "cluster_id": f.cluster_id,
            "uploaded_at": str(f.uploaded_at),
        }
        if f.cluster_id is not None:
            clusters.setdefault(f.cluster_id, []).append(entry)
        else:
            ungrouped.append(entry)

    for cid in clusters:
        clusters[cid].sort(
            key=lambda x: x["similarity_score"] if x["similarity_score"] is not None else 0,
            reverse=True,
        )

    groups = [
        {"cluster_id": cid, "files": members}
        for cid, members in sorted(clusters.items())
    ]

    return {
        "groups": groups,
        "ungrouped": ungrouped,
        "total_files": len(files),
        "total_groups": len(groups),
    }


# ── Pairwise similarity matrix ────────────────────────────────────────────────

@app.get("/files/similarity-matrix")
def similarity_matrix(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """Return all pairwise cosine similarity scores > 0.5 for the current user's files."""
    user = db.query(models.User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    files = db.query(models.File).filter_by(user_id=user.id).all()
    matrix = compute_similarity_matrix(files)

    file_map = {f.id: f.original_filename for f in files}

    return {
        "pairs": [
            {
                "file_a_id": a,
                "file_a_name": file_map.get(a),
                "file_b_id": b,
                "file_b_name": file_map.get(b),
                "similarity_score": score,
                "similarity_percent": f"{score * 100:.1f}%",
            }
            for (a, b), score in sorted(matrix.items(), key=lambda x: -x[1])
        ]
    }


# ── Delete file ───────────────────────────────────────────────────────────────

@app.delete("/files/{file_id}")
def delete_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    user = db.query(models.User).filter_by(username=current_user).first()
    file = db.query(models.File).filter_by(id=file_id, user_id=user.id).first()

    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    if os.path.exists(file.path):
        os.remove(file.path)

    db.query(models.File).filter_by(duplicate_of_id=file_id).update({
        "duplicate_of_id": None,
        "is_duplicate": False,
        "similarity_score": None,
        "similar_to_filename": None,
    })

    db.delete(file)
    db.commit()

    return {"message": f"File '{file.original_filename}' deleted successfully"}
