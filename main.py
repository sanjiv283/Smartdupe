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
from similarity_tiers import get_tier, tier_summary

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

        # 4. Two-stage hashing (text files normalised inside hashing.py)
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

        # 8. AI similarity — search ALL user files, always store best score
        best_similarity_score = None
        best_similarity_filename = None

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

        # 9. Determine tier
        tier = get_tier(best_similarity_score, is_exact_duplicate=is_duplicate)

        # 10. Save to database
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

        # 11. Re-run clustering across all user files
        all_user_files = db.query(models.File).filter_by(user_id=user.id).all()
        cluster_map = cluster_files(all_user_files)
        for fid, cid in cluster_map.items():
            db.query(models.File).filter_by(id=fid).update({"cluster_id": cid})
        db.commit()

        result = {
            "file": upload_file.filename,
            "type": file_type,
            "size_bytes": size,
            "cluster_id": cluster_map.get(file_record.id),
            "most_similar_to": best_similarity_filename,
            # ── Three-tier similarity result ──────────────────────────────
            "similarity": {
                "score": best_similarity_score,
                "percent": tier["percent"],
                "tier": tier["tier"],
                "label": tier["label"],
                "emoji": tier["emoji"],
                "action": tier["action"],
                "summary": tier_summary(tier),
            },
        }

        if duplicate_of_id:
            result["duplicate_of_id"] = duplicate_of_id

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

    def _file_entry(f):
        tier = get_tier(f.similarity_score, is_exact_duplicate=f.is_duplicate)
        return {
            "id": f.id,
            "filename": f.original_filename,
            "size": f.size,
            "type": f.file_type,
            "duplicate_of_id": f.duplicate_of_id,
            "similar_to_filename": f.similar_to_filename,
            "cluster_id": f.cluster_id,
            "uploaded_at": str(f.uploaded_at),
            "similarity": {
                "score": f.similarity_score,
                "percent": tier["percent"],
                "tier": tier["tier"],
                "label": tier["label"],
                "emoji": tier["emoji"],
                "action": tier["action"],
                "summary": tier_summary(tier),
            },
        }

    return {"files": [_file_entry(f) for f in files]}


# ── Grouped files view ────────────────────────────────────────────────────────

@app.get("/files/groups")
def list_file_groups(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """
    Files grouped by semantic cluster, with tier labels applied to each file.

    Group structure:
      - 🔴 Exact duplicates  → tier = exact
      - 🟡 Near duplicates   → tier = near   (90–99%)
      - 🟢 Related files     → tier = related (70–89%)
      - ⚪ Unique            → tier = unique  (<70%)
    """
    user = db.query(models.User).filter_by(username=current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    files = db.query(models.File).filter_by(user_id=user.id).all()

    clusters: dict[int, list] = {}
    ungrouped = []

    for f in files:
        tier = get_tier(f.similarity_score, is_exact_duplicate=f.is_duplicate)
        entry = {
            "id": f.id,
            "filename": f.original_filename,
            "size": f.size,
            "type": f.file_type,
            "duplicate_of_id": f.duplicate_of_id,
            "similar_to_filename": f.similar_to_filename,
            "cluster_id": f.cluster_id,
            "uploaded_at": str(f.uploaded_at),
            "similarity": {
                "score": f.similarity_score,
                "percent": tier["percent"],
                "tier": tier["tier"],
                "label": tier["label"],
                "emoji": tier["emoji"],
                "action": tier["action"],
                "summary": tier_summary(tier),
            },
        }
        if f.cluster_id is not None:
            clusters.setdefault(f.cluster_id, []).append(entry)
        else:
            ungrouped.append(entry)

    # Sort each cluster: exact → near → related → unique, then by score desc
    tier_order = {"exact": 0, "near": 1, "related": 2, "unique": 3}
    for cid in clusters:
        clusters[cid].sort(key=lambda x: (
            tier_order.get(x["similarity"]["tier"], 9),
            -(x["similarity"]["score"] or 0),
        ))

    groups = [
        {
            "cluster_id": cid,
            "size": len(members),
            "dominant_tier": members[0]["similarity"]["tier"] if members else "unique",
            "files": members,
        }
        for cid, members in sorted(clusters.items())
    ]

    return {
        "groups": groups,
        "ungrouped": ungrouped,
        "total_files": len(files),
        "total_groups": len(groups),
        "tier_legend": {
            "🔴 Exact Duplicate": "100% match — safe to delete",
            "🟡 Near Duplicate":  "90–99% — review before deleting",
            "🟢 Related File":    "70–89% — same topic, not a duplicate",
            "⚪ Unique":          "Below 70% — independent file",
        },
    }


# ── Pairwise similarity matrix ────────────────────────────────────────────────

@app.get("/files/similarity-matrix")
def similarity_matrix(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """All pairwise cosine similarity scores > 0.5, with tier labels."""
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
                **get_tier(score),
            }
            for (a, b), score in sorted(matrix.items(), key=lambda x: -x[1])
        ],
        "tier_legend": {
            "🔴 Exact Duplicate": "100%",
            "🟡 Near Duplicate":  "90–99%",
            "🟢 Related File":    "70–89%",
            "⚪ Unique":          "<70%",
        },
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
