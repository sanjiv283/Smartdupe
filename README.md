# SmartDupe — AI-Powered File Deduplication Engine

> **Hackathon Track: AI / ML**  
> SmartDupe uses semantic embeddings, cosine similarity, and unsupervised clustering to detect exact *and* near-duplicate files — even when filenames, casing, or whitespace differ.

---

## What Makes This AI/ML?

| Layer | Technique |
|---|---|
| Text embeddings | Cohere `embed-english-v3.0` (prod) / `all-MiniLM-L6-v2` (local) |
| Image similarity | Perceptual hashing (pHash) → embedded as text vector |
| Near-duplicate detection | Cosine similarity on sentence-transformer vectors |
| File grouping | KMeans clustering on embedding space |
| Pairwise similarity | Full cosine similarity matrix exposed via API |

---

## Problem Solved

Cloud storage and team shared drives accumulate duplicate files silently:
- Same document uploaded with different capitalisation → missed by naive hash
- Scanned PDFs that are 95% identical → not caught by exact hash
- Images resized or re-saved → missed by filename/size checks

SmartDupe catches **all three** by combining:
1. **Exact deduplication** — normalised xxHash (case/whitespace-insensitive for text)
2. **AI near-duplicate detection** — cosine similarity on semantic embeddings
3. **Semantic clustering** — groups similar files visually even if they aren't duplicates

---

## API Endpoints

### Auth
| Method | Path | Description |
|---|---|---|
| POST | `/register` | Create account |
| POST | `/login` | Get JWT bearer token |

### Files
| Method | Path | Description |
|---|---|---|
| POST | `/upload` | Upload 1–N files; returns similarity scores + cluster IDs |
| GET | `/files` | List all files with similarity % and cluster |
| GET | `/files/groups` | **Files grouped by semantic cluster** (AI/ML demo endpoint) |
| GET | `/files/similarity-matrix` | All pairwise cosine similarity scores > 0.5 |
| DELETE | `/files/{id}` | Delete a file |

---

## Upload Response (example)

```json
{
  "uploaded": [
    {
      "file": "Report_FINAL.pdf",
      "type": "pdf",
      "size_bytes": 204800,
      "is_exact_duplicate": false,
      "cluster_id": 2,
      "similarity_score": 0.9741,
      "similarity_percent": "97.4%",
      "most_similar_to": "report_final.pdf",
      "ai_near_duplicate": {
        "near_duplicate_of": "report_final.pdf",
        "similarity_score": 0.9741,
        "similarity_percent": "97.4%"
      }
    }
  ]
}
```

---

## Groups Endpoint (AI/ML demo)

`GET /files/groups` returns files grouped by semantic cluster:

```json
{
  "groups": [
    {
      "cluster_id": 0,
      "files": [
        { "filename": "contract_v1.pdf", "similarity_score": 0.97, "similarity_percent": "97.0%" },
        { "filename": "Contract_V1.pdf", "similarity_score": 0.97, "similarity_percent": "97.0%" }
      ]
    },
    {
      "cluster_id": 1,
      "files": [
        { "filename": "logo.png", "similarity_score": 0.88, "similarity_percent": "88.0%" }
      ]
    }
  ],
  "ungrouped": [],
  "total_files": 3,
  "total_groups": 2
}
```

---

## Key Bug Fixes (vs original)

### 1. Similarity % was not showing
**Root cause:** `best_similarity_score` was only stored when `best_score > 0.92`.  
**Fix:** Score is now **always stored** in the DB and returned in every response. The `> 0.92` threshold only controls the `ai_near_duplicate` alert flag.

### 2. Case change treated as unique file
**Root cause:** No text normalisation before hashing or embedding.  
**Fix (hashing):** Text files (`.txt`, `.md`, `.csv`, etc.) are lowercased and whitespace-collapsed before hashing — so `Report.txt` and `report.txt` with identical content produce the same hash.  
**Fix (embedding):** `normalise_text()` in `ai_similarity.py` lowercases + strips punctuation/whitespace before every embedding call — so case variants embed to nearly identical vectors and score ≥ 0.99 cosine similarity.

### 3. Similarity searched only size-grouped candidates
**Root cause:** Near-duplicate search was limited to size-grouped candidates, missing same-content files of slightly different sizes.  
**Fix:** Similarity now searches **all user files**, not just the size bucket.

---

## Local Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Visit http://localhost:8000/docs
```

Set `COHERE_API_KEY` in `.env` for production-quality embeddings, or leave unset to use the local `all-MiniLM-L6-v2` model.

---

## Render Deployment

1. Push to GitHub
2. Create a new **Web Service** on Render pointing to your repo
3. Use `render.yaml` — it provisions a free Postgres DB automatically
4. Add `COHERE_API_KEY` manually in the Render dashboard → Environment tab

---

## Tech Stack

- **FastAPI** — async REST API
- **SQLAlchemy + PostgreSQL** — persistent storage (SQLite for local dev)
- **Cohere / sentence-transformers** — semantic text embeddings
- **scikit-learn** — cosine similarity, KMeans clustering
- **imagehash + Pillow** — perceptual image hashing
- **PyMuPDF (fitz)** — PDF text extraction
- **xxhash** — fast exact-duplicate hashing
- **bcrypt + PyJWT** — auth
