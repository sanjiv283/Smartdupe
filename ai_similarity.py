import os
import json
import numpy as np
import fitz
from PIL import Image
import imagehash

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# ── Model loading ─────────────────────────────────────────────────────────────
# Try Cohere first (production/Render), fall back to sentence-transformers (local)

_cohere_client = None
_st_model = None

def _get_cohere():
    global _cohere_client
    if _cohere_client is None:
        api_key = os.getenv("COHERE_API_KEY", "")
        if api_key:
            try:
                import cohere
                _cohere_client = cohere.Client(api_key)
            except Exception:
                pass
    return _cohere_client

def _get_st_model():
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pass
    return _st_model


# ── Core embedding function ───────────────────────────────────────────────────

def get_text_embedding(text: str) -> np.ndarray | None:
    """
    Generate a semantic embedding vector for a text string.
    Uses Cohere if API key is available, otherwise falls back to
    sentence-transformers (all-MiniLM-L6-v2) with cosine similarity.
    Returns a numpy array or None if both fail.
    """
    text = text[:2048]  # truncate for token limits

    # Try Cohere first
    co = _get_cohere()
    if co:
        try:
            response = co.embed(
                texts=[text],
                model="embed-english-v3.0",
                input_type="search_document",
            )
            return np.array(response.embeddings[0])
        except Exception:
            pass

    # Fall back to sentence-transformers + cosine similarity
    model = _get_st_model()
    if model:
        try:
            return model.encode([text])[0]
        except Exception:
            pass

    return None


# ── Text (PDF) helpers ────────────────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        return "".join(page.get_text() for page in doc)
    except Exception as e:
        raise RuntimeError(f"Could not extract text from PDF: {pdf_path}") from e


def text_similarity(text1: str, text2: str) -> float:
    """
    Return cosine similarity (0–1) between two text strings.
    Uses sentence-transformers embeddings + sklearn cosine_similarity.
    """
    v1 = get_text_embedding(text1)
    v2 = get_text_embedding(text2)
    if v1 is None or v2 is None:
        return 0.0
    return float(cosine_similarity([v1], [v2])[0][0])


# ── Image helpers ─────────────────────────────────────────────────────────────

def image_phash(path: str) -> str:
    """Compute perceptual hash (pHash) of an image as a hex string."""
    try:
        img = Image.open(path)
        return str(imagehash.phash(img))
    except Exception as e:
        raise RuntimeError(f"Could not compute pHash for image: {path}") from e


def image_phash_similarity(hash1: str, hash2: str) -> float:
    """
    Compare two pHash strings using Hamming distance.
    Returns 0–1 similarity score (1 = identical, 0 = completely different).
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    hamming = h1 - h2
    return 1.0 - (hamming / 64.0)


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_files(file_records, n_clusters: int = None) -> dict:
    """
    Cluster files by their stored embedding vectors using KMeans.
    Uses cosine similarity implicitly through the embedding space.

    Args:
        file_records: list of File ORM objects with embedding_vector set.
        n_clusters:   number of clusters; defaults to sqrt(n/2) heuristic.

    Returns:
        dict mapping file.id -> cluster_id
    """
    files_with_embeddings = [f for f in file_records if f.embedding_vector]

    if len(files_with_embeddings) < 2:
        return {f.id: 0 for f in files_with_embeddings}

    vectors = np.array([
        json.loads(f.embedding_vector) for f in files_with_embeddings
    ])

    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(len(vectors) / 2)))

    n_clusters = min(n_clusters, len(vectors))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(vectors)

    return {f.id: int(label) for f, label in zip(files_with_embeddings, labels)}


# ── Entry point ───────────────────────────────────────────────────────────────

def get_embedding_for_file(file_path: str, file_type: str) -> list | None:
    """
    Generate a vector embedding for a file depending on its type.

    Pipeline:
    - PDF:   extract text → sentence-transformer embedding → cosine similarity ready
    - Image: compute pHash → embed as text string → cosine similarity ready
    - Other: returns None

    Returns a plain Python list (JSON-serialisable) or None if unsupported.
    """
    try:
        if file_type == "pdf":
            text = extract_text(file_path)
            if text.strip():
                vec = get_text_embedding(text)
                if vec is not None:
                    return vec.tolist()

        elif file_type == "image":
            phash_str = image_phash(file_path)
            # Encode pHash as a descriptive string so it enters the same
            # embedding space as text documents for unified clustering
            vec = get_text_embedding(f"image perceptual hash {phash_str}")
            if vec is not None:
                return vec.tolist()

    except Exception:
        pass

    return None
