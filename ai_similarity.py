import os
import json
import numpy as np
import fitz
from PIL import Image
import imagehash
import cohere

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Cohere client — API key must be set as COHERE_API_KEY environment variable
_co = cohere.Client(os.getenv("COHERE_API_KEY", ""))


# ── Text (PDF) helpers ────────────────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        return "".join(page.get_text() for page in doc)
    except Exception as e:
        raise RuntimeError(f"Could not extract text from PDF: {pdf_path}") from e


def text_embedding(text: str) -> np.ndarray:
    """
    Convert text to a semantic embedding vector using Cohere.
    Uses 'search_document' input type — best for comparing stored documents.
    Truncates to 2048 chars to stay within token limits on free tier.
    """
    truncated = text[:2048]
    response = _co.embed(
        texts=[truncated],
        model="embed-english-v3.0",
        input_type="search_document",
    )
    return np.array(response.embeddings[0])


def text_similarity(text1: str, text2: str) -> float:
    """Return cosine similarity (0–1) between two text strings."""
    v1 = text_embedding(text1)
    v2 = text_embedding(text2)
    return float(cosine_similarity([v1], [v2])[0][0])


# ── Image helpers ─────────────────────────────────────────────────────────────

def image_phash(path: str) -> str:
    """Compute perceptual hash of an image (as hex string)."""
    try:
        img = Image.open(path)
        return str(imagehash.phash(img))
    except Exception as e:
        raise RuntimeError(f"Could not compute pHash for image: {path}") from e


def image_phash_similarity(hash1: str, hash2: str) -> float:
    """
    Compare two pHash strings.
    Returns 0–1 similarity score (1 = identical, 0 = completely different).
    Max hamming distance for a 64-bit hash is 64.
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    hamming = h1 - h2
    return 1.0 - (hamming / 64.0)


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_files(file_records, n_clusters: int = None) -> dict:
    """
    Cluster files by their stored embedding vectors using KMeans.

    Args:
        file_records: list of File ORM objects that have embedding_vector set.
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


# ── Entry point: get embedding for any file type ──────────────────────────────

def get_embedding_for_file(file_path: str, file_type: str) -> list | None:
    """
    Generate a vector embedding for a file depending on its type.
    Returns a plain Python list (JSON-serialisable) or None if unsupported.

    - PDF:   extract text → Cohere embedding
    - Image: convert pHash to string → Cohere embedding
    - Other: returns None (no embedding, skipped in clustering)
    """
    try:
        if file_type == "pdf":
            text = extract_text(file_path)
            if text.strip():
                return text_embedding(text).tolist()

        elif file_type == "image":
            # Encode the perceptual hash as a short descriptive string
            # so Cohere can embed it into the same vector space
            phash_str = image_phash(file_path)
            return text_embedding(f"image perceptual hash {phash_str}").tolist()

    except Exception:
        pass  # If embedding fails, file is skipped in clustering

    return None
