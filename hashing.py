import re
import xxhash


def _normalise_bytes(data: bytes) -> bytes:
    """
    Normalise raw bytes for TEXT-like files so that case / whitespace
    changes don't produce a different hash.

    Strategy:
      - Decode as UTF-8 (ignore errors)
      - Lowercase
      - Collapse whitespace
      - Re-encode to UTF-8

    For binary files (PDF, images) this normalisation is skipped and the
    original bytes are hashed directly — content equality matters there.
    """
    try:
        text = data.decode("utf-8", errors="ignore")
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text.encode("utf-8")
    except Exception:
        return data


# ── Text-file extensions that should be normalised before hashing ─────────────
_TEXT_EXTS = {".txt", ".md", ".csv", ".html", ".htm", ".json", ".xml", ".log"}


def _should_normalise(file_path: str) -> bool:
    import os
    ext = os.path.splitext(file_path)[1].lower()
    return ext in _TEXT_EXTS


def partial_hash(file_path: str) -> str:
    """
    Hash only the first 4 KB of a file (after optional normalisation).
    Fast first-pass filter — files with different partial hashes are
    definitely not duplicates.
    """
    h = xxhash.xxh64()
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(4096)
        if _should_normalise(file_path):
            chunk = _normalise_bytes(chunk)
        h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        raise RuntimeError(f"Could not read file for partial hash: {file_path}") from e


def full_hash(file_path: str) -> str:
    """
    Hash the entire file (after optional normalisation for text files).
    Only called when partial hashes match — confirms an exact duplicate.
    """
    h = xxhash.xxh64()
    try:
        if _should_normalise(file_path):
            # For text files read everything, normalise, then hash
            with open(file_path, "rb") as f:
                data = f.read()
            h.update(_normalise_bytes(data))
        else:
            # Binary files: stream in chunks
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        raise RuntimeError(f"Could not read file for full hash: {file_path}") from e
