import xxhash


def partial_hash(file_path: str) -> str:
    """
    Hash only the first 4 KB of a file.
    Fast first-pass filter — files with different partial hashes are definitely not duplicates.
    """
    h = xxhash.xxh64()
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(4096)
        h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        raise RuntimeError(f"Could not read file for partial hash: {file_path}") from e


def full_hash(file_path: str) -> str:
    """
    Hash the entire file.
    Only called when partial hashes match — confirms an exact duplicate.
    """
    h = xxhash.xxh64()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        raise RuntimeError(f"Could not read file for full hash: {file_path}") from e
