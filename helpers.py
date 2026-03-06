import os
import uuid

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".txt", ".docx"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def get_file_type(filename: str) -> str:
    """Categorise file as image, pdf, or other."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "image"
    if ext == ".pdf":
        return "pdf"
    return "other"


def validate_file(filename: str, size: int):
    """Raise ValueError if file type or size is not allowed."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type '{ext}' is not allowed.")
    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File exceeds maximum size of 50 MB.")


async def save_file(file) -> tuple[str, str, int]:
    """
    Async-safe file save.
    Returns (stored_filename, full_path, file_size_bytes).
    Uses UUID prefix to prevent filename collisions.
    """
    contents = await file.read()
    size = len(contents)

    ext = os.path.splitext(file.filename)[1].lower()
    stored_filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_FOLDER, stored_filename)

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    with open(path, "wb") as f:
        f.write(contents)

    return stored_filename, path, size
