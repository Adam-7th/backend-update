# backend/utils/helpers.py
import re
from pathlib import Path
from fastapi import HTTPException

# -------------------------------
# File & Path Helpers
# -------------------------------

def sanitize_filename(filename: str) -> str:
    """
    Remove unsafe characters from filename to prevent path traversal attacks
    """
    # Only allow alphanumeric, dash, underscore, dot
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    return safe_name

def validate_file_extension(filename: str, allowed_extensions=None):
    """
    Ensure file type is allowed
    """
    if allowed_extensions is None:
        allowed_extensions = {".csv", ".txt", ".pdf", ".docx", ".xls", ".xlsx", ".png", ".jpg", ".jpeg"}
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

def ensure_dir(path: Path):
    """
    Ensure the directory exists
    """
    path.mkdir(parents=True, exist_ok=True)
    return path