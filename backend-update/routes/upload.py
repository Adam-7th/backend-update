from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import os
import aiofiles
import asyncio
from azure.storage.blob.aio import BlobServiceClient
from dotenv import load_dotenv
import re
from typing import List, Dict

# Load env
BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BACKEND_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

AZURE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "lab-data")

router = APIRouter()

# Local storage
PROJECT_ROOT = BACKEND_DIR.parent
LOCAL_UPLOAD_FOLDER = PROJECT_ROOT / "data" / "raw"
os.makedirs(LOCAL_UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".txt", ".pdf", ".docx", ".xls", ".xlsx", ".png", ".jpg", ".jpeg"}

def sanitize_filename(filename: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)

def validate_file_extension(filename: str):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")


async def _upload_to_azure(local_path: Path, safe_filename: str):
    try:
        blob_service_client = BlobServiceClient(
            account_url=f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net",
            credential=AZURE_ACCOUNT_KEY,
        )
        try:
            container_client = blob_service_client.get_container_client(CONTAINER_NAME)
            blob_client = container_client.get_blob_client(safe_filename)

            with open(local_path, "rb") as data_file:
                payload = data_file.read()

            await asyncio.wait_for(blob_client.upload_blob(payload, overwrite=True), timeout=120)
        finally:
            await blob_service_client.close()
    except Exception:
        return


@router.get("/files")
async def list_available_files():
    local_names: set[str] = set()
    azure_names: set[str] = set()

    for path in LOCAL_UPLOAD_FOLDER.iterdir():
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            local_names.add(path.name)

    if AZURE_ACCOUNT_NAME and AZURE_ACCOUNT_KEY:
        try:
            blob_service_client = BlobServiceClient(
                account_url=f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net",
                credential=AZURE_ACCOUNT_KEY,
            )
            try:
                container_client = blob_service_client.get_container_client(CONTAINER_NAME)
                async for blob in container_client.list_blobs():
                    blob_name = Path(blob.name).name
                    if Path(blob_name).suffix.lower() in ALLOWED_EXTENSIONS:
                        azure_names.add(blob_name)
            finally:
                await blob_service_client.close()
        except Exception:
            pass

    ordered_names = sorted(local_names | azure_names, key=lambda name: name.lower())
    files: List[Dict[str, object]] = [
        {
            "name": name,
            "in_local": name in local_names,
            "in_azure": name in azure_names,
            "source": "both" if (name in local_names and name in azure_names) else "local" if name in local_names else "azure",
        }
        for name in ordered_names
    ]

    return {
        "files": files,
        "count": len(files),
    }

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")
    
    safe_filename = sanitize_filename(file.filename)
    validate_file_extension(safe_filename)
    
    local_path = LOCAL_UPLOAD_FOLDER / safe_filename

    # Save locally
    try:
        async with aiofiles.open(local_path, "wb") as out_file:
            while content := await file.read(1024 * 1024):
                await out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local write failed: {e}")

    # Upload to Azure in background so request returns quickly
    asyncio.create_task(_upload_to_azure(local_path, safe_filename))

    return {
        "filename": safe_filename,
        "message": "Uploaded locally and queued for Azure sync",
        "azure_blob_url": f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/{safe_filename}"
    }