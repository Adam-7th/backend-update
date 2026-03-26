from fastapi import APIRouter, HTTPException
from pathlib import Path
from services.report_service import generate_dashboard_data
from utils.helpers import sanitize_filename
from azure.storage.blob import BlobServiceClient
import os
import tempfile

router = APIRouter()

# Load Azure Storage config from environment
AZURE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "lab-data")
RAW_FOLDER = Path(__file__).resolve().parents[2] / "data" / "raw"


def _resolve_source_filename(file_name: str) -> str:
    safe_name = sanitize_filename(file_name)
    if safe_name.endswith("_report.pdf"):
        return safe_name[: -len("_report.pdf")]
    return safe_name

@router.get("/dashboard/{file_name}")
async def dashboard(file_name: str):
    """
    Generate dashboard for a file stored in Azure Blob Storage.
    """
    source_name = _resolve_source_filename(file_name)
    tmp_path: Path | None = None
    source_local_path = RAW_FOLDER / source_name

    if source_local_path.exists():
        return generate_dashboard_data(source_local_path, source_name)

    if not AZURE_ACCOUNT_NAME or not AZURE_ACCOUNT_KEY:
        raise HTTPException(
            status_code=404,
            detail="File not found locally and Azure Storage is not configured",
        )

    blob_service_client = BlobServiceClient(
        account_url=f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net",
        credential=AZURE_ACCOUNT_KEY,
    )
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    # Download file from Azure to a temporary location
    try:
        blob_client = container_client.get_blob_client(source_name)
        if not blob_client.exists():
            raise HTTPException(status_code=404, detail="File not found in Azure container")

        suffix = Path(source_name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = Path(tmp_file.name)
            download_stream = blob_client.download_blob()
            tmp_file.write(download_stream.readall())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download file from Azure: {e}")
    finally:
        blob_service_client.close()

    # Generate dashboard data (charts + AI insights) using existing report_service
    dashboard_data = generate_dashboard_data(tmp_path, source_name)

    # Clean up temp file
    if tmp_path:
        tmp_path.unlink(missing_ok=True)

    return dashboard_data