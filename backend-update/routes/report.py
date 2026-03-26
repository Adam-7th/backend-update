from fastapi import APIRouter, HTTPException, Query, Response
from pathlib import Path
import os

from azure.storage.blob.aio import BlobServiceClient
from dotenv import load_dotenv

from services.report_pdf_service import generate_pdf_report
from utils.helpers import sanitize_filename

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

AZURE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "lab-data")
REPORTS_CONTAINER_NAME = os.getenv("AZURE_REPORTS_CONTAINER_NAME", "lab-reports")

TEMP_FOLDER = Path(__file__).resolve().parents[2] / "data" / "temp"
TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
RAW_FOLDER = Path(__file__).resolve().parents[2] / "data" / "raw"


def _resolve_source_filename(file_name: str) -> str:
    safe_name = sanitize_filename(file_name)
    if safe_name.endswith("_report.pdf"):
        return safe_name[: -len("_report.pdf")]
    return safe_name


@router.get("/report")
async def report_with_query(file_name: str = Query(...)):
    return await report(file_name)


@router.get("/report/download")
async def download_report(file_name: str = Query(...), inline: bool = Query(False)):
    source_name = _resolve_source_filename(file_name)
    blob_name = f"{source_name}_report.pdf"

    blob_service_client = BlobServiceClient(
        account_url=f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net",
        credential=AZURE_ACCOUNT_KEY,
    )
    container_client = blob_service_client.get_container_client(REPORTS_CONTAINER_NAME)

    try:
        blob_client = container_client.get_blob_client(blob_name)
        exists = await blob_client.exists()
        if not exists:
            raise HTTPException(status_code=404, detail="Report PDF not found in Azure container")

        stream = await blob_client.download_blob()
        content = await stream.readall()
        disposition_type = "inline" if inline else "attachment"
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'{disposition_type}; filename="{blob_name}"'},
        )
    finally:
        await blob_service_client.close()


@router.get("/report/{file_name}")
async def report(file_name: str):
    source_name = _resolve_source_filename(file_name)
    local_path = TEMP_FOLDER / source_name
    source_local_path = RAW_FOLDER / source_name

    if source_local_path.exists():
        try:
            pdf_url = await generate_pdf_report(str(source_local_path), source_name)
            return {"pdf_url": pdf_url}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to generate report from local file: {exc}")

    blob_service_client = BlobServiceClient(
        account_url=f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net",
        credential=AZURE_ACCOUNT_KEY,
    )
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    try:
        blob_client = container_client.get_blob_client(source_name)
        exists = await blob_client.exists()
        if not exists:
            raise HTTPException(status_code=404, detail="File not found in Azure container")

        download_stream = await blob_client.download_blob()
        with open(local_path, "wb") as file_handle:
            file_handle.write(await download_stream.readall())

        pdf_url = await generate_pdf_report(str(local_path), source_name)
        return {"pdf_url": pdf_url}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {exc}")
    finally:
        local_path.unlink(missing_ok=True)
        await blob_service_client.close()