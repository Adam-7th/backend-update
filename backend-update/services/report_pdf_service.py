import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import generate_blob_sas, BlobSasPermissions

from services.report_service import (
    generate_dashboard_data,
    generate_pdf_report as build_pdf_report,
)


AZURE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
CONTAINER_NAME = os.getenv("AZURE_REPORTS_CONTAINER_NAME", "lab-reports")


async def generate_pdf_report(file_path: str, file_name: str) -> str:
    dashboard_data = generate_dashboard_data(file_path, file_name)
    summary = dashboard_data.get("summary", {})
    data_type = dashboard_data.get("data_type", "analysis")
    pdf_path = build_pdf_report(
        file_name,
        summary,
        dashboard_payload=dashboard_data,
        data_type=data_type,
    )
    blob_name = Path(pdf_path).name

    blob_service_client = BlobServiceClient(
        account_url=f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net",
        credential=AZURE_ACCOUNT_KEY,
    )
    try:
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        try:
            await container_client.create_container()
        except ResourceExistsError:
            pass

        blob_client = container_client.get_blob_client(blob_name)

        with open(pdf_path, "rb") as report_file:
            await blob_client.upload_blob(report_file, overwrite=True)
    finally:
        await blob_service_client.close()

    base_url = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/{blob_name}"

    try:
        sas_token = generate_blob_sas(
            account_name=AZURE_ACCOUNT_NAME,
            account_key=AZURE_ACCOUNT_KEY,
            container_name=CONTAINER_NAME,
            blob_name=blob_name,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=2),
        )
        return f"{base_url}?{sas_token}"
    except Exception:
        return base_url