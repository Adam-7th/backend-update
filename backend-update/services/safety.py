# backend/services/safety.py
import os
import requests
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

COGNITIVE_ENDPOINT = os.getenv("AZURE_COGNITIVE_ENDPOINT")
COGNITIVE_KEY = os.getenv("AZURE_COGNITIVE_KEY")

def check_text_safety(text: str):
    """
    Check if text contains unsafe content using Azure Cognitive Content Safety
    """
    if not text.strip():
        return {"safe": True, "reason": "Empty text"}

    url = f"{COGNITIVE_ENDPOINT}/contentmoderator/moderate/v1.0/ProcessText/Screen"
    headers = {
        "Ocp-Apim-Subscription-Key": COGNITIVE_KEY,
        "Content-Type": "text/plain"
    }
    response = requests.post(url, headers=headers, data=text.encode("utf-8"))

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Content Safety API failed")

    result = response.json()
    # Example response fields: ClassificationType, Terms (flagged words), Category
    flagged = result.get("Classification", {}).get("ReviewRecommended", False)
    return {"safe": not flagged, "reason": result}

def check_file_safety(file_path: str):
    """
    Optional: Check file for unsafe content
    - For images: use Computer Vision or Content Moderator
    - For text/PDF/DOCX: extract text and call check_text_safety
    """
    from backend.services.data_service import read_file

    data_type, data = read_file(file_path)
    if data_type in ["txt", "pdf", "docx"]:
        return check_text_safety(data)
    elif data_type == "image":
        # Optional OCR + safety check
        ocr_text = data.get("ocr_text", "") if isinstance(data, dict) else ""
        return check_text_safety(ocr_text)
    else:
        return {"safe": True, "reason": "Non-text data"}