import os
import pandas as pd
from PIL import Image, ImageStat
import pdfplumber
from docx import Document
from collections import Counter
import re
import json
from pathlib import Path
from typing import Tuple, Union

def read_file(file_path: Union[str, Path]) -> Tuple[str, Union[pd.DataFrame, str, Image.Image, None]]:
    """
    Read CSV, TXT, IMAGE, PDF, DOCX, or Excel files.
    Returns (data_type, data)
    """
    try:
        file_path = str(file_path)
        if file_path.endswith(".csv"):
            return "csv", pd.read_csv(file_path)
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return "txt", f.read()
        elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(file_path)
            return "image", img
        elif file_path.lower().endswith(".pdf"):
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return "pdf", text
        elif file_path.lower().endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
            return "docx", text
        elif file_path.lower().endswith((".xls", ".xlsx")):
            return "excel", pd.read_excel(file_path)
        else:
            return "unknown", None
    except Exception as e:
        return "error", str(e)


def summarize_data(data_type: str, data, file_name="unknown", save_json=True) -> dict:
    """
    Summarize data differently based on type.
    Adds metadata, anomalies, correlations for numeric CSV/Excel,
    trends, focus areas, confidence, protocol text, keywords for text,
    image hints, and optional JSON saving.
    """
    summary_result = {}

    # ----------------- NUMERIC FILES -----------------
    if data_type in ["csv", "excel"]:
        numeric = data.select_dtypes(include='number')
        stats = numeric.describe().to_dict()
        missing = numeric.isnull().sum().to_dict()
        corr = numeric.corr().to_dict() if numeric.shape[1] > 1 else {}

        anomalies = {}
        for col in numeric.columns:
            mean = numeric[col].mean()
            std = numeric[col].std()
            anomalies[col] = int(((numeric[col] - mean).abs() > 3 * std).sum())

        metadata = {
            "rows": data.shape[0],
            "columns": data.shape[1],
            "missing_values": missing,
            "anomalies_detected": anomalies
        }

        # Focus areas, confidence, trends
        focus_areas, confidence, trends = {}, {}, {}
        for col in numeric.columns:
            notes = []
            if missing[col] > 0:
                notes.append(f"{missing[col]} missing values")
            if anomalies[col] > 0:
                notes.append(f"{anomalies[col]} anomalies")
            if numeric[col].std() > 0.1 * numeric[col].mean():
                notes.append("high variance")
            focus_areas[col] = ", ".join(notes) if notes else "no issues"
            confidence[col] = round(1 - (anomalies[col] / data.shape[0]), 2)

            if len(numeric[col]) > 3:
                diff = numeric[col].diff().fillna(0)
                up = (diff > 0).sum()
                down = (diff < 0).sum()
                trends[col] = "up" if up > down else "down" if down > up else "stable"
            else:
                trends[col] = "stable"

        summary_result = {
            "stats": stats,
            "correlations": corr,
            "metadata": metadata,
            "focus_areas": focus_areas,
            "confidence": confidence,
            "trends": trends,
            "protocol_text": ""
        }

    # ----------------- TEXT FILES -----------------
    elif data_type in ["txt", "pdf", "docx"]:
        text_preview = data[:500]
        word_count = len(data.split())
        metadata = {"length_chars": len(data), "word_count": word_count}
        words = re.findall(r'\w+', data.lower())
        keywords = [w for w, _ in Counter(words).most_common(10)]

        summary_result = {
            "text_preview": text_preview,
            "metadata": metadata,
            "protocol_text": data,
            "focus_areas": {},
            "confidence": {},
            "keywords": keywords
        }

    # ----------------- IMAGE FILES -----------------
    elif data_type == "image":
        width, height = data.size
        mode = data.mode
        try:
            stat = ImageStat.Stat(data)
            avg_intensity = stat.mean[0] if stat.mean else None
        except:
            avg_intensity = None

        try:
            colors = data.convert("RGB").getcolors(1000000)
            dominant_color = max(colors, key=lambda x: x[0])[1] if colors else None
        except:
            dominant_color = None

        summary_result = {
            "width": width,
            "height": height,
            "mode": mode,
            "avg_intensity": avg_intensity,
            "dominant_color": dominant_color,
            "focus_areas": {},
            "confidence": {},
            "protocol_text": ""
        }

    # ----------------- ERROR HANDLING -----------------
    elif data_type == "error":
        summary_result = {"error": data}
    else:
        summary_result = {"message": "Unsupported file type"}

    # ----------------- SAVE SUMMARY -----------------
    if save_json:
        project_root = Path(__file__).resolve().parents[2]
        processed_dir = project_root / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        summary_path = processed_dir / f"{file_name}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_result, f, ensure_ascii=False, indent=2)

    return summary_result