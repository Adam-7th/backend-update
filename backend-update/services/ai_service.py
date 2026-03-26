import os
import json
from typing import Any, Dict, List
from pathlib import Path

from openai import AzureOpenAI
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
REPORT_WRITING_STYLE = os.getenv("REPORT_WRITING_STYLE", "balanced").strip().lower()

JUDGING_CRITERIA = [
    "Performance 25%",
    "Innovation 25%",
    "Operational Impact 25%",
    "Responsible AI 25%",
]

if not OPENAI_ENDPOINT:
    OPENAI_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT")

if not OPENAI_KEY:
    OPENAI_KEY = os.getenv("FOUNDRY_API_KEY")


def _build_client() -> AzureOpenAI | None:
    if not OPENAI_ENDPOINT or not OPENAI_KEY:
        return None

    return AzureOpenAI(
        api_key=OPENAI_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version=OPENAI_API_VERSION,
    )


client = _build_client()


def _build_table_markdown(summary: Dict[str, Any]) -> str:
    stats = summary.get("stats", {}) or {}
    trends = summary.get("trends", {}) or {}
    focus = summary.get("focus_areas", {}) or {}
    confidence = summary.get("confidence", {}) or {}

    header = "| Metric | Mean | Std | Min | Max | Trend | Focus Area | Confidence |"
    separator = "|---|---:|---:|---:|---:|---|---|---:|"
    lines: List[str] = [header, separator]

    for metric_name, metric_stats in stats.items():
        lines.append(
            "| {metric} | {mean:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} | {trend} | {focus_area} | {conf} |".format(
                metric=metric_name,
                mean=float(metric_stats.get("mean", 0)),
                std=float(metric_stats.get("std", 0)),
                min_val=float(metric_stats.get("min", 0)),
                max_val=float(metric_stats.get("max", 0)),
                trend=trends.get(metric_name, "stable"),
                focus_area=focus.get(metric_name, "no issues"),
                conf=confidence.get(metric_name, "n/a"),
            )
        )

    return "\n".join(lines)


def _build_chart_ready_data(summary: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    stats = summary.get("stats", {}) or {}
    focus = summary.get("focus_areas", {}) or {}

    line = []
    histogram = []

    for metric_name, metric_stats in stats.items():
        line.append(
            {
                "metric": metric_name,
                "points": [
                    {"x": "min", "y": float(metric_stats.get("min", 0))},
                    {"x": "q1", "y": float(metric_stats.get("25%", 0))},
                    {"x": "median", "y": float(metric_stats.get("50%", 0))},
                    {"x": "q3", "y": float(metric_stats.get("75%", 0))},
                    {"x": "max", "y": float(metric_stats.get("max", 0))},
                ],
            }
        )
        histogram.append(
            {
                "metric": metric_name,
                "bins": [
                    {"range": "min-q1", "value": float(metric_stats.get("25%", 0)) - float(metric_stats.get("min", 0))},
                    {"range": "q1-median", "value": float(metric_stats.get("50%", 0)) - float(metric_stats.get("25%", 0))},
                    {"range": "median-q3", "value": float(metric_stats.get("75%", 0)) - float(metric_stats.get("50%", 0))},
                    {"range": "q3-max", "value": float(metric_stats.get("max", 0)) - float(metric_stats.get("75%", 0))},
                ],
            }
        )

    issue_count = sum(1 for _, issue in focus.items() if issue and issue != "no issues")
    ok_count = max(len(focus) - issue_count, 0)
    pie = [
        {"label": "Needs Attention", "value": issue_count},
        {"label": "Stable", "value": ok_count},
    ]

    return {"line": line, "histogram": histogram, "pie": pie}


def _fallback_dashboard_payload(file_name: str, file_type: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    metadata = summary.get("metadata", {}) or {}
    stats = summary.get("stats", {}) or {}
    trends = summary.get("trends", {}) or {}
    focus = summary.get("focus_areas", {}) or {}
    confidence = summary.get("confidence", {}) or {}

    notable_trends = ", ".join([f"{k}:{v}" for k, v in trends.items()]) if trends else "no dominant trend"
    focus_items = ", ".join([f"{k}:{v}" for k, v in focus.items()]) if focus else "no critical focus areas"

    formal_sections = [
        f"Executive Summary: This report explains the analysis results for {file_name} in clear language. The findings are based on the same data used in the website dashboard and do not introduce different calculations.",
        f"Data Overview: The dataset includes approximately {metadata.get('rows', 'unknown')} records and {metadata.get('columns', 'unknown')} variables. The analysis focused on trend behavior, consistency, unusual values, and confidence indicators.",
        f"Key Findings: Main focus areas include {focus_items}. Trend signals show {notable_trends}.",
        "AI-Powered Insights: The observed changes may reflect shifts in operating conditions, collection methods, or natural process variation. Results should be reviewed with context from the experiment timeline.",
        "Confidence & Reliability: The analysis is generally reliable for decision support, but confidence can be reduced when unusual values or missing entries are present.",
    ]
    text_explanation = "\n\n".join(formal_sections)

    metric_actions: List[str] = []
    for metric_name, metric_stats in stats.items():
        mean_value = float(metric_stats.get("mean", 0) or 0)
        std_value = float(metric_stats.get("std", 0) or 0)
        cv_value = abs(std_value / mean_value) if mean_value else 0.0
        trend_label = trends.get(metric_name, "stable")
        focus_note = str(focus.get(metric_name, "no issues"))
        confidence_value = confidence.get(metric_name, "n/a")

        if "anomal" in focus_note.lower() or "missing" in focus_note.lower() or cv_value >= 0.30:
            metric_actions.append(
                f"- {metric_name}: focus={focus_note}, trend={trend_label}, CV={cv_value:.3f}, confidence={confidence_value}. Prioritize calibration, anomaly review, and completeness checks before downstream decisions."
            )

    if not metric_actions:
        metric_actions.append("- No high-risk metric signals detected; maintain routine monitoring and periodic QA checks.")

    recommendations = (
        "Detailed Recommendations:\n"
        "Priority 1 - Immediate Monitoring:\n"
        + "\n".join(metric_actions[:6])
        + "\nPriority 2 - Investigation Actions:\n"
        "- Compare unusual spikes and drops against experiment logs, equipment events, and environment changes from the same time window.\n"
        "- Validate whether trend changes are expected process behavior or indicate drift needing correction.\n"
        "Priority 3 - Improvement Actions:\n"
        "- Standardize data collection checkpoints (calibration, timestamp validation, completeness checks) before each run.\n"
        "- Add weekly review of confidence and focus-area flags to improve reliability over time.\n"
        "Priority 4 - Governance and Safety:\n"
        "- Require human approval for decisions when confidence is low or quality warnings are present.\n"
        "- Keep a decision log linking each action to the supporting metric evidence and confidence value."
    )

    return {
        "text_explanation": text_explanation,
        "table_markdown": _build_table_markdown(summary),
        "charts_data": _build_chart_ready_data(summary),
        "recommendations": recommendations,
    }


def generate_lab_dashboard_payload(file_name: str, file_type: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    metadata = summary.get("metadata", {}) or {}
    stats = summary.get("stats", {}) or {}
    trends = summary.get("trends", {}) or {}
    focus_areas = summary.get("focus_areas", {}) or {}
    confidence = summary.get("confidence", {}) or {}
    keywords = summary.get("keywords", []) or []

    fallback = _fallback_dashboard_payload(file_name, file_type, summary)

    if client is None:
        return fallback

    prompt = f"""
You are Expelexia Lab Assistant AI.

Core identity to reflect in language:
"Expelexia Lab is an AI-powered data analysis platform that automatically analyzes user data and generates intelligent, human-friendly recommendations to support better decision-making."

Create website-consistent report content and return strict JSON only.

File info:
- Name: {file_name}
- Type: {file_type}
- Metadata: {metadata}
- Numeric Stats: {stats}
- Trends: {trends}
- Focus Areas: {focus_areas}
- Confidence: {confidence}
- Keywords / extracted insights: {keywords}

Critical consistency rule:
1) Use the same results already shown in the website analysis.
2) Do not add contradictory findings.
3) Expand explanation quality only.

Audience and tone requirements:
1) Audience is non-technical users.
2) Avoid technical jargon (for example: variance/correlation) unless rewritten in simple language.
3) Keep a formal, professional, human-friendly tone.
4) Prefer short, clear sentences with practical meaning.

text_explanation must include these labeled sections in order:
- Executive Summary
- Data Overview
- Visual Analysis
- Key Findings
- AI-Powered Insights
- Confidence & Reliability

Recommendation requirements (most important):
1) Provide practical recommendations with explicit action steps.
2) Use priority levels and include evidence + action + expected outcome.
3) Include these labels where relevant:
   - High Priority (Red): immediate attention required
   - Moderate Attention (Yellow): monitor and follow up
   - Low Concern (Green): stable, continue routine checks
4) Keep recommendations understandable to non-technical users.

Formatting requirements:
1) Build markdown table with columns: Metric, Mean, Std, Min, Max, Trend, Focus Area, Confidence.
2) Build charts_data JSON with keys: line, histogram, pie.
3) Base all claims only on provided inputs. If missing, explicitly state unavailable.
4) Writing style mode is {REPORT_WRITING_STYLE}: scientific=technical precision, judge-friendly=clarity/impact, balanced=blend both.

Return ONLY JSON with this exact shape:
{{
  "text_explanation": "...",
  "table_markdown": "...",
  "charts_data": {{"line": [], "histogram": [], "pie": []}},
  "recommendations": "..."
}}
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise lab analytics assistant that returns valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1200,
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content) if content else {}

        return {
            "text_explanation": parsed.get("text_explanation") or fallback["text_explanation"],
            "table_markdown": parsed.get("table_markdown") or fallback["table_markdown"],
            "charts_data": parsed.get("charts_data") or fallback["charts_data"],
            "recommendations": parsed.get("recommendations") or fallback["recommendations"],
        }
    except Exception:
        return fallback


def ai_generate_insights(file_name: str, summary_json: Dict[str, Any]) -> str:
    prompt = f"Provide human-readable insights for {file_name}:\n{summary_json}"

    if client is None:
        return "AI insight unavailable: Azure OpenAI credentials are not configured."

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.5,
        )
        message = response.choices[0].message.content
        return (message or "").strip()
    except Exception as exc:
        return f"AI insight generation failed: {exc}"