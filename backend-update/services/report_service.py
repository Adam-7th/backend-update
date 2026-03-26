from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import io
import os
from datetime import datetime
from typing import Dict, Any, Union, List, Tuple
import re
from copy import deepcopy
from threading import Lock
from xml.sax.saxutils import escape

from services.data_service import read_file, summarize_data
from services.ai_service import generate_lab_dashboard_payload


_DASHBOARD_CACHE: Dict[str, Dict[str, Any]] = {}
_DASHBOARD_CACHE_LOCK = Lock()


def _dashboard_cache_key(file_path: Union[str, Path], file_name: str) -> str:
    path_obj = Path(file_path)
    try:
        stat = path_obj.stat()
        return f"{file_name}|{path_obj.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except Exception:
        return f"{file_name}|{path_obj.resolve()}"

# ---------------- AI RECOMMENDATION ----------------
def generate_ai_recommendation(summary: dict) -> str:
    """Generate detailed, data-grounded recommendations for lab-style analysis output."""
    sections: List[str] = []
    metric_findings: List[str] = []
    prioritized_actions: List[str] = []

    metadata = summary.get("metadata", {}) or {}
    stats = summary.get("stats", {}) or {}
    trends = summary.get("trends", {}) or {}
    focus = summary.get("focus_areas", {}) or {}
    confidence = summary.get("confidence", {}) or {}

    rows = metadata.get("rows")
    columns = metadata.get("columns")
    if rows is not None and columns is not None:
        sections.append(
            f"Abstract: The analysis processed {rows} rows across {columns} columns to evaluate quality, trend behavior, and operational risk."
        )
    else:
        sections.append("Abstract: The analysis evaluated available signals for quality, trend behavior, and operational risk.")

    sections.append(
        "Introduction: Recommendations are mapped to Performance (25%), Innovation (25%), Operational Impact (25%), and Responsible AI (25%)."
    )

    high_risk_metrics: List[str] = []
    medium_risk_metrics: List[str] = []
    low_confidence_metrics: List[str] = []

    for metric_name, metric_stats in stats.items():
        mean_value = float(metric_stats.get("mean", 0) or 0)
        std_value = float(metric_stats.get("std", 0) or 0)
        min_value = float(metric_stats.get("min", 0) or 0)
        max_value = float(metric_stats.get("max", 0) or 0)
        q1_value = float(metric_stats.get("25%", min_value) or min_value)
        median_value = float(metric_stats.get("50%", q1_value) or q1_value)
        q3_value = float(metric_stats.get("75%", median_value) or median_value)

        spread = max_value - min_value
        iqr = q3_value - q1_value
        coefficient_of_variation = abs(std_value / mean_value) if mean_value else 0.0
        trend_label = str(trends.get(metric_name, "stable")).lower()
        focus_note = str(focus.get(metric_name, "no issues")).lower()
        confidence_score = confidence.get(metric_name)

        risk_score = 0
        if coefficient_of_variation >= 0.30:
            risk_score += 2
        elif coefficient_of_variation >= 0.15:
            risk_score += 1

        if "anomal" in focus_note or "missing" in focus_note:
            risk_score += 2
        elif "variance" in focus_note:
            risk_score += 1

        if trend_label in ("up", "down"):
            risk_score += 1

        if isinstance(confidence_score, (int, float)) and confidence_score < 0.90:
            risk_score += 2
            low_confidence_metrics.append(metric_name)

        if risk_score >= 4:
            high_risk_metrics.append(metric_name)
        elif risk_score >= 2:
            medium_risk_metrics.append(metric_name)

        metric_findings.append(
            f"- {metric_name}: mean={mean_value:.4f}, std={std_value:.4f}, range={min_value:.4f}..{max_value:.4f}, "
            f"IQR={iqr:.4f}, CV={coefficient_of_variation:.3f}, trend={trend_label}, focus={focus_note}, confidence={confidence_score if confidence_score is not None else 'n/a'}."
        )

        if "missing" in focus_note:
            prioritized_actions.append(
                f"[Performance 25%] {metric_name}: missing-value signal detected; apply completeness checks and imputation policy before retraining or threshold decisions."
            )
        if "anomal" in focus_note:
            prioritized_actions.append(
                f"[Performance 25%] {metric_name}: anomaly concentration detected; verify raw acquisition logs, sensor drift, and boundary thresholds."
            )
        if coefficient_of_variation >= 0.30:
            prioritized_actions.append(
                f"[Innovation 25%] {metric_name}: high variability (CV={coefficient_of_variation:.3f}); add adaptive control limits and confidence-banded recommendations."
            )
        if trend_label == "up":
            prioritized_actions.append(
                f"[Performance 25%] {metric_name}: upward trend detected; review upper tolerance limits and likely drift drivers."
            )
        elif trend_label == "down":
            prioritized_actions.append(
                f"[Performance 25%] {metric_name}: downward trend detected; verify if decline matches expected process behavior and acceptable bounds."
            )
        if isinstance(confidence_score, (int, float)) and confidence_score < 0.90:
            prioritized_actions.append(
                f"[Responsible AI 25%] {metric_name}: low confidence ({confidence_score:.3f}); require human review and explicit uncertainty disclosure before recommendations are operationalized."
            )

    if not stats:
        metric_findings.append("- Numeric metric statistics are unavailable; recommendations are limited to metadata-level checks.")

    if not prioritized_actions:
        prioritized_actions.append(
            "[Performance 25%] No high-risk indicator was detected in available statistics; continue routine monitoring with periodic recalibration checks."
        )

    sections.append("Materials and Methods: Data quality, distribution spread, quartiles, trend direction, focus-area tags, and confidence values were evaluated per metric.")
    sections.append("Results:")
    sections.extend(metric_findings)
    sections.append("Discussion:")
    sections.append(
        f"- Risk stratification: high risk={len(high_risk_metrics)}, medium risk={len(medium_risk_metrics)}, low-confidence={len(low_confidence_metrics)} metrics."
    )
    if high_risk_metrics:
        sections.append(f"- High-priority metrics: {', '.join(high_risk_metrics)}.")
    if medium_risk_metrics:
        sections.append(f"- Medium-priority metrics: {', '.join(medium_risk_metrics)}.")
    if low_confidence_metrics:
        sections.append(f"- Confidence-sensitive metrics: {', '.join(low_confidence_metrics)}.")
    sections.append(
        "- Recommendation confidence improves when calibration records, sampling context, and anomaly root-cause notes are attached to each run."
    )

    sections.append(
        "Conclusion: The dataset supports decision guidance when recommendations are executed with metric-level evidence, uncertainty disclosure, and safety checks."
    )
    sections.append("Recommended Next Actions (criteria-aligned):")

    deduplicated_actions: List[str] = []
    for action in prioritized_actions:
        if action not in deduplicated_actions:
            deduplicated_actions.append(action)

    for idx, action in enumerate(deduplicated_actions[:10], start=1):
        sections.append(f"{idx}) {action}")

    sections.append(
        f"{len(deduplicated_actions[:10]) + 1}) [Operational Impact 25%] Log metric-level evidence, confidence tags, and safety outcomes in a consistent audit trail for transparent reporting."
    )

    return "\n".join(sections)


def _build_table_preview(summary: dict) -> List[List[str]]:
    stats = summary.get("stats", {}) or {}
    rows: List[List[str]] = [["Metric", "Count", "Mean", "Std", "Min", "Max"]]

    for metric_name, metric_stats in stats.items():
        rows.append([
            str(metric_name),
            str(round(metric_stats.get("count", 0), 2)),
            str(round(metric_stats.get("mean", 0), 4)),
            str(round(metric_stats.get("std", 0), 4)),
            str(round(metric_stats.get("min", 0), 4)),
            str(round(metric_stats.get("max", 0), 4)),
        ])

    return rows


def _generate_chart_base64(summary: dict) -> Dict[str, str]:
    stats = summary.get("stats", {}) or {}
    if not stats:
        return {}

    metrics = []
    means = []
    for metric_name, metric_stats in stats.items():
        metrics.append(str(metric_name))
        means.append(float(metric_stats.get("mean", 0)))

    figure = plt.figure(figsize=(8, 3.6))
    axis = figure.add_subplot(111)
    axis.bar(metrics, means)
    axis.set_title("Average Value by Metric")
    axis.set_ylabel("Mean")
    axis.grid(axis="y", alpha=0.25)

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return {"mean_overview": f"data:image/png;base64,{encoded}"}


def _build_data_change_explanation(summary: dict) -> str:
    stats = summary.get("stats", {}) or {}
    trends = summary.get("trends", {}) or {}
    focus = summary.get("focus_areas", {}) or {}

    if not stats:
        return "Data-change analysis is not available for this document type."

    lines: List[str] = ["Data Change Analysis:"]
    for metric, metric_stats in stats.items():
        minimum = float(metric_stats.get("min", 0))
        maximum = float(metric_stats.get("max", 0))
        q1 = float(metric_stats.get("25%", minimum))
        median = float(metric_stats.get("50%", q1))
        q3 = float(metric_stats.get("75%", median))
        std_val = float(metric_stats.get("std", 0))
        mean_val = float(metric_stats.get("mean", 0))
        spread = maximum - minimum
        iqr = q3 - q1
        volatility = "high" if mean_val and std_val > abs(mean_val) * 0.1 else "moderate"
        trend = trends.get(metric, "stable")
        focus_note = focus.get(metric, "no issues")

        lines.append(
            f"- {metric}: value range {minimum:.3f} to {maximum:.3f} (spread {spread:.3f}), middle 50% spans {iqr:.3f}. "
            f"Trend is {trend}; volatility appears {volatility}; focus area is {focus_note}."
        )

    return "\n".join(lines)


def _build_chart_explanation(summary: dict) -> str:
    stats = summary.get("stats", {}) or {}
    trends = summary.get("trends", {}) or {}

    if not stats:
        return "Chart explanation is not available because no numeric metrics were detected."

    lines: List[str] = ["Chart Interpretation:"]
    for metric, metric_stats in stats.items():
        mean_val = float(metric_stats.get("mean", 0))
        median_val = float(metric_stats.get("50%", mean_val))
        trend = trends.get(metric, "stable")
        skew_hint = "right-skewed" if mean_val > median_val else "left-skewed" if mean_val < median_val else "balanced"
        lines.append(
            f"- {metric}: bar chart centers near mean {mean_val:.3f}; line trend indicates {trend}. "
            f"Mean/median relation suggests a {skew_hint} distribution."
        )

    lines.append("- Histogram bins show where value density is concentrated between quartile ranges.")
    lines.append("- Pie chart summarizes stable vs attention-needed metrics from focus-area rules.")
    return "\n".join(lines)


def _build_technical_review(summary: dict) -> str:
    stats = summary.get("stats", {}) or {}
    trends = summary.get("trends", {}) or {}
    focus = summary.get("focus_areas", {}) or {}
    confidence = summary.get("confidence", {}) or {}

    if not stats:
        return "Technical review is limited because structured numeric statistics are unavailable for this file type."

    lines: List[str] = ["Technical Review (Advanced):"]
    for metric_name, metric_stats in stats.items():
        mean_value = float(metric_stats.get("mean", 0) or 0)
        std_value = float(metric_stats.get("std", 0) or 0)
        min_value = float(metric_stats.get("min", 0) or 0)
        max_value = float(metric_stats.get("max", 0) or 0)
        q1_value = float(metric_stats.get("25%", min_value) or min_value)
        median_value = float(metric_stats.get("50%", q1_value) or q1_value)
        q3_value = float(metric_stats.get("75%", median_value) or median_value)
        iqr = q3_value - q1_value
        cv = abs(std_value / mean_value) if mean_value else 0.0
        trend = trends.get(metric_name, "stable")
        focus_note = focus.get(metric_name, "no issues")
        conf = confidence.get(metric_name, "n/a")

        lines.append(
            f"- {metric_name}: mean={mean_value:.4f}, std={std_value:.4f}, min={min_value:.4f}, max={max_value:.4f}, "
            f"IQR={iqr:.4f}, CV={cv:.4f}, trend={trend}, focus={focus_note}, confidence={conf}."
        )

    return "\n".join(lines)


def _build_technical_recommendations(summary: dict) -> str:
    stats = summary.get("stats", {}) or {}
    trends = summary.get("trends", {}) or {}
    focus = summary.get("focus_areas", {}) or {}
    confidence = summary.get("confidence", {}) or {}

    if not stats:
        return "Technical recommendations are limited to metadata checks due to lack of numeric metric statistics."

    actions: List[str] = []
    for metric_name, metric_stats in stats.items():
        mean_value = float(metric_stats.get("mean", 0) or 0)
        std_value = float(metric_stats.get("std", 0) or 0)
        cv = abs(std_value / mean_value) if mean_value else 0.0
        trend = str(trends.get(metric_name, "stable")).lower()
        focus_note = str(focus.get(metric_name, "no issues")).lower()
        conf = confidence.get(metric_name)

        if cv >= 0.30:
            actions.append(f"- {metric_name}: high coefficient of variation (CV={cv:.4f}); apply robust thresholding and recalibration checks.")
        if "anomal" in focus_note:
            actions.append(f"- {metric_name}: anomaly-heavy profile detected; run outlier root-cause analysis against raw capture logs.")
        if "missing" in focus_note:
            actions.append(f"- {metric_name}: missing-data signal detected; enforce completeness validation and imputation strategy before model use.")
        if trend == "up":
            actions.append(f"- {metric_name}: persistent upward trend; evaluate drift alarms and upper tolerance boundaries.")
        elif trend == "down":
            actions.append(f"- {metric_name}: persistent downward trend; validate process degradation assumptions and lower tolerance thresholds.")
        if isinstance(conf, (int, float)) and conf < 0.9:
            actions.append(f"- {metric_name}: low confidence ({conf:.3f}); require human validation before operational decisions.")

    if not actions:
        actions.append("- No severe technical risk signal detected; continue periodic calibration and drift monitoring.")

    return "Technical Recommendations (Advanced):\n" + "\n".join(actions[:12])


def _build_document_notes(file_name: str, data_type: str, summary: dict) -> str:
    metadata = summary.get("metadata", {}) or {}
    keywords = summary.get("keywords", []) or []

    lines = [
        "Document Context:",
        f"- Source file: {file_name}",
        f"- File type: {data_type}",
    ]

    if metadata:
        rows = metadata.get("rows", "n/a")
        columns = metadata.get("columns", "n/a")
        lines.append(f"- Structural profile: rows={rows}, columns={columns}")

    if keywords:
        lines.append(f"- Extracted keywords: {', '.join([str(k) for k in keywords[:12]])}")

    lines.append("- This report is generated for a general lab-review workflow and operational decision support.")
    return "\n".join(lines)


def _extract_formal_sections(text: str) -> Dict[str, str]:
    ordered_titles = [
        "Abstract",
        "Introduction",
        "Materials and Methods",
        "Results",
        "Discussion",
        "Conclusion",
    ]

    if not text:
        return {}

    pattern = re.compile(
        r"(Abstract|Introduction|Materials and Methods|Results|Discussion|Conclusion):",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return {}

    extracted: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        raw_title = match.group(1).strip()
        canonical_title = next((title for title in ordered_titles if title.lower() == raw_title.lower()), raw_title)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            extracted[canonical_title] = body

    return extracted


def _build_executive_summary(summary: dict, file_name: str) -> str:
    metadata = summary.get("metadata", {}) or {}
    focus = summary.get("focus_areas", {}) or {}
    trends = summary.get("trends", {}) or {}

    rows = metadata.get("rows", "unknown")
    columns = metadata.get("columns", "unknown")
    focus_items = [f"{metric}: {note}" for metric, note in list(focus.items())[:3]]
    trend_items = [f"{metric} {direction}" for metric, direction in list(trends.items())[:3]]

    points = [
        f"This report summarizes analysis results for {file_name} using the same data shown in the website dashboard.",
        f"The dataset contains {rows} records across {columns} variables.",
    ]

    if focus_items:
        points.append(f"Main areas needing attention: {', '.join(focus_items)}.")
    if trend_items:
        points.append(f"Observed trend signals: {', '.join(trend_items)}.")

    points.append("Recommendations focus on practical monitoring, early detection of unusual changes, and improving reliability of future data collection.")
    return "\n".join(points)


def _build_key_findings(summary: dict) -> List[str]:
    stats = summary.get("stats", {}) or {}
    trends = summary.get("trends", {}) or {}
    focus = summary.get("focus_areas", {}) or {}

    findings: List[str] = []
    for metric_name, metric_stats in list(stats.items())[:6]:
        mean_val = float(metric_stats.get("mean", 0) or 0)
        min_val = float(metric_stats.get("min", 0) or 0)
        max_val = float(metric_stats.get("max", 0) or 0)
        trend_val = trends.get(metric_name, "stable")
        focus_note = focus.get(metric_name, "no major issue")
        findings.append(
            f"{metric_name}: values range from {min_val:.2f} to {max_val:.2f}, average around {mean_val:.2f}. "
            f"Trend appears {trend_val}; review note: {focus_note}."
        )

    if not findings:
        findings.append("No detailed numeric findings were available for this file type, so interpretation is based on available metadata.")

    return findings


def _build_reliability_note(summary: dict) -> str:
    confidence = summary.get("confidence", {}) or {}
    focus = summary.get("focus_areas", {}) or {}

    low_confidence = [metric for metric, value in confidence.items() if isinstance(value, (int, float)) and value < 0.9]
    flagged_focus = [metric for metric, note in focus.items() if isinstance(note, str) and note.lower() != "no issues"]

    lines = ["The analysis is generally reliable when interpreted with the same context shown in the website dashboard."]

    if low_confidence:
        lines.append(f"Lower confidence was detected for: {', '.join(low_confidence)}. These areas should be reviewed by a human before final decisions.")
    if flagged_focus:
        lines.append(f"Attention is recommended for: {', '.join(flagged_focus)} due to unusual patterns or data quality concerns.")
    if not low_confidence and not flagged_focus:
        lines.append("No major reliability warning was detected in the available indicators.")

    lines.append("Missing values or unusual spikes can reduce certainty, so regular monitoring and data quality checks are advised.")
    return "\n".join(lines)


def _paragraphize_text(value: str) -> str:
    if not value:
        return ""
    safe = escape(str(value))
    return safe.replace("\n", "<br/>")


def _dashboard_alignment_rows(summary: dict) -> List[List[str]]:
    stats = summary.get("stats", {}) or {}
    trends = summary.get("trends", {}) or {}
    focus = summary.get("focus_areas", {}) or {}
    confidence = summary.get("confidence", {}) or {}

    rows: List[List[str]] = [["Metric", "Trend", "Focus Area", "Confidence"]]
    for metric_name in list(stats.keys())[:10]:
        conf = confidence.get(metric_name, "n/a")
        confidence_value = f"{conf:.3f}" if isinstance(conf, (int, float)) else str(conf)
        rows.append([
            str(metric_name),
            str(trends.get(metric_name, "stable")),
            str(focus.get(metric_name, "no issues")),
            confidence_value,
        ])
    return rows


def _parse_markdown_table(markdown_text: str) -> List[List[str]]:
    if not markdown_text:
        return []

    parsed_rows: List[List[str]] = []
    lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
    for line in lines:
        if "|" not in line:
            continue
        stripped = line.strip("|")
        cells = [cell.strip() for cell in stripped.split("|")]
        if not cells:
            continue

        is_separator = all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)
        if is_separator:
            continue
        parsed_rows.append(cells)

    if len(parsed_rows) <= 1:
        return []
    return parsed_rows


def _build_uniform_widths(total_width: float, columns: int, min_width: float = 48.0) -> List[float]:
    if columns <= 0:
        return []
    width = max(min_width, total_width / columns)
    widths = [width for _ in range(columns)]
    total = sum(widths)
    if total > total_width:
        ratio = total_width / total
        widths = [value * ratio for value in widths]
    return widths


def _build_apa_references(file_name: str) -> List[str]:
    current_year = datetime.now().year
    access_date = datetime.now().strftime("%B %d, %Y")
    return [
        f"Expelexia Analytics Team. ({current_year}). Automated dashboard analysis report for {file_name} [Internal report]. Expelexia.",
        "National Institute of Standards and Technology. (2023). Artificial Intelligence Risk Management Framework (AI RMF 1.0). NIST. Retrieved " + access_date + ", from https://www.nist.gov/itl/ai-risk-management-framework",
        "OECD. (2019). OECD principles on artificial intelligence. OECD AI Policy Observatory. Retrieved " + access_date + ", from https://oecd.ai/en/ai-principles",
        "ReportLab Inc. (n.d.). ReportLab user guide. Retrieved " + access_date + ", from https://www.reportlab.com/documentation/",
    ]


def _detect_report_domains(summary: dict, file_name: str, data_type: str) -> List[str]:
    keywords = [str(keyword).lower() for keyword in (summary.get("keywords", []) or [])]
    stats = summary.get("stats", {}) or {}
    metric_names = [str(name).lower() for name in stats.keys()]
    protocol_text = str(summary.get("protocol_text", "") or "").lower()
    metadata = summary.get("metadata", {}) or {}
    metadata_keys = [str(key).lower() for key in metadata.keys()]

    signal_text = " ".join(
        [
            str(file_name).lower(),
            data_type.lower(),
            " ".join(keywords),
            " ".join(metric_names),
            " ".join(metadata_keys),
            protocol_text[:1200],
        ]
    )

    detected: List[str] = []

    healthcare_terms = [
        "medical", "medicine", "patient", "diagnosis", "clinical", "hospital", "health", "ehr", "emr", "phi", "lab", "biomarker",
    ]
    finance_terms = [
        "finance", "financial", "bank", "transaction", "payment", "loan", "credit", "aml", "fraud", "ledger", "account",
    ]
    education_terms = [
        "education", "student", "school", "grade", "curriculum", "classroom", "university", "exam", "teacher",
    ]
    pii_terms = [
        "name", "email", "phone", "address", "ssn", "social security", "dob", "birth", "passport", "national id", "identifier",
    ]

    if any(term in signal_text for term in healthcare_terms):
        detected.append("healthcare")
    if any(term in signal_text for term in finance_terms):
        detected.append("finance")
    if any(term in signal_text for term in education_terms):
        detected.append("education")
    if any(term in signal_text for term in pii_terms):
        detected.append("personal-data")

    if data_type in ("image",):
        detected.append("image-data")

    if not detected:
        detected.append("general")

    deduplicated: List[str] = []
    for value in detected:
        if value not in deduplicated:
            deduplicated.append(value)
    return deduplicated


def _build_compliance_notes(domains: List[str]) -> List[str]:
    notes: List[str] = [
        "Apply recommendations with human oversight and retain an auditable decision log linking actions to evidence.",
        "Confirm data minimization, access control, and retention limits before operational use.",
    ]

    if "healthcare" in domains:
        notes.extend([
            "Healthcare-related signals detected: validate protected health information controls, consent basis, and role-based access before use.",
            "For clinical-support use cases, require qualified human review and document intended use limitations.",
        ])

    if "finance" in domains:
        notes.extend([
            "Finance-related signals detected: document model governance, risk controls, and review steps for high-impact decisions.",
            "Retain traceable evidence for anomaly-triggered actions and include periodic control testing.",
        ])

    if "education" in domains:
        notes.extend([
            "Education-related signals detected: protect learner records and apply role-scoped access to identifiable student information.",
            "Ensure recommendations are reviewed for fairness and context before affecting student outcomes.",
        ])

    if "personal-data" in domains:
        notes.extend([
            "Personal-data indicators detected: apply lawful basis checks, purpose limitation, and subject-rights handling workflows.",
            "Use de-identification or pseudonymization where feasible before sharing outputs.",
        ])

    if "image-data" in domains:
        notes.append("Image data detected: verify image handling policy, usage consent, and storage protection controls.")

    return notes


def _build_priority_recommendations(summary: dict, recommendations_text: str) -> List[Tuple[str, str, str]]:
    focus = summary.get("focus_areas", {}) or {}
    confidence = summary.get("confidence", {}) or {}

    extracted: List[Tuple[str, str, str]] = []

    lines = [line.strip(" -•\t") for line in str(recommendations_text or "").splitlines() if line.strip()]
    for line in lines:
        normalized = line.lower()
        if "high priority" in normalized or "priority 1" in normalized or "immediate" in normalized:
            extracted.append(("High Priority", "Immediate attention required", line))
        elif "moderate" in normalized or "priority 2" in normalized or "monitor" in normalized:
            extracted.append(("Moderate Attention", "Track closely and follow up", line))
        elif "low concern" in normalized or "priority 3" in normalized or "stable" in normalized:
            extracted.append(("Low Concern", "Routine monitoring is sufficient", line))

    if extracted:
        return extracted[:8]

    for metric, note in list(focus.items())[:8]:
        conf = confidence.get(metric, "n/a")
        conf_val = float(conf) if isinstance(conf, (int, float)) else None
        note_text = str(note).lower()

        if "anomal" in note_text or "missing" in note_text or (conf_val is not None and conf_val < 0.9):
            extracted.append((
                "High Priority",
                "Immediate attention required",
                f"{metric}: investigate unusual patterns, validate data quality, and review collection conditions before decisions.",
            ))
        elif "variance" in note_text:
            extracted.append((
                "Moderate Attention",
                "Track closely and follow up",
                f"{metric}: variation is noticeable; monitor trend movement and apply regular validation checks.",
            ))
        else:
            extracted.append((
                "Low Concern",
                "Routine monitoring is sufficient",
                f"{metric}: currently stable; continue periodic monitoring and keep audit notes updated.",
            ))

    if not extracted:
        extracted.append((
            "Low Concern",
            "Routine monitoring is sufficient",
            "No critical warning signal was detected in the available indicators.",
        ))

    return extracted[:8]


def _build_apa_references_for_domains(file_name: str, domains: List[str]) -> List[str]:
    current_year = datetime.now().year
    access_date = datetime.now().strftime("%B %d, %Y")

    references: List[str] = [
        f"Expelexia Analytics Team. ({current_year}). Automated dashboard analysis report for {file_name} [Internal report]. Expelexia.",
        "Microsoft. (n.d.). Azure OpenAI documentation. Microsoft Learn. Retrieved " + access_date + ", from https://learn.microsoft.com/azure/ai-services/openai/",
        "Microsoft. (n.d.). Responsible AI resources. Microsoft Learn. Retrieved " + access_date + ", from https://learn.microsoft.com/azure/ai-foundry/responsible-ai/",
        "National Institute of Standards and Technology. (2023). Artificial Intelligence Risk Management Framework (AI RMF 1.0). NIST. Retrieved " + access_date + ", from https://www.nist.gov/itl/ai-risk-management-framework",
        "Organisation for Economic Co-operation and Development. (2019). OECD principles on artificial intelligence. OECD AI Policy Observatory. Retrieved " + access_date + ", from https://oecd.ai/en/ai-principles",
    ]

    if "personal-data" in domains:
        references.append(
            "European Union. (2016). Regulation (EU) 2016/679 (General Data Protection Regulation). EUR-Lex. Retrieved " + access_date + ", from https://eur-lex.europa.eu/eli/reg/2016/679/oj"
        )
    if "healthcare" in domains:
        references.extend([
            "U.S. Department of Health & Human Services. (n.d.). Health information privacy (HIPAA). Retrieved " + access_date + ", from https://www.hhs.gov/hipaa/index.html",
            "U.S. Food and Drug Administration, Health Canada, & Medicines and Healthcare products Regulatory Agency. (2021). Good machine learning practice for medical device development. Retrieved " + access_date + ", from https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles",
        ])
    if "finance" in domains:
        references.extend([
            "Federal Financial Institutions Examination Council. (n.d.). IT examination handbook. Retrieved " + access_date + ", from https://ithandbook.ffiec.gov/",
            "U.S. Federal Trade Commission. (2021). Standards for safeguarding customer information (Safeguards Rule). Retrieved " + access_date + ", from https://www.ftc.gov/business-guidance/privacy-security/gramm-leach-bliley-act",
        ])
    if "education" in domains:
        references.append(
            "U.S. Department of Education. (n.d.). Family Educational Rights and Privacy Act (FERPA). Retrieved " + access_date + ", from https://studentprivacy.ed.gov/"
        )

    references.append(
        "ReportLab Inc. (n.d.). ReportLab user guide. Retrieved " + access_date + ", from https://www.reportlab.com/documentation/"
    )

    deduplicated: List[str] = []
    for reference in references:
        if reference not in deduplicated:
            deduplicated.append(reference)
    return deduplicated


def _extract_chart_image_from_payload(dashboard_payload: Dict[str, Any]) -> io.BytesIO | None:
    charts = dashboard_payload.get("charts", {}) if dashboard_payload else {}
    if not isinstance(charts, dict):
        return None

    data_url = charts.get("mean_overview")
    if not data_url or not isinstance(data_url, str) or "base64," not in data_url:
        return None

    try:
        encoded_part = data_url.split("base64,", 1)[1]
        binary = base64.b64decode(encoded_part)
        buffer = io.BytesIO(binary)
        buffer.seek(0)
        return buffer
    except Exception:
        return None


# ---------------- CHART GENERATION ----------------
def generate_chart(file_name: str, summary: dict):
    project_root = Path(__file__).resolve().parents[2]
    charts_dir = project_root / "data" / "reports"
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_path = charts_dir / f"{file_name}_chart.png"

    stats = summary.get("stats", {})

    if stats:
        col = list(stats.keys())[0]
        values = list(stats[col].values())

        plt.figure()
        plt.plot(values)
        plt.title(f"{col} trend overview")
        plt.savefig(chart_path)
        plt.close()

        return chart_path

    return None


def generate_dashboard_data(file_path: Union[str, Path], file_name: str) -> Dict[str, Any]:
    cache_key = _dashboard_cache_key(file_path, file_name)
    with _DASHBOARD_CACHE_LOCK:
        cached = _DASHBOARD_CACHE.get(cache_key)
    if cached is not None:
        return deepcopy(cached)

    data_type, data = read_file(file_path)
    summary = summarize_data(data_type, data, file_name=file_name)
    table_preview = _build_table_preview(summary)
    chart_payload = _generate_chart_base64(summary)
    ai_payload = generate_lab_dashboard_payload(file_name=file_name, file_type=data_type, summary=summary)
    text_explanation = ai_payload.get("text_explanation", "")
    recommendations = ai_payload.get("recommendations", "")
    data_change_explanation = _build_data_change_explanation(summary)
    chart_explanation = _build_chart_explanation(summary)
    technical_review = _build_technical_review(summary)
    technical_recommendations = _build_technical_recommendations(summary)
    document_notes = _build_document_notes(file_name, data_type, summary)
    project_name = os.getenv("REPORT_PROJECT_NAME", "Expelexia Lab")
    microsoft_link = os.getenv("REPORT_MICROSOFT_LINK", "https://www.microsoft.com/en-us/hackathon")

    payload = {
        "file_name": file_name,
        "data_type": data_type,
        "summary": summary,
        "table_preview": table_preview,
        "charts": chart_payload,
        "charts_data": ai_payload.get("charts_data", {}),
        "table_markdown": ai_payload.get("table_markdown", ""),
        "text_explanation": text_explanation,
        "recommendations": recommendations,
        "data_change_explanation": data_change_explanation,
        "chart_explanation": chart_explanation,
        "technical_review": technical_review,
        "technical_recommendations": technical_recommendations,
        "document_notes": document_notes,
        "project_name": project_name,
        "microsoft_link": microsoft_link,
        "ai_insight": text_explanation or generate_ai_recommendation(summary),
    }

    with _DASHBOARD_CACHE_LOCK:
        _DASHBOARD_CACHE[cache_key] = payload

    return deepcopy(payload)


# ---------------- PDF GENERATION ----------------
def generate_pdf_report(
    file_name: str,
    summary: dict,
    dashboard_payload: Dict[str, Any] | None = None,
    data_type: str = "analysis",
):
    project_root = Path(__file__).resolve().parents[2]
    reports_dir = project_root / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = reports_dir / f"{file_name}_report.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=64,
        bottomMargin=42,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyTight", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["Normal"], fontSize=9, leading=12, spaceAfter=4))
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], textColor=colors.HexColor("#1f2937"), spaceBefore=10, spaceAfter=6))
    content = []

    company_name = os.getenv("REPORT_COMPANY_NAME", "Expelexia")
    program_name = os.getenv("REPORT_PROGRAM_NAME", "AI Analytics Program")
    project_name = os.getenv("REPORT_PROJECT_NAME", "Expelexia Lab")
    logo_path = os.getenv("REPORT_LOGO_PATH", "")
    table_max_width = A4[0] - doc.leftMargin - doc.rightMargin

    def _safe_lines_to_paragraphs(block_text: str, style_name: str = "BodyTight"):
        if not block_text:
            return
        paragraph_chunks = [chunk.strip() for chunk in str(block_text).split("\n\n") if chunk.strip()]
        if not paragraph_chunks:
            paragraph_chunks = [str(block_text).strip()]
        for chunk in paragraph_chunks:
            content.append(Paragraph(_paragraphize_text(chunk), styles[style_name]))

    def _build_wrapped_table(rows: List[List[str]], col_widths: List[float] | None = None):
        if not rows:
            return None

        max_cols = max(len(row) for row in rows)
        normalized_rows: List[List[str]] = []
        for row in rows:
            padded = row + [""] * (max_cols - len(row))
            normalized_rows.append(padded)

        active_col_widths = col_widths or _build_uniform_widths(table_max_width, max_cols)
        wrapped_rows: List[List[Any]] = []
        for row_index, row in enumerate(normalized_rows):
            style_name = "BodySmall" if row_index > 0 else "BodyTight"
            wrapped_rows.append([Paragraph(_paragraphize_text(cell), styles[style_name]) for cell in row])

        table = Table(wrapped_rows, colWidths=active_col_widths, repeatRows=1, splitByRow=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    def _draw_header_footer(canvas, document):
        canvas.saveState()
        page_width, page_height = A4

        def _draw_microsoft_badge():
            start_x = page_width - 126
            start_y = page_height - 40
            square_size = 6
            gap = 2

            canvas.setFillColor(colors.HexColor("#F25022"))
            canvas.rect(start_x, start_y + square_size + gap, square_size, square_size, fill=1, stroke=0)
            canvas.setFillColor(colors.HexColor("#7FBA00"))
            canvas.rect(start_x + square_size + gap, start_y + square_size + gap, square_size, square_size, fill=1, stroke=0)
            canvas.setFillColor(colors.HexColor("#00A4EF"))
            canvas.rect(start_x, start_y, square_size, square_size, fill=1, stroke=0)
            canvas.setFillColor(colors.HexColor("#FFB900"))
            canvas.rect(start_x + square_size + gap, start_y, square_size, square_size, fill=1, stroke=0)
            canvas.setFillColor(colors.HexColor("#374151"))
            canvas.setFont("Helvetica-Bold", 8)
            canvas.drawString(start_x + (square_size * 2) + (gap * 2) + 4, start_y + 4, "Microsoft")

        canvas.setStrokeColor(colors.HexColor("#9ca3af"))
        canvas.line(30, page_height - 34, page_width - 30, page_height - 34)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(36, page_height - 28, f"{company_name} | {program_name} | Lab Analysis Report")

        if logo_path and os.path.exists(logo_path):
            try:
                canvas.drawImage(logo_path, page_width - 90, page_height - 40, width=48, height=22, preserveAspectRatio=True, mask='auto')
            except Exception:
                _draw_microsoft_badge()
        else:
            _draw_microsoft_badge()

        canvas.setStrokeColor(colors.HexColor("#9ca3af"))
        canvas.line(30, 30, page_width - 30, 30)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(36, 18, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        canvas.drawRightString(page_width - 36, 18, f"Page {document.page}")
        canvas.restoreState()

    content.append(Paragraph("Expelexia Lab - Data Analysis Report", styles["Title"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"<b>File Name:</b> {escape(str(file_name))}", styles["BodyTight"]))
    content.append(Paragraph(f"<b>Date of Analysis:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["BodyTight"]))
    content.append(Paragraph(f"<b>Prepared For:</b> {escape(str(company_name))} ({escape(str(program_name))})", styles["BodyTight"]))
    content.append(Paragraph(f"<b>Project:</b> {escape(str(project_name))}", styles["BodyTight"]))
    content.append(Spacer(1, 8))
    content.append(
        Paragraph(
            "This report provides a comprehensive analysis and AI-generated insights based on the uploaded data.",
            styles["BodyTight"],
        )
    )
    content.append(Spacer(1, 12))

    effective_dashboard = dashboard_payload or {}
    if not effective_dashboard:
        ai_payload = generate_lab_dashboard_payload(file_name=file_name, file_type=data_type, summary=summary)
        effective_dashboard = {
            "text_explanation": ai_payload.get("text_explanation", ""),
            "recommendations": ai_payload.get("recommendations", ""),
            "table_markdown": ai_payload.get("table_markdown", ""),
            "data_change_explanation": _build_data_change_explanation(summary),
            "chart_explanation": _build_chart_explanation(summary),
            "technical_review": _build_technical_review(summary),
            "technical_recommendations": _build_technical_recommendations(summary),
            "document_notes": _build_document_notes(file_name, data_type, summary),
        }

    ai_text = effective_dashboard.get("text_explanation") or generate_ai_recommendation(summary)
    recommendations_text = effective_dashboard.get("recommendations") or ""
    table_markdown = effective_dashboard.get("table_markdown") or ""
    data_change_explanation = effective_dashboard.get("data_change_explanation") or _build_data_change_explanation(summary)
    chart_explanation = effective_dashboard.get("chart_explanation") or _build_chart_explanation(summary)
    technical_review = effective_dashboard.get("technical_review") or _build_technical_review(summary)
    technical_recommendations = effective_dashboard.get("technical_recommendations") or _build_technical_recommendations(summary)
    document_notes = effective_dashboard.get("document_notes") or _build_document_notes(file_name, data_type, summary)
    detected_domains = _detect_report_domains(summary=summary, file_name=file_name, data_type=data_type)
    compliance_notes = _build_compliance_notes(detected_domains)
    ai_responsibility_statement = os.getenv(
        "REPORT_AI_RESPONSIBILITY",
        "This report is generated using AI technologies provided through Microsoft Azure. Recommendations are based on available data and are intended to support decision-making. Validate critical decisions with domain experts.",
    )

    content.append(Paragraph("Executive Summary", styles["SectionTitle"]))
    content.append(Spacer(1, 8))
    _safe_lines_to_paragraphs(_build_executive_summary(summary, file_name))
    content.append(Spacer(1, 12))

    metadata = summary.get("metadata", {}) or {}
    records = metadata.get("rows", "n/a")
    variables = metadata.get("columns", "n/a")
    missing_values = metadata.get("missing_values", "n/a")

    content.append(Paragraph("Data Overview", styles["SectionTitle"]))
    overview_rows = [
        ["Measure", "Value", "What it means"],
        ["Number of records", str(records), "How many entries were analyzed."],
        ["Number of variables", str(variables), "How many fields were evaluated."],
        ["Missing values", str(missing_values), "Missing entries can reduce result reliability if frequent."],
    ]
    overview_table = _build_wrapped_table(overview_rows, col_widths=[130, 110, table_max_width - 240])
    if overview_table is not None:
        content.append(overview_table)
        content.append(Spacer(1, 12))

    content.append(Paragraph("Website Dashboard Alignment", styles["SectionTitle"]))
    content.append(Paragraph("This table mirrors key trend, focus, and confidence indicators used in the website dashboard view.", styles["BodyTight"]))
    alignment_table = _build_wrapped_table(
        _dashboard_alignment_rows(summary),
        col_widths=[table_max_width * 0.25, table_max_width * 0.17, table_max_width * 0.40, table_max_width * 0.18],
    )
    if alignment_table is not None:
        content.append(alignment_table)
    content.append(Spacer(1, 12))

    content.append(Paragraph("Visual Analysis", styles["SectionTitle"]))
    content.append(Paragraph("The chart below shows how key values compare. It helps identify increases, decreases, and unusual patterns over time.", styles["BodyTight"]))
    content.append(Spacer(1, 8))

    chart_path = generate_chart(file_name, summary)
    if chart_path and os.path.exists(chart_path):
        content.append(RLImage(str(chart_path), width=400, height=200))
        content.append(Spacer(1, 10))

    embedded_chart = _extract_chart_image_from_payload(effective_dashboard)
    if embedded_chart is not None:
        try:
            content.append(RLImage(embedded_chart, width=430, height=190))
            content.append(Spacer(1, 10))
        except Exception:
            pass

    _safe_lines_to_paragraphs(chart_explanation)
    content.append(Spacer(1, 12))

    content.append(Paragraph("Data Change Explanation", styles["SectionTitle"]))
    _safe_lines_to_paragraphs(data_change_explanation)
    content.append(Spacer(1, 12))

    content.append(Paragraph("Key Findings", styles["SectionTitle"]))
    for finding in _build_key_findings(summary):
        content.append(Paragraph(f"• {finding}", styles["BodyTight"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("AI-Powered Insights", styles["SectionTitle"]))
    _safe_lines_to_paragraphs(ai_text)
    content.append(Spacer(1, 12))

    content.append(Paragraph("Technical Review (Advanced)", styles["SectionTitle"]))
    _safe_lines_to_paragraphs(technical_review)
    content.append(Spacer(1, 12))

    content.append(Paragraph("Technical Recommendations (Advanced)", styles["SectionTitle"]))
    _safe_lines_to_paragraphs(technical_recommendations)
    content.append(Spacer(1, 12))

    content.append(Paragraph("Special AI Recommendations", styles["SectionTitle"]))
    content.append(Paragraph("Priority actions below are generated from trend volatility, confidence, and focus-area signals.", styles["BodyTight"]))
    _safe_lines_to_paragraphs(generate_ai_recommendation(summary))
    content.append(Spacer(1, 12))

    if recommendations_text:
        content.append(Paragraph("Recommendations", styles["SectionTitle"]))
        _safe_lines_to_paragraphs(recommendations_text)
        content.append(Spacer(1, 12))

    content.append(Paragraph("Priority Recommendation Guide", styles["SectionTitle"]))
    priority_rows: List[List[str]] = [["Priority Level", "Meaning", "Recommended Action"]]
    priority_items = _build_priority_recommendations(summary, recommendations_text)
    for level, meaning, action in priority_items:
        priority_rows.append([level, meaning, action])

    priority_table = _build_wrapped_table(
        priority_rows,
        col_widths=[table_max_width * 0.20, table_max_width * 0.26, table_max_width * 0.54],
    )
    if priority_table is not None:
        priority_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#F3F4F6")),
        ]))
        for row_index, row in enumerate(priority_rows[1:], start=1):
            level_text = str(row[0]).lower()
            if "high" in level_text:
                priority_table.setStyle(TableStyle([("BACKGROUND", (0, row_index), (0, row_index), colors.HexColor("#FEE2E2"))]))
            elif "moderate" in level_text:
                priority_table.setStyle(TableStyle([("BACKGROUND", (0, row_index), (0, row_index), colors.HexColor("#FEF3C7"))]))
            else:
                priority_table.setStyle(TableStyle([("BACKGROUND", (0, row_index), (0, row_index), colors.HexColor("#DCFCE7"))]))
        content.append(priority_table)
    content.append(Spacer(1, 12))

    content.append(Paragraph("Confidence & Reliability", styles["SectionTitle"]))
    _safe_lines_to_paragraphs(_build_reliability_note(summary))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Document Notes", styles["SectionTitle"]))
    _safe_lines_to_paragraphs(document_notes)
    content.append(Spacer(1, 12))

    content.append(Paragraph("Legal & Compliance Context", styles["SectionTitle"]))
    content.append(Paragraph("The report automatically adapts reference guidance to detected data context. Domain tags: " + escape(", ".join(detected_domains)) + ".", styles["BodyTight"]))
    for note in compliance_notes:
        content.append(Paragraph("• " + _paragraphize_text(note), styles["BodyTight"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("AI Responsibility & Transparency", styles["SectionTitle"]))
    _safe_lines_to_paragraphs(ai_responsibility_statement)
    content.append(Spacer(1, 12))

    if table_markdown:
        content.append(Paragraph("Appendix", styles["SectionTitle"]))
        content.append(Paragraph("Additional summary details are provided below for reference.", styles["BodyTight"]))
        content.append(Spacer(1, 6))

        markdown_rows = _parse_markdown_table(table_markdown)
        if markdown_rows:
            markdown_table = _build_wrapped_table(markdown_rows)
            if markdown_table is not None:
                content.append(markdown_table)
        else:
            _safe_lines_to_paragraphs(table_markdown, style_name="BodySmall")
        content.append(Spacer(1, 12))

    table_rows = _build_table_preview(summary)
    if len(table_rows) > 1:
        content.append(Paragraph("Detailed Data Table", styles["SectionTitle"]))
        detailed_table = _build_wrapped_table(
            table_rows,
            col_widths=[table_max_width * 0.30, table_max_width * 0.14, table_max_width * 0.14, table_max_width * 0.14, table_max_width * 0.14, table_max_width * 0.14],
        )
        if detailed_table is not None:
            content.append(detailed_table)
        content.append(Spacer(1, 12))

    content.append(Paragraph("References (AI Use)", styles["SectionTitle"]))
    for reference in _build_apa_references_for_domains(file_name=file_name, domains=detected_domains):
        content.append(Paragraph(_paragraphize_text(reference), styles["BodySmall"]))
    content.append(Spacer(1, 10))

    doc.build(content, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)

    return pdf_path
