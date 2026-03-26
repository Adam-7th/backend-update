from pathlib import Path
from fastapi.testclient import TestClient

from main import app


def run_smoke(sample_name: str = "IOT-temp.csv") -> int:
    client = TestClient(app)

    print(f"ROOT: {client.get('/').status_code}")
    print(f"OPENAPI: {client.get('/openapi.json').status_code}")

    openapi = client.get("/openapi.json")
    paths = openapi.json().get("paths", {}) if openapi.status_code == 200 else {}
    expected_paths = [
        "/api/upload",
        "/api/analyze",
        "/api/report",
        "/api/report/{file_name}",
        "/api/dashboard/{file_name}",
    ]
    for route in expected_paths:
        print(f"ROUTE {route}: {'OK' if route in paths else 'MISSING'}")

    sample_path = Path(__file__).resolve().parents[1] / "data" / "raw" / sample_name
    if not sample_path.exists():
        print(f"UPLOAD: SKIPPED (sample file not found: {sample_path.name})")
        return 1

    with open(sample_path, "rb") as file_handle:
        upload_response = client.post(
            "/api/upload",
            files={"file": (sample_name, file_handle, "text/csv")},
        )

    analyze_response = client.post("/api/analyze", params={"file_name": sample_name})
    report_query_response = client.get("/api/report", params={"file_name": sample_name})
    report_path_response = client.get(f"/api/report/{sample_name}")
    dashboard_response = client.get(f"/api/dashboard/{sample_name}")

    print(f"UPLOAD: {upload_response.status_code}")
    print(f"ANALYZE: {analyze_response.status_code}")
    print(f"REPORT_QUERY: {report_query_response.status_code}")
    print(f"REPORT_PATH: {report_path_response.status_code}")
    print(f"DASHBOARD: {dashboard_response.status_code}")

    statuses = [
        upload_response.status_code,
        analyze_response.status_code,
        report_query_response.status_code,
        report_path_response.status_code,
        dashboard_response.status_code,
    ]
    return 0 if all(code == 200 for code in statuses) else 2


if __name__ == "__main__":
    raise SystemExit(run_smoke())
