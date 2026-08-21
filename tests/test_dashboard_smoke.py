from pathlib import Path


def test_dashboard_app_exists():
    assert Path("dashboards/app.py").exists()


def test_dashboard_has_review_controls():
    content = Path("dashboards/app.py").read_text(encoding="utf-8")

    assert "Reject reasons" in content
    assert "Clip Review" in content
    assert "Download filtered CSV" in content
    assert "Search captions / paths" in content


def test_dashboard_has_upload_demo_controls():
    content = Path("dashboards/app.py").read_text(encoding="utf-8")

    assert "Upload up to 10 short video clips" in content
    assert "Process uploaded videos" in content
    assert "Max accepted clip duration (seconds)" in content
    assert "VLM keyframes per clip" in content
    assert "Groq VLM model" in content
    assert "GROQ_API_KEY is required" in content
    assert "GroqVisionCaptioner" in content
    assert "captioner=captioner" in content
    assert "Download manifest JSONL" in content
    assert "use demo data" in content


def test_dashboard_has_large_run_manifest_path():
    content = Path("dashboards/app.py").read_text(encoding="utf-8")

    assert "upload manifest JSONL" in content
    assert "For 50+ clips" in content
    assert "MAX_UPLOAD_FILES = 10" in content
