from pathlib import Path


def test_dashboard_app_exists():
    assert Path("dashboards/app.py").exists()


def test_dashboard_has_review_controls():
    content = Path("dashboards/app.py").read_text(encoding="utf-8")

    assert "Reject reasons" in content
    assert "Clip Review" in content
    assert "Download filtered CSV" in content
