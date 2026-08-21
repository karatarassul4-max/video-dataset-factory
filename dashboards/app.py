from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

from video_dataset_factory.caption import GROQ_DEFAULT_VISION_MODEL, GroqVisionCaptioner
from video_dataset_factory.duplicates import mark_duplicates
from video_dataset_factory.pipeline import process_video
from video_dataset_factory.reporting import render_markdown_summary, summarize_manifest
from video_dataset_factory.schema import AppConfig, ClipRecord

SCORE_COLUMNS = [
    "blur_score",
    "brightness_score",
    "motion_score",
    "ocr_text_area_ratio",
    "aesthetic_score",
]
VIDEO_TYPES = ["mp4", "mov", "mkv", "avi", "webm"]
DEMO_MANIFEST_PATH = Path("examples/demo_manifest.jsonl")
MAX_UPLOAD_FILES = 10
MAX_TOTAL_UPLOAD_MB = 250
MAX_SINGLE_UPLOAD_MB = 50
DEFAULT_VLM_MODEL = GROQ_DEFAULT_VISION_MODEL


def normalize_records(records: pd.DataFrame) -> pd.DataFrame:
    normalized = records.copy()
    if "keep" not in normalized:
        normalized["keep"] = False
    if "reject_reasons" not in normalized:
        normalized["reject_reasons"] = [[] for _ in range(len(normalized))]
    if "caption" not in normalized:
        normalized["caption"] = ""
    if "duplicate_of" not in normalized:
        normalized["duplicate_of"] = None
    return normalized


def get_secret_or_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        secret = st.secrets.get(name)
    except Exception:  # noqa: BLE001
        return None
    return str(secret) if secret else None


def apply_filters(records: pd.DataFrame) -> pd.DataFrame:
    status = st.sidebar.radio("Status", ["all", "accepted", "rejected"], horizontal=True)
    search = st.sidebar.text_input("Search captions / paths", "")

    filtered = records.copy()
    if status == "accepted":
        filtered = filtered[filtered["keep"]]
    elif status == "rejected":
        filtered = filtered[~filtered["keep"]]

    all_reasons = sorted(
        reason
        for reason in records.explode("reject_reasons")["reject_reasons"].dropna().unique().tolist()
        if reason
    )
    selected_reasons = st.sidebar.multiselect("Reject reasons", all_reasons)
    if selected_reasons:
        filtered = filtered[
            filtered["reject_reasons"].apply(
                lambda reasons: any(reason in reasons for reason in selected_reasons)
            )
        ]

    if search:
        needle = search.lower()
        source = filtered.get("source_path", pd.Series([""] * len(filtered))).astype(str).str.lower()
        caption = filtered.get("caption", pd.Series([""] * len(filtered))).astype(str).str.lower()
        filtered = filtered[source.str.contains(needle, regex=False) | caption.str.contains(needle, regex=False)]

    return filtered


def render_upload_mode() -> list[ClipRecord] | None:
    st.info(
        "Online upload is for a small demo run. For 50+ clips, process videos locally "
        "with the CLI and upload the generated manifest JSONL."
    )
    uploaded_files = st.file_uploader(
        "Upload up to 10 short video clips",
        type=VIDEO_TYPES,
        accept_multiple_files=True,
    )
    max_duration_sec = st.slider(
        "Max accepted clip duration (seconds)",
        min_value=5,
        max_value=180,
        value=20,
        step=5,
        help="Videos longer than this are marked with duration_too_long.",
    )
    threshold = st.slider("Near-duplicate pHash threshold", 0, 16, 6)
    sample_frames = st.slider("Frames sampled per clip", 4, 16, 8, step=2)
    max_vlm_keyframes = st.slider("VLM keyframes per clip", 1, 5, 4)
    vlm_model = st.sidebar.text_input("Groq VLM model", get_secret_or_env("GROQ_MODEL") or DEFAULT_VLM_MODEL)
    st.sidebar.caption("Upload processing uses Groq vision. Set GROQ_API_KEY in app secrets.")

    if not uploaded_files:
        st.info("Upload clips to build a temporary manifest, or switch to demo data.")
        return st.session_state.get("uploaded_records")

    upload_error = validate_video_uploads(uploaded_files)
    if upload_error:
        st.error(upload_error)
        return st.session_state.get("uploaded_records")

    if st.button("Process uploaded videos", type="primary"):
        api_key = get_secret_or_env("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY is required because uploaded clips use real VLM captioning.")
            return st.session_state.get("uploaded_records")

        with st.spinner("Processing uploaded videos with Groq VLM captions..."):
            records = process_uploaded_files(
                uploaded_files,
                sample_frames=sample_frames,
                threshold=threshold,
                max_duration_sec=float(max_duration_sec),
                vlm_model=vlm_model,
                vlm_api_key=api_key,
                max_vlm_keyframes=max_vlm_keyframes,
            )
        st.session_state["uploaded_records"] = records
        st.success(f"Processed {len(records)} uploaded clip(s).")

    return st.session_state.get("uploaded_records")


def validate_video_uploads(uploaded_files) -> str | None:
    if len(uploaded_files) > MAX_UPLOAD_FILES:
        return (
            f"This public demo accepts up to {MAX_UPLOAD_FILES} videos at once. "
            "For 50+ videos, run the CLI locally and upload the manifest JSONL."
        )

    sizes = [getattr(uploaded_file, "size", 0) or 0 for uploaded_file in uploaded_files]
    max_size_mb = max(sizes, default=0) / 1024**2
    total_size_mb = sum(sizes) / 1024**2
    if max_size_mb > MAX_SINGLE_UPLOAD_MB:
        return f"One file is {max_size_mb:.1f} MB. Keep each demo upload under 50 MB."
    if total_size_mb > MAX_TOTAL_UPLOAD_MB:
        return f"Total upload is {total_size_mb:.1f} MB. Keep one demo run under 250 MB."
    return None


def render_manifest_upload_mode() -> list[ClipRecord] | None:
    st.info("Use this for large runs: upload a JSONL manifest produced by the CLI.")
    uploaded_manifest = st.file_uploader("Upload manifest JSONL", type=["jsonl", "json"])
    if uploaded_manifest is None:
        return None

    try:
        text = uploaded_manifest.getvalue().decode("utf-8")
        records = [ClipRecord.model_validate_json(line) for line in text.splitlines() if line.strip()]
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read manifest: {exc}")
        return None

    st.success(f"Loaded {len(records)} manifest row(s).")
    return records


def process_uploaded_files(
    uploaded_files,
    sample_frames: int,
    threshold: int,
    max_duration_sec: float,
    vlm_model: str,
    vlm_api_key: str,
    max_vlm_keyframes: int,
) -> list[ClipRecord]:
    upload_dir = get_upload_dir()
    config = AppConfig()
    config.pipeline.sample_frames = sample_frames
    config.quality.max_duration_sec = max_duration_sec
    config.captioning = {"provider": "manual"}
    config.aesthetic = {"provider": "heuristic"}
    captioner = GroqVisionCaptioner(
        api_key=vlm_api_key,
        model_name=vlm_model,
        max_keyframes=max_vlm_keyframes,
        max_new_tokens=180,
    )

    records: list[ClipRecord] = []
    progress = st.progress(0.0)
    for index, uploaded_file in enumerate(uploaded_files):
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        safe_name = f"{uuid4().hex}{suffix.lower()}"
        path = upload_dir / safe_name
        path.write_bytes(uploaded_file.getbuffer())
        try:
            record = process_video(path, config, captioner=captioner)
            records.append(record.model_copy(update={"source_path": str(path)}))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not process {uploaded_file.name}: {exc}")
        progress.progress((index + 1) / len(uploaded_files))

    return mark_duplicates(records, threshold=threshold)


def get_upload_dir() -> Path:
    if "upload_dir" not in st.session_state:
        st.session_state["upload_dir"] = str(Path(tempfile.gettempdir()) / f"vdf-{uuid4().hex}")
    upload_dir = Path(st.session_state["upload_dir"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def load_demo_records() -> list[ClipRecord]:
    return load_records_from_jsonl(DEMO_MANIFEST_PATH)


def load_records_from_jsonl(path: Path) -> list[ClipRecord]:
    with path.open("r", encoding="utf-8") as handle:
        return [ClipRecord.model_validate_json(line) for line in handle if line.strip()]


def records_to_dataframe(records: list[ClipRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.model_dump() for record in records])


def records_to_jsonl(records: list[ClipRecord]) -> bytes:
    lines = [json.dumps(record.model_dump(), ensure_ascii=False) for record in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_summary(records: pd.DataFrame, filtered: pd.DataFrame) -> None:
    accepted = int(records["keep"].sum())
    rejected = len(records) - accepted
    yield_rate = accepted / len(records) * 100 if len(records) else 0.0
    duplicates = int(records["duplicate_of"].notna().sum()) if "duplicate_of" in records else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total clips", len(records))
    c2.metric("Accepted", accepted)
    c3.metric("Rejected", rejected)
    c4.metric("Yield", f"{yield_rate:.1f}%")
    c5.metric("Duplicates", duplicates)

    st.caption(f"Showing {len(filtered)} filtered rows from {len(records)} total clips.")


def render_reject_reasons(records: pd.DataFrame) -> None:
    st.subheader("Reject Reasons")
    if "reject_reasons" not in records or records.empty:
        st.info("No reject reasons available.")
        return

    reasons = records.explode("reject_reasons")["reject_reasons"].dropna()
    reasons = reasons[reasons.astype(str).str.len() > 0].value_counts()
    if reasons.empty:
        st.success("No reject reasons in the current filter.")
        return
    st.bar_chart(reasons)


def render_score_distributions(records: pd.DataFrame) -> None:
    st.subheader("Quality & Motion Scores")
    available = [column for column in SCORE_COLUMNS if column in records.columns]
    if not available:
        st.info("No score columns available.")
        return

    selected = st.multiselect("Score columns", available, default=available[:4])
    if selected:
        st.line_chart(records[selected].reset_index(drop=True))


def render_gallery(records: pd.DataFrame) -> None:
    st.subheader("Clip Review")
    if records.empty:
        st.info("No clips match the current filters.")
        return

    page_size = st.slider("Clips per page", min_value=3, max_value=24, value=6, step=3)
    max_page = max((len(records) - 1) // page_size, 0)
    page = st.number_input("Page", min_value=0, max_value=max_page, value=0)
    window = records.iloc[page * page_size : (page + 1) * page_size]

    for row in window.to_dict(orient="records"):
        render_clip_card(row)


def render_clip_card(row: dict[str, Any]) -> None:
    keep = bool(row.get("keep"))
    label = "accepted" if keep else "rejected"
    title = f"{row.get('clip_id', 'unknown')} - {label}"

    with st.expander(title, expanded=False):
        source_path = str(row.get("source_path", ""))
        source = Path(source_path)
        left, right = st.columns([1, 2])

        with left:
            if source.exists() and source.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}:
                st.video(str(source))
            else:
                st.code(source_path or "missing source_path", language="text")

        with right:
            st.write(row.get("caption") or "No caption")
            motion_caption = row.get("motion_caption")
            if motion_caption:
                st.caption(motion_caption)

            metrics = {
                "duration_sec": row.get("duration_sec"),
                "fps": row.get("fps"),
                "resolution": f"{row.get('width')}x{row.get('height')}",
                "blur": row.get("blur_score"),
                "motion": row.get("motion_score"),
                "aesthetic": row.get("aesthetic_score"),
                "text_ratio": row.get("ocr_text_area_ratio"),
                "duplicate_of": row.get("duplicate_of"),
            }
            st.json(metrics)
            reasons = row.get("reject_reasons") or []
            if reasons:
                st.warning(", ".join(reasons))


def render_downloads(records: list[ClipRecord], filtered: pd.DataFrame) -> None:
    st.subheader("Exports")
    summary = summarize_manifest(records)
    summary_markdown = render_markdown_summary(summary).encode("utf-8")

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download manifest JSONL",
        records_to_jsonl(records),
        file_name="manifest.jsonl",
        mime="application/jsonl",
    )
    c2.download_button(
        "Download summary MD",
        summary_markdown,
        file_name="dataset_summary.md",
        mime="text/markdown",
    )
    c3.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_manifest.csv",
        mime="text/csv",
    )


def render_manifest_table(records: pd.DataFrame) -> None:
    st.subheader("Manifest Table")
    st.dataframe(records, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Video Dataset Factory", layout="wide")
    st.title("Video Dataset Factory")
    st.caption("Upload short clips for a temporary quality report, or inspect existing manifests.")

    mode = st.sidebar.radio("Input", ["upload videos", "upload manifest JSONL", "use demo data"])

    if mode == "upload videos":
        records = render_upload_mode()
        if not records:
            st.stop()
    elif mode == "upload manifest JSONL":
        records = render_manifest_upload_mode()
        if not records:
            st.stop()
    else:
        records = load_demo_records()

    frame = normalize_records(records_to_dataframe(records))
    if frame.empty:
        st.warning("Manifest is empty.")
        st.stop()

    filtered = apply_filters(frame)
    render_summary(frame, filtered)
    render_reject_reasons(filtered)
    render_score_distributions(filtered)
    render_gallery(filtered)
    render_manifest_table(filtered)
    render_downloads(records, filtered)


main()
