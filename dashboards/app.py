from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

SCORE_COLUMNS = [
    "blur_score",
    "brightness_score",
    "motion_score",
    "ocr_text_area_ratio",
    "aesthetic_score",
]


def normalize_records(records: pd.DataFrame) -> pd.DataFrame:
    normalized = records.copy()
    if "keep" not in normalized:
        normalized["keep"] = False
    if "reject_reasons" not in normalized:
        normalized["reject_reasons"] = [[] for _ in range(len(normalized))]
    if "caption" not in normalized:
        normalized["caption"] = ""
    return normalized


def apply_filters(records: pd.DataFrame) -> pd.DataFrame:
    status = st.sidebar.radio("Status", ["all", "accepted", "rejected"], horizontal=True)
    search = st.sidebar.text_input("Caption/source search", "")

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


def render_summary(records: pd.DataFrame, filtered: pd.DataFrame) -> None:
    accepted = int(records["keep"].sum())
    rejected = len(records) - accepted
    yield_rate = accepted / len(records) * 100 if len(records) else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total clips", len(records))
    c2.metric("Accepted", accepted)
    c3.metric("Rejected", rejected)
    c4.metric("Yield", f"{yield_rate:.1f}%")

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
    title = f"{row.get('clip_id', 'unknown')} · {label}"

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
                "text_ratio": row.get("ocr_text_area_ratio"),
            }
            st.json(metrics)
            reasons = row.get("reject_reasons") or []
            if reasons:
                st.warning(", ".join(reasons))


def render_manifest_table(records: pd.DataFrame) -> None:
    st.subheader("Manifest Table")
    st.dataframe(records, use_container_width=True)
    st.download_button(
        "Download filtered CSV",
        records.to_csv(index=False).encode("utf-8"),
        file_name="filtered_manifest.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Video Dataset Factory", layout="wide")
    st.title("Video Dataset Factory")

    manifest_path = st.sidebar.text_input("Manifest path", "outputs/manifest.jsonl")
    path = Path(manifest_path)

    if not path.exists():
        st.info("Run `vdf process-folder data/clips --output outputs/manifest.jsonl` first.")
        st.stop()

    records = pd.read_json(path, lines=True)
    if records.empty:
        st.warning("Manifest is empty.")
        st.stop()

    records = normalize_records(records)
    filtered = apply_filters(records)
    render_summary(records, filtered)
    render_reject_reasons(filtered)
    render_score_distributions(filtered)
    render_gallery(filtered)
    render_manifest_table(filtered)


main()
