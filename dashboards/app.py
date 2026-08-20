from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Video Dataset Factory", layout="wide")
st.title("Video Dataset Factory")

manifest_path = st.sidebar.text_input("Manifest path", "outputs/manifest.jsonl")
path = Path(manifest_path)

if not path.exists():
    st.info("Run `vdf process-folder data/raw --output outputs/manifest.jsonl` first.")
    st.stop()

records = pd.read_json(path, lines=True)
accepted = int(records["keep"].sum()) if "keep" in records else 0
rejected = len(records) - accepted

c1, c2, c3 = st.columns(3)
c1.metric("Clips", len(records))
c2.metric("Accepted", accepted)
c3.metric("Rejected", rejected)

st.subheader("Dataset Manifest")
st.dataframe(records, use_container_width=True)

if "reject_reasons" in records:
    st.subheader("Reject Reasons")
    reasons = records.explode("reject_reasons")["reject_reasons"].dropna().value_counts()
    st.bar_chart(reasons)
