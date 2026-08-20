# Streamlit Deployment

The repository is ready for Streamlit Community Cloud.

## Deploy Settings

Use these settings when creating the app at `share.streamlit.io`:

| Field | Value |
| --- | --- |
| Repository | `karatarassul4-max/video-dataset-factory` |
| Branch | `main` |
| Main file path | `dashboards/app.py` |

The deployment uses:

- `requirements.txt` for Python dependencies;
- `packages.txt` for Linux packages such as `ffmpeg`;
- `.streamlit/config.toml` for upload size and app settings.

## App Modes

- `demo manifest`: loads `examples/demo_manifest.jsonl` for a zero-setup public demo.
- `upload videos`: lets visitors upload short clips, runs the CPU pipeline, marks near-duplicates, and exports JSONL/CSV/Markdown summary files.
- `manifest path`: reads a manifest available on the server filesystem for local or private runs.

## Practical Limits

Streamlit Community Cloud is CPU-oriented and ephemeral. Uploaded files are stored in a temporary directory for the current app session. This is good for a public demo, but not for persistent production storage.

Use short clips for the online demo. For large datasets, run the CLI locally or on a GPU/CPU worker, then publish the generated manifest and summary.

## Recommended Demo Flow

1. Open the deployed app.
2. Start with `demo manifest` so viewers immediately see metrics.
3. Switch to `upload videos` and upload 2-5 short clips.
4. Download `manifest.jsonl` and `dataset_summary.md` from the Exports section.

## Better Production Path

For a serious hosted version, keep Streamlit as the UI and move processing to a backend worker:

- object storage for uploaded videos;
- queue for processing jobs;
- CPU/GPU worker for OpenCV/VLM work;
- database for manifests and review decisions.

That architecture is overkill for the portfolio demo, but it is the right next step if this becomes a real tool.
