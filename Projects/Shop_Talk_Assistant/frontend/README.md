# Frontend Container (`frontend/`)

This folder contains the Docker packaging for the Streamlit UI.

- Dockerfile: `frontend/Dockerfile`
- Runtime app code comes from:
  - `app/`
  - `src/` (shared search logic)

## Purpose

`frontend/` exists to keep container concerns separate from UI code:
- Base image and OS packages
- Python dependency install
- Container startup command for Streamlit

## Build and Run (direct)

From project root:

- Build: `docker build -f frontend/Dockerfile -t shoptalk-frontend .`
- Run: `docker run --rm -p 8501:8501 shoptalk-frontend`

In normal usage, run via compose:
- `docker compose up -d frontend`
