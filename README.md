# Trading Agent

This repository is the foundation of an AI trading research platform. It provides a minimal FastAPI service, PostgreSQL connection setup, Alembic migration configuration, and a local document ingestion pipeline. **Trading functionality has not been implemented.**

## Development stage

Phase 1 (foundation) and Phase 2A (knowledge ingestion foundation) are complete. Embeddings, retrieval, market analysis, and trading capabilities are not implemented.

## Technology stack

Python 3.12+, FastAPI, PostgreSQL, SQLAlchemy 2.x, Alembic, Pydantic Settings, pypdf, pytest, Docker, and Docker Compose.

## Project structure

- `backend/app` — API, configuration, database setup, and empty model package.
- `backend/app/knowledge` — local text, Markdown, and PDF loading, cleaning, chunking, and JSON output.
- `backend/tests` — API and ingestion tests that do not require a database.
- `backend/alembic` — migration environment; no migrations exist yet.
- `docs` — architecture, development rules, and roadmap.
- `data` — ignored local data directories.
- `scripts` — project scripts directory.

## Run locally

Use Python 3.12 or newer. From the repository root, create and activate a virtual environment, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
$env:PYTHONPATH = "backend"
Copy-Item backend/.env.example backend/.env
uvicorn app.main:app --app-dir backend --reload
```

The API is available at `http://localhost:8000`; interactive API docs are at `/docs`. The root and health endpoints do not require a live database. Database operations will require PostgreSQL at the configured `DATABASE_URL`.

## Ingest a local document

After installing dependencies, run from the repository root:

```sh
python scripts/ingest_document.py data/raw/sample_trading_notes.txt
```

The script writes source metadata and ordered text chunks as JSON under `data/processed/`. Supported inputs are `.txt`, `.md`, and text-based `.pdf` files. Image-only PDFs are reported as unsupported because OCR is not part of this phase.

## Run with Docker Compose

From the repository root:

```sh
docker compose up --build
```

Compose starts PostgreSQL and the API. For non-development deployments, set `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` in the shell or an untracked root `.env` file. Do not use the development defaults for deployment.

## Run tests

From the repository root:

```sh
python -m pip install -r backend/requirements.txt
python -m pytest backend/tests
```

The tests use FastAPI's test client and do not connect to PostgreSQL.
