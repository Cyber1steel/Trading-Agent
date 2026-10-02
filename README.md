# Trading Agent

This repository is the foundation of an AI trading research platform. It provides a minimal FastAPI service, PostgreSQL connection setup, Alembic migration configuration, and a local document ingestion pipeline. **Trading functionality has not been implemented.**

## Development stage

Phase 1, Phase 2A (local knowledge ingestion), and Phase 2B (local embeddings and semantic retrieval) are implemented. Market analysis and trading capabilities are not implemented.

## Technology stack

Python 3.12+, FastAPI, PostgreSQL with pgvector, SQLAlchemy 2.x, Alembic, FastEmbed, Pydantic Settings, pypdf, pytest, Docker, and Docker Compose.

## Project structure

- `backend/app` — API, configuration, database setup, SQLAlchemy models, embeddings, and knowledge services.
- `backend/app/knowledge` — local text, Markdown, and PDF loading, cleaning, chunking, JSON output, embedding ingestion, and retrieval.
- `backend/tests` — API and ingestion tests that do not require a database.
- `backend/alembic` — migration environment and pgvector embedding schema migration.
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

The API is available at `http://localhost:8000`; interactive API docs are at `/docs`. The root and health endpoints do not require a live database. Embedding storage requires PostgreSQL with pgvector and the migration applied. Embeddings use FastEmbed's local `BAAI/bge-small-en-v1.5` model (384 dimensions). Model files are not downloaded in offline mode by default. For a deliberate first download set `EMBEDDING_LOCAL_FILES_ONLY=false`, run the command below once, then restore it to `true` for offline use.

## Ingest a local document

After installing dependencies, run from the repository root:

```sh
python scripts/ingest_document.py data/raw/sample_trading_notes.txt
```

The script writes source metadata and ordered text chunks as JSON under `data/processed/`. Supported inputs are `.txt`, `.md`, and text-based `.pdf` files. Image-only PDFs are reported as unsupported because OCR is not part of this phase.

## Store embeddings and search

Start the pgvector database and apply the schema migration from `backend/`:

```powershell
docker compose up -d db
Set-Location backend
$env:PYTHONPATH = "."
alembic upgrade head
Set-Location ..
```

After Phase 2A has produced JSON files, embed them explicitly:

```powershell
python scripts/embed_documents.py data/processed
```

The command is rerunnable: unchanged chunk text is skipped, changed text is re-embedded, and provenance points to the Phase 2A JSON source of truth. `EmbeddingIngestionService` and `SemanticRetrievalService` are application services; they are not exposed as public HTTP routes in this phase. Retrieval returns structured chunks with cosine distance and source metadata. It currently uses exact cosine search, with filters for source/document ID, source type, path, and page. There is no keyword/hybrid search or evidence synthesis yet.

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

Unit tests use fakes for provider and service orchestration; they do not download model weights or connect to PostgreSQL. Database integration requires a separately migrated pgvector database.
