# Trading Agent

This repository is the foundation of an AI trading research platform. It provides a minimal FastAPI service, PostgreSQL connection setup, Alembic migration configuration, and a local document ingestion pipeline. **Trading functionality has not been implemented.**

## Development stage

Phase 1, Phase 2A (local knowledge ingestion), Phase 2B (embeddings and semantic retrieval), Phase 2C (lexical/hybrid retrieval and evaluation), Phase 2D (deterministic market-data/context foundations), and Phase 2E (deterministic market analysis) are implemented. Strategy, signal, and trading capabilities are not implemented.

## Technology stack

Python 3.12+, FastAPI, PostgreSQL with pgvector, SQLAlchemy 2.x, Alembic, FastEmbed, Pydantic Settings, pypdf, pytest, Docker, and Docker Compose.

## Project structure

- `backend/app` — API, configuration, database setup, SQLAlchemy models, embeddings, knowledge, market-data, market-context, and deterministic market-analysis services.
- `backend/app/knowledge` — local text, Markdown, and PDF loading, cleaning, chunking, JSON output, embedding ingestion, and retrieval.
- `backend/tests` — deterministic unit tests plus gated PostgreSQL integration tests.
- `backend/alembic` — migration environment, pgvector/retrieval schema, and market-data schema migrations.
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

The command is rerunnable: unchanged chunk text is skipped, changed text is re-embedded, and provenance points to the Phase 2A JSON source of truth. After applying the Phase 2C migration, backfill lexical vectors for existing embeddings from canonical JSON without loading the model:

```powershell
python scripts/index_lexical_documents.py data/processed
```

Semantic and lexical services are independently callable; `HybridRetrievalService` combines their bounded candidate lists using Reciprocal Rank Fusion (RRF). Candidate counts default to 20 per mode and `HYBRID_RRF_K` defaults to 60. Filters are applied by both search queries before their candidate limits. Results keep the shared structured provenance contract, score/rank evidence, and cosine distance when available. These services are not public HTTP routes.

Lexical search uses PostgreSQL's English full-text query/rank functions and a GIN-indexed derived `tsvector`. It searches chunks already registered by Phase 2B; the lexical backfill script indexes canonical Phase 2A JSON for existing rows without requiring model weights.

### Phase 2C validation status

PostgreSQL with pgvector is the intended persistence and retrieval backend for Phase 2C. The implementation has been checked with unit tests and static/offline checks. Live PostgreSQL/pgvector migration and retrieval execution have **not** been verified locally in the current Windows environment: a usable PostgreSQL test database is not configured/available, and the local psycopg binary is restricted by Windows Application Control. SQLite can support unrelated local checks, but it does not validate PostgreSQL-specific `TSVECTOR`, GIN, or pgvector behavior. The gated PostgreSQL integration tests remain available and should be run against a working PostgreSQL + pgvector test database. Live validation is pending; this environment limitation does not by itself indicate an implementation failure.

Compare all retrieval modes on the versioned bootstrap dataset after the model and database are ready:

```powershell
python scripts/evaluate_retrieval.py
```

The report provides per-query failure details and Recall@K, Precision@K, Hit Rate@K, and MRR@K. The initial dataset is a small check against the fictional sample note, not a representative quality benchmark. Hybrid retrieval is not reranking or RAG, and retrieval quality does not establish trading profitability.

## Market-data foundation (Phase 2D)

Phase 2D defines typed instruments, canonical timeframes, UTC-aware BAR-OPEN candles, provider contracts, normalization and quality reports, immutable dataset snapshots, explicit dataset-scoped range queries, and configurable timezone-aware session labels. A deterministic in-memory fixture provider is included; no external provider, market-data API route, or network download is implemented. Daily/weekly nominal durations do not determine exchange boundaries. Generic Asia/London/New York windows are configurable examples, not broker or exchange hours; without an explicit market calendar, trading status remains unknown.

PostgreSQL is the intended persistence backend. Phase 2D unit and static checks pass, but live PostgreSQL migration and persistence/range-query behavior have not been verified locally because a migrated Phase 2D PostgreSQL test database is unavailable and the Windows environment restricts the psycopg binary. SQLite checks do not validate PostgreSQL-specific behavior. The gated test module can be enabled with `MARKET_DATA_TEST_DATABASE_URL` pointing to a PostgreSQL database already migrated to head; it performs no migrations itself.

## Deterministic market analysis (Phase 2E)

`backend/app/market_analysis` performs read-only candle feature, true-range/ATR, confirmed-swing, structure, fixed-duration higher-timeframe, and session-context calculations over explicit Phase 2D dataset IDs and versions. Results carry source provenance, normalized selected-slice hashes, analysis version `2e.1.0`, parameters, ranges, cutoff, and a deterministic fingerprint. Warm-up history is bounded by the explicit `input_start`; analysis is not persisted and has no API route.

An observation is knowable at BAR-OPEN plus its fixed timeframe duration. Unclosed primary bars are rejected at the declared cutoff; swings become visible only after their right-side confirmation bars close. Parent bars are aligned only after their close, including equality at the observation time. Daily and weekly calendar-anchored bars are unsupported. Session labels delegate to the Phase 2D classifier; no exchange holidays are inferred. These timing rules are correctness requirements against future leakage.

Phase 2E adds no database schema or migration, external dependencies, strategies, signals, or trading behavior. PostgreSQL validation status for the Phase 2D source data remains as described above.

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

Unit tests use deterministic fixtures/fakes; they do not download model weights or contact market-data services. PostgreSQL integration tests remain gated and require a separately migrated PostgreSQL + pgvector database. Set `PGVECTOR_TEST_DATABASE_URL` for embedding integration and `MARKET_DATA_TEST_DATABASE_URL` for Phase 2D integration. SQLite does not validate PostgreSQL-specific types or constraints.
