# Trading Agent

This repository is the foundation of an AI trading research platform. It currently provides a minimal FastAPI service, PostgreSQL connection setup, and Alembic migration configuration. **Trading functionality has not been implemented.**

## Development stage

Phase 1 (foundation) is complete. Future research, analysis, and simulation capabilities are documented in `docs/`; they are not part of the current application.

## Technology stack

Python 3.12+, FastAPI, PostgreSQL, SQLAlchemy 2.x, Alembic, Pydantic Settings, pytest, Docker, and Docker Compose.

## Project structure

- `backend/app` — API, configuration, database setup, and empty model package.
- `backend/tests` — API tests that do not require a database.
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
