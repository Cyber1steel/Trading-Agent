# Architecture

## Current foundation

The current system is a small FastAPI service. Its API routes live separately from application configuration and database infrastructure. SQLAlchemy provides a PostgreSQL engine, session factory, and empty declarative metadata for future tables. Alembic is configured to consume that metadata. No application tables or trading behavior exist yet.

## Future components

These are architectural directions only; they are not implemented:

- **Knowledge/RAG** — ingestion and retrieval for research material.
- **Market Data** — historical and live market data adapters.
- **Analysis** — market structure and technical analysis.
- **Risk** — risk controls and position sizing research.
- **Agent** — language model assisted research workflows.
- **Decision Engine** — explicit orchestration of research outputs.
- **Backtesting** — historical strategy evaluation.
- **Chat Analysis** — analysis of messages and signals.
- **Journal** — records of research and simulated decisions.
- **Frontend** — a dashboard for research workflows.

Future components should be added as focused modules when their requirements are defined. The platform is intended to begin as research and analysis software, not an execution system.
