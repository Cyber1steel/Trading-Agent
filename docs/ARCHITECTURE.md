# Architecture

## Current foundation

The current system is a small FastAPI service. Its API routes live separately from application configuration and database infrastructure. SQLAlchemy provides a PostgreSQL engine, session factory, and empty declarative metadata for future tables. Alembic is configured to consume that metadata. No application tables or trading behavior exist yet.

## Knowledge ingestion (Phase 2A)

The implemented knowledge foundation accepts local `.txt`, `.md`, and text-based `.pdf` files. Format-specific loaders extract text and source metadata; a deterministic cleaner normalizes safe whitespace and extraction artifacts; a character-based chunker emits ordered chunks with source IDs, file paths, optional URLs, page numbers, and character offsets. The standalone ingestion pipeline writes structured JSON to `data/processed/`.

Ingestion is separated from retrieval so documents first become clean, structured, traceable knowledge. Embeddings and retrieval can then be introduced without coupling document parsing to a particular model or vector store. This phase does not include OCR, database tables, embeddings, or search.

## Future components

These are architectural directions only; they are not implemented:

- **Embeddings and vector search (Phase 2B)** — future representation and retrieval of ingested chunks.
- **Web/YouTube/podcast ingestion (Phase 2C)** — future source adapters; no download or scraping behavior exists yet.
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
