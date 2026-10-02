# Architecture

The project-wide engineering contract for future architecture and implementation is documented in [TRADING_AGENT_SPEC.md](TRADING_AGENT_SPEC.md).

## Current foundation

The current system is a small FastAPI service. Its API routes live separately from application configuration and database infrastructure. SQLAlchemy provides a PostgreSQL engine and session factory; Alembic manages the application schema. No trading behavior exists yet.

## Knowledge ingestion (Phase 2A)

The implemented knowledge foundation accepts local `.txt`, `.md`, and text-based `.pdf` files. Format-specific loaders extract text and source metadata; a deterministic cleaner normalizes safe whitespace and extraction artifacts; a character-based chunker emits ordered chunks with source IDs, file paths, optional URLs, page numbers, and character offsets. The standalone ingestion pipeline writes structured JSON to `data/processed/`.

Ingestion is separated from retrieval so documents first become clean, structured, traceable knowledge. The JSON output remains the canonical chunk content and provenance record.

## Embeddings and semantic retrieval (Phase 2B)

`app.embeddings.EmbeddingProvider` defines provider identity, model identity, dimensions, single-text embedding, and batch embedding. `FastEmbedProvider` is the initial local implementation, configured by `EMBEDDING_*` settings and using `BAAI/bge-small-en-v1.5` (384 dimensions) by default. It uses query/passage prefixes and defaults to local-files-only; model downloads are an explicit operator action. Provider abstraction leaves room for a future API implementation.

PostgreSQL with pgvector stores vectors and retrieval metadata. Alembic installs the `vector` extension and creates `embedding_spaces` and `knowledge_embeddings`. One provider/model pair is bound to one dimension; foreign keys and vector-dimension checks reject incompatible rows. A unique chunk/provider/model constraint makes ingestion idempotent. Search performs exact cosine-distance ordering and can filter by source/document ID, source type, source path, and page.

Run `scripts/embed_documents.py` against one Phase 2A JSON file or a directory. It reads bounded batches, hashes chunk content, skips unchanged vectors, updates changed vectors, and refreshes JSON references when provenance moves. Search resolves matched chunk text from that same Phase 2A JSON and checks its hash before returning a structured result with source, page, chunk index, offsets, path/URL, model identity, and cosine distance. A missing or changed JSON chunk is surfaced instead of returning stale evidence.

This phase does not include hybrid search, reranking, query expansion, LLM synthesis, public search endpoints, or OCR. Future retrieval enhancements can build on the provider interface, structured metadata, and result contract without changing Phase 2A ingestion.

## Future components

These are architectural directions only; they are not implemented:

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
