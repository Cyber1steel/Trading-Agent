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

## Hybrid retrieval and evaluation (Phase 2C)

Lexical retrieval uses PostgreSQL `websearch_to_tsquery('english', ...)`, `ts_rank_cd`, and a GIN index over a derived `tsvector` on `knowledge_embeddings`. The `search_vector` stores only PostgreSQL's searchable token representation; it does not store another copy of chunk text. Phase 2A JSON remains canonical. New/changed embeddings populate the vector during ingestion; after applying the Phase 2C migration, existing rows can be backfilled from processed JSON with `scripts/index_lexical_documents.py`. The backfill updates only rows whose indexed content hash still matches the canonical chunk. Lexical-only search does not embed queries, but searches chunks already registered by Phase 2B ingestion.

`LexicalRetrievalService`, `SemanticRetrievalService`, and `HybridRetrievalService` remain independently callable and return the shared `RetrievalResult` evidence structure. Filters are applied in each database candidate query before its candidate limit. Both methods resolve candidates through the Phase 2A JSON and reject missing or stale provenance. Lexical rows are deduplicated in PostgreSQL across embedding models with a stable model-identity choice before applying top-k.

Hybrid retrieval merges the bounded semantic and lexical result lists with Reciprocal Rank Fusion:

`score(chunk) = 1 / (rrf_k + semantic_rank) + 1 / (rrf_k + lexical_rank)`

Absent ranks contribute zero. The default `rrf_k` is 60; semantic and lexical candidate counts default to 20 each and are configurable with `HYBRID_RRF_K`, `HYBRID_SEMANTIC_CANDIDATES`, and `HYBRID_LEXICAL_CANDIDATES`. It does not add cosine distance to PostgreSQL's lexical score. Duplicate chunk IDs are merged while retaining both ranks, lexical score, semantic distance, and fused score. Ties are resolved by chunk ID then source ID. The fused list is the insertion point for a future reranker; no reranker is present in this phase.

The versioned bootstrap evaluation set is `backend/tests/data/retrieval_eval_v1.json`. It targets the existing fictional sample note using queries and expected Phase 2A chunk identities. Relative source paths are resolved to the repository's Phase 2A source IDs at run time so the dataset works from another checkout path. Its small fictional corpus is a plumbing check, not a representative retrieval benchmark. `scripts/evaluate_retrieval.py` compares lexical, semantic, and hybrid rankings on the same cases and reports per-case expected/retrieved IDs, relevant ranks, Recall@K, Precision@K, Hit Rate@K, and MRR@K. Precision@K divides relevant retrieved items by K (unfilled positions count as non-relevant); MRR@K uses the first relevant rank within K. It reports measurements without selecting a winning method.

### Phase 2C validation status

PostgreSQL with pgvector is the intended persistence and retrieval backend for this phase. Phase 2C has passed unit tests and static/offline checks. Live PostgreSQL/pgvector migration and retrieval execution have **not** been verified locally in the current Windows environment because a usable PostgreSQL test database is not configured/available and the local psycopg binary is restricted by Windows Application Control. SQLite checks must not be treated as validation of PostgreSQL-specific `TSVECTOR`, GIN, or pgvector behavior. The gated PostgreSQL integration tests remain available and should be run when a working PostgreSQL + pgvector test database is available. Live database validation is pending; no implementation failure is inferred from this environment limitation.

Hybrid retrieval is not reranking and is not RAG. Retrieval relevance does not establish evidence truth or trading profitability. Evaluation quality is limited by the small dataset and available indexed sources.

## Future components

These are architectural directions only; they are not implemented:

- **Web/YouTube/podcast ingestion** — future source adapters; Phase 2C implemented retrieval and evaluation, not external source downloading or scraping.
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
