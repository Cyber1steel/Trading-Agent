# Trading Agent

This repository is the foundation of an AI trading research platform. It provides a minimal FastAPI service, PostgreSQL connection setup, Alembic migration configuration, local document ingestion and retrieval, deterministic market analysis and setup evaluation, an in-process historical backtesting foundation, an independent deterministic risk calculator, an immutable trade-candidate/evidence boundary, and an evidence-bound LLM reasoning foundation. **These research foundations do not establish profitable performance; trading execution has not been implemented.**

## Development stage

Phases 1, 2A–2I, and 3A–3B are implemented as foundations: knowledge ingestion/retrieval, market data/context, deterministic analysis and strategy evaluation, in-process backtesting, independent risk calculations, deterministic candidate/evidence assembly, an evidence-bound reasoning boundary, and offline adversarial reasoning evaluation. These are limited research components, not evidence of strategy performance or trading readiness. No broker integration, order execution, paper trading, or live trading is implemented. The repository previously used Phase 2H for the risk engine; the candidate/evidence foundation therefore follows it as Phase 2I to preserve the committed phase history.

## Technology stack

Python 3.12+, FastAPI, PostgreSQL with pgvector, SQLAlchemy 2.x, Alembic, FastEmbed, Pydantic Settings, pypdf, pytest, Docker, and Docker Compose.

## Project structure

- `backend/app` — API, configuration, database setup, SQLAlchemy models, embeddings, knowledge, market-data/context, deterministic market analysis and strategy evaluation, backtesting, risk calculation, trade-candidate assembly, and evidence-bound reasoning.
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

Phase 2D defines typed instruments, canonical timeframes, UTC-aware BAR-OPEN candles, provider contracts, normalization and quality reports, versioned dataset snapshots with repository/service-enforced immutability, explicit dataset-scoped range queries, and configurable timezone-aware session labels. A deterministic in-memory fixture provider is included; no external provider, market-data API route, or network download is implemented. Daily/weekly nominal durations do not determine exchange boundaries. Generic Asia/London/New York windows are configurable examples, not broker or exchange hours; without an explicit market calendar, trading status remains unknown.

PostgreSQL is the intended persistence backend. Dataset immutability is enforced by repository/service operations; the schema does not add database triggers or constraints preventing direct updates/deletes. Phase 2D unit and static checks pass, but live PostgreSQL migration and persistence/range-query behavior have not been verified locally because a migrated Phase 2D PostgreSQL test database is unavailable and the Windows environment restricts the psycopg binary. SQLite checks do not validate PostgreSQL-specific behavior. The gated test module can be enabled with `MARKET_DATA_TEST_DATABASE_URL` pointing to a PostgreSQL database already migrated to head; it performs no migrations itself.

## Deterministic market analysis (Phase 2E)

`backend/app/market_analysis` performs read-only candle feature, true-range/ATR, confirmed-swing, structure, fixed-duration higher-timeframe, and session-context calculations over explicit Phase 2D dataset IDs and versions. Results carry source provenance, normalized selected-slice hashes, analysis version `2e.1.0`, parameters, ranges, cutoff, and a deterministic fingerprint. Warm-up history is bounded by the explicit `input_start`; analysis is not persisted and has no API route.

An observation is knowable at BAR-OPEN plus its fixed timeframe duration. Unclosed primary bars are rejected at the declared cutoff; swings become visible only after their right-side confirmation bars close. Parent bars are aligned only after their close, including equality at the observation time. Daily and weekly calendar-anchored bars are unsupported. Session labels delegate to the Phase 2D classifier; no exchange holidays are inferred. These timing rules are correctness requirements against future leakage.

Phase 2E adds no database schema or migration, external dependencies, strategies, signals, or trading behavior. PostgreSQL validation status for the Phase 2D source data remains as described above.

## Deterministic strategy/setup foundation (Phase 2F)

`backend/app/strategy` defines immutable, explicitly versioned strategy/setup contracts and evaluates their typed conditions against a supplied Phase 2E `AnalysisResult`. Evaluation uses three-valued logic, provenance-linked evidence, deterministic lifecycle transitions, exact UTC observation cutoffs, and SHA-256 fingerprints. The first supplied observation is left-censored: an already-true qualifying condition cannot create a candidate until a prior supplied observation establishes it was false. Phase 2F verifies the full Phase 2E fingerprint and separately derives a cutoff-visible prefix identity; full provenance is retained while future suffixes do not alter the decision fingerprint. It is a pure in-memory evaluation layer: it does not query datasets or persist definitions/results.

Phase 2F defines and evaluates deterministic setups but does not establish setup quality or trading performance. Phases 2G/2H/2I add limited backtest, single-trade risk, and immutable candidate/evidence foundations below; Phase 3A implements the downstream reasoning foundation described later. Walk-forward validation, Monte Carlo, portfolio controls, stop-loss/take-profit lifecycle management, broker integration, paper trading, live execution, persistence, and strategy/risk API routes remain future work. Phase 2F itself has no migration or dependency changes.

## Historical backtesting foundation (Phase 2G)

`backend/app/backtesting` replays a supplied Phase 2E analysis result and Phase 2F strategy/setup over an explicit UTC range. It checks dataset, analysis, strategy, setup, and provenance identities; evaluates lifecycle transitions once and replays them chronologically; and models entries/exits at the next primary bar-open with explicit fixed quantity, spread, slippage, and fees. The shared `backend/app/execution` contract contains assumptions only; it does not integrate with brokers. There are no intrabar stop/target rules, dynamic position sizing, optimization, walk-forward validation, broker integration, or persistence. An open position at the cutoff remains open; ending equity includes a conservative mark-to-market at the last closed candle with the configured assumed exit costs, while closed-trade metrics remain separate. Maximum drawdown uses closed-trade outcomes, not intratrade marks.

This is a deterministic simulation foundation, not a validated performance report. It does not establish fill realism, predictive value, or profitability. PostgreSQL is not used by the backtest engine; its source `AnalysisResult` must be supplied by the caller.

## Deterministic risk calculation (Phase 2H)

`backend/app/risk` evaluates an explicit trade proposal against immutable risk configuration using Decimal arithmetic. It requires a supplied positive PnL-to-account-currency conversion rate, includes modeled execution costs in downside sizing, rounds quantity down to the configured increment, and returns `RISK_VALID`, `RISK_REJECTED`, or `INSUFFICIENT_EVIDENCE` with a request-bound fingerprint. The engine does not select trades, create stops/targets, integrate with a broker, persist decisions, or authorize execution. Configuration and conversion inputs are caller-supplied evidence; this module does not fetch exchange rates or verify their source.

The Phase 2H calculator is not portfolio risk management: it does not model leverage/margin, correlated exposure, daily or weekly loss limits, drawdown controls, or dynamic account equity. A valid calculation only means the supplied inputs satisfy the implemented single-trade checks; it does not mean a trade is suitable or profitable.

## Deterministic trade candidate and evidence foundation (Phase 2I)

`backend/app/trade_candidate` composes a supplied Phase 2E analysis result, Phase 2F strategy/setup evaluation, Phase 2H risk request/result, and the same execution assumptions used by that risk request. It does not recompute setup conditions or position sizing. Before an `ACTIONABLE` candidate can be returned, the service checks artifact fingerprints, strategy/setup/dataset identities, the exact evaluation observation, source-linked evidence timestamps, risk-request binding, price relationships, stop distance, risk budget, and execution assumptions. Candidate quantity and downside are copied from the RiskResult; reward/risk is present only when a target is supplied.

Candidates and evidence packages are frozen, nested immutable contracts with deterministic fingerprints and an ID derived from that fingerprint. Statuses distinguish `ACTIONABLE`, `WAIT`, `INSUFFICIENT_EVIDENCE`, and `REJECTED`. The evidence package records dataset content/slice identities, analysis and visible-prefix identities, exact definitions and evaluation identity, source evidence references, risk configuration/request/result identities, and execution assumptions. The layer is research output only: it has no LLM, API, persistence, broker, paper-trading, or live-execution functionality. The prior committed roadmap assigned Phase 2H to risk; this subsequent candidate layer is numbered Phase 2I.

## LLM reasoning and evidence orchestration foundation (Phase 3A)

`backend/app/reasoning` consumes an already validated Phase 2I candidate, optional matching Phase 2E analysis, optional matching historical Phase 2G backtest evidence, and chunks returned by an injected existing Phase 2B/2C retrieval service. It builds frozen typed context, applies the evaluation-time cutoff, preserves full deterministic market observations, artifact fingerprints, retrieval method/ranks, source/document/chunk identity, source location, embedding model identity, publication time, and canonical chunk hash, then constructs a versioned deterministic prompt. Knowledge without a known publication time and knowledge published after the cutoff are omitted and disclosed as unavailable. The retrieval engine is not duplicated, and the LLM cannot search documents directly.

The evidence hierarchy is explicit: deterministic market observations; deterministic strategy/risk and candidate evidence; historical backtest evidence; retrieved educational/reference knowledge; and LLM inference. Historical backtest evidence includes dataset, strategy/setup, analysis, request, execution assumptions, engine version, sample counts, metrics, warnings, and an explicit `NOT_ESTABLISHED` out-of-sample state. It is historical simulation, not a forecast or proof of profitability.

Providers implement a small protocol returning raw structured JSON and provider/model/request metadata. There is no vendor SDK, API key requirement, default live provider, or network dependency. The core validates structured output, evidence and knowledge citations, cited numerical values, status semantics, and unsupported future-profitability language before creating a result. Malformed responses, provider errors/timeouts, and unsupported claims produce an auditable `INSUFFICIENT_EVIDENCE` result. Deterministic `REJECTED`, `WAIT`, and `INSUFFICIENT_EVIDENCE` states cannot be promoted by a provider. Results preserve deterministic candidate status separately, contain no mutable candidate fields, and are fingerprinted with the context, question, prompt contract, provider/model, citations, response hash, and output.

Phase 3A is an explanation and evidence-orchestration foundation only. It adds no API route, persistence, autonomous agent loop, reranker, strategy/signal generation, risk calculation, or execution capability. Free-form language is not a complete semantic fact checker; structured citations and numerical claims receive deterministic validation, while broader natural-language truth still requires human review. Source publication time does not establish when a local knowledge corpus was indexed, so this phase does not claim a knowledge source was present in the system at a historical decision time. The implementation does not establish strategy quality, future profitability, or trading readiness.

## Reasoning evaluation and adversarial validation (Phase 3B)

The offline scenario runner at `backend/app/reasoning/evaluation.py` exercises the real reasoning service with 13 immutable, reusable golden/adversarial cases and a simulated provider. The cases cover valid ACTIONABLE, WAIT, INSUFFICIENT_EVIDENCE, and REJECTED states; attacks cover risk/execution overrides, fabricated evidence and metrics, temporal leakage, malicious instructions inside retrieved documents, contradictory evidence, altered provenance, and provider failures. It reports safety-boundary counts rather than “AI accuracy.” Defined numeric, future-performance, override, citation, context-integrity, and response-size checks fail closed; deterministic WAIT/REJECTED/INSUFFICIENT_EVIDENCE does not invoke a provider. Retrieved passages are explicitly treated as untrusted content, never as instructions.

These checks validate only the specified attack classes. They do not prove the reasoning layer is hallucination-proof, fully fact-check arbitrary prose, calibrate model confidence, or predict trading profitability. Context fingerprints detect stale/inconsistent content but are unkeyed hashes, not authentication against a hostile caller that can fabricate and re-hash all inputs; source authenticity remains with trusted upstream artifact and retrieval producers. Evaluation runs offline with no paid model/API. Live provider behavior, strategy quality, out-of-sample performance, and profitability remain unverified.

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
