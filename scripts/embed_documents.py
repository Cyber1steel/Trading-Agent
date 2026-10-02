"""Explicitly embed Phase 2A JSON output and store it in PostgreSQL/pgvector."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.embeddings.fastembed_provider import FastEmbedProvider  # noqa: E402
from app.knowledge.embedding.ingestion import EmbeddingIngestionService  # noqa: E402
from app.knowledge.embedding.repository import EmbeddingRepository  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_path", type=Path, help="Phase 2A JSON file or directory of JSON files")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    source = args.processed_path
    paths = sorted(source.glob("*.json")) if source.is_dir() else [source]
    if not paths:
        parser.error(f"No processed JSON files found at {source}")
    service = EmbeddingIngestionService(
        FastEmbedProvider(get_settings()), EmbeddingRepository(), args.batch_size
    )
    report = service.ingest_files(paths)
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
