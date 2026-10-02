"""Backfill PostgreSQL full-text vectors from Phase 2A JSON without model loading."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.knowledge.embedding.lexical import LexicalIndexingService  # noqa: E402
from app.knowledge.embedding.repository import EmbeddingRepository  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_path", type=Path, help="Phase 2A JSON file or directory of JSON files")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    paths = sorted(args.processed_path.glob("*.json")) if args.processed_path.is_dir() else [args.processed_path]
    if not paths:
        parser.error(f"No processed JSON files found at {args.processed_path}")
    report = LexicalIndexingService(EmbeddingRepository(), args.batch_size).index_files(paths)
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
