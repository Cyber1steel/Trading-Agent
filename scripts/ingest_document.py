from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.knowledge.pipeline import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DocumentIngestionError,
    process_document,
    write_processed_document,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and chunk a local trading research document.")
    parser.add_argument("source", type=Path, help="Path to a .txt, .md, or text-based .pdf file")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        document = process_document(
            args.source,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        output_path = write_processed_document(document, args.output_dir)
    except (DocumentIngestionError, OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Processed: {document.source.title}")
    print(f"Chunks: {len(document.chunks)}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
