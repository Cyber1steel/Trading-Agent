"""Compare lexical, semantic, and hybrid retrieval on a versioned JSON dataset."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.embeddings.fastembed_provider import FastEmbedProvider  # noqa: E402
from app.knowledge.embedding.evaluation import (  # noqa: E402
    evaluate_rankings,
    load_evaluation_dataset,
)
from app.knowledge.embedding.hybrid import HybridRetrievalService  # noqa: E402
from app.knowledge.embedding.lexical import LexicalRetrievalService  # noqa: E402
from app.knowledge.embedding.repository import EmbeddingRepository  # noqa: E402
from app.knowledge.embedding.retrieval import SemanticRetrievalService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=ROOT / "backend" / "tests" / "data" / "retrieval_eval_v1.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ks", default="1,3,5", help="comma-separated metric cutoffs")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    try:
        ks = [int(item) for item in args.ks.split(",")]
    except ValueError:
        parser.error("--ks must be comma-separated positive integers")
    dataset = load_evaluation_dataset(args.dataset, ROOT)
    provider = FastEmbedProvider(get_settings())
    repository = EmbeddingRepository()
    semantic = SemanticRetrievalService(provider, repository)
    lexical = LexicalRetrievalService(repository)
    hybrid = HybridRetrievalService(semantic, lexical)
    services = {"semantic": semantic, "lexical": lexical, "hybrid": hybrid}

    reports = []
    for mode, service in services.items():
        rankings = {
            case.case_id: service.search(case.query, top_k=args.top_k)
            for case in dataset.cases
        }
        reports.append(evaluate_rankings(dataset, rankings, retrieval_mode=mode, ks=ks))
    print(json.dumps([report.model_dump(mode="json") for report in reports], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
