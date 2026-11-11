"""Produce real retrieval evidence for the release gate without downloading an LLM.

The first 40 original questions remain reserved for development. No parameters
are tuned here. Both systems run on the remaining 160 existing generated queries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from .corpus import ROOT, load
from .release_gate import atomic_json, digest, read_json, validate_run
from .retrieval import BM25

PROTOCOL = {
    "grader": "gold-document-reciprocal-rank-at-10/v1",
    "score_definition": "1/rank if the source abstract is in top 10, otherwise 0; higher is better",
    "latency_scope": "Single warm sequential query: query preprocessing, retrieval and top-10 ranking; excludes index construction and network",
    "cost_scope": "External API fees USD only; local CPU/electricity/hosting are unpriced and excluded",
}


def _jsonl(path: Path) -> list[dict]:
    import json
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_manifest(rows: list[dict]) -> list[dict]:
    manifest = []
    for row in rows:
        input_sha = digest(row["question"])
        reference_sha = digest({"gold_id": row["gold_id"], "reference": row["reference"]})
        manifest.append({"task_id": input_sha, "group_id": row["gold_id"],
                         "input_sha256": input_sha, "reference_sha256": reference_sha})
    if len({row["task_id"] for row in manifest}) != len(manifest):
        raise ValueError("duplicate input questions in evaluation dataset")
    return manifest


def reciprocal_rank(ranked: list[str], gold: str) -> float:
    if len(ranked) != len(set(ranked)):
        raise ValueError("retrieved documents must be unique")
    return 1.0 / (ranked.index(gold) + 1) if gold in ranked[:10] else 0.0


def evaluate(variant: str, corpus_path: Path = ROOT / "data/arxiv_corpus.jsonl",
             questions_path: Path = ROOT / "data/answerable.jsonl") -> dict:
    if variant not in {"tfidf", "bm25", "broken-bm25"}:
        raise ValueError("unsupported retriever variant")
    papers = load(corpus_path)
    paper_ids = [paper.id for paper in papers]
    if len(paper_ids) != len(set(paper_ids)):
        raise ValueError("duplicate corpus paper IDs")
    rows = _jsonl(questions_path)[40:]
    if not rows or any(row["gold_id"] not in paper_ids for row in rows):
        raise ValueError("test questions must be nonempty and all gold sources must exist")
    manifest = task_manifest(rows)
    corpus_hash = digest([vars(paper) for paper in papers])
    started = time.perf_counter()
    settings = {"variant": variant, "top_k": 10, "corpus_sha256": corpus_hash,
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "retrieval_source_sha256": hashlib.sha256((ROOT / "src/resassist/retrieval.py").read_bytes()).hexdigest(),
                "numpy": np.__version__}
    if variant == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, norm="l2")
        matrix = vectorizer.fit_transform([paper.text for paper in papers])
        settings.update({"sklearn": importlib.metadata.version("scikit-learn"),
                         "stop_words": "english", "norm": "l2"})

        def search(query):
            scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
            return [int(i) for i in np.argsort(-scores, kind="stable")[:10]]
    else:
        bm25 = BM25([paper.text for paper in papers])
        settings.update({"rank_bm25": importlib.metadata.version("rank-bm25"),
                         "k1": 1.5, "b": 0.75, "epsilon": 0.25})

        def search(query):
            indices = bm25.search(query, 10)
            # An explicitly bad change, exercised to prove the gate blocks it.
            return list(reversed(indices)) if variant == "broken-bm25" else indices
    index_ms = (time.perf_counter() - started) * 1000
    search("language model retrieval")  # common warm-up, not a measured test question
    cases = []
    for row, identity in zip(rows, manifest):
        start = time.perf_counter()
        ranked = [paper_ids[index] for index in search(row["question"])]
        elapsed = (time.perf_counter() - start) * 1000
        cases.append({**identity, "score": reciprocal_rank(ranked, row["gold_id"]),
                      "latency_ms": elapsed, "cost_usd": 0.0, "error": False,
                      "question": row["question"], "gold_id": row["gold_id"],
                      "reference": row["reference"], "ranked_ids": ranked})
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    run = {"schema_version": 1, "dataset_version": "arxiv-generated-answerable-test-v1",
           "dataset_sha256": digest(sorted(manifest, key=lambda item: item["task_id"])),
           "corpus_sha256": corpus_hash, "model_sha256": digest(settings),
           "prompt_sha256": digest("not-applicable: retrieval-only, no generation prompt"),
           "system_name": variant, "protocol": PROTOCOL, "cases": cases,
           "created_utc": datetime.now(timezone.utc).isoformat(), "code_revision": revision,
           "dirty_worktree": dirty, "model_configuration": settings,
           "environment": {"python": platform.python_version(), "platform": platform.platform(),
                           "processor": platform.processor(), "index_build_ms": index_ms},
           "source_file_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (corpus_path, questions_path)},
           "scope": {"corpus_documents": len(papers), "development_questions_excluded": 40,
                     "answer_generation": False, "labels": "Existing LLM-generated source-grounded questions; not independent human gold",
                     "failure_drill": variant == "broken-bm25"},
           "limitations": ["Queries were generated from source abstracts and favor lexical retrieval.",
                           "Single-process warm timings on this machine; not a concurrency or cloud-serving benchmark.",
                           "Zero external API fees excludes local compute cost.",
                           "This new run does not regenerate the historical LLM answer-quality tables."]}
    validate_run(run)
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["tfidf", "bm25", "broken-bm25"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run = evaluate(args.variant)
    atomic_json(args.output, run)
    # Round-trip the exact written bytes through the strict gate validator.
    validate_run(read_json(args.output))
    print(f"{args.variant}: {len(run['cases'])} tasks, "
          f"MRR@10={np.mean([case['score'] for case in run['cases']]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
