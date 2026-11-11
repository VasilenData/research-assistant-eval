"""Real-corpus retrieval comparison plus an explicitly destructive ranking drill."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from resassist.corpus import ROOT
from resassist.gate_retrieval import evaluate
from resassist.release_gate import atomic_json, main as gate_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "gate/evidence")
    parser.add_argument("--publish", type=Path, default=ROOT / "docs/gate/latest.json")
    args = parser.parse_args()
    for variant in ("tfidf", "bm25", "broken-bm25"):
        atomic_json(args.output / f"{variant}.json", evaluate(variant))
    common = ["--budget", str(ROOT / "gate/budget.json"), "--publish", str(args.publish)]
    passing = gate_main([str(args.output / "tfidf.json"), str(args.output / "bm25.json"),
                         "--report", str(args.output / "comparison.json"), *common])
    if passing != 0:
        raise SystemExit("The real candidate failed its configured gate; no new publication.")
    prior_bytes = args.publish.read_bytes()
    rejected = gate_main([str(args.output / "bm25.json"), str(args.output / "broken-bm25.json"),
                          "--report", str(args.output / "regression-drill.json"), *common])
    if rejected != 1 or args.publish.read_bytes() != prior_bytes:
        raise SystemExit("Regression drill did not reject or changed the last passing publication")
    atomic_json(args.output / "recovery-proof.json", {
        "scope": "Deliberately reversed real BM25 top-10 rankings; this is a recovery exercise, not an observed production incident",
        "regression_exit_code": rejected, "last_passing_preserved": True,
        "last_passing_sha256": hashlib.sha256(prior_bytes).hexdigest(),
    })
    print("Real retrieval comparison passed; intentional regression blocked; last passing report unchanged.")


if __name__ == "__main__":
    main()
