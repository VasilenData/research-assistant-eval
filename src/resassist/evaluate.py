"""Run the full evaluation and write results/results.json, results/report.md and results/samples.md.

    python -m resassist.evaluate

Stages: (1) retrieval quality of four retrievers, (2) abstention threshold chosen on a DEV split,
(3) answers on the TEST split for four systems, (4) validation of the NLI judge itself.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
import random
import re
import time
from pathlib import Path

import numpy as np

from . import metrics
from .assistant import ABSTAIN, Assistant, split_sentences, strip_citations
from .corpus import ROOT, load
from .evaluation_release import file_hashes, identify, publish, validate_tasks
from .retrieval import Retriever
from .verify import NLI, best_support, check_answer, guard

N_DEV = 40          # answerable questions used to choose the abstention threshold
N_DEV_UNANS = 10


def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def retrieval_eval(retriever: Retriever, answerable: list[dict], id_to_idx: dict[str, int]) -> dict:
    runs = {"bm25": [], "dense": [], "hybrid (RRF)": [], "hybrid + rerank": []}
    for row in answerable:
        q = row["question"]
        runs["bm25"].append(retriever.bm25.search(q, 10))
        runs["dense"].append(retriever.dense.search(q, 10))
        cands = retriever.candidates(q, 30)
        runs["hybrid (RRF)"].append(cands[:10])
        runs["hybrid + rerank"].append([i for i, _ in retriever.reranker.rerank(q, cands, 10)])
    golds = [id_to_idx[r["gold_id"]] for r in answerable]
    out = {}
    for name, ranked in runs.items():
        r5 = [metrics.recall_at_k(r, g, 5) for r, g in zip(ranked, golds)]
        out[name] = {
            "recall@1": metrics.mean([metrics.recall_at_k(r, g, 1) for r, g in zip(ranked, golds)]),
            "recall@5": metrics.mean(r5), "recall@5_ci": metrics.bootstrap_ci(r5),
            "recall@10": metrics.mean([metrics.recall_at_k(r, g, 10) for r, g in zip(ranked, golds)]),
            "mrr@10": metrics.mean([metrics.reciprocal_rank(r, g) for r, g in zip(ranked, golds)]),
        }
    return out


def mask_rare_terms(question: str, gold_text: str, df: dict[str, int], max_df: int = 3) -> str:
    """Drop the question words that are rare in the corpus and appear in the gold abstract.
    This removes the exact-term overlap that makes keyword search look perfect on generated questions."""
    gold_words = set(re.findall(r"[a-z0-9]+", gold_text.lower()))
    keep = [w for w in question.split() if not (re.sub(r"[^a-z0-9]", "", w.lower()) in gold_words
                                               and 0 < df.get(re.sub(r"[^a-z0-9]", "", w.lower()), 0) <= max_df)]
    return " ".join(keep)


def corpus_df(papers: list) -> dict[str, int]:
    df: dict[str, int] = {}
    for p in papers:
        for w in set(re.findall(r"[a-z0-9]+", p.text.lower())):
            df[w] = df.get(w, 0) + 1
    return df


def choose_threshold(retriever: Retriever, dev_ans: list[dict], dev_unans: list[dict]) -> tuple[float, dict]:
    """Retrieval-confidence gate: abstain when the best reranker score is below tau. Pick tau on DEV only."""
    if not dev_ans or not dev_unans:
        raise ValueError("both development classes are required")
    def top(question):
        hits = retriever.search(question, 1)
        if not hits or not np.isfinite(hits[0][1]):
            raise ValueError("development retrieval must return a finite top score")
        return hits[0][1]
    a = np.array([top(r["question"]) for r in dev_ans])
    u = np.array([top(r["question"]) for r in dev_unans])
    observed = np.unique(np.concatenate([a, u]))
    reject_all = np.nextafter(observed[-1], np.inf)
    if not np.isfinite(reject_all):
        raise ValueError("development scores exceed the supported finite range")
    # Include both extreme policies; ties prefer more abstention.
    candidates = np.r_[observed, reject_all]
    best = max(candidates, key=lambda t: (((a >= t).mean() + (u < t).mean()) / 2, t))
    return float(best), {"dev_answerable_pass": float((a >= best).mean()), "dev_unanswerable_blocked": float((u < best).mean())}



def judge_validation(nli: NLI, answerable: list[dict], papers: list, id_to_idx: dict[str, int], seed: int = 3) -> dict:
    """Is the NLI judge any good? Reference facts must be entailed by their own abstract and not by a random one."""
    rng = random.Random(seed)
    if len(papers) < 2 or not answerable:
        raise ValueError("judge validation requires examples and at least two papers")
    p = [best_support(nli, r["reference"], papers[id_to_idx[r["gold_id"]]]) for r in answerable]
    negatives = []
    for row in answerable:
        gold = id_to_idx[row["gold_id"]]
        other = rng.randrange(len(papers) - 1)
        negatives.append(papers[other + (other >= gold)])
    n = [best_support(nli, row["reference"], paper) for row, paper in zip(answerable, negatives)]
    return {"n_pairs": len(p), "supported_accepted": float(np.mean(np.array(p) >= 0.5)),
            "unsupported_rejected": float(np.mean(np.array(n) < 0.5)),
            "accuracy": float((np.sum(np.array(p) >= 0.5) + np.sum(np.array(n) < 0.5)) / (len(p) + len(n)))}


def summarize(name: str, rows: list[dict]) -> dict:
    ans = [r for r in rows if r["kind"] == "answerable"]
    una = [r for r in rows if r["kind"] != "answerable"]
    answered = [r for r in ans if not r["abstained"]]
    sent_total = sum(r["n_sentences"] for r in answered)
    out = {
        "system": name,
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "answers_reference_fact": metrics.mean([r["ref_entailed"] for r in ans]),
        "answers_reference_fact_ci": metrics.bootstrap_ci([r["ref_entailed"] for r in ans]),
        "false_abstain_rate": metrics.mean([float(r["abstained"]) for r in ans]),
        "correct_abstain_rate": metrics.mean([float(r["abstained"]) for r in una]),
        "unanswerable_answered_confidently": metrics.mean([float(not r["abstained"]) for r in una]),
        "faithfulness": (sum(r["n_supported"] for r in answered) / sent_total) if sent_total else None,
        "valid_citation_rate": (sum(r["n_valid_cite"] for r in answered) / sent_total) if sent_total else None,
        "mean_seconds": metrics.mean([r["seconds"] for r in rows]),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="Run a new immutable full evaluation; historical reports are preserved.")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output release already exists; choose a new --out path")
    root = ROOT
    source_paths = [root / "data" / name for name in ("arxiv_corpus.jsonl", "answerable.jsonl", "unanswerable.jsonl")]
    before_sources = file_hashes(source_paths)
    papers = load()
    id_to_idx = {p.id: i for i, p in enumerate(papers)}
    answerable, unanswerable = jl(root / "data" / "answerable.jsonl"), jl(root / "data" / "unanswerable.jsonl")
    answerable, unanswerable = identify(answerable, "answerable"), identify(unanswerable, "unanswerable")
    validate_tasks(answerable, unanswerable, [p.id for p in papers])
    retriever = Retriever([p.text for p in papers], cache=root / "data" / "embeddings.npy")
    results: dict = {"corpus_papers": len(papers), "answerable": len(answerable), "unanswerable": len(unanswerable)}

    print("retrieval...", flush=True)
    results["retrieval"] = retrieval_eval(retriever, answerable, id_to_idx)
    df = corpus_df(papers)
    masked = [dict(r, question=mask_rare_terms(r["question"], papers[id_to_idx[r["gold_id"]]].text, df)) for r in answerable]
    results["retrieval_masked"] = retrieval_eval(retriever, masked, id_to_idx)
    results["masked_example"] = {"original": answerable[0]["question"], "masked": masked[0]["question"]}

    dev_a, test_a = answerable[:N_DEV], answerable[N_DEV:]
    dev_u, test_u = unanswerable[:N_DEV_UNANS // 2] + unanswerable[15:15 + N_DEV_UNANS // 2], \
        unanswerable[N_DEV_UNANS // 2:15] + unanswerable[15 + N_DEV_UNANS // 2:]
    if not test_a or not test_u:
        raise ValueError("both test classes must remain after the fixed development split")
    tau, tau_info = choose_threshold(retriever, dev_a, dev_u)
    results["abstention_threshold"] = {"tau": tau, **tau_info, "dev_answerable": len(dev_a), "dev_unanswerable": len(dev_u)}

    from .llm import LLM
    llm, nli = LLM(), NLI()
    results["judge_validation"] = judge_validation(nli, answerable, papers, id_to_idx)
    asst = Assistant(papers, retriever, llm, top_k=4, min_score=None)

    items = [dict(r, kind="answerable") for r in test_a] + [dict(r, kind=r["kind"]) for r in test_u]
    systems: dict[str, list[dict]] = {k: [] for k in ("closed-book LLM", "RAG", "RAG + retrieval gate", "RAG + gate + claim guard")}
    samples = []
    for n, it in enumerate(items, 1):
        q = it["question"]
        gold = papers[id_to_idx[it["gold_id"]]] if it.get("gold_id") else None
        for system in systems:
            t0 = time.time()
            if system == "closed-book LLM":
                a = asst.ask_closed_book(q)
            elif system == "RAG":
                a = rag = asst.ask(q)
                rag_secs = time.time() - t0 + (llm.last_seconds - llm.last_wall if rag.sources else 0)
            elif system == "RAG + retrieval gate":
                a = rag if rag.scores and rag.scores[0] >= tau else type(rag)(q, ABSTAIN, rag.sources, rag.scores, True, "low retrieval score")
                gate_ans = a
            else:
                a = gate_ans
                if not a.abstained:
                    kept, _ = guard(nli, a.text, a.sources)
                    a = type(a)(q, kept or ABSTAIN, a.sources, a.scores, not kept, "claims unsupported" if not kept else "")
            secs = {"RAG + retrieval gate": rag_secs if system == "RAG + retrieval gate" else None}.get(system)
            if system == "RAG + gate + claim guard":
                secs = rag_secs + (time.time() - t0)
            elif system == "RAG":
                secs = rag_secs
            elif secs is None:
                secs = time.time() - t0 - llm.last_wall + llm.last_seconds
            # claims are judged against the sources the answer cites; closed-book has none, so judge it against the gold paper
            judge_sources = a.sources if a.sources else ([gold] if gold else [])
            if a.abstained or not judge_sources:
                checks = []
            elif system == "closed-book LLM":
                from .verify import SentenceCheck
                sents = split_sentences(a.text)
                scores = [best_support(nli, s, judge_sources[0]) for s in sents]
                checks = [SentenceCheck(s, [1], True, sc, sc >= 0.5, 1) for s, sc in zip(sents, scores)]
            else:
                checks = check_answer(nli, a.text, a.sources)
            ref_entailed = 0.0
            if it["kind"] == "answerable" and not a.abstained:
                ref_entailed = float(nli.entailment([(strip_citations(a.text), it["reference"])])[0] >= 0.5)
            systems[system].append({
                "task_id": it["task_id"], "input_sha256": it["input_sha256"], "reference_sha256": it["reference_sha256"],
                "answer": a.text, "source_ids": [p.id for p in a.sources], "retrieval_scores": a.scores,
                "kind": "answerable" if it["kind"] == "answerable" else it["kind"], "abstained": a.abstained,
                "ref_entailed": ref_entailed, "n_sentences": len(checks), "n_supported": sum(c.supported for c in checks),
                "n_valid_cite": sum(c.valid_citation for c in checks), "seconds": secs,
                "gold_in_sources": bool(gold and gold in a.sources)})
            if system == "RAG" and len(samples) < 14 and (n % 7 == 0):
                samples.append({"q": q, "kind": it["kind"], "answer": a.text, "sources": [p.title for p in a.sources],
                                "gold": gold.title if gold else None})
        if n % 10 == 0:
            print(f"{n}/{len(items)}", flush=True)

    # generation time of the RAG answer is shared by the gated/guarded systems; report it honestly
    results["systems"] = [summarize(k, v) for k, v in systems.items()]
    results["gold_in_top4_sources"] = metrics.mean([float(r["gold_in_sources"]) for r in systems["RAG"] if r["kind"] == "answerable"])
    results["test_split"] = {"answerable": len(test_a), "unanswerable": len(test_u)}
    if file_hashes(source_paths) != before_sources:
        raise ValueError("source data changed during evaluation; no release published")
    source_identity = {Path(name).name: value for name, value in before_sources.items()}
    code_identity = {Path(name).name: value for name, value in file_hashes(list(Path(__file__).parent.glob("*.py"))).items()}
    provenance = {"sources_sha256": source_identity, "code_sha256": code_identity, "python": platform.python_version(),
                  "numpy": np.__version__, "llm": llm.provenance,
                  "development_task_ids": [r["task_id"] for r in dev_a + dev_u],
                  "threshold_rule": "balanced development accuracy; ties favor abstention; includes reject-all policy",
                  "judging": "same NLI judge as the historical evaluator; no independent human labels"}
    publish(args.out, results, samples, systems, items, provenance)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
