# Evaluated AI research assistant

[![ci](https://github.com/VasilenData/research-assistant-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/VasilenData/research-assistant-eval/actions/workflows/ci.yml)

A small research assistant that answers questions from 2,835 arXiv abstracts, plus the evaluation that shows
**which parts of it actually help**. The assistant uses hybrid retrieval, a local 1.5B model, an abstention gate and a claim checker.
The point of the repo is the measurement: every component is added, switched off and compared on held-out questions, with confidence intervals, and the checker's own accuracy is measured too.

Everything runs on a CPU. No API keys, no paid model.

## What was measured

Corpus: 2,835 arXiv abstracts (cs.CL, cs.IR, cs.LG, cs.DB) from June to October 2026, selected to reduce overlap with the answer model's training period; publication dates alone do not prove the absence of memorized or related knowledge.
Test split: 160 answerable and 20 unanswerable questions. The threshold was chosen on a separate dev split (40 + 10).

| System | Answers contain the reference fact | Wrongly abstains | Correctly abstains on unanswerable | Judge-supported claims on answered, answerable questions |
|---|---|---|---|---|
| Closed-book model | 3.8% (95% CI 1.3 to 6.9) | 22.5% | 35% | 17% |
| RAG | **50.6%** (43.1 to 58.8) | 0% | 75% | 76% |
| RAG + retrieval gate | 45.6% (38.1 to 53.8) | 10% | **100%** | 77% |
| RAG + gate + claim guard | 35.6% (28.1 to 43.1) | 20.6% | 100% | **100%** |

What this says:

- **Retrieval is what makes the model useful.** Same model, same questions: 3.8% closed-book against 50.6% with retrieved abstracts.
- **The retrieval gate improved abstention on this test split.** Without it, 25% of unanswerable questions (off-topic, or naming models and benchmarks that do not exist) got a confident answer. With it, none did. The price is 10% of answerable questions refused.
- **The claim guard trades coverage for trust.** It removes sentences that this NLI judge does not rate as supported and adds an attributed `[n]` citation to the survivors. This is model-based checking, not independent verification. Supported-claim rate goes to 100% by construction, and the share of answers containing the reference fact drops from 45.6% to 35.6%. Use it where a wrong claim is worse than a missing one.
- **The 1.5B model almost never writes citations on its own.** Prompted to end each sentence with `[n]`, it did so for only about 7% of sentences. That is why the guard attributes each sentence to the best-supporting source after generation instead of trusting the model's citations.

### Retrieval quality

Recall of the one abstract each question was written from (200 questions):

| Retriever | Recall@1 | Recall@5 | MRR@10 | Recall@5 with rare terms removed |
|---|---|---|---|---|
| BM25 | 1.000 | 1.000 | 1.000 | 0.960 |
| Dense (bge-small) | 0.835 | 0.920 | 0.872 | 0.795 |
| Hybrid (RRF) | 0.955 | 0.995 | 0.973 | 0.955 |
| Hybrid + cross-encoder rerank | 0.990 | 1.000 | 0.995 | 0.965 |

The questions were generated from the abstracts and filtered to be specific, so they share exact terms with the answer and BM25 scores a perfect 1.000. That number is **inflated by construction** and should not be read as BM25 being perfect.
To show it, the last column removes from each question the words that are rare in the corpus and appear in the gold abstract. Keyword search then drops to 0.96, dense to 0.795, and the reranked hybrid holds at 0.965.
The masked experiment suggests less sensitivity to this particular lexical perturbation for the reranked system. It does not establish the best retriever for naturally written user questions, which were not evaluated.

### The judge is not ground truth

Claim checking uses an NLI model (`cross-encoder/nli-deberta-v3-small`). Its accuracy was measured on 200 true pairs (a reference fact against its own abstract) and 200 assumed mismatched pairs (against a random abstract; these were not manually proven to be unsupported):

| | |
|---|---|
| True facts accepted | 73% |
| False facts rejected | 98% |
| Accuracy | 85.5% |

Feeding the whole abstract as the premise gave only 61% accuracy and accepted 22.5% of true facts, so the judge uses overlapping two-sentence windows of the abstract instead. Both false negatives and false positives are possible, so the "contains the reference fact" and "supported" numbers are judge-dependent estimates, not proven lower bounds or ground truth.

## How it works

```
question -> BM25 + bge-small dense -> RRF fusion -> MiniLM cross-encoder rerank -> top 4 abstracts
         -> gate: best rerank score below tau? -> "I can't answer this from the provided papers."
         -> Qwen2.5-1.5B (greedy, fixed seed) answers using only the numbered sources
         -> guard: each sentence must be entailed by a source window, else it is dropped; survivors get [n]
```

| File | Purpose |
|---|---|
| `src/resassist/retrieval.py` | BM25, dense search, RRF, reranker |
| `src/resassist/assistant.py` | Prompts, abstention gate, answer object |
| `src/resassist/verify.py` | Windowed NLI claim checking, citation attribution, guard |
| `src/resassist/evaluate.py` | The whole experiment, writes `results/` |
| `src/resassist/metrics.py` | Recall, MRR, bootstrap confidence intervals |
| `scripts/fetch_corpus.py` | Rebuilds the corpus from the arXiv API |
| `data/answerable.jsonl`, `data/unanswerable.jsonl` | The 200 + 30 evaluation questions |
| `results/results.json`, `results/samples.json` | Raw output of the run reported above |

## Run it

```bash
pip install -r requirements.txt
# download the model once (Apache 2.0, about 1.1 GB):
#   Qwen2.5-1.5B-Instruct, file qwen2.5-1.5b-instruct-q4_k_m.gguf, from huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF
export RESASSIST_MODEL=/path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf
export RESASSIST_CACHE=results/.llm_cache.jsonl     # optional: resume an interrupted run
PYTHONPATH=src python -m resassist.evaluate         # about 1 hour on one CPU core
```

Tests need only numpy and pytest, because the model, retrievers and judge are replaced by fakes:

```bash
pip install -r requirements-dev.txt && PYTHONPATH=src pytest -q    # deterministic tests; no model downloads
```

## Limits

- **The questions are LLM-generated.** 200 answerable questions were written by a language model from the abstracts and filtered for specificity (at least two rare terms, no "this paper", not a copy of the title). I spot-checked 12: most are specific and a few are vague. Treat absolute numbers as indicative and the differences between systems as the finding.
- **The 30 unanswerable questions are hand-written**: 15 off-topic and 15 naming models or benchmarks that do not exist. Only 20 are in the test split, so their rates move in 5-point steps.
- **The generator and the system share a model family.** Questions came from an LLM and answers come from one, which can flatter fluency-based metrics. The retrieval table and the judge validation do not depend on this.
- **One small model, one corpus, one run.** Qwen2.5-1.5B with greedy decoding. Results are repeatable but not averaged over seeds, and a larger model would score higher.
- **Latency is CPU, one core**, about 5.9 s per RAG answer, almost all of it generation. The claim guard adds NLI checking time on top, which the evaluation does not report separately.
- Model weights and the embedding cache are not committed. The corpus is public arXiv metadata.

## Decisions

- **Hybrid before dense.** Dense alone missed one question in twelve at rank 5. Fusing it with BM25 recovers that, and reranking fixes the top position.
- **Threshold chosen on dev only.** The abstention threshold never saw the test questions.
- **Judge before trusting the guard.** The first version fed whole abstracts to the NLI model, and the guard then deleted nearly every answer (judge accuracy 61%). Measuring the judge exposed that, and switching to sentence windows fixed it.
- **Attribute citations after generation.** The small model does not cite reliably, so the guard assigns and verifies citations itself.

## Reliability checks
The assistant abstains without generation when retrieval returns no sources. Embedding caches are checked against the ordered corpus, model name and vector contents; legacy, mismatched or corrupt caches are rebuilt. Claim-score caches distinguish changed source text and judge instances. Negative judge-validation pairs exclude the gold document, although another document may still support the fact.

The stored model-evaluation tables are historical measurements, not regenerated by the lightweight regression suite. The guard is scored by the same NLI judge used to filter it, so a 100% judge-supported rate is circular and does not prove factual correctness.
