# Retrieval regression release gate

## What is implemented

`resassist.release_gate` compares baseline and candidate **per-task evidence**. It
requires exactly the same task identities, question/reference hashes, document
groups, dataset version, corpus fingerprint, grader and measurement scopes.
Missing/duplicate tasks, invalid hashes, NaN/infinity, negative timings, malformed
costs or a different evaluation protocol fail closed. Failed tasks must remain in
the denominator with score zero; omitting them is invalid.

The gate accepts any producer using its version-1 run contract and a bounded,
higher-is-better score. The included, actually executed producer is **retrieval**.
There is no new real-model answer-generation adapter or independently reviewed
human gold set in this change. Historical LLM tables are unchanged.

## Measured demonstration

The November 11, 2026 local run uses all 2,835 committed arXiv abstracts and the
160 existing answerable questions after the first 40 development questions.
TF-IDF cosine search is compared with the repository's BM25. Neither has a tuned
parameter search on these questions. The raw questions, source IDs, rankings,
scores and timings are in [`gate/evidence`](../gate/evidence).

| System | MRR@10 | p95 warm search |
| --- | ---: | ---: |
| TF-IDF | 0.9854 | 0.79 ms |
| BM25 | 1.0000 | 4.95 ms |
| Reversed BM25 rankings (deliberate failure drill) | 0.1000 | 5.97 ms |

The candidate-minus-baseline MRR difference is 0.01458, with a paired
source-document bootstrap interval [0.00313, 0.03021] (2,000 resamples, seed 17).
All 160 questions have distinct source documents in this set; the implementation
also keeps multiple questions from one source together when resampling. The
mean is question-weighted, not an equal-weight average of group means.

**Do not interpret perfect retrieval as general accuracy.** These existing
questions were LLM-generated from their answer abstracts, filtered for rare
terms, and were used in earlier portfolio work. They are a regression set, not a
new blind generalization test. Retrieval of the designated source does not
measure correctness of a generated answer or whether other documents could
answer the question. Independent, naturally written questions remain needed.

Timings include query processing and ranking after one common warm-up. They
exclude index construction, network, generation and concurrency; raw reports
record index-build time and Python/package/platform identity. External API fees
are zero because this run makes no API calls. CPU, energy and hosting are
**unpriced**, not free. Benchmark changes on another machine are expected.

## Decision policy

[`gate/budget.json`](../gate/budget.json) declares the policy independently of the
candidate output: candidate mean score at least 0.85; lower endpoint of the paired
95% percentile interval no worse than -0.02; p95 warm search at most 100 ms;
zero external API fees and zero task errors. These are explicit project budgets,
not employer SLAs. All five conditions must pass.

The paired interval is exploratory, not a calibrated repeated-release testing
procedure. Repeated tuning against this set can overfit it. Version and refresh
the evaluation suite, freeze decisions before independent assessment, and add
human review when a probabilistic judge supplies scores.

Model fingerprints bind the retrieval configuration, corpus, package versions,
adapter bytes and retrieval source bytes. The prompt hash explicitly identifies
that retrieval has no generation prompt. Reports also include source-file
checksums, Git revision and a dirty-worktree flag. A dirty local run is identified
by its recorded source hashes, not by claiming the Git revision alone reproduces
uncommitted changes. Hosted CI uses the checked-out PR/push revision.

## Reproduce

```bash
pip install -r requirements-gate.txt
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/check_release_gate.py
python -m http.server 8000 --directory docs
# Open http://localhost:8000/gate/
```

The script generates real TF-IDF/BM25 evidence, requires the real candidate to
pass, then runs an intentionally damaged retrieval variant. It requires that
variant to fail with exit code 1 and verifies byte-for-byte preservation of
`docs/gate/latest.json`. `gate/evidence/recovery-proof.json` records the SHA-256.
This is a failure exercise, not an observed production incident.

To run individual stages or integrate another producer:

```bash
PYTHONPATH=src python -m resassist.gate_retrieval --variant bm25 --output candidate.json
PYTHONPATH=src python -m resassist.release_gate baseline.json candidate.json \
  --budget gate/budget.json --report comparison.json --publish docs/gate/latest.json
```

Exit 0 means all checks passed; exit 1 means valid evidence failed policy; exit 2
means invalid/incomparable evidence or an I/O failure. Each attempted valid
comparison gets its own report; only a passing result can replace the published
snapshot. Use distinct `--report` and `--publish` paths. The HTML viewer is static
and reads that one complete JSON snapshot, so it cannot combine metrics from
different runs. Publishing evidence does not reload or promote a served model.
The file boundary is an atomic same-filesystem replacement with one writer,
not a distributed transaction or a guarantee against disk/power loss.

CI runs this real-corpus gate on pushes and PRs alongside unit tests. A separate
weekly/manual workflow repeats it and uploads evidence; it does not automatically
commit results, notify a fabricated Slack workspace, or deploy a model. Branch
protection is not configured by this code: requiring the CI check before merging
is a separate repository setting.

## Test approach and references

Tests use hand-computed score/rank examples and cover paired identity, malformed
evidence, each budget, CLI exit codes, deliberate regression, and interrupted
publication. They adapt GitHub's official
[Python testing prompt](https://docs.github.com/en/copilot/tutorials/customization-library/custom-instructions/testing-automation)
to this repository. The design also follows the distinction between graders,
trials and regression suites in
[Anthropic's evaluation guide](https://www.anthropic.com/research/demystifying-evals-for-ai-agents).
These references do not validate this dataset or replace the recorded checks.
