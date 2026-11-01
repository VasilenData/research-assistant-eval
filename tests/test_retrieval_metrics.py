from resassist import metrics
from resassist.evaluate import summarize
from resassist.retrieval import rrf, tokenize


def test_rrf_prefers_documents_ranked_well_by_both():
    assert rrf([[1, 2, 3], [3, 1, 4]])[:2] == [1, 3]


def test_tokenize_drops_stopwords():
    assert tokenize("What is the BM25 score?") == ["bm25", "score"]


def test_recall_and_mrr():
    assert metrics.recall_at_k([5, 6, 7], 6, 1) == 0 and metrics.recall_at_k([5, 6, 7], 6, 2) == 1
    assert metrics.reciprocal_rank([5, 6, 7], 7) == 1 / 3 and metrics.reciprocal_rank([5], 9) == 0


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = metrics.bootstrap_ci([1.0] * 50 + [0.0] * 50)
    assert lo < 0.5 < hi and hi - lo < 0.3


def row(kind, abstained, ref=0.0, n=1, sup=1, cite=1):
    return {"kind": kind, "abstained": abstained, "ref_entailed": ref, "n_sentences": n, "n_supported": sup,
            "n_valid_cite": cite, "seconds": 1.0}


def test_summarize_counts_abstentions_and_faithfulness():
    rows = [row("answerable", False, 1.0, 2, 2, 2), row("answerable", True, n=0, sup=0, cite=0),
            row("fabricated", True, n=0, sup=0, cite=0), row("off_topic", False, n=1, sup=0, cite=1)]
    s = summarize("x", rows)
    assert s["false_abstain_rate"] == 0.5 and s["correct_abstain_rate"] == 0.5
    assert s["faithfulness"] == 1.0 and s["answers_reference_fact"] == 0.5
