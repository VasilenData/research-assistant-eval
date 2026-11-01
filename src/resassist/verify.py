"""Claim verification: does the cited paper actually support each sentence of the answer?

Uses a natural-language-inference cross-encoder (premise = cited abstract, hypothesis = the sentence).
It is a judge, not ground truth. Its accuracy is measured separately in evaluate.py (judge_validation).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .assistant import citations, split_sentences, strip_citations
from .corpus import Paper

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"  # label order: contradiction, entailment, neutral


class NLI:
    def __init__(self, model: str = NLI_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model, max_length=512)

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        logits = np.asarray(self.model.predict(pairs, batch_size=8))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return [float(p) for p in probs[:, 1]]


def windows(paper: Paper, size: int = 2) -> list[str]:
    """Overlapping sentence windows of the abstract (plus the title). A claim is usually backed by one or two
    sentences, and a small NLI model is much more reliable on short premises than on a whole abstract."""
    sents = split_sentences(paper.abstract) or [paper.abstract]
    wins = [" ".join(sents[i:i + size]) for i in range(max(1, len(sents) - size + 1))]
    return [f"{paper.title}. {w}" for w in wins]


_WORD = re.compile(r"[a-z0-9]+")
_CACHE: dict[tuple, float] = {}


def best_support(nli, hypothesis: str, paper: Paper, top_windows: int = 3) -> float:
    """Max entailment over the few abstract windows that share the most words with the claim (a cheap lexical
    pre-filter keeps CPU cost bounded; results are memoised because several systems judge the same text)."""
    key = (id(nli), hypothesis, paper.id)
    if key not in _CACHE:
        h = set(_WORD.findall(hypothesis.lower()))
        wins = sorted(windows(paper), key=lambda w: -len(h & set(_WORD.findall(w.lower()))))[:top_windows]
        _CACHE[key] = max(nli.entailment([(w, hypothesis) for w in wins]), default=0.0)
    return _CACHE[key]


@dataclass
class SentenceCheck:
    sentence: str
    cited: list[int]
    valid_citation: bool      # the MODEL cited a source number that was really provided
    support: float            # best entailment over candidate sources
    supported: bool
    source: int = 0           # 1-based source that best supports it (cited one, or found post hoc)


def check_answer(nli, answer_text: str, sources: list[Paper], threshold: float = 0.5, attribute: bool = True) -> list[SentenceCheck]:
    """Judge each sentence. If the model cited valid sources, check those; otherwise (small models often forget)
    attribute the sentence to whichever provided source supports it best, when attribute=True."""
    out = []
    for s in split_sentences(answer_text):
        cited = citations(s)
        valid = bool(cited) and all(1 <= c <= len(sources) for c in cited)
        cands = cited if valid else (list(range(1, len(sources) + 1)) if attribute else [])
        hyp = strip_citations(s)
        best, best_c = 0.0, 0
        for c in cands:
            sc = best_support(nli, hyp, sources[c - 1])
            if sc > best:
                best, best_c = sc, c
        out.append(SentenceCheck(s, cited, valid, best, best >= threshold, best_c))
    return out


def _cite(sentence: str, n: int) -> str:
    if not sentence:
        return ""
    return f"{sentence[:-1]} [{n}]{sentence[-1]}" if sentence[-1:] in ".!?" else f"{sentence} [{n}]"


def guard(nli, answer_text: str, sources: list[Paper], threshold: float = 0.5) -> tuple[str, list[SentenceCheck]]:
    """Keep only supported sentences, each ending with a verified citation. If nothing survives, the caller abstains."""
    checks = check_answer(nli, answer_text, sources, threshold)
    kept = [_cite(strip_citations(c.sentence), c.source) for c in checks if c.supported and strip_citations(c.sentence)]
    return " ".join(kept), checks
