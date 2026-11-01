"""Hybrid retrieval: BM25 + dense embeddings fused with RRF, then a cross-encoder rerank."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an the of in on at to for and or is are was were be been by with as from that this it its what which who how why did does do has have had can".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def rrf(rankings: list[list[int]], k: int = 60, top: int = 50) -> list[int]:
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))][:top]


class BM25:
    def __init__(self, docs: list[str]):
        from rank_bm25 import BM25Okapi
        self.index = BM25Okapi([tokenize(d) for d in docs])

    def search(self, query: str, k: int = 50) -> list[int]:
        scores = self.index.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


class Dense:
    """Sentence embeddings with exact cosine search. 3,000 abstracts need no vector database."""

    def __init__(self, docs: list[str], model: str = "BAAI/bge-small-en-v1.5", cache: Path | None = None):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model)
        if cache and cache.exists():
            self.vecs = np.load(cache)
        else:
            self.vecs = np.array(list(self.model.embed(docs, batch_size=64)), dtype=np.float32)
            if cache:
                np.save(cache, self.vecs)
        self.vecs /= np.linalg.norm(self.vecs, axis=1, keepdims=True)

    def search(self, query: str, k: int = 50) -> list[int]:
        q = np.array(list(self.model.query_embed([query])))[0]
        q = q / np.linalg.norm(q)
        return [int(i) for i in np.argsort(-(self.vecs @ q), kind="stable")[:k]]


class Reranker:
    def __init__(self, docs: list[str], model: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.docs = docs
        self.model = TextCrossEncoder(model_name=model)

    def rerank(self, query: str, candidates: list[int], top: int = 5) -> list[tuple[int, float]]:
        scores = np.array(list(self.model.rerank(query, [self.docs[c] for c in candidates])))
        order = np.argsort(-scores, kind="stable")[:top]
        return [(candidates[i], float(scores[i])) for i in order]


class Retriever:
    """query -> [(doc_id, rerank_score)], best first."""

    def __init__(self, docs: list[str], cache: Path | None = None):
        self.docs = docs
        self.bm25, self.dense, self.reranker = BM25(docs), Dense(docs, cache=cache), Reranker(docs)

    def candidates(self, query: str, k: int = 30) -> list[int]:
        return rrf([self.bm25.search(query, k), self.dense.search(query, k)], top=k)

    def search(self, query: str, top: int = 5) -> list[tuple[int, float]]:
        return self.reranker.rerank(query, self.candidates(query), top)
