"""The assistant: retrieve, answer from the sources with citations, abstain when the sources do not cover the question."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .corpus import Paper

ABSTAIN = "I can't answer this from the provided papers."

SYSTEM = (
    "You are a research assistant. Answer the question using ONLY the numbered sources. "
    "Write 1 to 3 short sentences. End every sentence with the number of the source that supports it, like [2]. "
    f"If the sources do not contain the answer, reply exactly: {ABSTAIN}"
)
CLOSED_BOOK_SYSTEM = (
    "You are a research assistant. Answer the question in 1 to 3 short sentences. "
    f"If you do not know, reply exactly: {ABSTAIN}"
)

CITE = re.compile(r"\[(\d+)\]")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


def build_prompt(question: str, papers: list[Paper]) -> str:
    sources = "\n\n".join(f"[{i}] {p.title}. {p.abstract}" for i, p in enumerate(papers, start=1))
    return f"Sources:\n{sources}\n\nQuestion: {question}"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT.split(text.strip()) if s.strip()]


def citations(sentence: str) -> list[int]:
    return [int(n) for n in CITE.findall(sentence)]


def strip_citations(sentence: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", sentence).strip()


def is_abstention(answer: str) -> bool:
    return answer.strip().lower().startswith("i can't answer") or answer.strip().lower().startswith("i cannot answer")


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Paper] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    abstained: bool = False
    reason: str = ""


class Assistant:
    def __init__(self, papers: list[Paper], retriever, llm, top_k: int = 4, min_score: float | None = None):
        self.papers, self.retriever, self.llm = papers, retriever, llm
        self.top_k, self.min_score = top_k, min_score

    def ask(self, question: str) -> Answer:
        hits = self.retriever.search(question, self.top_k)
        sources = [self.papers[i] for i, _ in hits]
        scores = [s for _, s in hits]
        if not hits:
            return Answer(question, ABSTAIN, [], [], True, "no retrieved sources")
        # Gate 1: retrieval confidence. Nothing relevant was found, so do not let the model improvise.
        if self.min_score is not None and scores[0] < self.min_score:
            return Answer(question, ABSTAIN, sources, scores, True, "low retrieval score")
        text = self.llm.chat(SYSTEM, build_prompt(question, sources))
        return Answer(question, text, sources, scores, is_abstention(text), "model abstained" if is_abstention(text) else "")

    def ask_closed_book(self, question: str) -> Answer:
        text = self.llm.chat(CLOSED_BOOK_SYSTEM, f"Question: {question}")
        return Answer(question, text, [], [], is_abstention(text))
