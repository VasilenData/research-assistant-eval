"""Build the evaluation set.

Answerable questions: a local LLM reads a sampled abstract and writes one specific question plus a one-sentence
reference answer. Questions are filtered by simple rules (see `keep`). The source paper is the gold document.
Limits: other papers may also answer a question, so recall is a lower bound; and an LLM wrote the questions.
A hand check of a sample is documented in the README.

Unanswerable questions are written by hand (data/unanswerable.jsonl): off-topic ones and ones that name
models or benchmarks that do not exist.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .corpus import Paper, load
from .retrieval import tokenize

SYSTEM = "You write test questions for a literature-search system. Follow the output format exactly."
PROMPT = """Abstract:
{abstract}

Write ONE specific question that this abstract answers, and its short answer.
Rules:
- The question must make sense on its own, without seeing the paper.
- Name the specific topic, method, dataset or setting, so only a few papers could answer it.
- Do not use the phrases "this paper", "the authors", "the study", "the abstract".
- The answer is one sentence taken from the facts in the abstract.

Output exactly two lines:
QUESTION: <question>
ANSWER: <answer>"""

BANNED = re.compile(r"\b(this paper|the authors?|the study|the abstract|the paper|this work)\b", re.I)
PARSE = re.compile(r"QUESTION:\s*(.+?)\s*\n\s*ANSWER:\s*(.+)", re.S)


def parse(text: str) -> tuple[str, str] | None:
    m = PARSE.search(text.strip())
    return (m.group(1).strip(), m.group(2).strip().split("\n")[0]) if m else None


def doc_freq(papers: list[Paper]) -> dict[str, int]:
    df: dict[str, int] = {}
    for p in papers:
        for t in set(tokenize(p.text)):
            df[t] = df.get(t, 0) + 1
    return df


def keep(question: str, answer: str, paper: Paper, df: dict[str, int] | None = None, n_docs: int = 0) -> bool:
    words = question.split()
    if not (8 <= len(words) <= 40) or not question.endswith("?") or BANNED.search(question):
        return False
    if len(answer.split()) < 5 or "\n" in question:
        return False
    q, t = set(tokenize(question)), set(tokenize(paper.title))
    if t and len(q & t) / len(t) > 0.7:          # question is basically the title: retrieval would be trivial
        return False
    if df is not None:  # specific enough: at least 2 rare terms (in under 1% of papers) so few papers can answer it
        if sum(1 for t in q if 1 <= df.get(t, 0) < max(2, 0.01 * n_docs)) < 2:
            return False
    return len(q & set(tokenize(paper.abstract))) >= 3


def build(llm, n_keep: int = 120, n_try: int = 220, seed: int = 11) -> list[dict]:
    papers = load()
    df = doc_freq(papers)
    rng = random.Random(seed)
    order = list(range(len(papers)))
    rng.shuffle(order)
    out = []
    for i in order[:n_try]:
        p = papers[i]
        parsed = parse(llm.chat(SYSTEM, PROMPT.format(abstract=p.abstract), max_tokens=110))
        if parsed and keep(*parsed, p, df, len(papers)):
            out.append({"question": parsed[0], "reference": parsed[1], "gold_id": p.id, "category": p.category})
        if len(out) >= n_keep:
            break
    return out


def main():
    from .llm import LLM
    root = Path(__file__).resolve().parents[2]
    rows = build(LLM(), n_keep=200, n_try=650)
    (root / "data" / "answerable.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(len(rows), "answerable questions written")


if __name__ == "__main__":
    main()
