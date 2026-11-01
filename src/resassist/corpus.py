from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    abstract: str
    published: str
    category: str

    @property
    def text(self) -> str:
        return f"{self.title}. {self.abstract}"

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.id}"


def load(path: str | Path = ROOT / "data" / "arxiv_corpus.jsonl") -> list[Paper]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [Paper(**r) for r in rows]
