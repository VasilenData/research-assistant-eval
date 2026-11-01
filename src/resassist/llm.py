"""Local LLM through llama.cpp (GGUF, CPU). Greedy decoding, fixed seed, so every eval run is repeatable."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

DEFAULT_MODEL = Path(os.environ.get("RESASSIST_MODEL", Path.home() / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"))


class LLM:
    def __init__(self, path: Path = DEFAULT_MODEL, n_ctx: int = 4096, threads: int | None = None):
        from llama_cpp import Llama
        self.cache_path = Path(os.environ.get("RESASSIST_CACHE", "")) if os.environ.get("RESASSIST_CACHE") else None
        self.cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["k"]] = row
        self.last_seconds = self.last_wall = 0.0
        self.llm = Llama(model_path=str(path), n_ctx=n_ctx, n_threads=threads or os.cpu_count(), seed=7, verbose=False)

    def chat(self, system: str, user: str, max_tokens: int = 220) -> str:
        """Greedy decoding is deterministic, so answers can be cached on disk to resume an interrupted evaluation.
        The cache stores the original generation time, which the evaluation reports instead of the lookup time."""
        key = hashlib.sha256(json.dumps([system, user, max_tokens]).encode()).hexdigest()
        t0 = time.time()
        if key in self.cache:
            self.last_seconds, text = self.cache[key]["s"], self.cache[key]["t"]
        else:
            out = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=0.0)
            text = out["choices"][0]["message"]["content"].strip()
            self.last_seconds = time.time() - t0
            if self.cache_path:
                with self.cache_path.open("a") as f:
                    f.write(json.dumps({"k": key, "t": text, "s": self.last_seconds}) + "\n")
        self.last_wall = time.time() - t0
        return text
