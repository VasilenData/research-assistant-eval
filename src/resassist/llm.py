"""Local LLM through llama.cpp (GGUF, CPU), with model-specific resumable response caching."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

DEFAULT_MODEL = Path(os.environ.get("RESASSIST_MODEL", Path.home() / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"))


class LLM:
    def __init__(self, path: Path = DEFAULT_MODEL, n_ctx: int = 4096, threads: int | None = None):
        import llama_cpp
        path = Path(path)
        threads = (os.cpu_count() or 1) if threads is None else threads
        for name, value in (("n_ctx", n_ctx), ("threads", threads)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.cache_path = Path(os.environ.get("RESASSIST_CACHE", "")) if os.environ.get("RESASSIST_CACHE") else None
        if self.cache_path and self.cache_path.resolve() == path.resolve():
            raise ValueError("response cache must not overwrite the model file")
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for block in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(block)
        identity = {"format_version": 2, "model_sha256": digest.hexdigest(), "n_ctx": n_ctx,
                    "threads": threads, "runtime": getattr(llama_cpp, "__version__", "unknown"),
                    "seed": 7, "temperature": 0.0}
        self.provenance = identity.copy()
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                lines = self.cache_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                lines = []
            for line in lines:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # Interrupted legacy append or corrupt line: regenerate, never trust it.
                if (isinstance(row, dict) and row.get("identity") == self.identity
                        and isinstance(row.get("k"), str) and isinstance(row.get("t"), str)
                        and isinstance(row.get("s"), (int, float)) and not isinstance(row["s"], bool)
                        and math.isfinite(row["s"]) and row["s"] >= 0):
                    self.cache[row["k"]] = row
        self.last_seconds = self.last_wall = 0.0
        self.llm = llama_cpp.Llama(model_path=str(path), n_ctx=n_ctx, n_threads=threads, seed=7, verbose=False)

    def _persist(self, cache: dict[str, dict]) -> None:
        """Replace one complete cache snapshot; callers must serialize writers to this path."""
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.cache_path.parent,
                                             prefix=".llm-cache-", suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                for row in cache.values():
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.cache_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def chat(self, system: str, user: str, max_tokens: int = 220) -> str:
        """Resume identical requests only for the same model bytes and generation configuration.
        The cache stores the original generation time, which the evaluation reports instead of the lookup time."""
        if not isinstance(system, str) or not isinstance(user, str):
            raise ValueError("system and user prompts must be strings")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        key = hashlib.sha256(json.dumps([self.identity, system, user, max_tokens]).encode()).hexdigest()
        t0 = time.perf_counter()
        if key in self.cache:
            self.last_seconds, text = self.cache[key]["s"], self.cache[key]["t"]
        else:
            out = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=0.0)
            text = out["choices"][0]["message"]["content"].strip()
            self.last_seconds = time.perf_counter() - t0
            updated = {**self.cache, key: {"identity": self.identity, "k": key, "t": text, "s": self.last_seconds}}
            self._persist(updated)
            self.cache = updated
        self.last_wall = time.perf_counter() - t0
        return text
