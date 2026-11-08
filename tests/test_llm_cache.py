"""Response-cache provenance and failure safety without loading or downloading model weights."""
import json
import sys
from types import SimpleNamespace

import pytest

from resassist.llm import LLM


@pytest.fixture
def setup_model(tmp_path, monkeypatch):
    class FakeLlama:
        calls = 0
        def __init__(self, **kwargs):
            self.config = kwargs
        def create_chat_completion(self, **kwargs):
            type(self).calls += 1
            return {"choices": [{"message": {"content": f"answer {type(self).calls}"}}]}
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama, __version__="test-1"))
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"weights-v1")
    cache = tmp_path / "nested" / "responses.jsonl"
    monkeypatch.setenv("RESASSIST_CACHE", str(cache))
    return weights, cache, FakeLlama


def test_responses_are_reused_in_memory_and_after_restart(setup_model):
    weights, _, fake = setup_model
    first = LLM(weights, threads=1)
    answer = first.chat("system", "question")
    original_seconds = first.last_seconds
    assert first.chat("system", "question") == answer
    restarted = LLM(weights, threads=1)
    assert restarted.chat("system", "question") == answer
    assert restarted.last_seconds == original_seconds and fake.calls == 1


def test_changing_weights_at_the_same_path_invalidates_old_answers(setup_model):
    weights, _, fake = setup_model
    old = LLM(weights, threads=1).chat("system", "question")
    weights.write_bytes(b"weights-v2")
    assert LLM(weights, threads=1).chat("system", "question") != old
    assert fake.calls == 2


def test_context_runtime_and_generation_parameters_are_part_of_cache_identity(setup_model, monkeypatch):
    weights, _, fake = setup_model
    LLM(weights, n_ctx=100, threads=1).chat("system", "question", max_tokens=5)
    LLM(weights, n_ctx=200, threads=1).chat("system", "question", max_tokens=5)
    monkeypatch.setattr(sys.modules["llama_cpp"], "__version__", "test-2")
    model = LLM(weights, n_ctx=200, threads=1)
    model.chat("system", "question", max_tokens=5)
    model.chat("system", "question", max_tokens=6)
    assert fake.calls == 4


@pytest.mark.parametrize("contents", [b'{"k":"legacy","t":"stale","s":1}\n', b'{"broken', b'\xff'])
def test_legacy_and_corrupt_caches_are_regenerated(setup_model, contents):
    weights, cache, fake = setup_model
    cache.parent.mkdir(parents=True)
    cache.write_bytes(contents)
    assert LLM(weights, threads=1).chat("system", "question") == "answer 1"
    row = json.loads(cache.read_text())
    assert row["identity"] and fake.calls == 1


def test_failed_cache_replacement_preserves_previous_snapshot(setup_model, monkeypatch):
    from resassist import llm
    weights, cache, _ = setup_model
    model = LLM(weights, threads=1)
    model.chat("system", "first")
    before = cache.read_bytes()
    def fail_replace(*args):
        raise OSError("injected write failure")
    monkeypatch.setattr(llm.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        model.chat("system", "second")
    assert cache.read_bytes() == before and len(model.cache) == 1
    assert not list(cache.parent.glob("*.tmp"))


def test_invalid_cached_duration_cannot_pollute_evaluation_timing(setup_model):
    weights, cache, fake = setup_model
    LLM(weights, threads=1).chat("system", "question")
    row = json.loads(cache.read_text())
    row["s"] = float("nan")
    cache.write_text(json.dumps(row), encoding="utf-8")
    LLM(weights, threads=1).chat("system", "question")
    assert fake.calls == 2


def test_cache_cannot_replace_model_weights(setup_model, monkeypatch):
    weights, _, _ = setup_model
    monkeypatch.setenv("RESASSIST_CACHE", str(weights))
    with pytest.raises(ValueError, match="overwrite"):
        LLM(weights)
    assert weights.read_bytes() == b"weights-v1"
