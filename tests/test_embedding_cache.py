"""Cache identity is tested without downloading or running an embedding model."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from resassist.retrieval import Dense, _vector_digest


@pytest.fixture
def embedder(monkeypatch):
    class FakeEmbedding:
        calls = 0

        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, docs, batch_size):
            type(self).calls += 1
            return [[len(doc), sum(map(ord, doc))] for doc in docs]

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeEmbedding))
    return FakeEmbedding


def test_cache_reused_only_for_the_same_ordered_corpus_and_model(tmp_path, embedder):
    cache = tmp_path / "embeddings.npy"
    first = Dense(["cat", "dogs"], cache=cache).vecs.copy()
    assert np.allclose(Dense(["cat", "dogs"], cache=cache).vecs, first)
    assert embedder.calls == 1
    reordered = Dense(["dogs", "cat"], cache=cache).vecs
    assert embedder.calls == 2 and np.allclose(reordered, first[::-1])
    Dense(["dogs", "cat"], model="different-model", cache=cache)
    assert embedder.calls == 3


@pytest.mark.parametrize("metadata", [None, [], "metadata"])
def test_non_object_metadata_is_rebuilt(tmp_path, embedder, metadata):
    cache = tmp_path / "embeddings.npy"
    expected = Dense(["cat", "dogs"], cache=cache).vecs.copy()
    cache.with_suffix(".npy.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert np.allclose(Dense(["cat", "dogs"], cache=cache).vecs, expected)
    assert embedder.calls == 2


@pytest.mark.parametrize("bad_vectors", [
    np.array([["cat", "dog"], ["dog", "cat"]]),
    np.full((2, 2), 1 + 2j),
    np.full((2, 2), np.finfo(np.float64).max),
])
def test_invalid_vector_types_or_ranges_are_rebuilt(tmp_path, embedder, bad_vectors):
    cache = tmp_path / "embeddings.npy"
    expected = Dense(["cat", "dogs"], cache=cache).vecs.copy()
    np.save(cache, bad_vectors)
    assert np.allclose(Dense(["cat", "dogs"], cache=cache).vecs, expected)
    assert embedder.calls == 2


def test_large_finite_vectors_normalize_without_float32_overflow(tmp_path, embedder):
    cache = tmp_path / "embeddings.npy"
    Dense(["cat", "dogs"], cache=cache)
    vectors = np.full((2, 2), 1e30, dtype=np.float32)
    np.save(cache, vectors)
    metadata_path = cache.with_suffix(".npy.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["vectors_digest"] = _vector_digest(vectors)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result = Dense(["cat", "dogs"], cache=cache).vecs
    assert embedder.calls == 1
    assert np.isfinite(result).all()
    assert np.allclose(np.linalg.norm(result, axis=1), 1)


def test_legacy_and_corrupt_vectors_are_rebuilt(tmp_path, embedder):
    cache = tmp_path / "embeddings.npy"
    np.save(cache, np.ones((2, 2)))  # No corpus identity: never silently trust it.
    Dense(["cat", "dogs"], cache=cache)
    assert embedder.calls == 1
    np.save(cache, np.array([[float("nan"), 1.], [2., 3.]]))
    assert np.isfinite(Dense(["cat", "dogs"], cache=cache).vecs).all()
    assert embedder.calls == 2
    # A crash between writing new vectors and their metadata must not mix corpora.
    np.save(cache, np.ones((2, 2), dtype=np.float32))
    Dense(["cat", "dogs"], cache=cache)
    assert embedder.calls == 3
