import copy
import json

import pytest

from resassist.evaluation_release import identify, publish, validate_tasks
from resassist.evaluate import choose_threshold


def tasks():
    return identify([{"question": "How does the system recover?", "gold_id": "paper-1", "reference": "It replays events."}], "answerable")


def systems():
    row = {**tasks()[0], "answer": "It replays events [1].", "source_ids": ["paper-1"], "seconds": .1}
    return {"baseline": [copy.deepcopy(row)], "candidate": [copy.deepcopy(row)]}


def test_task_identity_changes_with_reference_or_input_but_not_order():
    original = tasks()[0]
    changed = identify([{**original, "reference": "Another fact"}], "answerable")[0]
    assert original["input_sha256"] == changed["input_sha256"]
    assert original["reference_sha256"] != changed["reference_sha256"]
    assert original["task_id"] != changed["task_id"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_tasks(tasks() * 2, [], ["paper-1"])
    with pytest.raises(ValueError, match="gold"):
        validate_tasks(tasks(), [], ["another-paper"])


@pytest.mark.parametrize("mutation", [lambda s: s["candidate"].clear(),
                                      lambda s: s["candidate"].append(copy.deepcopy(s["candidate"][0])),
                                      lambda s: s["candidate"][0].update(reference_sha256="different"),
                                      lambda s: s["candidate"][0].update(seconds=float("nan"))])
def test_incomplete_or_incomparable_generation_evidence_cannot_publish(tmp_path, mutation):
    rows = systems()
    mutation(rows)
    with pytest.raises(ValueError):
        publish(tmp_path / "release", {}, [], rows, tasks(), {})
    assert not (tmp_path / "release").exists()


def test_atomic_bundle_contains_all_files_and_rejects_overwrite(tmp_path, monkeypatch):
    from resassist import evaluation_release
    old = tmp_path / "old"
    publish(old, {"systems": []}, [], systems(), tasks(), {"model": "fake test double"})
    expected = {p.name: p.read_bytes() for p in old.iterdir()}
    assert set(expected) == {"results.json", "samples.json", "per_task.json", "protocol.json", "manifest.json"}
    with pytest.raises(FileExistsError):
        publish(old, {}, [], systems(), tasks(), {})
    def fail(*args):
        raise OSError("injected final rename interruption")
    monkeypatch.setattr(evaluation_release.os, "rename", fail)
    with pytest.raises(OSError):
        publish(tmp_path / "new", {}, [], systems(), tasks(), {})
    assert {p.name: p.read_bytes() for p in old.iterdir()} == expected
    assert list(tmp_path.iterdir()) == [old]


def test_threshold_ties_allow_abstaining_on_every_question():
    class Retriever:
        def search(self, question, k):
            return [(0, 1.)]
    threshold, info = choose_threshold(Retriever(), [{"question": "a"}], [{"question": "u"}])
    assert threshold > 1 and info == {"dev_answerable_pass": 0, "dev_unanswerable_blocked": 1}


def test_full_evaluator_integration_uses_fake_models_and_does_not_touch_history(tmp_path, monkeypatch):
    from resassist import evaluate, llm
    from resassist.corpus import Paper
    data = tmp_path / "data"
    data.mkdir()
    ans = [{"question": f"Question {i}?", "gold_id": "paper-1", "reference": "Reference fact."} for i in range(41)]
    una = [{"question": f"Unsupported {i}?", "kind": "off_topic" if i < 15 else "fabricated"} for i in range(30)]
    for name, rows in [("answerable.jsonl", ans), ("unanswerable.jsonl", una), ("arxiv_corpus.jsonl", [])]:
        (data / name).write_text("\n".join(json.dumps(row) for row in rows))
    papers = [Paper("paper-1", "Title", "Reference fact.", "2026-01-01", "test"),
              Paper("paper-2", "Other", "Different fact.", "2026-01-01", "test")]
    class Retriever:
        def __init__(self, *args, **kwargs):
            self.bm25 = self.dense = self.reranker = self
        def search(self, q, k):
            return [(0, 1.)] if k == 1 or k == 4 else [0, 1]
        def candidates(self, q, k):
            return [0, 1]
        def rerank(self, q, candidates, k):
            return [(0, 1.), (1, .5)]
    class LLM:
        last_wall = last_seconds = 0.
        provenance = {"model": "fake test double; no quality benchmark"}
        def chat(self, *args):
            return "Reference fact [1]."
    class NLI:
        def entailment(self, pairs):
            return [1.] * len(pairs)
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "load", lambda: papers)
    monkeypatch.setattr(evaluate, "Retriever", Retriever)
    monkeypatch.setattr(evaluate, "NLI", NLI)
    monkeypatch.setattr(llm, "LLM", LLM)
    history = tmp_path / "results" / "results.json"
    history.parent.mkdir()
    history.write_text("historical evidence")
    destination = tmp_path / "new"
    monkeypatch.setattr("sys.argv", ["evaluate", "--out", str(destination)])
    evaluate.main()
    rows = json.loads((destination / "per_task.json").read_text())
    assert len(rows) == 4 and all(len(records) == 21 for records in rows.values())
    assert history.read_text() == "historical evidence"
    assert json.loads((destination / "protocol.json").read_text())["provenance"]["llm"]["model"].startswith("fake")
