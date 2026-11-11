from copy import deepcopy
import json

import pytest

from resassist.release_gate import (
    Budget, atomic_json, compare, digest, main, publish_passing, validate_run,
)
from resassist.gate_retrieval import reciprocal_rank, task_manifest


def run(scores=(1.0, 0.5, 0.0, 1.0), name="baseline"):
    cases = [{"task_id": str(i), "group_id": f"paper-{i // 2}",
              "input_sha256": digest(f"question-{i}"), "reference_sha256": digest(f"answer-{i}"),
              "score": score, "latency_ms": 10.0, "cost_usd": 0.0, "error": False}
             for i, score in enumerate(scores)]
    manifest = [{key: item[key] for key in ("task_id", "group_id", "input_sha256", "reference_sha256")}
                for item in cases]
    return {"schema_version": 1, "dataset_version": "test-v1", "dataset_sha256": digest(manifest),
            "corpus_sha256": digest("corpus"), "model_sha256": digest(name), "prompt_sha256": digest("prompt"),
            "system_name": name, "created_utc": "2026-11-11T00:00:00Z", "code_revision": "fixture",
            "protocol": {"grader": "fixture-v1", "score_definition": "hand-computed scores",
                         "latency_scope": "whole request", "cost_scope": "API only"}, "cases": cases}


def budget(**changes):
    return Budget(**{"min_quality": 0.5, "bootstrap_iterations": 100, **changes})


def test_hand_computed_scores_and_paired_zero_difference():
    report = compare(run(), run(name="candidate"), budget())
    assert report["passed"]
    assert report["baseline"]["mean_quality"] == 0.625
    assert report["quality_delta"] == 0
    assert report["paired_cluster_95pct_interval"] == [0, 0]
    assert report["candidate"]["groups"] == 2


def test_identity_is_paired_by_task_id_not_input_order():
    candidate = run(name="candidate")
    candidate["cases"].reverse()
    assert compare(run(), candidate, budget())["paired_cluster_95pct_interval"] == [0, 0]


def test_regression_fails_quality_checks():
    report = compare(run(), run((0, 0, 0, 0), "broken"), budget())
    assert report["passed"] is False
    assert report["checks"]["absolute_quality"] is False
    assert report["checks"]["paired_quality_lower_bound"] is False


@pytest.mark.parametrize("field,value", [("latency_ms", 101), ("cost_usd", 0.01), ("error", True)])
def test_operating_budgets_reject_quality_preserving_changes(field, value):
    candidate = run()
    for case in candidate["cases"]:
        case[field] = value
        if field == "error":
            case["score"] = 0
    report = compare(run(), candidate, budget())
    assert report["passed"] is False
    check = {"latency_ms": "p95_latency", "cost_usd": "mean_cost", "error": "error_rate"}[field]
    assert report["checks"][check] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "input", "reference", "group", "dataset", "protocol", "corpus"])
def test_changed_or_incomplete_evidence_fails_closed(mutation):
    candidate = run()
    if mutation == "missing":
        candidate["cases"].pop()
    elif mutation == "duplicate":
        candidate["cases"].append(candidate["cases"][0])
    elif mutation in {"input", "reference"}:
        candidate["cases"][0][mutation + "_sha256"] = digest("changed")
    elif mutation == "group":
        candidate["cases"][0]["group_id"] = "different"
    elif mutation == "dataset":
        candidate["dataset_version"] = "other"
    elif mutation == "protocol":
        candidate["protocol"]["grader"] = "different"
    else:
        candidate["corpus_sha256"] = digest("other")
    with pytest.raises(ValueError):
        compare(run(), candidate, budget())


@pytest.mark.parametrize("field,value", [("score", float("nan")), ("score", 1.01),
                                         ("latency_ms", -1), ("cost_usd", float("inf")),
                                         ("error", "false"), ("score", True)])
def test_invalid_measurements_cannot_pass(field, value):
    candidate = run()
    candidate["cases"][0][field] = value
    with pytest.raises(ValueError):
        validate_run(candidate)


@pytest.mark.parametrize("changes", [{"allowed_quality_drop": -0.1}, {"min_quality": float("nan")},
                                     {"max_p95_latency_ms": float("inf")},
                                     {"bootstrap_iterations": 0}, {"seed": True}])
def test_invalid_policy_cannot_pass(changes):
    with pytest.raises(ValueError):
        compare(run(), run(), budget(**changes))


def test_rank_grader_has_an_independent_small_oracle():
    assert reciprocal_rank(["a", "b", "c"], "b") == 0.5
    assert reciprocal_rank(["a", "b", "c"], "missing") == 0
    with pytest.raises(ValueError):
        reciprocal_rank(["a", "a"], "a")


def test_duplicate_questions_are_rejected_before_execution():
    row = {"question": "Q", "reference": "A", "gold_id": "paper"}
    with pytest.raises(ValueError, match="duplicate"):
        task_manifest([row, row])


def test_publication_failure_preserves_previous_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "last-passing.json"
    atomic_json(path, {"old": True})
    before = path.read_bytes()

    def fail(*args):
        raise OSError("injected replace failure")

    monkeypatch.setattr("resassist.release_gate.os.replace", fail)
    with pytest.raises(OSError):
        atomic_json(path, {"new": True})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_rejected_report_cannot_replace_last_passing_report(tmp_path):
    path = tmp_path / "current.json"
    passing = compare(run(), run(), budget())
    publish_passing(path, passing)
    before = path.read_bytes()
    rejected = compare(run(), run((0, 0, 0, 0)), budget())
    with pytest.raises(ValueError):
        publish_passing(path, rejected)
    with pytest.raises(ValueError):
        publish_passing(path, {"passed": True, "checks": {}})
    assert path.read_bytes() == before


def test_cli_distinguishes_policy_failure_from_invalid_evidence(tmp_path):
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    policy, report, current = (tmp_path / name for name in ("budget.json", "report.json", "current.json"))
    atomic_json(baseline, run())
    atomic_json(candidate, run())
    atomic_json(policy, {"min_quality": 0.5, "bootstrap_iterations": 100})
    args = [str(baseline), str(candidate), "--budget", str(policy), "--report", str(report), "--publish", str(current)]
    assert main(args) == 0
    before = current.read_bytes()
    atomic_json(candidate, run((0, 0, 0, 0)))
    assert main(args) == 1
    assert json.loads(report.read_text())["passed"] is False
    assert current.read_bytes() == before
    invalid = deepcopy(run())
    invalid["cases"].pop()
    atomic_json(candidate, invalid)
    assert main(args) == 2
    assert current.read_bytes() == before


def test_cli_rejects_output_alias_before_writing(tmp_path):
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    policy, current = tmp_path / "budget.json", tmp_path / "current.json"
    atomic_json(baseline, run())
    atomic_json(candidate, run((0, 0, 0, 0)))
    atomic_json(policy, {"min_quality": 0.5, "bootstrap_iterations": 100})
    atomic_json(current, {"previous": True})
    before = current.read_bytes()
    assert main([str(baseline), str(candidate), "--budget", str(policy), "--report", str(current),
                 "--publish", str(current)]) == 2
    assert current.read_bytes() == before


def test_group_bootstrap_keeps_correlated_questions_together():
    baseline = run((0, 0, 0, 0))
    candidate = run((1, 1, 0, 0))
    result = compare(baseline, candidate, budget(bootstrap_iterations=2000))
    # Resampling two whole groups yields exactly 0, 0.5 or 1, not four
    # independent-question draws. Both endpoint groups have mass 25%.
    assert result["quality_delta"] == 0.5
    assert result["paired_cluster_95pct_interval"] == [0, 1]


def test_single_source_group_cannot_claim_uncertainty():
    source = run((1, 1))
    with pytest.raises(ValueError, match="two independent source groups"):
        compare(source, source, budget())


def test_cli_cannot_replace_its_input_evidence(tmp_path):
    baseline, candidate, policy = (tmp_path / name for name in ("baseline.json", "candidate.json", "budget.json"))
    atomic_json(baseline, run())
    atomic_json(candidate, run())
    atomic_json(policy, {"min_quality": 0.5, "bootstrap_iterations": 100})
    before = baseline.read_bytes()
    assert main([str(baseline), str(candidate), "--budget", str(policy), "--report", str(baseline)]) == 2
    assert baseline.read_bytes() == before
