"""Versioned, paired evaluation reports and a fail-closed release gate.

The gate accepts a bounded [0, 1] higher-is-better score per task. The producer
must declare the grader and its limitations; the gate does not make a judge
independent or turn a generated question set into human ground truth.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _number(value: object, name: str, lower: float = 0, upper: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    return float(value)


def _hash(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def validate_run(run: dict) -> dict[str, dict]:
    """Reject incomplete evidence; never silently drop failed or missing tasks."""
    try:
        if type(run["schema_version"]) is not int or run["schema_version"] != 1:
            raise ValueError("unsupported run schema")
        for field in ("dataset_version", "system_name", "created_utc", "code_revision"):
            _text(run[field], field)
        for field in ("dataset_sha256", "corpus_sha256", "model_sha256", "prompt_sha256"):
            _hash(run[field], field)
        for field in ("grader", "score_definition", "latency_scope", "cost_scope"):
            _text(run["protocol"][field], field)
        if not isinstance(run["cases"], list) or not run["cases"]:
            raise ValueError("cases must be a nonempty list")
        cases = {}
        manifest = []
        for case in run["cases"]:
            for field in ("task_id", "group_id"):
                _text(case[field], field)
            for field in ("input_sha256", "reference_sha256"):
                _hash(case[field], field)
            task_id = case["task_id"]
            if task_id in cases:
                raise ValueError(f"duplicate task: {task_id}")
            _number(case["score"], "score", upper=1)
            _number(case["latency_ms"], "latency_ms")
            _number(case["cost_usd"], "cost_usd")
            if not isinstance(case["error"], bool):
                raise ValueError("error must be a boolean")
            if case["error"] and case["score"] != 0:
                raise ValueError("failed tasks must score zero")
            cases[task_id] = case
            manifest.append({key: case[key] for key in
                             ("task_id", "group_id", "input_sha256", "reference_sha256")})
        if digest(sorted(manifest, key=lambda row: row["task_id"])) != run["dataset_sha256"]:
            raise ValueError("dataset fingerprint does not match task manifest")
        return cases
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed run: {exc}") from exc


@dataclass(frozen=True)
class Budget:
    min_quality: float = 0.85
    allowed_quality_drop: float = 0.02
    max_p95_latency_ms: float = 100.0
    max_mean_cost_usd: float = 0.0
    max_error_rate: float = 0.0
    bootstrap_iterations: int = 2000
    seed: int = 17

    def validate(self) -> None:
        for key in ("min_quality", "allowed_quality_drop", "max_error_rate"):
            _number(getattr(self, key), key, upper=1)
        _number(self.max_p95_latency_ms, "max_p95_latency_ms")
        _number(self.max_mean_cost_usd, "max_mean_cost_usd")
        if (type(self.bootstrap_iterations) is not int or
                not 100 <= self.bootstrap_iterations <= 100_000):
            raise ValueError("bootstrap_iterations must be an integer in [100, 100000]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")


def paired_interval(base: dict[str, dict], candidate: dict[str, dict], budget: Budget) -> list[float]:
    """Question-weighted delta, resampling source-document groups as whole units."""
    grouped: dict[str, list[float]] = {}
    for task_id in sorted(base):
        grouped.setdefault(base[task_id]["group_id"], []).append(
            candidate[task_id]["score"] - base[task_id]["score"])
    if len(grouped) < 2:
        raise ValueError("at least two independent source groups are required")
    groups = [grouped[key] for key in sorted(grouped)]
    totals = np.array([sum(group) for group in groups])
    counts = np.array([len(group) for group in groups])
    rng = np.random.default_rng(budget.seed)
    means = []
    for _ in range(budget.bootstrap_iterations):
        selected = rng.integers(0, len(groups), len(groups))
        means.append(float(totals[selected].sum() / counts[selected].sum()))
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def summarize(cases: dict[str, dict]) -> dict:
    rows = list(cases.values())
    return {"tasks": len(rows), "groups": len({row["group_id"] for row in rows}),
            "mean_quality": float(np.mean([row["score"] for row in rows])),
            "p50_latency_ms": float(np.quantile([row["latency_ms"] for row in rows], 0.5)),
            "p95_latency_ms": float(np.quantile([row["latency_ms"] for row in rows], 0.95)),
            "mean_cost_usd": float(np.mean([row["cost_usd"] for row in rows])),
            "error_rate": float(np.mean([row["error"] for row in rows]))}


def compare(baseline: dict, candidate: dict, budget: Budget) -> dict:
    budget.validate()
    base, cand = validate_run(baseline), validate_run(candidate)
    for field in ("dataset_version", "dataset_sha256", "corpus_sha256", "protocol"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"incomparable evidence: {field} differs")
    if set(base) != set(cand):
        raise ValueError("incomparable task sets")
    for task in base:
        for field in ("group_id", "input_sha256", "reference_sha256"):
            if base[task][field] != cand[task][field]:
                raise ValueError(f"task {task}: {field} differs")
    interval = paired_interval(base, cand, budget)
    b, c = summarize(base), summarize(cand)
    checks = {
        "absolute_quality": c["mean_quality"] >= budget.min_quality,
        "paired_quality_lower_bound": interval[0] >= -budget.allowed_quality_drop,
        "p95_latency": c["p95_latency_ms"] <= budget.max_p95_latency_ms,
        "mean_cost": c["mean_cost_usd"] <= budget.max_mean_cost_usd,
        "error_rate": c["error_rate"] <= budget.max_error_rate,
    }
    return {"schema_version": 1, "passed": bool(all(checks.values())),
            "checks": {key: bool(value) for key, value in checks.items()},
            "baseline": {"name": baseline["system_name"], **b},
            "candidate": {"name": candidate["system_name"], **c},
            "quality_delta": c["mean_quality"] - b["mean_quality"],
            "paired_cluster_95pct_interval": interval, "budget": asdict(budget),
            "dataset_version": baseline["dataset_version"],
            "dataset_sha256": baseline["dataset_sha256"],
            "corpus_sha256": baseline["corpus_sha256"], "protocol": baseline["protocol"],
            "baseline_sha256": digest(baseline), "candidate_sha256": digest(candidate),
            "gate_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "baseline_system": {key: baseline.get(key) for key in
                                ("model_sha256", "prompt_sha256", "code_revision", "dirty_worktree", "created_utc")},
            "candidate_system": {key: candidate.get(key) for key in
                                 ("model_sha256", "prompt_sha256", "code_revision", "dirty_worktree", "created_utc")},
            "limits": ["Paired source-group percentile bootstrap, exploratory and unadjusted for repeated candidate selection.",
                       "Passing is a configured evidence policy, not proof of generalization or production readiness.",
                       "The producer's grader and dataset limitations still apply."]}


def atomic_json(path: Path, value: dict) -> None:
    """One complete UTF-8 snapshot; failures preserve the previous visible file."""
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_passing(path: Path, report: dict) -> None:
    expected = {"absolute_quality", "paired_quality_lower_bound", "p95_latency", "mean_cost", "error_rate"}
    checks = report.get("checks", {})
    if (report.get("passed") is not True or set(checks) != expected or
            any(value is not True for value in checks.values())):
        raise ValueError("only a passing report may replace the last passing publication")
    # This is publication of evidence, not promotion/reloading of the served model.
    atomic_json(path, report)


def read_json(path: Path) -> dict:
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _same_path(left: Path, right: Path) -> bool:
    return (left.resolve() == right.resolve() or
            (left.exists() and right.exists() and left.samefile(right)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--publish", type=Path, help="Replace last passing report only after every check passes")
    args = parser.parse_args(argv)
    try:
        if args.publish and _same_path(args.report, args.publish):
            raise ValueError("--report and --publish must be distinct files")
        for output in [args.report] + ([args.publish] if args.publish else []):
            if any(_same_path(output, source) for source in
                   (args.baseline, args.candidate, args.budget)):
                raise ValueError("outputs must not overwrite input evidence or policy")
        report = compare(read_json(args.baseline), read_json(args.candidate),
                         Budget(**read_json(args.budget)))
        atomic_json(args.report, report)
        if report["passed"] and args.publish:
            publish_passing(args.publish, report)
        print(json.dumps({"passed": report["passed"], "checks": report["checks"]}))
        return 0 if report["passed"] else 1
    except (ValueError, TypeError, OSError) as exc:
        print(f"Invalid evaluation evidence: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
