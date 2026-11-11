"""Full-generation evaluation identities and immutable all-or-nothing result releases."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def identify(rows, kind):
    output = []
    for row in rows:
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("every evaluation task needs a nonempty question")
        identity = {"question": row["question"], "kind": kind if kind == "answerable" else row.get("kind"),
                    "gold_id": row.get("gold_id"), "reference": row.get("reference")}
        if kind == "answerable" and (not identity["gold_id"] or not identity["reference"]):
            raise ValueError("answerable tasks require gold identity and reference")
        if kind != "answerable" and identity["kind"] not in {"off_topic", "fabricated"}:
            raise ValueError("unanswerable kind must be off_topic or fabricated")
        output.append({**row, "task_id": digest(identity), "input_sha256": digest(row["question"]),
                       "reference_sha256": digest({"gold_id": identity["gold_id"], "reference": identity["reference"]})})
    return output


def validate_tasks(answerable, unanswerable, paper_ids):
    questions = [" ".join(row["question"].lower().split()) for row in answerable + unanswerable]
    if len(questions) != len(set(questions)):
        raise ValueError("duplicate questions could cross development/test boundaries")
    if len(paper_ids) != len(set(paper_ids)) or any(row["gold_id"] not in paper_ids for row in answerable):
        raise ValueError("corpus ids must be unique and every gold paper must exist")


def file_hashes(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def publish(destination: Path, results, samples, systems, tasks, provenance):
    """Publish summary, sample answers, all per-task results and checksums together."""
    if destination.exists():
        raise FileExistsError("full evaluation releases are immutable; choose a new --out path")
    expected = {row["task_id"]: row for row in tasks}
    if not expected or len(expected) != len(tasks) or len(systems) < 2:
        raise ValueError("unique test tasks and multiple compared systems are required")
    for name, rows in systems.items():
        if len(rows) != len(expected) or {row.get("task_id") for row in rows} != set(expected):
            raise ValueError(f"{name}: every test task must occur exactly once")
        for row in rows:
            task = expected[row["task_id"]]
            if any(row.get(field) != task[field] for field in ("input_sha256", "reference_sha256")):
                raise ValueError("task/reference identities do not match the declared test set")
            if not isinstance(row.get("answer"), str) or not isinstance(row.get("source_ids"), list):
                raise ValueError("answer text and source identities must be retained")
            seconds = row.get("seconds")
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("latency must be finite and nonnegative")
    payloads = {"results.json": results, "samples.json": samples, "per_task.json": systems,
                "protocol.json": {"schema_version": 1, "test_tasks": tasks, "provenance": provenance,
                                  "limits": "Generated labels and model-based judging; not independent human gold. Historical runs are unchanged."}}
    encoded = {name: json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode() for name, value in payloads.items()}
    manifest = {name: hashlib.sha256(value).hexdigest() for name, value in encoded.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".full-evaluation-", dir=destination.parent))
    try:
        for name, value in encoded.items():
            with (stage / name).open("wb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
