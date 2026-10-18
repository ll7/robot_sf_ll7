"""Presence and types in both outcome formats; inherited full receipt comparisons."""

import copy
import json

import pytest

from scripts.ci import default_path_equivalence as eq
from scripts.ci import make_behaviour_receipt as producer
from tests.validation.test_default_path_equivalence import _episode, _refresh_equivalence_binding
from tests.validation.test_make_behaviour_receipt import _raw_row, _sweep_rows
from tests.validation.test_receipt_outcomes import (
    FIELDS,
    RAW_FIELDS,
    SIDES,
    _edit_outcomes,
    _sides,
    scheduler_endpoint,  # noqa: F401
)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("side", SIDES)
def test_collection_requires_each_summary_outcome(tmp_path, field, side):
    roots = {
        "head": _sweep_rows(tmp_path, "head", job="11"),
        "baseline": _sweep_rows(tmp_path, "baseline", source_sha="b" * 40, job="12"),
    }
    for root in roots.values():
        _edit_outcomes(root)
        _refresh_equivalence_binding(root)
        assert producer._load_sweep_rows(root, preserve_raw=True)
    for selected in _sides(side):
        root = roots[selected]
        path = root / "episodes_main.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row.pop(field)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        _refresh_equivalence_binding(root)
        with pytest.raises(ValueError, match=rf"outcome schema refused: missing summary {field}"):
            producer._load_sweep_rows(root, preserve_raw=True)


@pytest.mark.parametrize("field", RAW_FIELDS)
@pytest.mark.parametrize("side", SIDES)
def test_full_classification_requires_each_raw_outcome(field, side):
    scope = {"arms": ["goal"], "maps": ["open"]}
    base = {("goal", "open", seed): _raw_row(seed) for seed in range(1001, 1031)}
    for row in base.values():
        row["metrics"]["success"] = False
    head = copy.deepcopy(base)
    assert (
        len(
            producer.build_rows_and_classifications(
                head,
                base,
                scope,
                classification_class="tooling-only",
                evidence_base_uri="https://evidence.example/outcomes",
            )[0]
        )
        == 30
    )
    for selected in _sides(side):
        for row in (head if selected == "head" else base).values():
            (row if field == "steps" else row["metrics"]).pop(field)
    with pytest.raises(ValueError, match=rf"outcome schema refused: missing raw .*{field}"):
        producer.build_rows_and_classifications(
            head,
            base,
            scope,
            classification_class="tooling-only",
            evidence_base_uri="https://evidence.example/outcomes",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("success", 0),
        ("success", 1),
        ("success", "false"),
        ("success", None),
        ("collisions", False),
        ("collisions", -1),
        ("collisions", 0.0),
        ("collisions", float("nan")),
        ("collisions", float("inf")),
        ("steps", True),
        ("steps", 0),
        ("steps", 1.0),
        ("steps", float("-inf")),
    ],
)
def test_comparison_requires_typed_finite_outcomes(field, value):
    row = _episode()
    (row if field == "steps" else row["metrics"])[field] = value
    rows = {("goal", "open", seed): {**row, "seed": seed} for seed in (1001, 1002, 1003)}
    with pytest.raises(ValueError, match=rf"outcome schema refused: invalid raw .*{field}"):
        eq.compare_rows(rows, rows, {"arms": ["goal"], "maps": ["open"]})


@pytest.mark.parametrize(
    "alias", ["success", "route_complete", "collisions", "total_collision_count"]
)
def test_comparison_refuses_contradictory_outcome_alias(alias):
    row = _episode()
    if alias == "route_complete":
        row["outcome"] = {alias: False}
    elif alias == "total_collision_count":
        row["metrics"][alias] = 1
    else:
        row[alias] = False if alias == "success" else 1
    rows = {("goal", "open", seed): {**row, "seed": seed} for seed in (1001, 1002, 1003)}
    with pytest.raises(ValueError, match="outcome schema refused: contradictory"):
        eq.compare_rows(rows, rows, {"arms": ["goal"], "maps": ["open"]})


def test_collection_crosschecks_summary_step_count(tmp_path):
    root = _sweep_rows(tmp_path, "head", job="11")
    assert producer._load_sweep_rows(root, preserve_raw=True)
    path = root / "episodes_main.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["steps"] = 2
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_equivalence_binding(root)
    with pytest.raises(ValueError, match="raw episode outcome differs from summary"):
        producer._load_sweep_rows(root, preserve_raw=True)


@pytest.mark.parametrize(
    ("field", "value"), [("success", "true"), ("collisions", 0.0), ("steps", True)]
)
def test_collection_requires_typed_summary_outcomes(tmp_path, field, value):
    root = _sweep_rows(tmp_path, "head", job="11")
    assert producer._load_sweep_rows(root, preserve_raw=True)
    path = root / "episodes_main.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1][field] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_equivalence_binding(root)
    with pytest.raises(ValueError, match=rf"outcome schema refused: invalid summary {field}"):
        producer._load_sweep_rows(root, preserve_raw=True)
