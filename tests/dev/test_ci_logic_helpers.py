"""Deterministic tests for extracted CI workflow logic helpers (issue #7666)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "dev"))

from ci_logic import (
    evaluate_required_jobs,
    merge_duration_artifacts,
    merge_duration_stores,
)


def _store(dir_path: Path, shard: int, durations: dict[str, float]) -> Path:
    shard_dir = dir_path / f"pytest-durations-{shard}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    target = shard_dir / ".test_durations"
    target.write_text(json.dumps(durations), encoding="utf-8")
    return target


def _four_valid_stores(tmp_path: Path) -> list[Path]:
    return [
        _store(tmp_path, 1, {"test_a": 0.5}),
        _store(tmp_path, 2, {"test_b": 1.0}),
        _store(tmp_path, 3, {"test_c": 2.25}),
        _store(tmp_path, 4, {"test_d": 3}),
    ]


def test_merge_four_valid_shards_is_sorted_and_deterministic(tmp_path: Path):
    files = _four_valid_stores(tmp_path)
    merged = merge_duration_stores(files)
    assert merged == {
        "test_a": 0.5,
        "test_b": 1.0,
        "test_c": 2.25,
        "test_d": 3,
    }
    assert list(merged) == sorted(merged)


def test_merge_missing_shard_fails_closed(tmp_path: Path):
    files = _four_valid_stores(tmp_path)
    del files[2]
    with pytest.raises(SystemExit, match="missing=\\['pytest-durations-3'\\]"):
        merge_duration_stores(files)


def test_merge_unexpected_shard_fails_closed(tmp_path: Path):
    files = _four_valid_stores(tmp_path)
    files.append(_store(tmp_path, 9, {"extra": 1.0}))
    with pytest.raises(SystemExit, match="unexpected=\\['pytest-durations-9'\\]"):
        merge_duration_stores(files)


def test_merge_duplicate_store_entry_fails_closed_on_overlap(tmp_path: Path):
    """A duplicated shard path re-claims node ids and must hit overlap rejection."""
    files = _four_valid_stores(tmp_path)
    files.append(files[0])
    with pytest.raises(SystemExit, match="Overlapping pytest duration stores"):
        merge_duration_stores(files)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ('{"a": "fast"}', "Invalid pytest duration store"),
        ('{"a": true}', "Invalid pytest duration store"),
        ('{"a": null}', "Invalid pytest duration store"),
        ('{"a": -1.0}', "Invalid pytest duration store"),
        ("not json at all", "Invalid pytest duration store"),
    ],
)
def test_merge_malformed_values_fail_closed(tmp_path: Path, payload: str, reason: str):
    files = _four_valid_stores(tmp_path)
    files[0].write_text(payload, encoding="utf-8")
    with pytest.raises(SystemExit, match=reason):
        merge_duration_stores(files)


def test_merge_non_finite_duration_fails_closed(tmp_path: Path):
    files = _four_valid_stores(tmp_path)
    files[1].write_text('{"test_b": Infinity}', encoding="utf-8")
    with pytest.raises(SystemExit, match="Invalid pytest duration store"):
        merge_duration_stores(files)


def test_merge_overlapping_node_ids_fail_closed(tmp_path: Path):
    files = _four_valid_stores(tmp_path)
    files[3] = _store(tmp_path, 4, {"test_a": 9.0})
    with pytest.raises(SystemExit, match="Overlapping pytest duration stores"):
        merge_duration_stores(files)


def test_merge_artifacts_writes_sorted_json_with_trailing_newline(tmp_path: Path):
    _four_valid_stores(tmp_path)
    out = tmp_path / "out" / ".test_durations"
    count = merge_duration_artifacts(tmp_path, out)
    assert count == 4
    text = out.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    data = json.loads(text)
    assert set(data) == {"test_a", "test_b", "test_c", "test_d"}
    assert text == json.dumps(data, indent=4, sort_keys=True) + "\n"


ALL_SUCCESS = {job: "success" for job, _, _ in __import__("ci_logic").REQUIRED_JOB_RULES}


def test_aggregate_all_success_passes_on_every_event():
    for event in ("pull_request", "merge_group", "push", "workflow_dispatch"):
        assert evaluate_required_jobs(dict(ALL_SUCCESS), event) == []


def test_aggregate_coverage_gate_not_required_on_pull_request():
    results = dict(ALL_SUCCESS, **{"coverage-gate": "skipped"})
    assert evaluate_required_jobs(results, "pull_request") == []
    failures = evaluate_required_jobs(results, "merge_group")
    assert any(f.startswith("coverage-gate ") for f in failures)


def test_aggregate_changed_coverage_gate_only_for_pr_and_merge_group():
    results = dict(ALL_SUCCESS, **{"changed-coverage-gate": "skipped"})
    for event in ("pull_request", "merge_group"):
        assert any(
            f.startswith("changed-coverage-gate ") for f in evaluate_required_jobs(results, event)
        )
    assert evaluate_required_jobs(results, "push") == []


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_aggregate_required_job_non_success_fails(result: str):
    results = dict(ALL_SUCCESS, **{"determinism-gate": result})
    failures = evaluate_required_jobs(results, "pull_request")
    assert failures == [f"determinism-gate finished with {result}"]


def test_aggregate_missing_job_fails_closed():
    results = dict(ALL_SUCCESS)
    del results["wheel-smoke-install"]
    failures = evaluate_required_jobs(results, "push")
    assert failures == ["wheel-smoke-install finished with unknown result"]


def test_aggregate_unknown_job_in_results_fails_closed():
    results = dict(ALL_SUCCESS, **{"rogue-job": "success"})
    failures = evaluate_required_jobs(results, "push")
    assert any("rogue-job is not part of the required-check manifest" in f for f in failures)


def test_model_cache_key_matches_reference_derivation(monkeypatch, tmp_path: Path):
    """The helper reproduces the pre-refactor inline derivation byte-for-byte."""
    import hashlib

    import yaml
    from ci_logic import derive_model_cache_key

    from robot_sf.models import preflight, registry

    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({"models": ["m_a", "m_b"]}), encoding="utf-8")
    monkeypatch.setattr(preflight, "required_model_ids_for_config", lambda cfg: list(cfg["models"]))
    monkeypatch.setattr(
        registry,
        "get_registry_entry",
        lambda model_id: {"github_release": {"sha256": "aaa" if model_id == "m_a" else "bbb"}},
    )

    expected = hashlib.sha256(b"aaa|bbb").hexdigest()[:16]
    assert derive_model_cache_key(config) == expected


def test_model_cache_key_fails_closed_on_missing_digest(monkeypatch, tmp_path: Path):
    import yaml
    from ci_logic import derive_model_cache_key

    from robot_sf.models import preflight, registry

    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({"models": ["m_x"]}), encoding="utf-8")
    monkeypatch.setattr(preflight, "required_model_ids_for_config", lambda cfg: list(cfg["models"]))
    monkeypatch.setattr(registry, "get_registry_entry", lambda model_id: {})

    with pytest.raises(SystemExit, match="Missing registry-pinned"):
        derive_model_cache_key(config)
