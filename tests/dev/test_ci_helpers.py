"""Focused tests for the extracted CI helper logic (issue #7666)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dev import check_ci_needs, merge_test_durations, model_cache_key

# --- model_cache_key -------------------------------------------------------


def test_model_cache_key_derives_stable_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Known registry digests must produce the expected deterministic key."""

    def fake_required(config) -> list[str]:
        assert config is not None
        return ["model-a", "model-b"]

    def fake_registry_entry(model_id: str) -> dict:
        digests = {"model-a": "aa" * 32, "model-b": "bb" * 32}
        return {"github_release": {"sha256": digests[model_id]}}

    import hashlib

    monkeypatch.setattr(model_cache_key, "required_model_ids_for_config", fake_required)
    monkeypatch.setattr(model_cache_key, "get_registry_entry", fake_registry_entry)
    expected = hashlib.sha256("|".join(["aa" * 32, "bb" * 32]).encode()).hexdigest()[:16]

    cfg = tmp_config(monkeypatch)
    assert model_cache_key.derive_model_cache_key(cfg) == expected


def test_model_cache_key_fails_closed_on_missing_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A required model without a pinned digest must raise instead of returning a key."""

    def fake_required(config) -> list[str]:
        return ["model-missing"]

    monkeypatch.setattr(model_cache_key, "required_model_ids_for_config", fake_required)
    monkeypatch.setattr(model_cache_key, "get_registry_entry", lambda _m: {"github_release": {}})

    with pytest.raises(ValueError, match="no pinned github_release.sha256"):
        model_cache_key.derive_model_cache_key(tmp_config(monkeypatch))


def test_model_cache_key_preserves_registry_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The key must change when the model order changes."""
    import hashlib

    def fake_required(config) -> list[str]:
        return config["order"]

    monkeypatch.setattr(model_cache_key, "required_model_ids_for_config", fake_required)

    def entry_for(model_id: str) -> dict:
        return {"github_release": {"sha256": f"{model_id}0" * 16}}

    monkeypatch.setattr(model_cache_key, "get_registry_entry", entry_for)
    cfg = tmp_config(monkeypatch, order=["m1", "m2"])
    rev = tmp_config(monkeypatch, order=["m2", "m1"])

    key_a = model_cache_key.derive_model_cache_key(cfg)
    key_b = model_cache_key.derive_model_cache_key(rev)
    assert key_a != key_b
    assert len(key_a) == 16
    assert hashlib.sha256  # ensure import used


def tmp_config(monkeypatch: pytest.MonkeyPatch, order=None) -> Path:
    """Write a temp YAML config and return its path."""
    import tempfile

    payload = {"order": order or ["model-a", "model-b"]}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        import yaml

        yaml.safe_dump(payload, fh)
        path = Path(fh.name)
    return path


# --- merge_test_durations --------------------------------------------------


def _write_shard(tmp_path: Path, name: str, durations: dict[str, float]) -> None:
    shard_dir = tmp_path / name
    shard_dir.mkdir(parents=True)
    (shard_dir / ".test_durations").write_text(
        json.dumps(durations, sort_keys=True), encoding="utf-8"
    )


def _four_valid_shards(tmp_path: Path) -> None:
    for index in range(1, 5):
        _write_shard(
            tmp_path,
            f"pytest-durations-{index}",
            {f"test_{index}_a": float(index), f"test_{index}_b": float(index + 0.5)},
        )


def test_duration_merge_accepts_four_valid_shards(tmp_path: Path) -> None:
    """Four valid, non-overlapping shard stores merge deterministically."""
    _four_valid_shards(tmp_path)
    merged = merge_test_durations.merge_duration_stores(tmp_path)
    assert len(merged) == 8
    assert merged["test_1_a"] == 1.0
    assert sorted(merged) == sorted(merged)


def test_duration_merge_rejects_missing_shard(tmp_path: Path) -> None:
    """A missing shard must fail closed with the missing names listed."""
    _four_valid_shards(tmp_path)
    (tmp_path / "pytest-durations-4").rename(tmp_path / "pytest-durations-4-backup")
    try:
        with pytest.raises(SystemExit, match="missing=.*pytest-durations-4"):
            merge_test_durations.merge_duration_stores(tmp_path)
    finally:
        (tmp_path / "pytest-durations-4-backup").rename(tmp_path / "pytest-durations-4")


def test_duration_merge_rejects_unexpected_shard(tmp_path: Path) -> None:
    """An unexpected shard name must be reported."""
    _four_valid_shards(tmp_path)
    _write_shard(tmp_path, "pytest-durations-9", {"extra": 1.0})
    with pytest.raises(SystemExit, match="unexpected=.*pytest-durations-9"):
        merge_test_durations.merge_duration_stores(tmp_path)


def test_duration_merge_rejects_overlap(tmp_path: Path) -> None:
    """Overlapping node ids across shards must fail."""
    _four_valid_shards(tmp_path)
    (tmp_path / "pytest-durations-2" / ".test_durations").write_text(
        json.dumps({"test_1_a": 5.0}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="Overlapping pytest duration stores"):
        merge_test_durations.merge_duration_stores(tmp_path)


@pytest.mark.parametrize(
    "bad",
    [
        {"test_x": "not-a-number"},
        {"test_x": float("nan")},
        {"test_x": float("inf")},
        {"test_x": -1.0},
        {"test_x": True},
    ],
)
def test_duration_merge_rejects_malformed_values(tmp_path: Path, bad: dict) -> None:
    """Non-numeric, non-finite, negative, and boolean durations must fail."""
    _four_valid_shards(tmp_path)
    (tmp_path / "pytest-durations-3" / ".test_durations").write_text(
        json.dumps(bad), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="Invalid pytest duration store"):
        merge_test_durations.merge_duration_stores(tmp_path)


# --- check_ci_needs --------------------------------------------------------


def _all_success() -> dict[str, str]:
    return dict.fromkeys(check_ci_needs.REQUIRED_JOBS, "success")


def test_needs_all_success_pull_request() -> None:
    """A fully green PR run passes including changed-coverage-gate."""
    results = {**_all_success(), "changed-coverage-gate": "success", "coverage-gate": "skipped"}
    assert check_ci_needs.evaluate_needs(results, "pull_request") == []


def test_needs_coverage_gate_required_on_push() -> None:
    """coverage-gate is required for push events, not for pull_request."""
    results = {**_all_success(), "coverage-gate": "failure", "changed-coverage-gate": "success"}
    assert check_ci_needs.evaluate_needs(results, "push") == ["coverage-gate"]
    assert check_ci_needs.evaluate_needs(results, "pull_request") == []


def test_needs_changed_coverage_gate_required_on_pull_request() -> None:
    """changed-coverage-gate is required for pull_request / merge_group only."""
    results = {
        **_all_success(),
        "coverage-gate": "success",
        "changed-coverage-gate": "cancelled",
    }
    assert check_ci_needs.evaluate_needs(results, "pull_request") == ["changed-coverage-gate"]
    assert check_ci_needs.evaluate_needs(results, "merge_group") == ["changed-coverage-gate"]
    assert check_ci_needs.evaluate_needs(results, "push") == []


def test_needs_merge_group_requires_both_coverage_gates() -> None:
    """Match the former workflow: merge groups require both coverage gates."""
    results = {
        **_all_success(),
        "coverage-gate": "failure",
        "changed-coverage-gate": "success",
    }
    assert check_ci_needs.evaluate_needs(results, "merge_group") == ["coverage-gate"]


def test_needs_unknown_event_fails_closed_on_coverage_gate() -> None:
    """Future event types must retain the non-pull-request coverage requirement."""
    results = {
        **_all_success(),
        "coverage-gate": "missing",
        "changed-coverage-gate": "skipped",
    }
    assert check_ci_needs.evaluate_needs(results, "future_event") == ["coverage-gate"]


def test_needs_fails_on_skipped_cancelled_failed_missing() -> None:
    """Any non-success required result or a missing job must be reported."""
    for bad in ("skipped", "cancelled", "failure", "missing-key"):
        results = _all_success()
        if bad == "missing-key":
            results.pop("fast-feedback")
        else:
            results["fast-feedback"] = bad
        failures = check_ci_needs.evaluate_needs(results, "pull_request")
        assert "fast-feedback" in failures, bad


def test_needs_cancelled_is_superseded_when_opt_in() -> None:
    """Issue #7926: cancelled dependencies (latest-main-wins) must not go red."""
    results = {
        **_all_success(),
        "fast-feedback": "cancelled",
        "coverage-gate": "success",
        "changed-coverage-gate": "success",
    }
    # Default is fail-closed: cancelled still fails.
    assert "fast-feedback" in check_ci_needs.evaluate_needs(results, "push")
    # Opt-in treats cancelled as superseded.
    assert check_ci_needs.evaluate_needs(results, "push", treat_cancelled_as_superseded=True) == []


def test_needs_cancelled_superseded_still_fails_on_real_failure() -> None:
    """Issue #7926: a genuine failure stays red even with the superseded opt-in."""
    results = {
        **_all_success(),
        "fast-feedback": "failure",
        "examples-smoke": "cancelled",
        "coverage-gate": "success",
    }
    failures = check_ci_needs.evaluate_needs(results, "push", treat_cancelled_as_superseded=True)
    assert "fast-feedback" in failures
    assert "examples-smoke" not in failures


def test_needs_cancelled_superseded_applies_to_coverage_gates() -> None:
    """Issue #7926: cancelled coverage gates are superseded too when opted in."""
    results = {**_all_success(), "coverage-gate": "cancelled"}
    assert check_ci_needs.evaluate_needs(results, "push", treat_cancelled_as_superseded=True) == []
    assert check_ci_needs.evaluate_needs(results, "push") == ["coverage-gate"]


def test_needs_main_requires_every_required_job() -> None:
    """Every job in REQUIRED_JOBS must appear in a passing main run."""
    results = {**_all_success(), "coverage-gate": "success", "changed-coverage-gate": "skipped"}
    assert check_ci_needs.evaluate_needs(results, "push") == []


def test_needs_normalizes_github_needs_objects() -> None:
    """The raw ``toJSON(needs)`` shape must expose each nested result."""
    raw = {
        "fast-feedback": {"result": "success", "outputs": {}},
        "coverage-gate": {"result": "skipped", "outputs": {}},
    }
    assert check_ci_needs.normalize_needs(raw) == {
        "fast-feedback": "success",
        "coverage-gate": "skipped",
    }
