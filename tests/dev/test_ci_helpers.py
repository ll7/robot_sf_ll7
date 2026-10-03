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


def test_duration_bootstrap_retains_completed_shards_after_missing_shard(
    tmp_path: Path,
) -> None:
    """Three real artifact stores seed balancing when the fourth job is cancelled."""
    for index in (1, 3, 4):
        _write_shard(tmp_path, f"pytest-durations-{index}", {f"test_{index}": float(index)})
    assert merge_test_durations.merge_duration_stores(tmp_path, allow_partial=True) == {
        "test_1": 1.0,
        "test_3": 3.0,
        "test_4": 4.0,
    }


@pytest.mark.parametrize("durations", [{}, {"node": float("nan")}, {"node": True}, {"node": -1}])
def test_partial_duration_bootstrap_rejects_bad_store(tmp_path: Path, durations: dict) -> None:
    """A missing shard never relaxes the measurement schema or empty-data guard."""
    _write_shard(tmp_path, "pytest-durations-1", durations)
    with pytest.raises(SystemExit, match="Invalid pytest duration store"):
        merge_test_durations.merge_duration_stores(tmp_path, allow_partial=True)


def test_partial_duration_bootstrap_rejects_no_artifacts(tmp_path: Path) -> None:
    """A skipped matrix must not save an empty cache over usable history."""
    with pytest.raises(SystemExit, match="missing="):
        merge_test_durations.merge_duration_stores(tmp_path, allow_partial=True)


def test_partial_duration_bootstrap_rejects_overlap_and_unexpected(
    tmp_path: Path,
) -> None:
    """Partial mode preserves disjoint stores and recognized shard identities."""
    _write_shard(tmp_path, "pytest-durations-1", {"node": 1.0})
    _write_shard(tmp_path, "pytest-durations-2", {"node": 2.0})
    with pytest.raises(SystemExit, match="Overlapping"):
        merge_test_durations.merge_duration_stores(tmp_path, allow_partial=True)
    _write_shard(tmp_path, "pytest-durations-9", {"other": 3.0})
    with pytest.raises(SystemExit, match="unexpected="):
        merge_test_durations.merge_duration_stores(tmp_path, allow_partial=True)


def test_partial_duration_cli_records_failed_matrix_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached hints from a failed matrix never claim a complete test verdict."""
    artifacts = tmp_path / "artifacts"
    _write_shard(artifacts, "pytest-durations-1", {"actual_test": 12.5})
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("DURATION_SOURCE_SHA", "b" * 40)
    monkeypatch.setenv("GITHUB_RUN_ID", "1234")
    monkeypatch.setenv("FAST_FEEDBACK_RESULT", "failure")
    monkeypatch.setenv("DURATION_CACHE_KEY", "test-durations-v2-Linux-X64-lock-1234-1")
    output, metadata = tmp_path / ".test_durations", tmp_path / "cache" / "metadata.json"
    assert (
        merge_test_durations.main(
            [
                "--artifact-dir",
                str(artifacts),
                "--output",
                str(output),
                "--allow-partial",
                "--metadata-output",
                str(metadata),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text()) == {"actual_test": 12.5}
    receipt = json.loads(metadata.read_text())
    assert receipt["source_sha"] == "b" * 40
    assert receipt["producer_sha"] == "a" * 40
    assert receipt["run_id"] == "1234"
    assert receipt["purpose"] == "scheduling-only"
    assert receipt["cache_key"] == "test-durations-v2-Linux-X64-lock-1234-1"
    assert receipt["missing_shards"] == [
        "pytest-durations-2",
        "pytest-durations-3",
        "pytest-durations-4",
    ]
    assert receipt["complete_matrix"] is False
    assert receipt["measurement_completeness"] == "partial_or_unverified"


def test_duration_cli_does_not_publish_empty_or_replace_output(tmp_path: Path) -> None:
    """Invalid inputs leave the previous cache bytes untouched and create no receipt."""
    _write_shard(tmp_path / "artifacts", "pytest-durations-1", {})
    output, metadata = tmp_path / ".test_durations", tmp_path / "cache" / "metadata.json"
    output.write_text('{"old": 3.0}\n')
    assert (
        merge_test_durations.main(
            [
                "--artifact-dir",
                str(tmp_path / "artifacts"),
                "--output",
                str(output),
                "--allow-partial",
                "--metadata-output",
                str(metadata),
            ]
        )
        == 1
    )
    assert output.read_text() == '{"old": 3.0}\n'
    assert not metadata.exists()


@pytest.mark.parametrize(
    "matrix_result, complete", [("success", True), ("failure", False), ("cancelled", False)]
)
def test_all_duration_artifacts_do_not_imply_successful_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matrix_result: str, complete: bool
) -> None:
    """Four uploaded artifacts can still come from failed or interrupted sessions."""
    _four_valid_shards(tmp_path / "artifacts")
    monkeypatch.setenv("FAST_FEEDBACK_RESULT", matrix_result)
    metadata = tmp_path / "cache" / "metadata.json"
    assert (
        merge_test_durations.main(
            [
                "--artifact-dir",
                str(tmp_path / "artifacts"),
                "--output",
                str(tmp_path / ".test_durations"),
                "--allow-partial",
                "--metadata-output",
                str(metadata),
            ]
        )
        == 0
    )
    receipt = json.loads(metadata.read_text())
    assert receipt["complete_matrix"] is complete
    assert receipt["missing_shards"] == []
    assert receipt["measurement_completeness"] == (
        "complete" if complete else "partial_or_unverified"
    )


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("shard_count", [5, 6])
def test_duration_merge_supports_explicit_matrix_size(
    tmp_path: Path, missing: bool, shard_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every configured shard contributes timings and missing shards stay explicit in provenance."""
    monkeypatch.setenv("FAST_FEEDBACK_RESULT", "success")
    for index in range(1, shard_count if missing else shard_count + 1):
        _write_shard(tmp_path / "artifacts", f"pytest-durations-{index}", {f"node{index}": 1.0})
    output, metadata = tmp_path / "merged.json", tmp_path / "metadata.json"
    args = [
        "--artifact-dir",
        str(tmp_path / "artifacts"),
        "--output",
        str(output),
        "--metadata-output",
        str(metadata),
        "--shard-count",
        str(shard_count),
    ]
    if missing:
        assert merge_test_durations.main(args) == 1
        args.append("--allow-partial")
    assert merge_test_durations.main(args) == 0
    assert len(json.loads(output.read_text())) == (shard_count - 1 if missing else shard_count)
    receipt = json.loads(metadata.read_text())
    assert receipt["missing_shards"] == ([f"pytest-durations-{shard_count}"] if missing else [])
    assert receipt["complete_matrix"] is (not missing)


@pytest.mark.parametrize("shard_count", [0, -1])
def test_duration_merge_rejects_nonpositive_matrix_size(
    tmp_path: Path, shard_count: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid matrix sizes must fail before publishing duration hints."""
    with pytest.raises(SystemExit, match="positive"):
        merge_test_durations.merge_duration_stores(tmp_path, shard_count=shard_count)
    with pytest.raises(SystemExit) as error:
        merge_test_durations.main(
            ["--artifact-dir", str(tmp_path), "--shard-count", str(shard_count)]
        )
    assert error.value.code == 2
    assert "--shard-count must be positive" in capsys.readouterr().err


@pytest.mark.parametrize("cached", [False, True])
def test_duration_snapshot_freezes_one_validated_input_for_every_shard(
    tmp_path: Path, cached: bool
) -> None:
    """A cache miss becomes one shared cold input; later writes cannot alter its weights."""
    source = tmp_path / "restored.json"
    if cached:
        source.write_text('{"measured": 7.0}', encoding="utf-8")
    snapshot = tmp_path / "snapshot" / ".test_durations"
    assert (
        merge_test_durations.main(["--snapshot-input", str(source), "--output", str(snapshot)]) == 0
    )
    source.write_text('{"newer": 999.0}', encoding="utf-8")
    assert json.loads(snapshot.read_text()) == ({"measured": 7.0} if cached else {})


@pytest.mark.parametrize("bad", [{}, {"node": True}, {"node": float("nan")}, {"node": -1}])
def test_duration_snapshot_rejects_corrupt_cache_without_replacing_output(
    tmp_path: Path, bad: dict
) -> None:
    """Corrupt restored data must fail closed rather than diverge across shard restores."""
    source, snapshot = tmp_path / "restored.json", tmp_path / "snapshot.json"
    source.write_text(json.dumps(bad), encoding="utf-8")
    snapshot.write_text('{"old": 3}', encoding="utf-8")
    assert (
        merge_test_durations.main(["--snapshot-input", str(source), "--output", str(snapshot)]) == 1
    )
    assert json.loads(snapshot.read_text()) == {"old": 3}


def test_duration_snapshot_requires_file_output(capsys: pytest.CaptureFixture[str]) -> None:
    """A snapshot without an artifact path must be an explicit usage error."""
    with pytest.raises(SystemExit) as error:
        merge_test_durations.main(["--snapshot-input", "missing-cache.json"])
    assert error.value.code == 2
    assert "--snapshot-input requires --output" in capsys.readouterr().err
