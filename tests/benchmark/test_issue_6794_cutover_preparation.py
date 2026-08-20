"""Tests for the Issue #6794 checkpoint-cutover preparation contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark import issue_6794_cutover_preparation as preparation_module
from robot_sf.benchmark.issue_6794_cutover_preparation import (
    compare_parity_rows,
    main,
    validate_preparation_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/benchmarks/issue_6794_phase_c_parity_preparation_v1.yaml"


def test_preparation_contract_freezes_current_bytes_and_load_paths() -> None:
    """The checked-in packet validates without staging or running a campaign."""
    report = validate_preparation_contract(REPO_ROOT, CONFIG)

    assert report["status"] == "prepared_not_executed"
    assert report["claim_boundary"] == "provenance_and_protocol_only"
    assert set(report["checkpoints"]) == {"default_ppo", "ga3c_cadrl"}
    assert len(report["load_paths"]) == 8
    assert report["parity_protocol"]["seeds"] == [111, 112, 113]


def test_preparation_contract_accepts_relative_config_path() -> None:
    """The public validator supports its documented repository-relative default."""
    report = validate_preparation_contract(REPO_ROOT)

    assert report["status"] == "prepared_not_executed"


def test_preparation_contract_rejects_escape_paths_and_malformed_arms(tmp_path: Path) -> None:
    """Declared inputs and protocol arms fail closed before any file is consumed."""
    outside = tmp_path.parent / "outside-checkpoint.txt"
    outside.write_text("not a checkpoint", encoding="utf-8")
    link = tmp_path / "checkpoint.txt"
    link.symlink_to(outside)

    with pytest.raises(ValueError, match="resolve within the repository"):
        preparation_module._repo_declared_path(tmp_path, "checkpoint.txt", name="checkpoint")
    with pytest.raises(ValueError, match="repository-relative"):
        preparation_module._repo_declared_path(
            tmp_path, "../outside-checkpoint.txt", name="checkpoint"
        )
    with pytest.raises(ValueError, match="two mapping arms"):
        preparation_module._validate_protocol_arms({"planner_arms": [{"key": "ppo"}, "malformed"]})


def test_load_path_inventory_rejects_escape_and_unknown_checkpoint() -> None:
    """Load-path selectors stay confined and reference a declared checkpoint."""
    with pytest.raises(ValueError, match="repository-relative"):
        preparation_module._validate_load_paths(
            REPO_ROOT,
            {
                "checkpoint_snapshots": {"default": {}},
                "load_path_inventory": [
                    {
                        "id": "escape",
                        "checkpoint": "default",
                        "path": "../outside.py",
                        "selector": "anything",
                    }
                ],
            },
        )
    with pytest.raises(ValueError, match="unknown checkpoint"):
        preparation_module._validate_load_paths(
            REPO_ROOT,
            {
                "checkpoint_snapshots": {"default": {}},
                "load_path_inventory": [
                    {
                        "id": "unknown",
                        "checkpoint": "missing",
                        "path": "README.md",
                        "selector": "anything",
                    }
                ],
            },
        )


def test_preparation_contract_rejects_protocol_type_coercion() -> None:
    """Boolean lookalikes cannot satisfy frozen numeric or arm contracts."""
    config = preparation_module.yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["parity_protocol"]["workers"] = True
    with pytest.raises(ValueError, match="workers must be an integer"):
        preparation_module._validate_protocol(REPO_ROOT, config["parity_protocol"])

    config = preparation_module.yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["parity_protocol"]["comparison"]["float_rel_tolerance"] = False
    with pytest.raises(ValueError, match="finite number"):
        preparation_module._validate_protocol(REPO_ROOT, config["parity_protocol"])

    config = preparation_module.yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["parity_protocol"]["planner_arms"][0]["execution_mode"] = "adapter"
    with pytest.raises(ValueError, match="unexpected execution_mode"):
        preparation_module._validate_protocol(REPO_ROOT, config["parity_protocol"])


def test_release_component_set_must_match_declared_sources() -> None:
    """A bundle cannot silently omit or add registry components."""
    digest = "a" * 64
    with pytest.raises(ValueError, match="component set"):
        preparation_module._verify_release_components(
            "bundle",
            {
                "registry_per_file_sha256": {
                    "one.data": digest,
                    "one.meta": digest,
                    "extra.meta": digest,
                }
            },
            {
                "per_file_sha256": {
                    "one.data": digest,
                    "one.meta": digest,
                    "extra.meta": digest,
                }
            },
            {"model/one.data": digest, "model/one.meta": digest},
        )


def _row(seed: int, *, delta: float = 0.0, status: str = "native") -> dict:
    """Return one complete synthetic parity row."""
    return {
        "planner_key": "ppo",
        "scenario_id": "fixture.scenario",
        "seed": seed,
        "row_status": status,
        "benchmark_success": True,
        "benchmark_success_basis": "all",
        "termination_reason": "success",
        "metrics": {
            "success": 1.0,
            "collisions": 0.0,
            "near_misses": 1.0 + delta,
            "time_to_goal_norm": 0.4,
            "snqi": 0.2,
        },
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    """Write test JSONL rows."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compare_parity_rows_accepts_unchanged_native_fixture(tmp_path: Path) -> None:
    """The future parity harness accepts identical rows without executing a campaign."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    rows = [_row(111), _row(112)]
    _write_rows(before, rows)
    _write_rows(after, rows)

    report = compare_parity_rows(before, after)

    assert report["status"] == "passed"
    assert report["compared_rows"] == 2
    assert all(delta["delta"] == 0.0 for delta in report["metric_deltas"])


def test_compare_parity_rows_rejects_status_and_metric_drift(tmp_path: Path) -> None:
    """Any status drift or metric delta beyond the frozen tolerance fails closed."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    _write_rows(before, [_row(111)])
    _write_rows(after, [_row(111, delta=1e-6, status="fallback")])

    report = compare_parity_rows(before, after)

    assert report["status"] == "failed"
    assert any("non-native" in blocker for blocker in report["blockers"])
    assert any("metric drift" in blocker for blocker in report["blockers"])


def test_compare_parity_rows_rejects_coerced_identity_and_status_types(tmp_path: Path) -> None:
    """Self-consistent boolean/string lookalikes cannot pass parity comparison."""
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    malformed_before = _row(111)
    malformed_after = _row(111)
    malformed_before["seed"] = True
    malformed_after["seed"] = True
    _write_rows(before, [malformed_before])
    _write_rows(after, [malformed_after])
    with pytest.raises(ValueError, match="seed"):
        compare_parity_rows(before, after)

    malformed_before = _row(111)
    malformed_after = _row(111)
    malformed_before["benchmark_success"] = "true"
    malformed_after["benchmark_success"] = "true"
    _write_rows(before, [malformed_before])
    _write_rows(after, [malformed_after])
    report = compare_parity_rows(before, after)
    assert report["status"] == "failed"
    assert any("invalid status field type" in blocker for blocker in report["blockers"])


def test_cli_rejects_one_sided_comparison_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI requires both before and after outputs for a comparison."""
    exit_code = main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--config",
            str(CONFIG),
            "--before-episodes",
            str(tmp_path / "before.jsonl"),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "failed"
    assert "supplied together" in payload["blockers"][0]
