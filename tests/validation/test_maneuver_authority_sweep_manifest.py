"""Tests for the issue #3213 maneuver-authority sweep manifest preflight."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.validation import check_maneuver_authority_sweep_manifest as preflight


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _touch_repo_file(repo_root: Path, relative_path: str) -> None:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# fixture\n", encoding="utf-8")


def _valid_payload() -> dict:
    return {
        "schema_version": preflight.SCHEMA_VERSION,
        "issue": 3213,
        "hard_seed_manifest": "configs/benchmarks/predictive_hard_seeds_v1.yaml",
        "expected_output": {
            "root": "output/benchmarks/issue_3213_maneuver_authority",
            "summary": "output/benchmarks/issue_3213_maneuver_authority/campaign_summary.json",
            "report": "output/benchmarks/issue_3213_maneuver_authority/campaign_report.md",
        },
        "variants": [
            {
                "name": "baseline",
                "algo_config": "configs/algos/hardcase_authority/baseline.yaml",
                "expected_output": {
                    "hard_jsonl_pattern": "output/benchmarks/issue_3213_maneuver_authority/hard__baseline__<checkpoint-token>.jsonl",
                    "global_jsonl_pattern": "output/benchmarks/issue_3213_maneuver_authority/global__baseline__<checkpoint-token>.jsonl",
                },
                "authority_metadata": {
                    "action_lattice": {"inherits_defaults": True},
                    "turn_authority": {"max_angular_speed_rad_s": 1.2},
                    "kinematic_adapter": {
                        "mode": "predictive_planner_default",
                        "config_source": "configs/algos/hardcase_authority/baseline.yaml",
                        "changed_params": [],
                    },
                },
                "params": {},
            },
            {
                "name": "high_angular",
                "algo_config": "configs/algos/hardcase_authority/high_angular.yaml",
                "expected_output": {
                    "hard_jsonl_pattern": "output/benchmarks/issue_3213_maneuver_authority/hard__high_angular__<checkpoint-token>.jsonl",
                    "global_jsonl_pattern": "output/benchmarks/issue_3213_maneuver_authority/global__high_angular__<checkpoint-token>.jsonl",
                },
                "authority_metadata": {
                    "action_lattice": {"candidate_heading_deltas_rad": [-1.0, 0.0, 1.0]},
                    "turn_authority": {"max_angular_speed_rad_s": 1.8},
                    "kinematic_adapter": {
                        "mode": "predictive_planner_config_overrides",
                        "config_source": "configs/algos/hardcase_authority/high_angular.yaml",
                        "changed_params": ["max_angular_speed"],
                    },
                },
                "params": {"max_angular_speed": 1.8},
            },
            {
                "name": "dense_lattice",
                "algo_config": "configs/algos/hardcase_authority/dense_lattice.yaml",
                "expected_output": {
                    "hard_jsonl_pattern": "output/benchmarks/issue_3213_maneuver_authority/hard__dense_lattice__<checkpoint-token>.jsonl",
                    "global_jsonl_pattern": "output/benchmarks/issue_3213_maneuver_authority/global__dense_lattice__<checkpoint-token>.jsonl",
                },
                "authority_metadata": {
                    "action_lattice": {"candidate_speed_samples_m_s": [0.0, 0.5, 1.0]},
                    "turn_authority": {"max_angular_speed_rad_s": 1.2},
                    "kinematic_adapter": {
                        "mode": "predictive_planner_config_overrides",
                        "config_source": "configs/algos/hardcase_authority/dense_lattice.yaml",
                        "changed_params": ["predictive_candidate_speeds"],
                    },
                },
                "params": {"predictive_candidate_speeds": [0.0, 0.5, 1.0]},
            },
        ],
    }


def test_checked_in_issue_3213_manifest_passes_preflight() -> None:
    """The committed #3213 sweep manifest has complete metadata but runs no benchmark."""
    report = preflight.build_report(preflight.DEFAULT_MANIFEST, Path.cwd())

    assert report["status"] == "ok"
    assert report["variant_count"] >= 3
    assert report["errors"] == []


def test_missing_algo_config_fails_closed(tmp_path: Path) -> None:
    """Missing referenced configs are explicit preflight errors."""
    payload = _valid_payload()
    repo_root = tmp_path
    _touch_repo_file(repo_root, payload["hard_seed_manifest"])
    for variant in payload["variants"]:
        if variant["name"] != "high_angular":
            _touch_repo_file(repo_root, variant["algo_config"])
            _touch_repo_file(
                repo_root,
                variant["authority_metadata"]["kinematic_adapter"]["config_source"],
            )
    manifest = _write(repo_root / "manifest.yaml", payload)

    report = preflight.build_report(manifest, repo_root)

    assert report["status"] == "failed"
    assert any("high_angular.algo_config does not exist" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda payload: payload["variants"][1].pop("authority_metadata"),
            "high_angular.authority_metadata must be a mapping",
        ),
        (
            lambda payload: payload["variants"][1]["authority_metadata"][
                "kinematic_adapter"
            ].update({"changed_params": []}),
            "changed_params does not cover params",
        ),
    ],
)
def test_malformed_sweep_arm_metadata_fails_closed(
    tmp_path: Path,
    mutation,
    expected_error: str,
) -> None:
    """Malformed arm metadata fails before a campaign can be mistaken for evidence."""
    payload = _valid_payload()
    mutation(payload)
    repo_root = tmp_path
    _touch_repo_file(repo_root, payload["hard_seed_manifest"])
    for variant in payload["variants"]:
        _touch_repo_file(repo_root, variant["algo_config"])
        adapter = variant.get("authority_metadata", {}).get("kinematic_adapter", {})
        if "config_source" in adapter:
            _touch_repo_file(repo_root, adapter["config_source"])
    manifest = _write(repo_root / "manifest.yaml", payload)

    report = preflight.build_report(manifest, repo_root)

    assert report["status"] == "failed"
    assert any(expected_error in error for error in report["errors"])
