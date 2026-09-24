"""Tests for the fail-closed issue #6642 to #6643 summary composer."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

import pytest

import robot_sf.benchmark.radius_sweep_summary as composer
from robot_sf.benchmark import result_provenance
from robot_sf.benchmark.radius_sweep_summary import (
    RadiusSweepSummaryError,
    compose_radius_sweep_summary,
    write_radius_sweep_summary,
)

if TYPE_CHECKING:
    from pathlib import Path

_GATE1_SURFACES = (
    "simulator_collision_geometry",
    "obstacle_pedestrian_contact_logic",
    "feasibility_oracle",
    "metric_metadata_and_output_rows",
    "planner_inputs",
)
_GATE1_RECEIPT_PAYLOAD = {
    "schema": "radius_binding_canary_report.v1",
    "canary_schema": "radius_binding_canary.v1",
    "issue": 6641,
    "parent_issue": 6600,
    "radii_m": [0.5, 0.8, 1.0],
    "go": True,
    "verdicts": [
        {
            "schema": "radius_binding_canary.v1",
            "target_radius_m": radius,
            "go": True,
            "surfaces": [{"surface": surface, "bound": True} for surface in _GATE1_SURFACES],
        }
        for radius in (0.5, 0.8, 1.0)
    ],
}
_GATE1_RECEIPT_BYTES = json.dumps(_GATE1_RECEIPT_PAYLOAD).encode()
_GATE1_RECEIPT_SHA256 = sha256(_GATE1_RECEIPT_BYTES).hexdigest()
_FIXTURE_CONFIG_PATHS = {
    "r0p5": "configs/benchmarks/fixture_arm_0p5.yaml",
    "r0p8": "configs/benchmarks/fixture_arm_0p8.yaml",
    "r1p0": "configs/benchmarks/fixture_arm_1p0.yaml",
}
_FIXTURE_SCENARIO_MATRIX_PATH = "configs/scenarios/classic_interactions_francis2023.yaml"
_FIXTURE_PLANNER_CONFIG_PATH = "configs/algos/fixture_goal.yaml"
_FIXTURE_FAMILY_CAMPAIGN_CONFIG_PATH = "configs/benchmarks/fixture_family_roster.yaml"
_FIXTURE_FAMILY_CONFIG_PATHS = {
    "scenario_adaptive_hybrid_orca_v1": "configs/algos/fixture_hybrid_v1.yaml",
    "scenario_adaptive_hybrid_orca_v2_collision_guard": "configs/algos/fixture_hybrid_v2.yaml",
    "hybrid_rule_v3_fast_progress_static_escape": "configs/algos/fixture_hybrid_v3_static.yaml",
    "hybrid_rule_v3_fast_progress_static_escape_continuous": (
        "configs/algos/fixture_hybrid_v3_continuous.yaml"
    ),
}


def _fixture_family_feasibility_evaluator(
    _root: object,
    radius: float,
    _episodes: object,
    _definition_id: str,
    _authority_sha256: str,
) -> tuple[str, dict[str, str]]:
    """Synthetic unit-test plumbing; never campaign evidence or a production rule."""
    return (
        "fixture-authoritative family feasibility",
        {"narrow_doorway": "feasible" if radius < 1.0 else "infeasible"},
    )


@dataclass(frozen=True, slots=True)
class _GitConfigFixture:
    root: Path
    commit: str
    config_paths: dict[str, str]
    config_sha256: dict[str, str]


@pytest.fixture(scope="module")
def git_config_fixture(tmp_path_factory: pytest.TempPathFactory) -> _GitConfigFixture:
    """Create exact committed config bytes for Git-blob provenance tests."""
    root = tmp_path_factory.mktemp("radius-sweep-source")
    scenario_matrix = root / _FIXTURE_SCENARIO_MATRIX_PATH
    scenario_matrix.parent.mkdir(parents=True, exist_ok=True)
    scenario_matrix.write_text(
        "scenarios:\n  - scenario_id: case_a\n  - scenario_id: case_b\n",
        encoding="utf-8",
    )
    planner_config = root / _FIXTURE_PLANNER_CONFIG_PATH
    planner_config.parent.mkdir(parents=True, exist_ok=True)
    planner_config.write_text("fixture: goal\n", encoding="utf-8")
    for arm_key, relative in _FIXTURE_CONFIG_PATHS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"fixture_arm: {arm_key}\n"
            "planners:\n"
            "  - key: goal\n"
            "    algo: goal\n"
            f"    algo_config: {_FIXTURE_PLANNER_CONFIG_PATH}\n",
            encoding="utf-8",
        )
    for planner_key, relative in _FIXTURE_FAMILY_CONFIG_PATHS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture_variant: {planner_key}\n", encoding="utf-8")
    family_campaign = root / _FIXTURE_FAMILY_CAMPAIGN_CONFIG_PATH
    family_campaign.parent.mkdir(parents=True, exist_ok=True)
    family_lines = ["planners:"]
    for planner_key, relative in _FIXTURE_FAMILY_CONFIG_PATHS.items():
        family_lines.extend(
            (
                f"  - key: {planner_key}",
                "    algo: hybrid_rule_local_planner",
                f"    algo_config: {relative}",
            )
        )
    family_campaign.write_text("\n".join(family_lines) + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "init", "--quiet"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "add",
            "--",
            _FIXTURE_SCENARIO_MATRIX_PATH,
            *_FIXTURE_CONFIG_PATHS.values(),
            _FIXTURE_PLANNER_CONFIG_PATH,
            _FIXTURE_FAMILY_CAMPAIGN_CONFIG_PATH,
            *_FIXTURE_FAMILY_CONFIG_PATHS.values(),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Robot SF test fixture",
            "-c",
            "user.email=robot-sf-test-fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture campaign source",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    config_sha256 = {
        arm_key: sha256((root / relative).read_bytes()).hexdigest()
        for arm_key, relative in _FIXTURE_CONFIG_PATHS.items()
    }
    return _GitConfigFixture(root, commit, _FIXTURE_CONFIG_PATHS.copy(), config_sha256)


@pytest.fixture
def compact_scope(monkeypatch: pytest.MonkeyPatch, git_config_fixture: _GitConfigFixture) -> None:
    """Shrink the frozen matrix while preserving all structural checks."""
    monkeypatch.setattr(composer, "RELEASE_PLANNER_KEYS", ("goal",))
    monkeypatch.setattr(composer, "EXPECTED_SCENARIO_NAMES", ("case_a", "case_b"))
    monkeypatch.setattr(composer, "EXPECTED_SEEDS", (111, 112))
    monkeypatch.setattr(composer, "EXPECTED_ROWS_PER_ARM", 4)
    monkeypatch.setattr(composer, "EXPECTED_GATE1_RECEIPT_SHA256", _GATE1_RECEIPT_SHA256)
    monkeypatch.setattr(composer, "SOURCE_REPOSITORY_ROOT", git_config_fixture.root)
    monkeypatch.setattr(composer, "EXPECTED_CAMPAIGN_GIT_COMMIT", git_config_fixture.commit)
    monkeypatch.setattr(result_provenance, "_git_hash_fallback", lambda: git_config_fixture.commit)
    monkeypatch.setattr(
        composer,
        "EXPECTED_FAMILY_FEASIBILITY_DEFINITION_ID",
        "fixture-family-rule.v1",
    )
    monkeypatch.setattr(composer, "EXPECTED_FAMILY_FEASIBILITY_AUTHORITY_SHA256", "f" * 64)
    # Explicit synthetic unit-test plumbing; production keeps this evaluator unset.
    monkeypatch.setattr(
        composer, "_FAMILY_FEASIBILITY_EVALUATOR", _fixture_family_feasibility_evaluator
    )
    monkeypatch.setattr(
        composer,
        "EXPECTED_ARM_CAMPAIGN_CONFIGS",
        git_config_fixture.config_paths,
    )
    monkeypatch.setattr(
        composer,
        "EXPECTED_ARM_CAMPAIGN_CONFIG_SHA256",
        git_config_fixture.config_sha256,
    )


def _write_arm(
    parent: Path,
    radius: float,
    *,
    commit: str | None = None,
    gate1_receipt_sha256: str,
    include_family: bool = True,
    degraded: bool = False,
) -> Path:
    if commit is None:
        commit = composer.EXPECTED_CAMPAIGN_GIT_COMMIT
    arm_key = composer.RADIUS_TO_ARM_KEY[radius]
    root = parent / arm_key
    (root / "preflight").mkdir(parents=True)
    (root / "reports").mkdir()
    binding = {
        "arm_key": arm_key,
        "radius_m": radius,
        "gate1_receipt_sha256": gate1_receipt_sha256,
    }
    manifest = {
        "schema_version": composer.CAMPAIGN_SCHEMA,
        "campaign_id": f"campaign-{arm_key}",
        "scenario_matrix": composer.EXPECTED_SCENARIO_MATRIX,
        "scenario_matrix_hash": composer._config_hash(
            [{"scenario_id": scenario} for scenario in composer.EXPECTED_SCENARIO_NAMES]
        ),
        "git": {"commit": commit},
        "radius_binding": binding,
        "seed_policy": {"resolved_seeds": list(composer.EXPECTED_SEEDS)},
    }
    preflight = {
        "config_path": composer.EXPECTED_ARM_CAMPAIGN_CONFIGS[arm_key],
        "config_sha256": composer.EXPECTED_ARM_CAMPAIGN_CONFIG_SHA256[arm_key],
        "radius_binding": binding,
    }
    campaign_summary = {
        "campaign": {
            "campaign_id": manifest["campaign_id"],
            "git_hash": commit,
            "scenario_matrix": composer.EXPECTED_SCENARIO_MATRIX,
            "scenario_matrix_hash": manifest["scenario_matrix_hash"],
            "campaign_execution_status": "completed",
            "evidence_status": "valid",
            "benchmark_success": True,
            "total_episodes": composer.EXPECTED_ROWS_PER_ARM,
            "total_runs": 1,
            "successful_runs": 1,
            "non_success_runs": 0,
            "unexpected_failed_runs": 0,
            "row_status_summary": {"fallback_or_degraded_rows": int(degraded)},
        },
        "campaign_integrity": {"status": "valid", "benchmark_success_allowed": True},
        "planner_rows": [
            {
                "planner_key": "goal",
                "status": "ok",
                "availability_status": "available",
                "benchmark_success": "true",
                "readiness_status": "native",
                "failed_jobs": 0,
                "episodes": composer.EXPECTED_ROWS_PER_ARM,
                "success_mean": "0.5000",
                "ped_collision_count_mean": "0.5000",
                "obstacle_collision_count_mean": "0.0000",
                "total_collision_count_mean": "0.5000",
                "snqi_mean": f"{radius + 0.1115:.4f}",
            }
        ],
    }
    if include_family:
        family_feasibility = {
            "schema_version": composer.FAMILY_FEASIBILITY_SCHEMA,
            "radius_m": radius,
            "source_campaign_id": manifest["campaign_id"],
            "source_campaign_commit": commit,
            "source_config_sha256": preflight["config_sha256"],
            "approved_rule": {
                "definition_id": composer.EXPECTED_FAMILY_FEASIBILITY_DEFINITION_ID,
                "authority_sha256": composer.EXPECTED_FAMILY_FEASIBILITY_AUTHORITY_SHA256,
            },
            "definition": "fixture-authoritative family feasibility",
            "families": {"narrow_doorway": "feasible" if radius < 1.0 else "infeasible"},
        }
    for path, payload in (
        (root / "campaign_manifest.json", manifest),
        (root / "preflight/validate_config.json", preflight),
        (root / "reports/campaign_summary.json", campaign_summary),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    if include_family:
        (root / "reports/radius_family_feasibility.json").write_text(
            json.dumps(family_feasibility), encoding="utf-8"
        )

    run = root / "runs/goal__differential_drive"
    run.mkdir(parents=True)
    rows = []
    for scenario_index, scenario in enumerate(composer.EXPECTED_SCENARIO_NAMES):
        for seed in composer.EXPECTED_SEEDS:
            rows.append(
                {
                    "episode_id": f"goal-{scenario}-{seed}",
                    "config_hash": composer._config_hash({"fixture": "goal"}),
                    "algo": "goal",
                    "algorithm_metadata": {
                        "algorithm": "goal",
                        "canonical_algorithm": "goal",
                    },
                    "result_provenance": {"planner_key": "goal"},
                    "scenario_id": scenario,
                    "seed": seed,
                    "git_hash": commit,
                    "horizon": composer.EXPECTED_HORIZON,
                    "integrity": {
                        "contradictions": [],
                        "effective_view": {"degraded": False},
                    },
                    "scenario_params": {
                        "algo": "goal",
                        "algo_config_hash": composer._config_hash({"fixture": "goal"}),
                        "run_horizon": composer.EXPECTED_HORIZON,
                        "run_dt": composer.EXPECTED_DT,
                        "record_forces": composer.EXPECTED_RECORD_FORCES,
                        "robot_config": {"radius": radius},
                    },
                    "metrics": {
                        "success": scenario_index == 0,
                        "ped_collision_count": scenario_index,
                        "obstacle_collision_count": 0,
                        "total_collision_count": scenario_index,
                        "snqi": radius + seed / 1000,
                    },
                }
            )
    (run / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    _write_runner_receipt(run / "episodes.jsonl", rows)
    return root


def _write_runner_receipt(
    episodes_path: Path,
    rows: list[dict] | None = None,
    *,
    algorithm: str = "goal",
) -> None:
    """Write a synthetic receipt through the production v1 builder, for plumbing tests only."""
    if rows is None:
        rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
    schema_path = episodes_path.parents[2] / "episode-schema.json"
    schema_path.write_text("{}\n", encoding="utf-8")
    payload = result_provenance.build_result_provenance_manifest(
        out_path=episodes_path,
        episode_records=rows,
        schema_path=schema_path,
        scenario_path=(composer.SOURCE_REPOSITORY_ROOT / _FIXTURE_SCENARIO_MATRIX_PATH),
        scenarios=[{"scenario_id": scenario} for scenario in composer.EXPECTED_SCENARIO_NAMES],
        algo=algorithm,
        algo_config_path=None,
        benchmark_profile="fixture",
        suite_key=composer.EXPECTED_SUITE_KEY,
        total_jobs=len(rows),
        written=len(rows),
        horizon=composer.EXPECTED_HORIZON,
        dt=composer.EXPECTED_DT,
        record_forces=composer.EXPECTED_RECORD_FORCES,
        active_observation_mode=None,
        active_observation_level=None,
    )
    result_provenance.write_result_provenance_manifest(
        result_provenance.manifest_path_for_result_jsonl(episodes_path), payload
    )


def _write_gate1_receipt(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / "gate1-canary-receipt.json"
    path.write_bytes(_GATE1_RECEIPT_BYTES)
    return path


def _write_triplet(parent: Path, **arm_kwargs: object) -> tuple[list[Path], Path]:
    receipt = _write_gate1_receipt(parent)
    digest = sha256(receipt.read_bytes()).hexdigest()
    roots = [
        _write_arm(parent, radius, gate1_receipt_sha256=digest, **arm_kwargs)
        for radius in (0.5, 0.8, 1.0)
    ]
    return roots, receipt


def test_compose_summary_is_deterministic_and_preserves_per_arm_configs(
    tmp_path: Path, compact_scope: None
) -> None:
    """A complete triplet produces canonical tables, pairs, and distinct config hashes."""
    roots, receipt = _write_triplet(tmp_path)
    roots.reverse()
    first = compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)
    second = compose_radius_sweep_summary(list(reversed(roots)), gate1_canary_receipt=receipt)
    assert first == second
    assert first["row_accounting"]["0.5"] == {
        "declared": 4,
        "present": 4,
        "excluded_by_reason": {},
    }
    assert first["metric_tables"]["0.5"]["goal"] == pytest.approx(
        {"success": 0.5, "typed_collisions": 0.5, "snqi": 0.6115}
    )
    assert first["paired_observations"]["0.5"]["goal"]["success"] == {
        "111": 0.5,
        "112": 0.5,
    }
    digests = {row["config_sha256"] for row in first["campaign_provenance"].values()}
    assert len(digests) == 3
    arm_provenance = first["campaign_provenance"]["0.5"]
    arm_root = next(root for root in roots if root.name == "r0p5")
    episode_path = arm_root / "runs/goal__differential_drive/episodes.jsonl"
    receipt_path = episode_path.with_suffix(".jsonl.provenance.json")
    assert arm_provenance["episode_artifacts"]["goal"] == {
        "episodes_path": "runs/goal__differential_drive/episodes.jsonl",
        "episodes_sha256": sha256(episode_path.read_bytes()).hexdigest(),
        "receipt_path": "runs/goal__differential_drive/episodes.jsonl.provenance.json",
        "receipt_sha256": sha256(receipt_path.read_bytes()).hexdigest(),
    }
    assert arm_provenance["source_integrity_status"] == "runner_receipt_matched"
    assert arm_provenance["artifact_custody_status"] == "unattested"
    assert arm_provenance["promotion_allowed"] is False
    assert arm_provenance["promotion_blockers"] == ["artifact_custody_unattested"]

    first_path = write_radius_sweep_summary(first, tmp_path / "first.json")
    second_path = write_radius_sweep_summary(second, tmp_path / "second.json")
    assert first_path.read_bytes() == second_path.read_bytes()


@pytest.mark.parametrize(
    "mutation",
    ("missing", "malformed", "wrong_schema", "wrong_runner", "wrong_commit", "wrong_path"),
)
def test_composer_rejects_missing_or_invalid_runner_receipt(
    tmp_path: Path, compact_scope: None, mutation: str
) -> None:
    """Every planner's exact-byte receipt is required and checked against its frozen run."""
    roots, receipt = _write_triplet(tmp_path)
    episodes_path = roots[0] / "runs/goal__differential_drive/episodes.jsonl"
    receipt_path = episodes_path.with_suffix(".jsonl.provenance.json")
    if mutation == "missing":
        receipt_path.unlink()
    elif mutation == "malformed":
        receipt_path.write_bytes(b"not-json\n")
    else:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if mutation == "wrong_schema":
            payload["schema_version"] = "untrusted.v0"
        elif mutation == "wrong_runner":
            payload["run"]["runner"] = "other.runner"
        elif mutation == "wrong_commit":
            payload["run"]["repo_commit"] = "b" * 40
        else:
            payload["raw_artifacts"][0]["path"] = "runs/orca__differential_drive/episodes.jsonl"
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RadiusSweepSummaryError, match="runner provenance receipt"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_rejects_runner_receipt_for_wrong_frozen_algorithm(
    tmp_path: Path, compact_scope: None
) -> None:
    """A schema-valid runner receipt must still match the committed planner algorithm."""
    roots, receipt = _write_triplet(tmp_path)
    episodes_path = roots[0] / "runs/goal__differential_drive/episodes.jsonl"
    _write_runner_receipt(episodes_path, algorithm="orca")

    with pytest.raises(
        RadiusSweepSummaryError, match="runner provenance receipt algorithm mismatch"
    ):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("matrix_path", "scenario matrix identity mismatch"),
        ("matrix_digest", "invalid runner provenance receipt"),
        ("matrix_hash", "scenario-matrix hash mismatch"),
        ("suite_key", "suite identity mismatch"),
    ),
)
def test_composer_binds_runner_receipt_to_frozen_matrix_and_suite(
    tmp_path: Path,
    compact_scope: None,
    mutation: str,
    message: str,
) -> None:
    """A valid artifact digest is insufficient if its frozen input/suite identity differs."""
    roots, receipt = _write_triplet(tmp_path)
    episodes_path = roots[0] / "runs/goal__differential_drive/episodes.jsonl"
    receipt_path = episodes_path.with_suffix(".jsonl.provenance.json")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "matrix_path":
        payload["inputs"]["scenario_matrix"]["path"] = "configs/scenarios/wrong.yaml"
    elif mutation == "matrix_digest":
        payload["inputs"]["scenario_matrix"]["sha256"] = "0" * 64
    elif mutation == "matrix_hash":
        payload["campaign_identity"]["scenario_matrix_hash"] = "0" * 64
    else:
        identity = payload["campaign_identity"]
        identity["suite_key"] = "unfrozen-suite"
        identity["input_bundle_sha256"] = result_provenance._canonical_input_bundle_sha256(
            inputs=payload["inputs"],
            algo=identity["algorithm"],
            protocol_version=payload["run"]["protocol_version"],
            suite_key=identity["suite_key"],
        )
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RadiusSweepSummaryError, match=message):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_requires_campaign_manifest_and_summary_matrix_hashes_to_match(
    tmp_path: Path, compact_scope: None
) -> None:
    """Campaign inputs cannot advertise different matrix expansions across surfaces."""
    roots, receipt = _write_triplet(tmp_path)
    manifest_path = roots[0] / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scenario_matrix_hash"] = "0" * 16
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RadiusSweepSummaryError, match="campaign identity mismatch"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


@pytest.mark.parametrize(
    ("location", "expected_error"),
    (
        ("status", "fallback/degraded runtime marker"),
        ("evidence_status", "fallback/degraded runtime marker"),
        ("execution_status", "fallback/degraded runtime marker"),
        ("planner_runtime", "fallback/degraded runtime marker"),
        ("planner_diagnostics", "fallback/degraded runtime marker"),
        ("planner_diagnostic_reasons", "fallback/degraded runtime marker"),
        ("planner_diagnostic_reasons_null", "fallback/degraded runtime marker"),
        ("algorithm_ineligible", "ineligible algorithm metadata"),
        ("algorithm_eligibility_invalid", "invalid algorithm evidence_eligible metadata"),
        ("foresight_ineligible", "ineligible foresight metadata"),
        ("foresight_eligibility_invalid", "invalid foresight evidence_eligible metadata"),
    ),
)
def test_composer_rejects_row_fallback_even_with_matching_receipt_and_summary(
    tmp_path: Path,
    compact_scope: None,
    location: str,
    expected_error: str,
) -> None:
    """Row-level runtime markers cannot be hidden by a clean aggregate status count."""
    roots, receipt = _write_triplet(tmp_path)
    episodes_path = roots[0] / "runs/goal__differential_drive/episodes.jsonl"
    rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
    metadata = rows[0]["algorithm_metadata"]
    if location == "status":
        metadata["status"] = "fallback"
    elif location == "evidence_status":
        rows[0]["evidence_status"] = "fallback"
    elif location == "execution_status":
        rows[0]["execution_status"] = "degraded"
    elif location == "planner_runtime":
        metadata["planner_runtime"] = {"fallback_triggered": True}
    elif location == "planner_diagnostics":
        metadata["planner_diagnostics"] = {"fallback": True, "fallback_count": 1}
    elif location == "planner_diagnostic_reasons":
        metadata["planner_diagnostics"] = {
            "fallback": False,
            "fallback_count": 0,
            "fallback_reason": None,
            "fallback_reasons": {"wrapper_exception": 1},
        }
    elif location == "planner_diagnostic_reasons_null":
        metadata["planner_diagnostics"] = {"fallback_reasons": None}
    elif location == "algorithm_ineligible":
        metadata["evidence_eligible"] = False
    elif location == "algorithm_eligibility_invalid":
        metadata["evidence_eligible"] = "false"
    else:
        metadata["foresight_prediction"] = {
            "evidence_eligible": 0 if location == "foresight_eligibility_invalid" else False
        }
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    # The row receipt matches the edited artifact. The campaign summary still claims zero
    # fallback/degraded rows, so only checking that aggregate would miss this marker.
    _write_runner_receipt(episodes_path, rows)

    with pytest.raises(RadiusSweepSummaryError, match=expected_error):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_rejects_non_object_planner_diagnostics_even_with_matching_receipt(
    tmp_path: Path, compact_scope: None
) -> None:
    """A malformed diagnostics container cannot hide row-level fallback markers."""
    roots, receipt = _write_triplet(tmp_path)
    episodes_path = roots[0] / "runs/goal__differential_drive/episodes.jsonl"
    rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["algorithm_metadata"]["planner_diagnostics"] = [{"fallback": True}]
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_runner_receipt(episodes_path, rows)

    with pytest.raises(
        RadiusSweepSummaryError,
        match="invalid algorithm_metadata.planner_diagnostics",
    ):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


@pytest.mark.parametrize(
    ("diagnostics", "expected_marker"),
    (
        pytest.param(
            {"fallback_reason": None},
            "algorithm_metadata.planner_diagnostics.fallback_reason=invalid",
            id="reason-without-explicit-false-flag",
        ),
        pytest.param(
            {"planner_fallback": "used"},
            "algorithm_metadata.planner_diagnostics.planner_fallback=invalid",
            id="unknown-fallback-status",
        ),
        pytest.param(
            {"fast_pysf_wrapper": {"fallback_reason": None}},
            "algorithm_metadata.planner_diagnostics.fast_pysf_wrapper.fallback_reason=invalid",
            id="nested-reason-without-explicit-false-flag",
        ),
        pytest.param(
            {"fast_pysf_wrapper": {"planner_fallback": "used"}},
            "algorithm_metadata.planner_diagnostics.fast_pysf_wrapper.planner_fallback=invalid",
            id="nested-unknown-fallback-status",
        ),
        pytest.param(
            {"fast_pysf_wrapper": {"fallback_reasons": {"wrapper_exception": 1}}},
            "algorithm_metadata.planner_diagnostics.fast_pysf_wrapper.fallback_reasons="
            "non-empty-or-invalid",
            id="nested-non-empty-fallback-reasons",
        ),
        pytest.param(
            {"fast_pysf_wrapper": {"fallback_reasons": []}},
            "algorithm_metadata.planner_diagnostics.fast_pysf_wrapper.fallback_reasons="
            "non-empty-or-invalid",
            id="nested-non-mapping-fallback-reasons",
        ),
    ),
)
def test_composer_rejects_unvalidated_planner_diagnostic_markers(
    tmp_path: Path,
    compact_scope: None,
    diagnostics: dict[str, object],
    expected_marker: str,
) -> None:
    """Unrecognized fallback fields and ambiguous empty reasons cannot be silently dropped."""
    roots, receipt = _write_triplet(tmp_path)
    episodes_path = roots[0] / "runs/goal__differential_drive/episodes.jsonl"
    rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["algorithm_metadata"]["planner_diagnostics"] = diagnostics
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_runner_receipt(episodes_path, rows)

    with pytest.raises(RadiusSweepSummaryError, match=expected_marker):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_episode_status_allows_empty_social_force_diagnostics() -> None:
    """Empty planner diagnostic maps are not mistaken for runtime fallback markers."""
    composer._validate_episode_runtime_status(
        {
            "status": "success",
            "algorithm_metadata": {
                "algorithm": "social_force",
                "canonical_algorithm": "social_force",
                "status": "ok",
                "planner_diagnostics": {
                    "planner_type": "SocialForcePlanner",
                    "fallback": False,
                    "fallback_count": 0,
                    "fallback_reason": None,
                    "fallback_reasons": {},
                },
            },
        },
        planner="social_force",
        expected_algorithm="social_force",
        radius=0.5,
    )


def test_episode_status_allows_nested_clean_fast_pysf_diagnostics() -> None:
    """The clean nested FastPysfWrapper producer shape remains admissible."""
    diagnostics = {
        "planner_type": "FastPysfPlannerPolicy",
        "fallback": False,
        "fallback_count": 0,
        "fallback_reason": None,
        "fallback_reasons": {},
        "fast_pysf_wrapper": {
            "planner_type": "FastPysfWrapper",
            "fallback": False,
            "fallback_count": 0,
            "fallback_reason": None,
            "fallback_reasons": {},
        },
    }
    composer._validate_episode_runtime_status(
        {
            "status": "success",
            "algorithm_metadata": {
                "algorithm": "social_force",
                "canonical_algorithm": "social_force",
                "status": "ok",
                "planner_diagnostics": diagnostics,
            },
        },
        planner="social_force",
        expected_algorithm="social_force",
        radius=0.5,
    )
    assert diagnostics["fast_pysf_wrapper"]["fallback_reasons"] == {}
    assert "fallback_used" not in diagnostics["fast_pysf_wrapper"]


@pytest.mark.parametrize("field", ("jsonl_line", "seed"))
def test_runner_row_binding_rejects_boolean_numeric_identities(field: str, tmp_path: Path) -> None:
    """JSON booleans are not accepted as integer row identifiers."""
    episode = {
        "episode_id": "episode-1",
        "scenario_id": "case-a",
        "seed": 1,
        "config_hash": "config-hash",
        "git_hash": "a" * 40,
    }
    receipt_row = {
        "episode_id": "episode-1",
        "scenario_id": "case-a",
        "seed": 1,
        "config_hash": "config-hash",
        "repo_commit": "a" * 40,
        "jsonl_line": 1,
    }
    receipt_row[field] = True

    with pytest.raises(RadiusSweepSummaryError, match="numeric identity has invalid types"):
        composer._validate_runner_row_binding(
            receipt_row,
            episode,
            jsonl_line=1,
            episodes_path=tmp_path / "episodes.jsonl",
        )


def test_runner_row_binding_requires_receipt_seed_to_match_episode_seed(tmp_path: Path) -> None:
    """Receipt seed identity is compared with the episode record, not itself."""
    episode = {
        "episode_id": "episode-1",
        "scenario_id": "case-a",
        "seed": 1,
        "config_hash": "config-hash",
        "git_hash": "a" * 40,
    }
    receipt_row = {
        "episode_id": "episode-1",
        "scenario_id": "case-a",
        "seed": 2,
        "config_hash": "config-hash",
        "repo_commit": "a" * 40,
        "jsonl_line": 1,
    }

    with pytest.raises(RadiusSweepSummaryError, match="runner provenance row identity mismatch"):
        composer._validate_runner_row_binding(
            receipt_row,
            episode,
            jsonl_line=1,
            episodes_path=tmp_path / "episodes.jsonl",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("horizon", 599, "row horizon mismatch"),
        ("dt", 0.2, "row dt mismatch"),
        ("record_forces", False, "row record_forces mismatch"),
    ),
)
def test_runner_row_binding_requires_frozen_simulator_settings(
    field: str, value: object, message: str, tmp_path: Path
) -> None:
    """Receipt rows cannot contradict the frozen h600/dt/force settings."""
    episode = {
        "episode_id": "episode-1",
        "scenario_id": "case-a",
        "seed": 1,
        "config_hash": "config-hash",
        "git_hash": "a" * 40,
    }
    receipt_row = {
        "episode_id": "episode-1",
        "scenario_id": "case-a",
        "seed": 1,
        "config_hash": "config-hash",
        "repo_commit": "a" * 40,
        "jsonl_line": 1,
        "simulator_settings": {
            "horizon": composer.EXPECTED_HORIZON,
            "dt": composer.EXPECTED_DT,
            "record_forces": composer.EXPECTED_RECORD_FORCES,
        },
    }
    receipt_row["simulator_settings"][field] = value

    with pytest.raises(RadiusSweepSummaryError, match=message):
        composer._validate_runner_row_binding(
            receipt_row,
            episode,
            jsonl_line=1,
            episodes_path=tmp_path / "episodes.jsonl",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("horizon", 599, "episode horizon mismatch"),
        ("run_horizon", 599, "episode horizon mismatch"),
        ("run_dt", 0.2, "episode dt mismatch"),
        ("record_forces", False, "episode record_forces mismatch"),
    ),
)
def test_episode_record_requires_frozen_simulator_settings(
    tmp_path: Path,
    compact_scope: None,
    field: str,
    value: object,
    message: str,
) -> None:
    """Episode bytes themselves must agree with the frozen simulator settings."""
    roots, _receipt = _write_triplet(tmp_path)
    episode_path = roots[0] / "runs/goal__differential_drive/episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    if field == "horizon":
        rows[0][field] = value
    else:
        rows[0]["scenario_params"][field] = value
    episode_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_runner_receipt(episode_path, rows)

    with pytest.raises(RadiusSweepSummaryError, match=message):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=_receipt)


@pytest.mark.parametrize("symlink_parent", ("runs", "planner"))
def test_composer_rejects_symlinked_episode_path_ancestors(
    tmp_path: Path, compact_scope: None, symlink_parent: str
) -> None:
    """The checked episode and sidecar cannot be reached through a symlinked parent."""
    roots, receipt = _write_triplet(tmp_path)
    root = roots[0]
    runs_path = root / "runs"
    if symlink_parent == "runs":
        real_parent = root / "runs-target"
        runs_path.rename(real_parent)
        runs_path.symlink_to(real_parent, target_is_directory=True)
    else:
        run_path = runs_path / "goal__differential_drive"
        real_parent = root / "goal-run-target"
        run_path.rename(real_parent)
        run_path.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(RadiusSweepSummaryError, match="runs path|planner run directory"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_rejects_coordinated_episode_and_planner_aggregate_edit_with_stale_receipt(
    tmp_path: Path, compact_scope: None
) -> None:
    """Changing source metrics and their serialized aggregate cannot bypass stale receipt SHA."""
    roots, receipt = _write_triplet(tmp_path)
    root = roots[0]
    episodes_path = root / "runs/goal__differential_drive/episodes.jsonl"
    episode_lines = episodes_path.read_text(encoding="utf-8").splitlines()
    episode = json.loads(episode_lines[0])
    episode["metrics"]["success"] = False
    episode_lines[0] = json.dumps(episode)
    episodes_path.write_text("\n".join(episode_lines) + "\n", encoding="utf-8")

    campaign_summary_path = root / "reports/campaign_summary.json"
    campaign_summary = json.loads(campaign_summary_path.read_text(encoding="utf-8"))
    campaign_summary["planner_rows"][0]["success_mean"] = "0.2500"
    campaign_summary_path.write_text(json.dumps(campaign_summary), encoding="utf-8")

    with pytest.raises(RadiusSweepSummaryError, match="episode JSONL digest mismatch"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


@pytest.mark.parametrize(
    "carrier",
    [
        "algo",
        "algorithm_metadata.algorithm",
        "algorithm_metadata.canonical_algorithm",
        "scenario_params.algo",
        "result_provenance.planner_key",
    ],
)
def test_composer_rejects_each_mismatched_episode_planner_carrier(
    tmp_path: Path, compact_scope: None, carrier: str
) -> None:
    """Every explicit planner identity must agree with the run-directory planner."""
    roots, receipt = _write_triplet(tmp_path)
    episode_path = roots[0] / "runs/goal__differential_drive/episodes.jsonl"
    lines = episode_path.read_text(encoding="utf-8").splitlines()
    episode = json.loads(lines[0])
    if carrier == "result_provenance.planner_key":
        episode["result_provenance"]["planner_key"] = "orca"
    else:
        episode.pop("algo")
        episode.pop("algorithm_metadata")
        episode["scenario_params"].pop("algo")
    if carrier == "algo":
        episode["algo"] = "orca"
    elif carrier.startswith("algorithm_metadata."):
        field = carrier.rsplit(".", maxsplit=1)[-1]
        episode["algorithm_metadata"] = {field: "orca"}
    elif carrier == "scenario_params.algo":
        episode["scenario_params"]["algo"] = "orca"
    lines[0] = json.dumps(episode)
    episode_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_runner_receipt(episode_path)

    with pytest.raises(RadiusSweepSummaryError, match="episode planner identity mismatch"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_rejects_conflicting_episode_planner_carriers(
    tmp_path: Path, compact_scope: None
) -> None:
    """A conflicting pair of payload identities cannot be hidden by the run directory."""
    roots, receipt = _write_triplet(tmp_path)
    episode_path = roots[0] / "runs/goal__differential_drive/episodes.jsonl"
    lines = episode_path.read_text(encoding="utf-8").splitlines()
    episode = json.loads(lines[0])
    episode["algorithm_metadata"]["canonical_algorithm"] = "orca"
    lines[0] = json.dumps(episode)
    episode_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_runner_receipt(episode_path)

    with pytest.raises(RadiusSweepSummaryError, match="planner identity carriers conflict"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_rejects_episode_without_planner_identity_carriers(
    tmp_path: Path, compact_scope: None
) -> None:
    """An episode cannot inherit its planner solely from the run-directory name."""
    roots, receipt = _write_triplet(tmp_path)
    episode_path = roots[0] / "runs/goal__differential_drive/episodes.jsonl"
    lines = episode_path.read_text(encoding="utf-8").splitlines()
    episode = json.loads(lines[0])
    episode.pop("algo")
    episode["algorithm_metadata"].pop("algorithm")
    episode["algorithm_metadata"].pop("canonical_algorithm")
    episode["result_provenance"].pop("planner_key")
    episode["scenario_params"].pop("algo")
    lines[0] = json.dumps(episode)
    episode_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_runner_receipt(episode_path)

    with pytest.raises(RadiusSweepSummaryError, match="no embedded planner identity carrier"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_accepts_canonical_alias_for_episode_algorithm(
    tmp_path: Path, compact_scope: None
) -> None:
    """Only aliases recognized by the canonical algorithm registry normalize to the directory."""
    roots, receipt = _write_triplet(tmp_path)
    episode_path = roots[0] / "runs/goal__differential_drive/episodes.jsonl"
    lines = episode_path.read_text(encoding="utf-8").splitlines()
    episode = json.loads(lines[0])
    episode["algo"] = "simple_policy"
    lines[0] = json.dumps(episode)
    episode_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_runner_receipt(episode_path)

    compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_frozen_hybrid_roster_algorithms_and_configs_bind_each_alias(
    monkeypatch: pytest.MonkeyPatch, git_config_fixture: _GitConfigFixture
) -> None:
    """Shared family algorithms remain distinct through committed per-arm config hashes."""
    planner_keys = tuple(_FIXTURE_FAMILY_CONFIG_PATHS)
    monkeypatch.setattr(composer, "SOURCE_REPOSITORY_ROOT", git_config_fixture.root)
    monkeypatch.setattr(composer, "RELEASE_PLANNER_KEYS", planner_keys)
    identities = composer._committed_planner_identities(
        git_config_fixture.commit, _FIXTURE_FAMILY_CAMPAIGN_CONFIG_PATH
    )
    scenario_id = composer.EXPECTED_SCENARIO_NAMES[0]

    assert tuple(identities) == planner_keys
    assert {
        identity.algorithm
        for planner_identities in identities.values()
        for identity in planner_identities.values()
    } == {"hybrid_rule_local_planner"}
    assert len(
        {
            planner_identities[scenario_id].algo_config_hash
            for planner_identities in identities.values()
        }
    ) == len(planner_keys)

    def episode_row(planner_key: str, scenario: str = scenario_id) -> dict[str, object]:
        identity = identities[planner_key][scenario]
        return {
            "algo": identity.algorithm,
            "algorithm_metadata": {
                "algorithm": identity.algorithm,
                "canonical_algorithm": identity.algorithm,
            },
            "scenario_params": {
                "algo": identity.algorithm,
                "algo_config_hash": identity.algo_config_hash,
            },
            "result_provenance": {"planner_key": planner_key},
        }

    for planner_key in planner_keys:
        composer._validate_episode_planner_identity(
            episode_row(planner_key),
            planner=planner_key,
            expected=identities[planner_key][scenario_id],
            radius=1.0,
        )

    first_key, second_key = planner_keys[:2]
    shared_identity = composer._PlannerIdentity(
        algorithm=identities[first_key][scenario_id].algorithm,
        algo_config_hash=identities[first_key][scenario_id].algo_config_hash,
        runner_algorithm=identities[first_key][scenario_id].runner_algorithm,
        planner_key_required=True,
    )
    no_key_row = episode_row(first_key)
    no_key_row.pop("result_provenance")
    with pytest.raises(RadiusSweepSummaryError, match="no embedded planner_key carrier"):
        composer._validate_episode_planner_identity(
            no_key_row,
            planner=first_key,
            expected=shared_identity,
            radius=1.0,
        )

    with pytest.raises(RadiusSweepSummaryError, match="algo_config_hash"):
        composer._validate_episode_planner_identity(
            episode_row(second_key),
            planner=first_key,
            expected=identities[first_key][scenario_id],
            radius=1.0,
        )

    wrong_key_row = episode_row(first_key)
    wrong_key_row["result_provenance"] = {"planner_key": second_key}
    with pytest.raises(RadiusSweepSummaryError, match="planner-key carriers"):
        composer._validate_episode_planner_identity(
            wrong_key_row,
            planner=first_key,
            expected=identities[first_key][scenario_id],
            radius=1.0,
        )


def test_frozen_hybrid_manifests_resolve_per_scenario_runtime_identity() -> None:
    """Pinned hybrid manifests use runtime merges, overrides, and fail-closed alias checks."""
    commit = composer.EXPECTED_CAMPAIGN_GIT_COMMIT
    campaign_path = composer.EXPECTED_ARM_CAMPAIGN_CONFIGS["r1p0"]
    identities = composer._committed_planner_identities(commit, campaign_path)
    v1_key = "scenario_adaptive_hybrid_orca_v1"
    v2_key = "scenario_adaptive_hybrid_orca_v2_collision_guard"
    v1_manifest_path = "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v1.yaml"
    v1_manifest = composer._committed_yaml_mapping(commit, v1_manifest_path, "fixture manifest")

    ordinary_scenario = "francis2023_narrow_doorway"
    ordinary_identity = identities[v1_key][ordinary_scenario]
    assert ordinary_identity.algorithm == "hybrid_rule_local_planner"
    base_config = composer._committed_manifest_reference_yaml(
        commit,
        v1_manifest_path,
        v1_manifest.get("base_config_path"),
        label="fixture base config",
    )
    expected_config = dict(base_config)
    expected_config.update(v1_manifest["params"])
    assert ordinary_identity.algo_config_hash == composer._config_hash(expected_config)

    overridden_scenario = "francis2023_perpendicular_traffic"
    overridden_identity = identities[v1_key][overridden_scenario]
    expected_override_config = dict(expected_config)
    expected_override_config.update(v1_manifest["scenario_overrides"][overridden_scenario])
    assert overridden_identity.algorithm == "hybrid_rule_local_planner"
    assert overridden_identity.algo_config_hash == composer._config_hash(expected_override_config)
    assert overridden_identity.algo_config_hash != ordinary_identity.algo_config_hash

    orca_scenario = "francis2023_leave_group"
    v1_orca_identity = identities[v1_key][orca_scenario]
    v2_orca_identity = identities[v2_key][orca_scenario]
    orca_override = v1_manifest["scenario_algo_overrides"][orca_scenario]
    orca_base = composer._committed_manifest_reference_yaml(
        commit,
        v1_manifest_path,
        orca_override.get("base_config_path"),
        label="fixture ORCA base config",
    )
    expected_orca_config = dict(orca_base)
    expected_orca_config.update(orca_override["params"])
    expected_orca_hash = composer._config_hash(expected_orca_config)
    assert v1_orca_identity.algorithm == v2_orca_identity.algorithm == "orca"
    assert v1_orca_identity.algo_config_hash == v2_orca_identity.algo_config_hash
    assert v1_orca_identity.algo_config_hash == expected_orca_hash
    assert v1_orca_identity.planner_key_required
    assert v2_orca_identity.planner_key_required
    assert (
        identities[v1_key][ordinary_scenario].algo_config_hash
        == identities[v2_key][ordinary_scenario].algo_config_hash
    )
    assert identities[v1_key][ordinary_scenario].planner_key_required
    assert identities[v2_key][ordinary_scenario].planner_key_required
    shared_scenarios = [
        scenario
        for scenario in composer.EXPECTED_SCENARIO_NAMES
        if (
            identities[v1_key][scenario].algorithm,
            identities[v1_key][scenario].algo_config_hash,
        )
        == (
            identities[v2_key][scenario].algorithm,
            identities[v2_key][scenario].algo_config_hash,
        )
    ]
    assert len(shared_scenarios) == len(composer.EXPECTED_SCENARIO_NAMES) - 1
    assert "classic_merging_low" not in shared_scenarios
    assert not identities[v1_key]["classic_merging_low"].planner_key_required
    assert not identities[v2_key]["classic_merging_low"].planner_key_required

    missing_key_row = {
        "algo": "orca",
        "scenario_params": {"algo_config_hash": v1_orca_identity.algo_config_hash},
    }
    with pytest.raises(RadiusSweepSummaryError, match="no embedded planner_key carrier"):
        composer._validate_episode_planner_identity(
            missing_key_row,
            planner=v1_key,
            expected=v1_orca_identity,
            radius=1.0,
        )

    distinct_key = "hybrid_rule_v3_fast_progress_static_escape"
    assert (
        identities[distinct_key][ordinary_scenario].algo_config_hash
        != ordinary_identity.algo_config_hash
    )
    swapped_row = {
        "algo": identities[distinct_key][ordinary_scenario].algorithm,
        "scenario_params": {
            "algo_config_hash": identities[distinct_key][ordinary_scenario].algo_config_hash
        },
    }
    with pytest.raises(RadiusSweepSummaryError, match="algo_config_hash"):
        composer._validate_episode_planner_identity(
            swapped_row,
            planner=v1_key,
            expected=ordinary_identity,
            radius=1.0,
        )


def test_composer_rejects_missing_authoritative_family_semantics(
    tmp_path: Path, compact_scope: None
) -> None:
    """Campaign outcomes are not silently reinterpreted as family feasibility."""
    roots, receipt = _write_triplet(tmp_path)
    (roots[-1] / "reports/radius_family_feasibility.json").unlink()
    with pytest.raises(RadiusSweepSummaryError, match="lacks authoritative family_feasibility"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_rejects_family_receipts_without_a_pinned_owner_approved_rule(
    tmp_path: Path, compact_scope: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt cannot establish its own family-feasibility authority."""
    roots, receipt = _write_triplet(tmp_path)
    monkeypatch.setattr(composer, "EXPECTED_FAMILY_FEASIBILITY_DEFINITION_ID", None)
    monkeypatch.setattr(composer, "EXPECTED_FAMILY_FEASIBILITY_AUTHORITY_SHA256", None)
    with pytest.raises(RadiusSweepSummaryError, match="no owner-approved family-feasibility"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("definition_id", "self-approved.v1"),
        ("authority_sha256", "e" * 64),
    ],
)
def test_composer_rejects_unpinned_family_rule_identity(
    tmp_path: Path, compact_scope: None, field: str, value: str
) -> None:
    """Family feasibility must carry the exact rule identity pinned by source code."""
    roots, receipt = _write_triplet(tmp_path)
    family_path = roots[0] / "reports/radius_family_feasibility.json"
    family = json.loads(family_path.read_text(encoding="utf-8"))
    family["approved_rule"][field] = value
    family_path.write_text(json.dumps(family), encoding="utf-8")
    with pytest.raises(RadiusSweepSummaryError, match="pinned approved rule identity"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_production_family_feasibility_has_no_rule_pins_or_evaluator(tmp_path: Path) -> None:
    """Production cannot admit a family receipt from identity strings alone."""
    assert composer.EXPECTED_FAMILY_FEASIBILITY_DEFINITION_ID is None
    assert composer.EXPECTED_FAMILY_FEASIBILITY_AUTHORITY_SHA256 is None
    assert composer._FAMILY_FEASIBILITY_EVALUATOR is None
    with pytest.raises(RadiusSweepSummaryError, match="no owner-approved family-feasibility"):
        composer._family_feasibility(
            tmp_path,
            radius=0.5,
            campaign_id="campaign-r0p5",
            campaign_commit="a" * 40,
            config_sha256="b" * 64,
            episodes=(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("family_label", "without an in-tree evaluator"),
        ("family_status", "without an in-tree evaluator"),
        ("definition", "without an in-tree evaluator"),
        ("source_campaign_id", "source provenance mismatch"),
        ("source_campaign_commit", "source provenance mismatch"),
        ("source_config_sha256", "source provenance mismatch"),
    ],
)
def test_family_receipt_mutations_stay_blocked_without_evaluator(
    tmp_path: Path,
    compact_scope: None,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    """Fixture-pinned rule strings cannot admit edited labels, definitions, or bindings."""
    roots, receipt = _write_triplet(tmp_path)
    monkeypatch.setattr(composer, "_FAMILY_FEASIBILITY_EVALUATOR", None)
    family_path = roots[0] / "reports/radius_family_feasibility.json"
    family = json.loads(family_path.read_text(encoding="utf-8"))
    if mutation == "family_label":
        family["families"]["invented_family"] = "feasible"
    elif mutation == "family_status":
        family["families"]["narrow_doorway"] = "infeasible"
    elif mutation == "definition":
        family["definition"] = "changed self-declared definition"
    elif mutation == "source_campaign_id":
        family["source_campaign_id"] = "different-campaign"
    elif mutation == "source_campaign_commit":
        family["source_campaign_commit"] = "b" * 40
    else:
        family["source_config_sha256"] = "e" * 64
    family_path.write_text(json.dumps(family), encoding="utf-8")

    with pytest.raises(RadiusSweepSummaryError, match=message):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


@pytest.mark.parametrize("mutation", ["family_label", "family_status", "definition"])
def test_family_evaluator_rejects_receipt_result_mismatch(
    tmp_path: Path, compact_scope: None, mutation: str
) -> None:
    """The test-only evaluator exercises comparison plumbing, not scientific evidence."""
    roots, receipt = _write_triplet(tmp_path)
    family_path = roots[0] / "reports/radius_family_feasibility.json"
    family = json.loads(family_path.read_text(encoding="utf-8"))
    if mutation == "family_label":
        family["families"]["invented_family"] = "feasible"
    elif mutation == "family_status":
        family["families"]["narrow_doorway"] = "infeasible"
    else:
        family["definition"] = "changed self-declared definition"
    family_path.write_text(json.dumps(family), encoding="utf-8")

    with pytest.raises(
        RadiusSweepSummaryError, match="does not match independently evaluated source-row results"
    ):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_family_evaluator_wraps_declared_summary_errors(
    tmp_path: Path,
    compact_scope: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expected evaluator validation failures retain radius context."""
    roots, receipt = _write_triplet(tmp_path)

    def fail_with_contract_error(*_args: object) -> tuple[str, dict[str, str]]:
        raise RadiusSweepSummaryError("source rows are invalid")

    monkeypatch.setattr(composer, "_FAMILY_FEASIBILITY_EVALUATOR", fail_with_contract_error)

    with pytest.raises(
        RadiusSweepSummaryError,
        match=r"radius 0\.5 family-feasibility evaluator failed: source rows are invalid",
    ):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_family_evaluator_does_not_mask_unexpected_exceptions(
    tmp_path: Path,
    compact_scope: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected evaluator bugs propagate instead of being normalized as data errors."""
    roots, receipt = _write_triplet(tmp_path)

    def fail_with_bug(*_args: object) -> tuple[str, dict[str, str]]:
        raise RuntimeError("fixture evaluator bug")

    monkeypatch.setattr(composer, "_FAMILY_FEASIBILITY_EVALUATOR", fail_with_bug)

    with pytest.raises(RuntimeError, match="fixture evaluator bug"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_rejects_campaign_commit_outside_frozen_source(
    tmp_path: Path, compact_scope: None
) -> None:
    """A self-consistent but unpinned commit cannot stand in for the frozen campaign source."""
    roots, receipt = _write_triplet(tmp_path)
    root = roots[-1]
    manifest_path = root / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["git"]["commit"] = "b" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary_path = root / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["campaign"]["git_hash"] = "b" * 40
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    family_path = root / "reports/radius_family_feasibility.json"
    family = json.loads(family_path.read_text(encoding="utf-8"))
    family["source_campaign_commit"] = "b" * 40
    family_path.write_text(json.dumps(family), encoding="utf-8")
    episodes_path = root / "runs/goal__differential_drive/episodes.jsonl"
    episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
    for episode in episodes:
        episode["git_hash"] = "b" * 40
    episodes_path.write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes), encoding="utf-8"
    )
    with pytest.raises(RadiusSweepSummaryError, match="frozen #6642 source commit"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_hashes_config_blob_at_campaign_commit_not_worktree_bytes(
    tmp_path: Path,
    compact_scope: None,
    git_config_fixture: _GitConfigFixture,
) -> None:
    """Config identity comes from the frozen Git object, not the mutable current file."""
    roots, receipt = _write_triplet(tmp_path)
    config_path = git_config_fixture.root / git_config_fixture.config_paths["r0p5"]
    original_bytes = config_path.read_bytes()
    config_path.write_bytes(b"changed working-tree bytes\n")
    try:
        summary = compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)
    finally:
        config_path.write_bytes(original_bytes)
    assert (
        summary["campaign_provenance"]["0.5"]["config_sha256"]
        == (git_config_fixture.config_sha256["r0p5"])
    )


def test_composer_rejects_fallback_or_degraded_rows(tmp_path: Path, compact_scope: None) -> None:
    """A campaign-level degraded row count blocks composition before interpretation."""
    roots, receipt = _write_triplet(tmp_path)
    summary_path = roots[0] / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["campaign"]["row_status_summary"]["fallback_or_degraded_rows"] = 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RadiusSweepSummaryError, match="fallback/degraded"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_rejects_duplicate_episode_identity(tmp_path: Path, compact_scope: None) -> None:
    """Duplicate planner/scenario/seed rows fail closed even when counts still look plausible."""
    roots, receipt = _write_triplet(tmp_path)
    episodes = roots[0] / "runs/goal__differential_drive/episodes.jsonl"
    lines = episodes.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[0]
    episodes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_runner_receipt(episodes)
    with pytest.raises(RadiusSweepSummaryError, match="duplicate row identity"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


@pytest.mark.parametrize(
    "field",
    [
        "success_mean",
        "ped_collision_count_mean",
        "obstacle_collision_count_mean",
        "total_collision_count_mean",
        "snqi_mean",
    ],
)
def test_composer_rejects_planner_aggregate_contradictions(
    tmp_path: Path, compact_scope: None, field: str
) -> None:
    """Every camera-ready mean must reconcile with the admitted episode population."""
    roots, receipt = _write_triplet(tmp_path)
    summary_path = roots[0] / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["planner_rows"][0][field] = "9.9999"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RadiusSweepSummaryError, match=f"aggregate mismatch:{field}"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_rejects_coordinated_config_digest_tampering(
    tmp_path: Path, compact_scope: None
) -> None:
    """A family receipt cannot echo an arbitrary replacement arm-config digest."""
    roots, receipt = _write_triplet(tmp_path)
    replacement_digest = "f" * 64
    for root in roots:
        preflight_path = root / "preflight/validate_config.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight["config_sha256"] = replacement_digest
        preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

        family_path = root / "reports/radius_family_feasibility.json"
        family = json.loads(family_path.read_text(encoding="utf-8"))
        family["source_config_sha256"] = replacement_digest
        family_path.write_text(json.dumps(family), encoding="utf-8")
    with pytest.raises(
        RadiusSweepSummaryError, match="does not match config bytes at the campaign commit"
    ):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_requires_exact_original_gate1_receipt(
    tmp_path: Path, compact_scope: None
) -> None:
    """A passing but byte-different replay cannot replace the campaign-bound receipt."""
    roots, receipt = _write_triplet(tmp_path)
    replay = tmp_path / "replayed-receipt.json"
    replay.write_text(receipt.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RadiusSweepSummaryError, match="frozen Gate 2 receipt digest"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=replay)


def test_composer_rejects_coordinated_gate1_receipt_and_metadata_tampering(
    tmp_path: Path, compact_scope: None
) -> None:
    """Self-consistent metadata cannot replace the frozen Gate 1 artifact identity."""
    roots, receipt = _write_triplet(tmp_path)
    tampered_payload = {**_GATE1_RECEIPT_PAYLOAD, "generated_at": "different-replay"}
    receipt.write_text(json.dumps(tampered_payload), encoding="utf-8")
    tampered_digest = sha256(receipt.read_bytes()).hexdigest()
    for root in roots:
        for relative_path in ("campaign_manifest.json", "preflight/validate_config.json"):
            path = root / relative_path
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["radius_binding"]["gate1_receipt_sha256"] = tampered_digest
            path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RadiusSweepSummaryError, match="frozen Gate 2 receipt digest"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composed_summary_is_independent_of_campaign_root_paths(
    tmp_path: Path, compact_scope: None
) -> None:
    """Relocating byte-identical campaign trees does not alter serialized evidence."""
    roots, receipt = _write_triplet(tmp_path / "source")
    relocated_parent = tmp_path / "relocated"
    relocated_roots = []
    for root in roots:
        destination = relocated_parent / root.name
        shutil.copytree(root, destination)
        relocated_roots.append(destination)
    first = compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)
    second = compose_radius_sweep_summary(relocated_roots, gate1_canary_receipt=receipt)
    assert first == second
    assert all("campaign_root" not in arm for arm in first["campaign_provenance"].values())


def test_composer_rejects_out_of_scope_episode_files(tmp_path: Path, compact_scope: None) -> None:
    """An extra kinematics run cannot be silently ignored by the frozen-scope composer."""
    roots, receipt = _write_triplet(tmp_path)
    extra = roots[0] / "runs/goal__holonomic/episodes.jsonl"
    extra.parent.mkdir()
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RadiusSweepSummaryError, match="run scope mismatch"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)
