"""Tests for the fail-closed issue #6642 to #6643 summary composer."""

from __future__ import annotations

import json
import shutil
from hashlib import sha256
from typing import TYPE_CHECKING

import pytest

import robot_sf.benchmark.radius_sweep_summary as composer
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


@pytest.fixture
def compact_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the frozen matrix while preserving all structural checks."""
    monkeypatch.setattr(composer, "RELEASE_PLANNER_KEYS", ("goal",))
    monkeypatch.setattr(composer, "EXPECTED_SCENARIO_NAMES", ("case_a", "case_b"))
    monkeypatch.setattr(composer, "EXPECTED_SEEDS", (111, 112))
    monkeypatch.setattr(composer, "EXPECTED_ROWS_PER_ARM", 4)
    monkeypatch.setattr(composer, "EXPECTED_GATE1_RECEIPT_SHA256", _GATE1_RECEIPT_SHA256)
    monkeypatch.setattr(
        composer,
        "EXPECTED_ARM_CAMPAIGN_CONFIGS",
        {
            "r0p5": "configs/arm_0p5.yaml",
            "r0p8": "configs/arm_0p8.yaml",
            "r1p0": "configs/arm_1p0.yaml",
        },
    )
    monkeypatch.setattr(
        composer,
        "EXPECTED_ARM_CAMPAIGN_CONFIG_SHA256",
        {"r0p5": "1" * 64, "r0p8": "2" * 64, "r1p0": "3" * 64},
    )


def _write_arm(
    parent: Path,
    radius: float,
    *,
    commit: str = "a" * 40,
    gate1_receipt_sha256: str,
    include_family: bool = True,
    degraded: bool = False,
) -> Path:
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
        "git": {"commit": commit},
        "radius_binding": binding,
        "seed_policy": {"resolved_seeds": list(composer.EXPECTED_SEEDS)},
    }
    preflight = {
        "config_path": composer.EXPECTED_ARM_CAMPAIGN_CONFIGS[arm_key],
        "config_sha256": ({"r0p5": "1", "r0p8": "2", "r1p0": "3"}[arm_key] * 64),
        "radius_binding": binding,
    }
    campaign_summary = {
        "campaign": {
            "campaign_id": manifest["campaign_id"],
            "git_hash": commit,
            "scenario_matrix": composer.EXPECTED_SCENARIO_MATRIX,
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
                    "scenario_id": scenario,
                    "seed": seed,
                    "git_hash": commit,
                    "integrity": {
                        "contradictions": [],
                        "effective_view": {"degraded": False},
                    },
                    "scenario_params": {"robot_config": {"radius": radius}},
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
    return root


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

    first_path = write_radius_sweep_summary(first, tmp_path / "first.json")
    second_path = write_radius_sweep_summary(second, tmp_path / "second.json")
    assert first_path.read_bytes() == second_path.read_bytes()


def test_composer_rejects_missing_authoritative_family_semantics(
    tmp_path: Path, compact_scope: None
) -> None:
    """Campaign outcomes are not silently reinterpreted as family feasibility."""
    roots, receipt = _write_triplet(tmp_path)
    (roots[-1] / "reports/radius_family_feasibility.json").unlink()
    with pytest.raises(RadiusSweepSummaryError, match="lacks authoritative family_feasibility"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


def test_composer_rejects_mixed_commits(tmp_path: Path, compact_scope: None) -> None:
    """Recovered arms remain admissible only at the same immutable source commit."""
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
    with pytest.raises(RadiusSweepSummaryError, match="mixed commits"):
        compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


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
    with pytest.raises(RadiusSweepSummaryError, match="frozen campaign config bytes"):
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
