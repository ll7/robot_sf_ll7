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
    for arm_key, relative in _FIXTURE_CONFIG_PATHS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture_arm: {arm_key}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "init", "--quiet"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "add", "--", *_FIXTURE_CONFIG_PATHS.values()], check=True
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
                    "algo": "goal",
                    "algorithm_metadata": {
                        "algorithm": "goal",
                        "canonical_algorithm": "goal",
                    },
                    "result_provenance": {"planner_key": "goal"},
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


@pytest.mark.parametrize(
    "carrier",
    [
        "algo",
        "algorithm_metadata.algorithm",
        "algorithm_metadata.canonical_algorithm",
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
    episode.pop("algo")
    episode.pop("algorithm_metadata")
    episode.pop("result_provenance")
    if carrier == "algo":
        episode["algo"] = "orca"
    elif carrier.startswith("algorithm_metadata."):
        field = carrier.rsplit(".", maxsplit=1)[-1]
        episode["algorithm_metadata"] = {field: "orca"}
    else:
        episode["result_provenance"] = {"planner_key": "orca"}
    lines[0] = json.dumps(episode)
    episode_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    lines[0] = json.dumps(episode)
    episode_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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

    compose_radius_sweep_summary(roots, gate1_canary_receipt=receipt)


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
