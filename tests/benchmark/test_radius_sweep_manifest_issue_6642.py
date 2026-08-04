"""Tests for the issue #6642 collision-envelope radius-sweep preparation manifest.

These tests cover the dry-run preparation manifest builder and checker only. They
do not run any benchmark episodes or submit SLURM compute, and they assert the
fail-closed contract: degraded/fallback/failed/missing rows are never evidence,
production submission stays blocked while the Gate 1 binding canary (#6641) is
pending, and all radius arms stay pinned to one immutable campaign commit.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark.radius_sweep_manifest import (
    BASELINE_RADIUS,
    CLAIM_BOUNDARY,
    EVIDENCE_STATUS,
    EXPECTED_ARM_CAMPAIGN_CONFIGS,
    EXPECTED_ARM_RELEASE_TAGS,
    EXPECTED_DT,
    EXPECTED_HORIZON,
    EXPECTED_KINEMATICS,
    EXPECTED_MANIFEST_CONFIG,
    EXPECTED_SCENARIO_COUNT,
    EXPECTED_SCENARIO_NAMES,
    EXPECTED_SEED_RANGE,
    EXPECTED_SEED_SET,
    GATE1_CANARY_ISSUE,
    GATE1_STATUS_NOT_YET_PASSED,
    MANIFEST_STATUS,
    PRODUCTION_RADII,
    PRODUCTION_RADIUS_KEYS,
    RADIUS_SWEEP_MANIFEST_CHECK_SCHEMA,
    RADIUS_SWEEP_MANIFEST_SCHEMA,
    RELEASE_PLANNER_KEYS,
    RUNTIME_BINDING_PENDING_GATE1,
    ArmCampaignIdentity,
    FixedFactors,
    RadiusSweepManifestError,
    build_radius_sweep_manifest,
    check_radius_sweep_manifest,
    validate_arm_campaign_payload,
    validate_arm_fixed_factors,
    write_radius_sweep_manifest,
    write_radius_sweep_manifest_check,
)
from scripts.benchmark.build_radius_sweep_manifest_issue_6642 import (
    _resolve_all_arms,
    _resolve_arm,
    _resolve_arm_fixed_factors,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CONFIG_PATH = REPO_ROOT / "configs/benchmarks/issue_6642_radius_sweep_manifest_v1.yaml"
ARM_CONFIG_PATH = REPO_ROOT / "configs/benchmarks/issue_6642_radius_sweep_arm_1p0m.yaml"
_TEST_GIT_HEAD = "a" * 40

_SEEDS = tuple(range(EXPECTED_SEED_RANGE[0], EXPECTED_SEED_RANGE[1] + 1))
_SCENARIO_NAMES = EXPECTED_SCENARIO_NAMES


def _fixed_factors(
    *,
    planner_keys: tuple[str, ...] = RELEASE_PLANNER_KEYS,
    scenario_names: tuple[str, ...] = _SCENARIO_NAMES,
    seeds: tuple[int, ...] = _SEEDS,
    horizon: int = EXPECTED_HORIZON,
    dt: float = EXPECTED_DT,
    kinematics: str = EXPECTED_KINEMATICS,
    seed_set: str = EXPECTED_SEED_SET,
    release_tag: str = "issue-6642-radius-sweep-1p0m",
) -> FixedFactors:
    """Return a contract-faithful FixedFactors instance for unit tests."""
    return FixedFactors(
        scenario_matrix="configs/scenarios/classic_interactions_francis2023.yaml",
        scenario_count=len(scenario_names),
        scenario_names=scenario_names,
        planner_keys=planner_keys,
        seed_set=seed_set,
        seeds=seeds,
        horizon=horizon,
        dt=dt,
        kinematics=kinematics,
        release_tag=release_tag,
    )


def _manifest_config() -> dict[str, Any]:
    """Load the tracked manifest config, asserting its schema version."""
    payload = yaml.safe_load(MANIFEST_CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == RADIUS_SWEEP_MANIFEST_SCHEMA
    return payload


def _arm_identities() -> list[ArmCampaignIdentity]:
    """Return contract-faithful arm campaign identities in radius order."""
    return [
        ArmCampaignIdentity(
            arm_key=arm_key,
            campaign_config=EXPECTED_ARM_CAMPAIGN_CONFIGS[arm_key],
            release_tag=EXPECTED_ARM_RELEASE_TAGS[arm_key],
        )
        for arm_key in PRODUCTION_RADIUS_KEYS
    ]


def _build(
    manifest_config: Mapping[str, Any] | None = None,
    fixed_factors: FixedFactors | None = None,
    arm_identities: Sequence[ArmCampaignIdentity] | None = None,
    git_head: str = _TEST_GIT_HEAD,
) -> dict[str, Any]:
    """Build a manifest from the tracked config and contract-faithful factors."""
    return build_radius_sweep_manifest(
        manifest_config if manifest_config is not None else _manifest_config(),
        fixed_factors=fixed_factors or _fixed_factors(),
        arm_identities=arm_identities if arm_identities is not None else _arm_identities(),
        options=_options(git_head=git_head),
    )


def _options(git_head: str = _TEST_GIT_HEAD):
    from robot_sf.benchmark.radius_sweep_manifest import ManifestOptions

    return ManifestOptions(
        config_path="configs/benchmarks/issue_6642_radius_sweep_manifest_v1.yaml",
        git_head=git_head,
    )


# ---------------------------------------------------------------------------
# Build contract
# ---------------------------------------------------------------------------


def test_build_enumerates_three_radius_arms_with_release_baseline() -> None:
    """The manifest enumerates exactly 0.5/0.8/1.0 m with 1.0 m as the sole baseline."""
    manifest = _build()

    assert manifest["schema_version"] == RADIUS_SWEEP_MANIFEST_SCHEMA
    assert manifest["status"] == MANIFEST_STATUS
    assert manifest["evidence_status"] == EVIDENCE_STATUS
    assert manifest["dry_run"] is True
    assert manifest["issue"] == 6642
    assert manifest["parent_issue"] == 6600
    assert [arm["radius_m"] for arm in manifest["arms"]] == list(PRODUCTION_RADII)
    assert [arm["baseline"] for arm in manifest["arms"]] == [False, False, True]
    baseline_arms = [arm for arm in manifest["arms"] if arm["baseline"]]
    assert len(baseline_arms) == 1
    assert baseline_arms[0]["radius_m"] == BASELINE_RADIUS


def test_build_stamps_pending_gate1_binding_on_every_arm() -> None:
    """While Gate 1 is pending, no arm may claim a bound radius."""
    manifest = _build()

    for arm in manifest["arms"]:
        assert arm["runtime_binding_status"] == RUNTIME_BINDING_PENDING_GATE1
        assert arm["planner_keys"] == list(RELEASE_PLANNER_KEYS)
        assert arm["planner_count"] == len(RELEASE_PLANNER_KEYS)
        assert arm["scenario_count"] == EXPECTED_SCENARIO_COUNT
        # 14 planners x 48 cells x 30 seeds = 20160 expected episodes per arm.
        assert arm["expected_episode_count"] == 14 * 48 * 30


def test_build_records_gate1_block_and_one_immutable_commit() -> None:
    """The manifest pins one commit across arms and keeps production blocked."""
    manifest = _build()

    gate = manifest["gate_preconditions"]
    assert gate["gate1_canary_issue"] == GATE1_CANARY_ISSUE
    assert gate["gate1_canary_status"] == GATE1_STATUS_NOT_YET_PASSED
    assert gate["production_submission_authorized"] is False
    commit = manifest["immutable_campaign_commit"]
    assert commit["one_commit_across_all_arms"] is True
    assert commit["git_head"] == _TEST_GIT_HEAD


def test_build_row_identity_ledger_template_counts_total_rows() -> None:
    """The row-identity ledger carries the four dimensions and 60480 total rows."""
    manifest = _build()

    ledger = manifest["row_identity_ledger_template"]
    assert ledger["dimensions"] == ("radius_arm", "planner_key", "scenario_name", "seed")
    assert ledger["completeness"] == "template_only_no_episodes_run"
    # 3 radii x 14 planners x 48 cells x 30 seeds = 60480 expected total rows.
    assert ledger["expected_total_rows"] == 3 * 14 * 48 * 30
    assert ledger["expected_rows_per_arm"] == 14 * 48 * 30


def test_build_rejects_non_dry_run_options() -> None:
    """The builder only supports dry-run preparation manifests."""
    from robot_sf.benchmark.radius_sweep_manifest import ManifestOptions

    with pytest.raises(RadiusSweepManifestError, match="dry-run"):
        build_radius_sweep_manifest(
            _manifest_config(),
            fixed_factors=_fixed_factors(),
            arm_identities=_arm_identities(),
            options=ManifestOptions(config_path="x", git_head=_TEST_GIT_HEAD, dry_run=False),
        )


def test_build_rejects_invalid_fixed_factor_input() -> None:
    """The builder rejects an unresolved or wrong-typed fixed-factor object."""
    with pytest.raises(RadiusSweepManifestError, match="FixedFactors"):
        build_radius_sweep_manifest(
            _manifest_config(),
            fixed_factors=None,
            arm_identities=_arm_identities(),
            options=_options(),  # type: ignore[arg-type]
        )


def test_build_rejects_resolved_fixed_factor_drift() -> None:
    """Resolved arm factors cannot bypass the frozen release contract."""
    drifted = replace(
        _fixed_factors(),
        scenario_matrix="configs/other.yaml",
        scenario_count=1,
        scenario_names=("other",),
        planner_keys=("other",),
        seed_set="other",
        seeds=(1,),
        horizon=1,
        dt=0.2,
        kinematics="holonomic",
        release_tag="",
    )

    with pytest.raises(RadiusSweepManifestError, match="resolved fixed factors"):
        build_radius_sweep_manifest(
            _manifest_config(),
            fixed_factors=drifted,
            arm_identities=_arm_identities(),
            options=_options(),
        )


@pytest.mark.parametrize(
    ("config_mutation", "match"),
    [
        (lambda config: config.update({"schema_version": "wrong"}), "schema_version"),
        (lambda config: config.update({"radii": []}), "radii"),
        (
            lambda config: config.update({"release_baseline_config": ""}),
            "release_baseline_config",
        ),
        (
            lambda config: config.update({"arm_campaign_config_0p5m": "configs/other-arm.yaml"}),
            "arm_campaign_config_0p5m",
        ),
        (
            lambda config: config.pop("arm_campaign_config_0p8m"),
            "arm_campaign_config_0p8m",
        ),
        (lambda config: config.update({"fixed_factors": None}), "fixed_factors mapping"),
        (
            lambda config: config["fixed_factors"].pop("horizon"),
            "missing required keys",
        ),
    ],
)
def test_build_rejects_manifest_config_shape_drift(config_mutation, match: str) -> None:
    """Malformed preparation configs fail before any arm is enumerated."""
    config = _manifest_config()
    config_mutation(config)

    with pytest.raises(RadiusSweepManifestError, match=match):
        build_radius_sweep_manifest(
            config,
            fixed_factors=_fixed_factors(),
            arm_identities=_arm_identities(),
            options=_options(),
        )


@pytest.mark.parametrize(
    "bad_radii,match",
    [
        # Missing the 1.0 m arm.
        (
            [
                {"key": "r0p5", "radius_m": 0.5, "baseline": False},
                {"key": "r0p8", "radius_m": 0.8, "baseline": False},
            ],
            "must declare exactly",
        ),
        # Wrong radius value.
        (
            [
                {"key": "r0p5", "radius_m": 0.5, "baseline": False},
                {"key": "r0p8", "radius_m": 0.9, "baseline": False},
                {"key": "r1p0", "radius_m": 1.0, "baseline": True},
            ],
            "must be",
        ),
        # Two baselines.
        (
            [
                {"key": "r0p5", "radius_m": 0.5, "baseline": True},
                {"key": "r0p8", "radius_m": 0.8, "baseline": False},
                {"key": "r1p0", "radius_m": 1.0, "baseline": True},
            ],
            "baseline",
        ),
        # Baseline not on the 1.0 m arm.
        (
            [
                {"key": "r0p5", "radius_m": 0.5, "baseline": True},
                {"key": "r0p8", "radius_m": 0.8, "baseline": False},
                {"key": "r1p0", "radius_m": 1.0, "baseline": False},
            ],
            "baseline radius must be",
        ),
    ],
)
def test_build_rejects_malformed_radius_treatment(bad_radii: list[dict], match: str) -> None:
    """Malformed radius treatments fail closed at build time."""
    config = _manifest_config()
    config = copy.deepcopy(config)
    config["radii"] = bad_radii
    with pytest.raises(RadiusSweepManifestError, match=match):
        build_radius_sweep_manifest(
            config,
            fixed_factors=_fixed_factors(),
            arm_identities=_arm_identities(),
            options=_options(),
        )


# ---------------------------------------------------------------------------
# Arm campaign identities: one tracked config and release tag per radius arm
# ---------------------------------------------------------------------------


def test_build_stamps_per_arm_campaign_config_and_release_tag() -> None:
    """Each arm carries its own tracked campaign config and issue-scoped release tag."""
    manifest = _build()

    assert [arm["key"] for arm in manifest["arms"]] == list(PRODUCTION_RADIUS_KEYS)
    for arm in manifest["arms"]:
        assert arm["arm_campaign_config"] == EXPECTED_ARM_CAMPAIGN_CONFIGS[arm["key"]]
        assert arm["release_tag"] == EXPECTED_ARM_RELEASE_TAGS[arm["key"]]


def test_build_rejects_missing_arm_identity() -> None:
    """The builder requires a resolved campaign identity for every radius arm."""
    with pytest.raises(RadiusSweepManifestError, match="ArmCampaignIdentity entries"):
        _build(arm_identities=_arm_identities()[:2])


def test_build_rejects_non_identity_entries() -> None:
    """Non-ArmCampaignIdentity entries fail closed instead of being trusted."""
    identities = _arm_identities()
    identities[1] = "not-an-identity"  # type: ignore[list-item]
    with pytest.raises(RadiusSweepManifestError, match="must be an ArmCampaignIdentity"):
        _build(arm_identities=identities)


def test_build_rejects_out_of_order_arm_identities() -> None:
    """Arm identities must follow radius order (0.5/0.8/1.0 m)."""
    identities = _arm_identities()
    identities[0], identities[1] = identities[1], identities[0]
    with pytest.raises(RadiusSweepManifestError, match="arm_key must be"):
        _build(arm_identities=identities)


@pytest.mark.parametrize("arm_index", [0, 1, 2])
def test_build_rejects_arm_identity_campaign_config_drift(arm_index: int) -> None:
    """An arm identity cannot point at another campaign config."""
    identities = _arm_identities()
    identities[arm_index] = replace(identities[arm_index], campaign_config="configs/other.yaml")
    with pytest.raises(RadiusSweepManifestError, match="campaign_config must be"):
        _build(arm_identities=identities)


@pytest.mark.parametrize("arm_index", [0, 1, 2])
def test_build_rejects_arm_identity_release_tag_drift(arm_index: int) -> None:
    """An arm identity cannot carry a foreign release tag."""
    identities = _arm_identities()
    identities[arm_index] = replace(identities[arm_index], release_tag="unrelated-release")
    with pytest.raises(RadiusSweepManifestError, match="release_tag must be"):
        _build(arm_identities=identities)


# ---------------------------------------------------------------------------
# Check contract: fail-closed boundary violations
# ---------------------------------------------------------------------------


def test_check_passes_on_valid_manifest() -> None:
    """A freshly built contract-faithful manifest passes the checker."""
    check = check_radius_sweep_manifest(_build())
    assert check["schema_version"] == RADIUS_SWEEP_MANIFEST_CHECK_SCHEMA
    assert check["passes"] is True
    assert check["violations"] == []
    assert check["arm_count"] == 3
    assert check["expected_total_rows"] == 3 * 14 * 48 * 30


def test_check_rejects_non_mapping_payload() -> None:
    """The checker rejects a non-serialized-manifest payload explicitly."""
    with pytest.raises(RadiusSweepManifestError, match="must be a mapping"):
        check_radius_sweep_manifest(None)  # type: ignore[arg-type]


def test_check_fails_when_production_submission_authorized_while_gate1_pending() -> None:
    """Authorizing production compute before Gate 1 passes violates the hard precondition."""
    manifest = _build()
    manifest["gate_preconditions"]["production_submission_authorized"] = True
    check = check_radius_sweep_manifest(manifest)
    assert check["passes"] is False
    assert any("production_submission_authorized" in v for v in check["violations"])


def test_check_fails_when_gate1_status_is_unrecognized() -> None:
    """An unknown Gate 1 status cannot bypass the preparation-only block."""
    manifest = _build()
    manifest["gate_preconditions"]["gate1_canary_status"] = "unknown"
    manifest["gate_preconditions"]["production_submission_authorized"] = True

    check = check_radius_sweep_manifest(manifest)

    assert check["passes"] is False
    assert any("gate1_canary_status" in v for v in check["violations"])
    assert any("production_submission_authorized" in v for v in check["violations"])


def test_check_fails_when_arm_claims_bound_radius() -> None:
    """An arm may not claim a bound radius while Gate 1 is pending."""
    manifest = _build()
    manifest["arms"][0]["runtime_binding_status"] = "bound"
    check = check_radius_sweep_manifest(manifest)
    assert check["passes"] is False
    assert any("runtime_binding_status" in v for v in check["violations"])


def test_check_fails_when_arm_campaign_config_drifts() -> None:
    """A serialized arm cannot point at another campaign config."""
    manifest = _build()
    manifest["arms"][0]["arm_campaign_config"] = "configs/other-arm.yaml"

    check = check_radius_sweep_manifest(manifest)

    assert check["passes"] is False
    assert any("arm_campaign_config" in v for v in check["violations"])


def test_check_fails_when_arm_release_tag_drifts() -> None:
    """A serialized arm cannot carry a foreign release tag."""
    manifest = _build()
    manifest["arms"][1]["release_tag"] = "unrelated-release"

    check = check_radius_sweep_manifest(manifest)

    assert check["passes"] is False
    assert any("release_tag" in v for v in check["violations"])


def test_check_fails_when_fixed_factors_drift_from_release() -> None:
    """Fixed factors must match the release baseline except the radius treatment."""
    manifest = _build()
    manifest["fixed_factors"]["horizon"] = 100
    check = check_radius_sweep_manifest(manifest)
    assert check["passes"] is False
    assert any("horizon" in v for v in check["violations"])


def test_check_fails_when_seed_roster_changes_without_range_change() -> None:
    """A duplicate or gap inside the declared seed range must fail closed."""
    manifest = _build()
    manifest["fixed_factors"]["seeds"][5] = manifest["fixed_factors"]["seeds"][4]

    check = check_radius_sweep_manifest(manifest)

    assert check["passes"] is False
    assert any("fixed_factors.seeds" in v for v in check["violations"])


def test_check_fails_when_scenario_roster_changes_without_count_change() -> None:
    """Replacing one scenario while retaining 48 entries must fail closed."""
    manifest = _build()
    manifest["fixed_factors"]["scenario_names"][0] = "unregistered_scenario"

    check = check_radius_sweep_manifest(manifest)

    assert check["passes"] is False
    assert any("fixed_factors.scenario_names" in v for v in check["violations"])


def test_check_fails_when_planner_roster_changes() -> None:
    """The 14-key release roster must be preserved in order on every arm."""
    manifest = _build()
    manifest["arms"][0]["planner_keys"] = list(RELEASE_PLANNER_KEYS[1:])
    manifest["arms"][0]["planner_count"] = len(RELEASE_PLANNER_KEYS) - 1
    check = check_radius_sweep_manifest(manifest)
    assert check["passes"] is False
    assert any("planner_keys" in v or "planner_count" in v for v in check["violations"])


def test_check_fails_when_commit_not_pinned_once() -> None:
    """All arms must share one immutable campaign commit."""
    manifest = _build()
    manifest["immutable_campaign_commit"]["one_commit_across_all_arms"] = False
    check = check_radius_sweep_manifest(manifest)
    assert check["passes"] is False
    assert any("one_commit_across_all_arms" in v for v in check["violations"])


@pytest.mark.parametrize("git_head", ["pending_launch", "abc1234", "g" * 40])
def test_check_fails_when_commit_is_not_a_full_git_sha(git_head: str) -> None:
    """A preparation manifest must carry an actual immutable commit identity."""
    manifest = _build()
    manifest["immutable_campaign_commit"]["git_head"] = git_head

    check = check_radius_sweep_manifest(manifest)

    assert check["passes"] is False
    assert any("40-character lowercase git SHA" in v for v in check["violations"])


def test_check_fails_when_top_level_git_head_does_not_match_immutable_commit() -> None:
    """Duplicated git identity fields must refer to one immutable campaign commit."""
    manifest = _build()
    manifest["git_head"] = "b" * 40

    check = check_radius_sweep_manifest(manifest)

    assert check["passes"] is False
    assert any("git_head" in v for v in check["violations"])


def test_check_fails_when_top_level_config_path_drifts() -> None:
    """The serialized provenance path must remain the canonical manifest config."""
    manifest = _build()
    manifest["config_path"] = "configs/other.yaml"

    check = check_radius_sweep_manifest(manifest)

    assert check["passes"] is False
    assert any("config_path" in v for v in check["violations"])


def test_check_fails_when_missingness_exclusions_weakened() -> None:
    """Dropping any evidence-exclusion class weakens the fail-closed contract."""
    manifest = _build()
    manifest["missingness_policy"]["evidence_exclusions"] = ["unavailable", "failed"]
    check = check_radius_sweep_manifest(manifest)
    assert check["passes"] is False
    assert any("evidence_exclusions" in v for v in check["violations"])


def test_check_fails_when_claim_boundary_weakened() -> None:
    """The no-evidence claim boundary phrases must all remain present."""
    manifest = _build()
    manifest["claim_boundary"] = "this is benchmark evidence"
    check = check_radius_sweep_manifest(manifest)
    assert check["passes"] is False
    assert any("claim_boundary" in v for v in check["violations"])


def test_check_fails_when_claim_boundary_contains_contradictory_text() -> None:
    """Contradictory evidence language cannot be appended to the canonical boundary."""
    manifest = _build()
    manifest["claim_boundary"] = f"This is benchmark evidence; {CLAIM_BOUNDARY}"

    check = check_radius_sweep_manifest(manifest)

    assert check["passes"] is False
    assert any("claim_boundary" in v for v in check["violations"])


def test_check_fails_when_row_identity_contract_is_negated() -> None:
    """A serialized manifest cannot negate its complete-row fail-closed contract."""
    manifest = _build()
    manifest["missingness_policy"]["row_identity_contract"] = (
        "not complete_row_identities and not missingness_ledger"
    )

    check = check_radius_sweep_manifest(manifest)

    assert check["passes"] is False
    assert any("row_identity_contract" in v for v in check["violations"])


@pytest.mark.parametrize("release_tag", ["", "unrelated-release"])
def test_check_fails_when_release_tag_is_not_issue_scoped(release_tag: str) -> None:
    """The serialized fixed factors must retain the issue-scoped arm release tag."""
    manifest = _build()
    manifest["fixed_factors"]["release_tag"] = release_tag

    check = check_radius_sweep_manifest(manifest)

    assert check["passes"] is False
    assert any("release_tag" in v for v in check["violations"])


def test_check_fails_when_radii_drift() -> None:
    """The radii treatment must remain exactly 0.5/0.8/1.0 m."""
    manifest = _build()
    manifest["radii"][1]["radius_m"] = 0.9
    check = check_radius_sweep_manifest(manifest)
    assert check["passes"] is False
    assert any("radii must be exactly" in v for v in check["violations"])


def test_check_fails_when_row_identity_ledger_drops_dimension() -> None:
    """The row-identity ledger must carry all four identity dimensions."""
    manifest = _build()
    manifest["row_identity_ledger_template"]["dimensions"] = ("planner_key", "seed")
    check = check_radius_sweep_manifest(manifest)
    assert check["passes"] is False
    assert any("dimensions" in v for v in check["violations"])


@pytest.mark.parametrize(
    "path",
    [
        ("row_identity_ledger_template", "dimensions"),
        ("fixed_factors", "planner_keys"),
        ("fixed_factors", "seeds"),
    ],
)
def test_check_reports_malformed_sequences_without_raising(path: tuple[str, str]) -> None:
    """Malformed serialized collections fail closed as violations, not checker crashes."""
    manifest = _build()
    manifest[path[0]][path[1]] = None

    check = check_radius_sweep_manifest(manifest)

    assert check["passes"] is False
    assert check["violations"]


def test_check_fails_when_row_identity_totals_drift() -> None:
    """The checker must enforce both the per-arm and full-grid row totals."""
    manifest = _build()
    manifest["arms"][0]["expected_episode_count"] -= 1
    manifest["row_identity_ledger_template"]["expected_total_rows"] -= 1
    manifest["row_identity_ledger_template"]["expected_rows_per_arm"] -= 1

    check = check_radius_sweep_manifest(manifest)

    assert check["passes"] is False
    assert any("expected_episode_count" in v for v in check["violations"])
    assert any("expected_total_rows" in v for v in check["violations"])
    assert any("expected_rows_per_arm" in v for v in check["violations"])


# ---------------------------------------------------------------------------
# Integration: the tracked configs resolve to the release contract
# ---------------------------------------------------------------------------


def test_tracked_arm_config_resolves_release_fixed_factors() -> None:
    """The 1.0 m arm campaign config resolves the exact release fixed factors."""
    manifest_config = _manifest_config()
    fixed = _resolve_arm_fixed_factors(manifest_config, REPO_ROOT)

    assert fixed.planner_keys == RELEASE_PLANNER_KEYS
    assert fixed.scenario_count == EXPECTED_SCENARIO_COUNT
    assert fixed.seed_set == EXPECTED_SEED_SET
    assert (min(fixed.seeds), max(fixed.seeds)) == EXPECTED_SEED_RANGE
    assert len(fixed.seeds) == 30
    assert fixed.horizon == EXPECTED_HORIZON
    assert fixed.dt == EXPECTED_DT
    assert fixed.kinematics == EXPECTED_KINEMATICS
    # The 1.0 m arm is pinned to its own issue-scoped release tag, distinct from
    # the frozen 0.0.3.post1 release so all arms share one campaign commit.
    assert fixed.release_tag == "issue-6642-radius-sweep-1p0m"


@pytest.mark.parametrize(
    ("arm_key", "radius_m", "baseline", "expected_release_tag"),
    [
        ("r0p5", 0.5, False, "issue-6642-radius-sweep-0p5m"),
        ("r0p8", 0.8, False, "issue-6642-radius-sweep-0p8m"),
        ("r1p0", 1.0, True, "issue-6642-radius-sweep-1p0m"),
    ],
)
def test_tracked_arm_configs_resolve_contract_factors(
    arm_key: str, radius_m: float, baseline: bool, expected_release_tag: str
) -> None:
    """Every tracked arm config resolves the frozen fixed factors and its own identity."""
    manifest_config = _manifest_config()

    factors, identity = _resolve_arm(
        manifest_config, REPO_ROOT, arm_key=arm_key, radius_m=radius_m, baseline=baseline
    )

    assert identity.arm_key == arm_key
    assert identity.campaign_config == EXPECTED_ARM_CAMPAIGN_CONFIGS[arm_key]
    assert identity.release_tag == expected_release_tag
    assert factors.planner_keys == RELEASE_PLANNER_KEYS
    assert factors.scenario_count == EXPECTED_SCENARIO_COUNT
    assert factors.scenario_names == EXPECTED_SCENARIO_NAMES
    assert factors.seed_set == EXPECTED_SEED_SET
    assert (min(factors.seeds), max(factors.seeds)) == EXPECTED_SEED_RANGE
    assert len(factors.seeds) == 30
    assert factors.horizon == EXPECTED_HORIZON
    assert factors.dt == EXPECTED_DT
    assert factors.kinematics == EXPECTED_KINEMATICS
    assert factors.release_tag == expected_release_tag


def test_resolve_all_arms_keeps_non_radius_factors_fixed() -> None:
    """All three arm configs resolve identical non-radius factors in radius order."""
    manifest_config = _manifest_config()

    baseline_factors, identities = _resolve_all_arms(manifest_config, REPO_ROOT)

    assert [identity.arm_key for identity in identities] == list(PRODUCTION_RADIUS_KEYS)
    assert baseline_factors.release_tag == "issue-6642-radius-sweep-1p0m"
    assert "francis2023_narrow_doorway" in baseline_factors.scenario_names


def test_tracked_manifest_config_builds_and_checks_clean() -> None:
    """Building from the tracked manifest config against the real arm configs passes."""
    manifest_config = _manifest_config()
    fixed, identities = _resolve_all_arms(manifest_config, REPO_ROOT)
    manifest = build_radius_sweep_manifest(
        manifest_config,
        fixed_factors=fixed,
        arm_identities=identities,
        options=_options(git_head="b" * 40),
    )
    check = check_radius_sweep_manifest(manifest)
    assert check["passes"] is True, check["violations"]
    assert manifest["claim_boundary"] == CLAIM_BOUNDARY
    assert manifest["config_path"] == EXPECTED_MANIFEST_CONFIG
    assert manifest["immutable_campaign_commit"]["git_head"] == "b" * 40
    # The 48-cell narrow-doorway family cell is present in the resolved roster.
    assert "francis2023_narrow_doorway" in fixed.scenario_names
    # Every arm entry carries its own tracked campaign config and release tag.
    for arm in manifest["arms"]:
        assert arm["arm_campaign_config"] == EXPECTED_ARM_CAMPAIGN_CONFIGS[arm["key"]]
        assert arm["release_tag"] == EXPECTED_ARM_RELEASE_TAGS[arm["key"]]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("issue", 9999),
        ("parent_issue", 1),
        ("release_baseline_config", "configs/other.yaml"),
        ("arm_campaign_config_0p5m", "configs/other-arm.yaml"),
        ("arm_campaign_config_0p8m", "configs/other-arm.yaml"),
        ("arm_campaign_config_1p0m", "configs/other-arm.yaml"),
    ],
)
def test_build_rejects_manifest_source_identity_drift(field: str, value: object) -> None:
    """The preparation config cannot silently retarget another campaign surface."""
    manifest_config = _manifest_config()
    manifest_config[field] = value

    with pytest.raises(RadiusSweepManifestError, match=field):
        build_radius_sweep_manifest(
            manifest_config,
            fixed_factors=_fixed_factors(),
            arm_identities=_arm_identities(),
            options=_options(),
        )


def test_build_rejects_declared_fixed_factor_drift() -> None:
    """Manifest-declared expectations must agree with the resolved arm config."""
    manifest_config = _manifest_config()
    manifest_config["fixed_factors"]["horizon"] = 100

    with pytest.raises(RadiusSweepManifestError, match="fixed_factors"):
        build_radius_sweep_manifest(
            manifest_config,
            fixed_factors=_fixed_factors(),
            arm_identities=_arm_identities(),
            options=_options(),
        )


def test_write_manifest_and_check_artifacts(tmp_path: Path) -> None:
    """The normal writer path emits both deterministic JSON artifacts."""
    manifest = _build()
    check = check_radius_sweep_manifest(manifest)

    manifest_path = write_radius_sweep_manifest(manifest, tmp_path)
    check_path = write_radius_sweep_manifest_check(check, tmp_path)

    assert manifest_path.name == "radius_sweep_manifest.json"
    assert check_path.name == "radius_sweep_manifest_check.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["dry_run"] is True
    assert json.loads(check_path.read_text(encoding="utf-8"))["passes"] is True


def test_cli_check_only_prints_summary_without_writing_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--check-only prints the checker summary without writing manifest artifacts."""
    from scripts.benchmark.build_radius_sweep_manifest_issue_6642 import main

    out_dir = tmp_path / "manifest"
    exit_code = main(["--check-only", "--out", str(out_dir)])

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["passes"] is True
    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# Arm campaign payload and factor validation (radius_sweep metadata contract)
# ---------------------------------------------------------------------------

_ARM_PAYLOAD_CASES = (
    ("r0p5", "configs/benchmarks/issue_6642_radius_sweep_arm_0p5m.yaml", 0.5, False),
    ("r0p8", "configs/benchmarks/issue_6642_radius_sweep_arm_0p8m.yaml", 0.8, False),
    ("r1p0", "configs/benchmarks/issue_6642_radius_sweep_arm_1p0m.yaml", 1.0, True),
)


@pytest.mark.parametrize(("arm_key", "config_relpath", "radius_m", "baseline"), _ARM_PAYLOAD_CASES)
def test_tracked_arm_payloads_declare_their_exact_treatment(
    arm_key: str, config_relpath: str, radius_m: float, baseline: bool
) -> None:
    """Every tracked arm config declares its own radius treatment, still pending Gate 1."""
    payload = yaml.safe_load((REPO_ROOT / config_relpath).read_text(encoding="utf-8"))

    validate_arm_campaign_payload(payload, arm_key=arm_key, radius_m=radius_m, baseline=baseline)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload["radius_sweep"].update({"arm_key": "r0p8"}), "arm_key"),
        (lambda payload: payload["radius_sweep"].update({"radius_m": 0.9}), "radius_m"),
        (lambda payload: payload["radius_sweep"].update({"baseline_arm": True}), "baseline_arm"),
        (
            lambda payload: payload["radius_sweep"].update({"runtime_binding_status": "bound"}),
            "runtime_binding_status",
        ),
        (lambda payload: payload["radius_sweep"].update({"issue": 6600}), "issue must be"),
        (
            lambda payload: payload["radius_sweep"].update({"parent_issue": 6642}),
            "parent_issue must be",
        ),
        (lambda payload: payload.pop("radius_sweep"), "radius_sweep metadata mapping"),
    ],
)
def test_arm_payload_validation_fails_closed_on_metadata_drift(mutation, match: str) -> None:
    """A silently divergent radius treatment declaration must fail closed."""
    payload = yaml.safe_load(
        (REPO_ROOT / "configs/benchmarks/issue_6642_radius_sweep_arm_0p5m.yaml").read_text(
            encoding="utf-8"
        )
    )
    mutation(payload)

    with pytest.raises(RadiusSweepManifestError, match=match):
        validate_arm_campaign_payload(payload, arm_key="r0p5", radius_m=0.5, baseline=False)


def test_validate_arm_fixed_factors_rejects_unknown_arm_key() -> None:
    """An unknown arm key cannot select a release-tag expectation."""
    with pytest.raises(RadiusSweepManifestError, match="unknown radius arm key"):
        validate_arm_fixed_factors(_fixed_factors(), arm_key="r9p9")


@pytest.mark.parametrize(
    ("arm_key", "expected_release_tag"),
    [
        ("r0p5", "issue-6642-radius-sweep-0p5m"),
        ("r0p8", "issue-6642-radius-sweep-0p8m"),
        ("r1p0", "issue-6642-radius-sweep-1p0m"),
    ],
)
def test_validate_arm_fixed_factors_enforces_per_arm_release_tag(
    arm_key: str, expected_release_tag: str
) -> None:
    """Each arm's factors must carry that arm's issue-scoped release tag."""
    factors = _fixed_factors(release_tag=expected_release_tag)
    validate_arm_fixed_factors(factors, arm_key=arm_key)

    drifted = replace(factors, release_tag="issue-6642-radius-sweep-other")
    with pytest.raises(RadiusSweepManifestError, match="release_tag must be"):
        validate_arm_fixed_factors(drifted, arm_key=arm_key)


def test_arm_config_factor_drift_fails_closed_in_isolated_tree(tmp_path: Path) -> None:
    """Dropping a planner from one arm config stops the whole sweep, fail-closed."""
    import shutil

    arm_relpath = "configs/benchmarks/issue_6642_radius_sweep_arm_0p8m.yaml"
    for _arm_key, source_relpath, _radius, _baseline in _ARM_PAYLOAD_CASES:
        target = tmp_path / source_relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / source_relpath, target)

    drifted = tmp_path / arm_relpath
    payload = yaml.safe_load(drifted.read_text(encoding="utf-8"))
    payload["planners"] = payload["planners"][:-1]
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(RadiusSweepManifestError, match="planner_keys"):
        _resolve_all_arms(_manifest_config(), tmp_path)


def test_arm_config_treatment_drift_fails_closed_in_isolated_tree(tmp_path: Path) -> None:
    """A divergent radius treatment in one arm config stops the sweep, fail-closed."""
    import shutil

    for _arm_key, source_relpath, _radius, _baseline in _ARM_PAYLOAD_CASES:
        target = tmp_path / source_relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / source_relpath, target)

    drifted = tmp_path / "configs/benchmarks/issue_6642_radius_sweep_arm_0p8m.yaml"
    payload = yaml.safe_load(drifted.read_text(encoding="utf-8"))
    payload["radius_sweep"]["radius_m"] = 0.9
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(RadiusSweepManifestError, match="radius_m"):
        _resolve_all_arms(_manifest_config(), tmp_path)
