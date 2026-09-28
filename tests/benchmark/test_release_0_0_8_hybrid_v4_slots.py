"""Issue #9751: v4-named hybrid slots in the 0.0.8 roster, behind a fail-closed freeze guard.

The author amendment on #9668 (2026-09-28) keeps 14 arm slots, replaces the four
hybrid slots with hybrid v4 arms under new v4-named keys, forbids a key that names
a version from running a different version, and freezes v4 parameters only after
#9748. These tests pin that contract on the tracked templates.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from robot_sf.benchmark.camera_ready._preflight import prepare_campaign_preflight
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
    _parse_algo_config,
)
from robot_sf.benchmark.policy_search_manifest import resolve_candidate_manifest_runtime
from robot_sf.benchmark.release_parameter_freeze import (
    ARM_SLOTS_0_0_7_TO_0_0_8,
    COMPARISON_IMPLEMENTATION_REPLACED,
    COMPARISON_PAIRED,
    RELEASE_PARAMETER_FREEZE_KEY,
    UnfrozenReleaseParametersError,
    arm_slot_for_0_0_8_key,
    release_parameter_freeze_blocker,
    unfrozen_planner_config_blockers,
)
from robot_sf.benchmark.release_protocol import validate_release_planner_roster

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_TEMPLATE = (
    REPO_ROOT
    / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
)
RELEASE_TEMPLATE = (
    REPO_ROOT / "configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml"
)
RELEASE_0_0_7_MANIFEST = (
    REPO_ROOT / "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml"
)
HYBRID_ALGO = "hybrid_rule_local_planner"
# The hybrid core that every 0.0.7 hybrid arm ran; a legacy key without a core
# version in its name may only ever run this core.
FROZEN_0_0_7_HYBRID_CORE_VERSION = 3
REPLACED_0_0_7_KEYS = (
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
)
_VERSION_TOKEN = re.compile(r"(?:^|_)(v\d+)(?=_|$)")
_HYBRID_RULE_CORE = re.compile(r"hybrid_rule_v(\d+)")
_SCENARIO_ADAPTIVE_CORE_SUFFIX = re.compile(r"^scenario_adaptive_hybrid_orca_v\d+_.+_v(\d+)$")


def _load_yaml(path: Path | str) -> dict[str, Any]:
    payload = yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8")) or {}
    assert isinstance(payload, dict), path
    return payload


def _load_base_config(raw_path: object) -> dict[str, Any]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {}
    return _load_yaml(raw_path)


def _campaign_planners(path: Path) -> list[dict[str, Any]]:
    planners = _load_yaml(path)["planners"]
    assert isinstance(planners, list)
    return [row for row in planners if row.get("enabled", True)]


def _canonical_0_0_7_campaign() -> Path:
    manifest = _load_yaml(RELEASE_0_0_7_MANIFEST)
    return (RELEASE_0_0_7_MANIFEST.parent / manifest["canonical_campaign_config"]).resolve()


def _implementation_identity(row: dict[str, Any]) -> tuple[list[str], int | None]:
    """Return resolved runtime version fields and the hybrid core version.

    Returns:
        Runtime version fields and the hybrid core version (``None`` for
        non-hybrid rows).
    """
    algo_config = row.get("algo_config")
    if not algo_config:
        return [], None
    manifest = _load_yaml(algo_config)
    freeze = manifest.get(RELEASE_PARAMETER_FREEZE_KEY)
    if isinstance(freeze, dict) and "unfrozen_candidate" in freeze:
        # Placeholders cannot run; inspect the candidate that would be frozen.
        manifest = _load_yaml(freeze["unfrozen_candidate"])
    _algo, runtime = resolve_candidate_manifest_runtime(
        default_algo=str(row["algo"]),
        manifest=manifest,
        scenario={"name": "__default__"},
        load_config=_load_base_config,
    )
    identities = [
        str(value)
        for key, value in runtime.items()
        if key == "planner_variant" or key.endswith("_version")
    ]
    core = None
    if row["algo"] == HYBRID_ALGO:
        match = _HYBRID_RULE_CORE.search(str(runtime.get("planner_variant", "")))
        assert match, f"{row['key']}: hybrid arm without a hybrid_rule_vN planner_variant"
        core = int(match.group(1))
    return identities, core


def version_naming_violations(rows: list[dict[str, Any]]) -> list[str]:
    """Return every roster row whose key names a version it does not run.

    Two rules, applied to every row:

    * every runtime-version ``vN`` token in a key must appear in the resolved
      ``planner_variant`` or ``*_version`` fields;
    * for hybrid arms, the hybrid core version named by the key (``hybrid_rule_vN`` or
      the ``_vN`` suffix of a scenario-adaptive key) must equal the core actually run,
      and a key that names no core may only run the frozen 0.0.7 core.

    Returns:
        Human-readable violations; empty when the roster is honest.
    """
    violations: list[str] = []
    for row in rows:
        key = str(row["key"])
        identities, core = _implementation_identity(row)
        tokens = _VERSION_TOKEN.findall(key)
        if key.startswith("scenario_adaptive_hybrid_orca_v2_"):
            # The first v2 names this historical adaptation family, not the
            # hybrid core. The trailing vN (if present) names the core.
            tokens = tokens[1:]
        for token in tokens:
            if not any(re.search(rf"(?:^|[_/]){token}(?=_|\.|$)", ident) for ident in identities):
                violations.append(f"{key}: names {token} but runs {identities}")
        if core is None:
            continue
        claimed = _HYBRID_RULE_CORE.search(key) or _SCENARIO_ADAPTIVE_CORE_SUFFIX.match(key)
        claimed_core = int(claimed.group(1)) if claimed else None
        if claimed_core is None and core != FROZEN_0_0_7_HYBRID_CORE_VERSION:
            violations.append(f"{key}: names no hybrid core but runs hybrid v{core}")
        if claimed_core is not None and claimed_core != core:
            violations.append(f"{key}: names hybrid v{claimed_core} but runs hybrid v{core}")
    return violations


def test_0_0_8_roster_has_exactly_fourteen_slot_keys() -> None:
    """The campaign and release templates carry the same 14 keys, one per 0.0.7 slot."""
    template_keys = [row["key"] for row in _campaign_planners(CAMPAIGN_TEMPLATE)]
    release = _load_yaml(RELEASE_TEMPLATE)
    expected = [slot.key_0_0_8 for slot in ARM_SLOTS_0_0_7_TO_0_0_8]

    assert len(template_keys) == len(set(template_keys)) == 14
    assert template_keys == expected
    assert release["planners"]["keys"] == expected
    assert set(release["planners"]["groups"]) == set(expected)
    assert release["matrix"]["planner_arms"] == 14
    assert release["matrix"]["expected_episode_cells"] == 14 * 48 * 30 == 20160
    assert release["artifact_provenance"]["full_acceptance"] == (
        "require-14-arms-and-20160-exact-identities"
    )


def test_slot_mapping_replaces_exactly_the_four_hybrid_slots_under_v4_keys() -> None:
    """0.0.7 keys map 1:1 to 0.0.8 keys; only the hybrid slots change, to v4-named keys."""
    frozen_keys = _load_yaml(RELEASE_0_0_7_MANIFEST)["planners"]["keys"]
    assert [slot.key_0_0_7 for slot in ARM_SLOTS_0_0_7_TO_0_0_8] == frozen_keys

    replaced = [slot for slot in ARM_SLOTS_0_0_7_TO_0_0_8 if slot.key_0_0_7 != slot.key_0_0_8]
    assert tuple(slot.key_0_0_7 for slot in replaced) == REPLACED_0_0_7_KEYS
    for slot in ARM_SLOTS_0_0_7_TO_0_0_8:
        if slot in replaced:
            assert slot.comparison == COMPARISON_IMPLEMENTATION_REPLACED
            assert "v4" in _VERSION_TOKEN.findall(slot.key_0_0_8), slot
            assert slot.key_0_0_8 not in frozen_keys
        else:
            assert slot.comparison == COMPARISON_PAIRED
    assert COMPARISON_IMPLEMENTATION_REPLACED == "implementation replaced"
    assert arm_slot_for_0_0_8_key("hybrid_rule_v4_fast_progress_static_escape").key_0_0_7 == (
        "hybrid_rule_v3_fast_progress_static_escape"
    )
    with pytest.raises(KeyError):
        arm_slot_for_0_0_8_key("hybrid_rule_v3_fast_progress_static_escape")


@pytest.mark.parametrize(
    "campaign",
    [CAMPAIGN_TEMPLATE, _canonical_0_0_7_campaign()],
    ids=["0_0_8_template", "0_0_7_canonical"],
)
def test_no_roster_key_names_a_version_it_does_not_run(campaign: Path) -> None:
    """Generic check: a key that names a version never runs a different version."""
    assert version_naming_violations(_campaign_planners(campaign)) == []


@pytest.mark.parametrize(
    ("key", "algo_config"),
    [
        (
            "hybrid_rule_v3_fast_progress_static_escape",
            "configs/policy_search/candidates/"
            "hybrid_rule_v4_fast_progress_static_escape_s30_h600_release.yaml",
        ),
        (
            "scenario_adaptive_hybrid_orca_v2_collision_guard",
            "configs/policy_search/candidates/"
            "scenario_adaptive_hybrid_orca_v2_collision_guard_v4_s30_h600_release.yaml",
        ),
        (
            "hybrid_rule_v4_fast_progress_static_escape",
            "configs/policy_search/candidates/"
            "hybrid_rule_v3_fast_progress_static_escape_s30_h600_release.yaml",
        ),
    ],
    ids=["v3_key_runs_v4", "legacy_key_runs_v4_twin", "v4_key_runs_v3"],
)
def test_version_naming_check_rejects_relabelled_arms(key: str, algo_config: str) -> None:
    """The generic check catches a silent relabel in either direction."""
    row = {"key": key, "algo": HYBRID_ALGO, "algo_config": algo_config}
    assert version_naming_violations([row])


def test_version_naming_ignores_v2_filename_when_runtime_is_v3(tmp_path: Path) -> None:
    """A v2 name cannot certify a runtime config whose planner variant is v3."""
    config = tmp_path / "example_v2.yaml"
    config.write_text("name: example_v2\nplanner_variant: example_v3\n", encoding="utf-8")
    row = {"key": "example_v2", "algo": "example", "algo_config": str(config)}
    assert version_naming_violations([row]) == ["example_v2: names v2 but runs ['example_v3']"]


def test_v4_slots_bind_placeholders_for_real_v4_twins() -> None:
    """Each v4 slot binds an unfrozen placeholder whose candidate is a genuine v4 twin."""
    rows = {row["key"]: row for row in _campaign_planners(CAMPAIGN_TEMPLATE)}
    for slot in ARM_SLOTS_0_0_7_TO_0_0_8:
        if slot.comparison != COMPARISON_IMPLEMENTATION_REPLACED:
            continue
        row = rows[slot.key_0_0_8]
        placeholder = _load_yaml(row["algo_config"])
        freeze = placeholder[RELEASE_PARAMETER_FREEZE_KEY]
        assert freeze["status"] == "unfrozen"
        assert freeze["required_gate"] == "ll7/robot_sf_ll7#9748"
        assert freeze["replaces_0_0_7_slot"] == slot.key_0_0_7
        assert freeze["implementation_family"] == "hybrid_rule_v4_clearance_braking"
        assert placeholder["algo"] == row["algo"] == HYBRID_ALGO
        # A placeholder carries no runnable planner parameters.
        assert not {"base_config_path", "params", "scenario_overrides"} & set(placeholder)

        candidate = _load_yaml(freeze["unfrozen_candidate"])
        assert (
            candidate["base_config_path"] == "configs/algos/hybrid_rule_v4_clearance_braking.yaml"
        )
        _algo, runtime = resolve_candidate_manifest_runtime(
            default_algo=HYBRID_ALGO,
            manifest=candidate,
            scenario={"name": "__default__"},
            load_config=_load_base_config,
        )
        assert runtime["planner_variant"] == freeze["implementation_family"]
        # The twin keeps the replaced arm's scenario-override set (same scenarios).
        predecessor = _load_yaml(
            f"configs/policy_search/candidates/{slot.key_0_0_7}_s30_h600_release.yaml"
        )
        assert predecessor["base_config_path"].endswith("hybrid_rule_v3_teb_like_rollout.yaml")
        for block in ("scenario_overrides", "scenario_algo_overrides"):
            assert set(candidate.get(block) or {}) == set(predecessor.get(block) or {}), block
        assert candidate.get("scenario_algo_overrides") == predecessor.get(
            "scenario_algo_overrides"
        )


def _placeholder_paths() -> list[Path]:
    return sorted((REPO_ROOT / "configs/policy_search/release_0_0_8_placeholders").glob("*.yaml"))


def test_placeholders_fail_closed_in_map_runner_parser_and_resolver() -> None:
    """Runtime config parsing and candidate resolution both refuse every placeholder."""
    paths = _placeholder_paths()
    assert len(paths) == 4
    for path in paths:
        with pytest.raises(UnfrozenReleaseParametersError, match="#9748"):
            _parse_algo_config(str(path))
        with pytest.raises(UnfrozenReleaseParametersError, match="not frozen"):
            resolve_candidate_manifest_runtime(
                default_algo=HYBRID_ALGO,
                manifest=_load_yaml(path),
                scenario={"name": "classic_doorway_medium"},
                load_config=_load_base_config,
            )


def test_campaign_preflight_refuses_template_before_creating_output(tmp_path: Path) -> None:
    """The template still loads, but preflight refuses it before any output directory exists."""
    cfg = load_campaign_config(CAMPAIGN_TEMPLATE)
    assert len(cfg.planners) == 14
    blockers = unfrozen_planner_config_blockers(cfg.planners)
    assert len(blockers) == 4
    output_root = tmp_path / "campaigns"
    with pytest.raises(UnfrozenReleaseParametersError, match="campaign preflight refused"):
        prepare_campaign_preflight(cfg, output_root=output_root, label="issue-9751-guard")
    assert not output_root.exists()


def test_release_roster_admission_blocks_exactly_the_four_unfrozen_slots() -> None:
    """Release planner-roster admission reports the four unfrozen v4 slots and nothing else."""
    cfg = load_campaign_config(CAMPAIGN_TEMPLATE)
    release = _load_yaml(RELEASE_TEMPLATE)
    manifest = SimpleNamespace(
        planner_keys=tuple(release["planners"]["keys"]),
        planner_groups=dict(release["planners"]["groups"]),
        expected_kinematics_matrix=tuple(release["kinematics"]["matrix"]),
    )
    admission = validate_release_planner_roster(manifest, cfg)
    assert admission["status"] == "invalid"
    replaced = [
        slot.key_0_0_8
        for slot in ARM_SLOTS_0_0_7_TO_0_0_8
        if slot.comparison == COMPARISON_IMPLEMENTATION_REPLACED
    ]
    assert len(admission["blockers"]) == len(replaced) == 4
    for key in replaced:
        assert any(
            b.startswith(f"planner {key}: release parameters are not frozen")
            for b in admission["blockers"]
        )


@pytest.mark.parametrize(
    ("block", "blocked"),
    [
        ({"status": "frozen"}, False),
        ({"status": "unfrozen", "required_gate": "ll7/robot_sf_ll7#9748"}, True),
        ({}, True),
        ("frozen", True),
    ],
    ids=["frozen", "unfrozen", "missing_status", "malformed"],
)
def test_freeze_guard_admits_only_an_explicit_frozen_status(block: object, blocked: bool) -> None:
    """Status governs generic configs; configs without a block are not governed."""
    config = {"algo": HYBRID_ALGO, RELEASE_PARAMETER_FREEZE_KEY: block}
    assert (release_parameter_freeze_blocker(config, label="x") is not None) is blocked
    assert release_parameter_freeze_blocker({"algo": HYBRID_ALGO}, label="x") is None


def test_freeze_guard_permanently_rejects_placeholder_marker() -> None:
    """Changing only placeholder status to frozen cannot make it runnable."""
    placeholder = _load_yaml(_placeholder_paths()[0])
    placeholder[RELEASE_PARAMETER_FREEZE_KEY]["status"] = "frozen"
    assert "unfrozen_candidate" in release_parameter_freeze_blocker(placeholder, label="x")
    with pytest.raises(UnfrozenReleaseParametersError, match="unfrozen_candidate"):
        resolve_candidate_manifest_runtime(
            default_algo=HYBRID_ALGO,
            manifest=placeholder,
            scenario={"name": "__default__"},
            load_config=_load_base_config,
        )


def test_frozen_v4_slot_requires_resolved_v4_variant() -> None:
    """A status edit plus missing base would otherwise run the default v0."""
    config = {
        "algo": HYBRID_ALGO,
        RELEASE_PARAMETER_FREEZE_KEY: {
            "status": "frozen",
            "implementation_family": "hybrid_rule_v4_clearance_braking",
        },
    }
    assert "planner_variant=None" in release_parameter_freeze_blocker(config, label="x")
    config["base_config_path"] = "configs/algos/hybrid_rule_v3_teb_like_rollout.yaml"
    assert "hybrid_rule_v3_teb_like_rollout" in release_parameter_freeze_blocker(config, label="x")
    config["base_config_path"] = "configs/algos/hybrid_rule_v4_clearance_braking.yaml"
    assert release_parameter_freeze_blocker(config, label="x") is None
