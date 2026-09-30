"""Issue #9751: v4-named hybrid slots in the 0.0.8 roster, behind a fail-closed freeze guard.

The author amendment on #9668 (2026-09-28) keeps 14 arm slots, replaces the four
hybrid slots with hybrid v4 arms under new v4-named keys, forbids a key that names
a version from running a different version, and freezes v4 parameters only after
#9748. These tests pin that contract on the tracked templates.
"""

# evidence-writer-exempt: this test writes only a pytest tmp_path YAML fixture to
# check version naming; it does not create or alter repository evidence artifacts.

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

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


def test_0_0_8_release_selects_active_waypoint_for_all_risk_dwa_paths() -> None:
    """The release template must bind v2 for both arms and guarded PPO fallback."""
    planners = {row["key"]: row for row in _campaign_planners(CAMPAIGN_TEMPLATE)}
    expected_paths = {
        "risk_dwa": "configs/algos/risk_dwa_camera_ready_goal_v2.yaml",
        "predictive_mppi": "configs/algos/predictive_mppi_camera_ready_goal_v2.yaml",
        "guarded_ppo": "configs/algos/guarded_ppo_camera_ready_cpu_goal_v2.yaml",
    }
    for key, path in expected_paths.items():
        assert planners[key]["algo_config"] == path
        config = _load_yaml(path)
        selector = config.get("fallback_risk_dwa", config).get("goal_target_version")
        assert selector == "active_waypoint_v2", key


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


def test_v4_slots_bind_reviewed_frozen_configs_for_real_v4_twins() -> None:
    """The four release slots use explicit freezes derived from the v4 candidates."""
    rows = {row["key"]: row for row in _campaign_planners(CAMPAIGN_TEMPLATE)}
    log_path = REPO_ROOT / "docs/context/evidence/issue_9748_v4_tuning_log_v1.json"
    log = json.loads(log_path.read_text(encoding="utf-8"))
    log_hash = hashlib.sha256(log_path.read_bytes()).hexdigest()
    selected = {
        "hybrid_rule_v4_fast_progress_static_escape": "goal_progress_weight_4p5",
        "hybrid_rule_v4_fast_progress_static_escape_continuous": "baseline",
    }
    for slot in ARM_SLOTS_0_0_7_TO_0_0_8:
        if slot.comparison != COMPARISON_IMPLEMENTATION_REPLACED:
            continue
        row = rows[slot.key_0_0_8]
        frozen = _load_yaml(row["algo_config"])
        freeze = frozen[RELEASE_PARAMETER_FREEZE_KEY]
        assert freeze["status"] == "frozen"
        assert freeze["implementation_family"] == "hybrid_rule_v4_clearance_braking"
        assert freeze["source_tuning_log_path"] == log_path.relative_to(REPO_ROOT).as_posix()
        assert freeze["source_tuning_log_sha256"] == log_hash
        assert freeze["freeze_decision_url"] == (
            "https://github.com/ll7/robot_sf_ll7/issues/9748#issuecomment-5884539753"
        )
        assert frozen["algo"] == row["algo"] == HYBRID_ALGO
        assert release_parameter_freeze_blocker(frozen, label=row["key"]) is None
        candidate = _load_yaml(
            f"configs/policy_search/candidates/{slot.key_0_0_8}_s30_h600_release.yaml"
        )
        assert (
            candidate["base_config_path"] == "configs/algos/hybrid_rule_v4_clearance_braking.yaml"
        )
        expected_candidate = dict(candidate)
        expected_candidate["name"] = frozen["name"]
        if row["key"] in selected:
            trial_id = selected[row["key"]]
            assert freeze["trial_id"] == trial_id
            entries = [
                entry
                for entry in log["entries"]
                if entry["candidate"] == candidate["name"] and entry["trial_id"] == trial_id
            ]
            assert len(entries) == 4
            assert {entry["effective_config_sha256"] for entry in entries} == {
                freeze["effective_config_sha256"]
            }
            expected_candidate["params"] = {
                **candidate["params"],
                **entries[0]["parameter_overrides"],
            }
        else:
            assert freeze["trial_id"] == "untuned_v4_base"
            assert freeze["tuning_status"] == "untuned"
            assert freeze["tuning_scope"] == "outside_pre_registered_9748_search"
        assert {
            key: value for key, value in frozen.items() if key != RELEASE_PARAMETER_FREEZE_KEY
        } == (expected_candidate)
        _algo, runtime = resolve_candidate_manifest_runtime(
            default_algo=HYBRID_ALGO,
            manifest=frozen,
            scenario={"name": "__default__"},
            load_config=_load_base_config,
        )
        assert runtime["planner_variant"] == freeze["implementation_family"]
        assert (
            hashlib.sha256(
                json.dumps(runtime, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            == freeze["effective_config_sha256"]
        )
        if row["key"] in selected:
            for entry in entries:
                _, resolved = resolve_candidate_manifest_runtime(
                    default_algo=HYBRID_ALGO,
                    manifest=frozen,
                    scenario={"name": entry["scenario_id"]},
                    load_config=_load_base_config,
                )
                assert (
                    hashlib.sha256(
                        json.dumps(
                            resolved, sort_keys=True, separators=(",", ":"), allow_nan=False
                        ).encode()
                    ).hexdigest()
                    == entry["effective_config_sha256"]
                )
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


def test_campaign_template_has_no_unfrozen_planner_blockers() -> None:
    """The selected frozen v4 configs pass the release parameter gate."""
    cfg = load_campaign_config(CAMPAIGN_TEMPLATE)
    assert len(cfg.planners) == 14
    assert unfrozen_planner_config_blockers(cfg.planners) == []


def test_release_roster_admission_accepts_the_four_frozen_slots() -> None:
    """Release planner-roster admission accepts the reviewed v4 freezes."""
    cfg = load_campaign_config(CAMPAIGN_TEMPLATE)
    release = _load_yaml(RELEASE_TEMPLATE)
    manifest = SimpleNamespace(
        planner_keys=tuple(release["planners"]["keys"]),
        planner_groups=dict(release["planners"]["groups"]),
        expected_kinematics_matrix=tuple(release["kinematics"]["matrix"]),
    )
    admission = validate_release_planner_roster(manifest, cfg)
    assert admission["status"] == "valid"
    assert admission["blockers"] == []


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
