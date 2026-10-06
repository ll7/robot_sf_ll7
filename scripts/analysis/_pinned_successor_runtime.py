"""Resolve successor campaign identity inside its detached source checkout.

This file is launched with ``python -I``. The checkout path is installed before
any project import, so production helpers come from the named source commit.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


def _assert_pinned_modules(checkout: Path) -> None:
    for name in (
        "robot_sf.benchmark.camera_ready._util",
        "robot_sf.benchmark.camera_ready._config",
        "robot_sf.benchmark.camera_ready._preflight",
        "robot_sf.benchmark.runner",
        "robot_sf.baselines.ppo",
        "robot_sf.benchmark.map_runner.map_runner",
        "robot_sf.benchmark.map_runner.map_runner_identity",
        "robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution",
        "robot_sf.benchmark.utils",
        "robot_sf.benchmark.algorithm_metadata",
        "robot_sf.benchmark.observation_noise",
        "robot_sf.benchmark.release_candidate",
        "robot_sf.benchmark.release_parameter_freeze",
        "pysocialforce",
    ):
        module = sys.modules[name]
        if not Path(module.__file__).resolve().is_relative_to(checkout):
            raise ValueError(f"runtime module {name} did not load from pinned successor source")


def main() -> None:  # noqa: C901, PLR0912, PLR0915 - pinned resolution stays together
    checkout = Path.cwd().resolve()
    sys.path.insert(0, str(checkout))
    sys.path.insert(1, str(checkout / "fast-pysf"))
    for package in ("robot_sf", "fast-pysf/pysocialforce"):
        if not (checkout / package / "__init__.py").is_file():
            raise ValueError(f"pinned successor source lacks runtime package {package}")

    import pysocialforce  # noqa: F401 - assert its origin with the project modules below

    from robot_sf.baselines.ppo import PPOPlanner
    from robot_sf.benchmark.algorithm_metadata import (
        enrich_algorithm_metadata,
        resolve_learned_checkpoint_observation_contract,
    )
    from robot_sf.benchmark.camera_ready import _util
    from robot_sf.benchmark.camera_ready._config import (
        _apply_fixed_campaign_horizon,
        _load_campaign_scenarios,
        _scenario_with_kinematics,
        load_campaign_config,
    )
    from robot_sf.benchmark.camera_ready._preflight import _scenario_matrix_hash
    from robot_sf.benchmark.map_runner.map_runner import _ppo_planner_config
    from robot_sf.benchmark.map_runner.map_runner_identity import (
        _resolve_seed_list,
        _scenario_identity_payload,
        _scenario_with_episode_seed_defaults,
        _select_seeds,
        _suite_key,
    )
    from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
        _apply_planner_selector_v2_context,
        _apply_scenario_uncertainty_envelope_config,
        _parse_algo_config,
        _resolve_config_path,
        _resolve_policy_search_candidate_runtime,
    )
    from robot_sf.benchmark.observation_noise import (
        normalize_observation_noise_spec,
        observation_noise_hash,
    )
    from robot_sf.benchmark.release_candidate import (
        _APPROVED_008_HYBRID_ALGO_OVERRIDES as APPROVED_008_HYBRID_ALGO_OVERRIDES,
    )
    from robot_sf.benchmark.release_parameter_freeze import (
        ARM_SLOTS_0_0_7_TO_0_0_8,
        V4_HYBRID_VARIANT,
    )
    from robot_sf.benchmark.runner import _apply_track_metadata_to_scenarios
    from robot_sf.benchmark.utils import _config_hash

    _assert_pinned_modules(checkout)

    request = json.load(sys.stdin)
    with patch.object(_util, "get_repository_root", return_value=checkout):
        if "resolved_identity_path" in request:
            from robot_sf.benchmark.release_notes import gate_manifest
            from robot_sf.benchmark.release_protocol import (
                load_release_campaign_config,
                load_release_manifest,
            )

            manifest = load_release_manifest(
                request["resolved_identity_path"], repository_root=checkout
            )
            gate_manifest(manifest, repository_root=checkout)
            cfg = load_release_campaign_config(manifest, repository_root=checkout)
            if request.get("snqi_v2_anchors"):
                from robot_sf.benchmark.snqi.v2_binding import bind_acquired_anchors

                cfg = bind_acquired_anchors(
                    cfg,
                    anchors_path=Path(request["snqi_v2_anchors"]),
                    calibration_root=None,
                    source_commit=manifest.source_sha,
                    diagnostic=False,
                )
        else:
            cfg = load_campaign_config(checkout / request["config_path"], repository_root=checkout)
        publication = request.get("publication_identity")
        if publication is not None:
            if (
                not isinstance(publication, dict)
                or set(publication) != {"release_tag", "doi"}
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in publication.values()
                )
            ):
                raise ValueError(
                    "publication_identity requires only release_tag and doi as nonempty strings"
                )
            # Match load_release_campaign_config: scientific inputs stay source-bound.
            cfg = replace(cfg, **publication)
        if "development_rehearsal_seeds" in request:
            from types import SimpleNamespace

            from robot_sf.benchmark.release_protocol import _development_campaign_config

            cfg = _development_campaign_config(
                SimpleNamespace(
                    release_kind="development_rehearsal",
                    resolved_seeds=request["development_rehearsal_seeds"],
                ),
                cfg,
            )
        for path in (cfg.source_config_path, cfg.scenario_matrix_path):
            if path is None or not path.resolve().is_relative_to(checkout):
                raise ValueError("successor config source escapes pinned checkout")
        scenarios = _load_campaign_scenarios(cfg, repository_root=checkout)
        config_hash = _config_hash(_util._config_hash_payload(cfg))
        scenario_hash = _scenario_matrix_hash(scenarios)

    scenario_by_id = {}
    for scenario in scenarios:
        identity = scenario.get("name") or scenario.get("scenario_id") or scenario.get("id")
        if not isinstance(identity, str) or not identity or identity in scenario_by_id:
            raise ValueError(f"invalid or duplicate pinned scenario identity: {identity!r}")
        scenario_by_id[identity] = _util._jsonable_repo_relative(scenario)

    template_path = (
        checkout
        / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
    )
    import yaml

    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    template_planners = {item["key"]: item for item in template["planners"]}
    versioned_keys = set(request["versioned_keys"])
    reviewed_slots = {
        slot.key_0_0_8: slot.key_0_0_7
        for slot in ARM_SLOTS_0_0_7_TO_0_0_8
        if slot.key_0_0_8 != slot.key_0_0_7
    }
    if versioned_keys != reviewed_slots.keys() or not versioned_keys <= template_planners.keys():
        raise ValueError("pinned 0.0.8 template lacks reviewed v4 slots")

    planners = {}
    for planner in cfg.planners:
        if not planner.enabled:
            continue
        path = planner.algo_config_path
        if path is not None and not path.resolve().is_relative_to(checkout):
            raise ValueError(f"successor planner config escapes pinned checkout: {planner.key}")
        raw = _parse_algo_config(str(path)) if path else {}
        if not isinstance(raw, dict):
            raise ValueError(f"successor planner config must be a mapping: {planner.key}")
        if path is not None:
            references = [raw.get("base_config_path")]
            overrides = raw.get("scenario_algo_overrides")
            if isinstance(overrides, dict):
                references.extend(
                    entry.get("base_config_path")
                    for entry in overrides.values()
                    if isinstance(entry, dict)
                )
            for reference in references:
                if reference is None:
                    continue
                resolved = _resolve_config_path(path.parent, reference)
                if (
                    resolved is None
                    or not resolved.is_file()
                    or not resolved.is_relative_to(checkout)
                ):
                    raise ValueError(
                        f"successor planner referenced config escapes pinned checkout: {planner.key}"
                    )
        planners[planner.key] = {
            "algo": planner.algo,
            "config": raw,
            "path": path.relative_to(checkout).as_posix() if path else None,
            "absolute_path": str(path) if path else None,
            "observation_mode": planner.observation_mode or cfg.observation_mode,
            "spec": planner,
        }
        if planner.key in versioned_keys:
            template_config = yaml.safe_load(
                (checkout / template_planners[planner.key]["algo_config"]).read_text(
                    encoding="utf-8"
                )
            )
            template_freeze = template_config.get("release_parameter_freeze", {})
            freeze = raw.get("release_parameter_freeze", {})
            if (
                template_planners[planner.key].get("algo") != planner.algo
                or template_planners[planner.key].get("algo_config")
                != planners[planner.key]["path"]
                or template_freeze.get("status") != "frozen"
                or template_freeze.get("implementation_family") != V4_HYBRID_VARIANT
                # Release slot lineage belongs to the source-pinned arm map;
                # frozen parameter bytes need not repeat it. Any declaration
                # present in either config must still agree with that map.
                or template_freeze.get("replaces_0_0_7_slot", reviewed_slots[planner.key])
                != reviewed_slots[planner.key]
                or freeze.get("status") != "frozen"
                or "unfrozen_candidate" in freeze
                or freeze.get("implementation_family")
                != template_freeze.get("implementation_family")
                or freeze.get("replaces_0_0_7_slot", reviewed_slots[planner.key])
                != reviewed_slots[planner.key]
            ):
                raise ValueError(
                    f"successor planner binding lacks reviewed v4 lineage: {planner.key}"
                )
            base_path = _resolve_config_path(path.parent, raw.get("base_config_path"))
            if base_path is None or not base_path.is_relative_to(checkout):
                raise ValueError(f"successor planner binding lacks pinned v4 base: {planner.key}")
            base = _parse_algo_config(str(base_path))
            if base.get("planner_variant") != V4_HYBRID_VARIANT:
                raise ValueError(f"successor planner binding resolves non-v4 base: {planner.key}")
            _, resolved = _resolve_policy_search_candidate_runtime(
                default_algo=planner.algo,
                algo_config_path=str(path),
                scenario={"name": "__default__"},
                algo_config=raw,
            )
            if resolved.get("planner_variant") != V4_HYBRID_VARIANT:
                raise ValueError(f"successor planner binding resolves non-v4 config: {planner.key}")

    expected_slots = set()
    runner_scenario_by_slot = {}
    scoped_hashes = []
    suite_seeds = _resolve_seed_list(checkout / "configs/benchmarks/seed_list_v1.yaml")
    suite_key = _suite_key(cfg.scenario_matrix_path)
    for kinematics in _util._kinematics_matrix_or_default(cfg.kinematics_matrix):
        scoped = [
            _scenario_with_kinematics(
                scenario,
                kinematics=kinematics,
                holonomic_command_mode=cfg.holonomic_command_mode,
            )
            for scenario in scenarios
        ]
        if cfg.telemetry is not None:
            for scenario in scoped:
                scenario["telemetry"] = dict(cfg.telemetry)
        for key, planner in planners.items():
            spec = planner["spec"]
            arm_horizon = (
                spec.horizon_override if spec.horizon_override is not None else cfg.horizon
            )
            runner_scenarios = _apply_track_metadata_to_scenarios(
                _apply_fixed_campaign_horizon(
                    scoped,
                    horizon=arm_horizon,
                    horizon_policy=cfg.horizon_policy,
                    protocol_version=cfg.protocol_version,
                ),
                observation_mode=planner["observation_mode"],
                observation_level=None,
                benchmark_track=None,
                track_schema_version=None,
                telemetry=cfg.telemetry,
            )
            scoped_hashes.append(
                {"planner": key, "kinematics": kinematics, "hash": _config_hash(runner_scenarios)}
            )
            for scenario in runner_scenarios:
                identity = scenario.get("name") or scenario.get("scenario_id") or scenario.get("id")
                seeds = _select_seeds(scenario, suite_seeds=suite_seeds, suite_key=suite_key)
                track = scenario.get("benchmark_track") or ""
                for seed in seeds:
                    slot = (key, kinematics, identity, seed, track)
                    if slot in expected_slots:
                        raise ValueError(f"duplicate pinned successor slot: {slot}")
                    expected_slots.add(slot)
                    runner_scenario_by_slot[slot] = scenario

    runtime_rows = []
    for item in request["rows"]:
        slot = item["slot"]
        if tuple(slot) not in expected_slots:
            continue
        planner = planners.get(slot[0])
        if planner is None:
            raise ValueError(f"0.0.8 row planner is absent from verified successor config: {slot}")
        matrix_scenario = scenario_by_id.get(slot[2])
        if matrix_scenario is None:
            raise ValueError(f"0.0.8 row scenario is absent from pinned matrix: {slot}")
        scoped_scenario = runner_scenario_by_slot[tuple(slot)]
        seeds = matrix_scenario.get("seeds")
        if isinstance(seeds, list) and slot[3] not in seeds:
            raise ValueError(f"0.0.8 row seed is absent from pinned scenario: {slot}")
        scenario = item["scenario_params"]
        if not isinstance(scenario, dict):
            raise ValueError(f"0.0.8 row lacks scenario provenance at {slot}")
        algo, effective = _resolve_policy_search_candidate_runtime(
            default_algo=planner["algo"],
            algo_config_path=planner["absolute_path"],
            scenario=scoped_scenario,
            algo_config=planner["config"],
        )
        effective = _apply_planner_selector_v2_context(
            algo, effective, scenario=scoped_scenario, seed=slot[3]
        )
        effective = _apply_scenario_uncertainty_envelope_config(algo, effective, scoped_scenario)
        if slot[0] in versioned_keys:
            overrides = planner["config"].get("scenario_algo_overrides") or {}
            override = overrides.get(slot[2]) if isinstance(overrides, dict) else None
            approved = APPROVED_008_HYBRID_ALGO_OVERRIDES.get(slot[0], {}).get(slot[2])
            # Exempt only the reviewed planner/scenario/algorithm/base tuple.
            # Checking the effective algorithm alone would admit arbitrary ORCA
            # bases, and checking the declaration alone would trust wrong resolution.
            approved_handoff = (
                approved is not None
                and isinstance(override, dict)
                and (override.get("algo"), override.get("base_config_path")) == approved
                and algo == approved[0]
                and _resolve_config_path(
                    Path(planner["absolute_path"]).parent, override["base_config_path"]
                )
                == (checkout / approved[1]).resolve()
            )
            if algo != planner["algo"] and not approved_handoff:
                raise ValueError(f"successor v4 slot resolves wrong algorithm: {slot}")
            if override is not None and not approved_handoff:
                raise ValueError(
                    f"successor v4 slot has unapproved algorithm/base override: {slot}"
                )
            if not approved_handoff and effective.get("planner_variant") != V4_HYBRID_VARIANT:
                raise ValueError(f"successor planner row resolves non-v4 config: {slot}")
        spec = planner["spec"]
        effective_dt = spec.dt_override if spec.dt_override is not None else cfg.dt
        observation = resolve_learned_checkpoint_observation_contract(
            algo,
            effective,
            observation_mode=planner["observation_mode"],
            observation_level=None,
        )
        active_mode = str(observation["active_observation_mode"])
        metadata = enrich_algorithm_metadata(
            algo=algo,
            metadata={},
            robot_kinematics=slot[1],
            observation_mode=active_mode,
            observation_level=observation.get("observation_level_key"),
        )
        safety_wrapper = (
            spec.safety_wrapper if spec.safety_wrapper is not None else cfg.safety_wrapper
        )
        seeded_scenario = _scenario_with_episode_seed_defaults(scoped_scenario, seed=slot[3])
        controls = _scenario_identity_payload(
            seeded_scenario,
            algo=algo,
            algo_config=effective,
            horizon=spec.horizon_override if spec.horizon_override is not None else cfg.horizon,
            dt=effective_dt,
            record_forces=cfg.record_forces,
            observation_mode=active_mode,
            observation_level=str(metadata["observation_level"]["key"]),
            benchmark_track=slot[4] or None,
            track_schema_version=scoped_scenario.get("track_schema_version"),
            observation_noise=cfg.observation_noise,
            synthetic_actuation_profile=_util._synthetic_actuation_metadata(
                cfg.synthetic_actuation_profile
            ),
            latency_stress_profile=(
                cfg.latency_stress_profile.to_metadata(dt=effective_dt or 0.1)
                if cfg.latency_stress_profile is not None
                else None
            ),
            safety_wrapper=safety_wrapper,
            record_planner_decision_trace=cfg.record_planner_decision_trace,
            record_simulation_step_trace=cfg.record_simulation_step_trace,
        )
        control_fields = (
            "run_horizon",
            "run_dt",
            "record_forces",
            "record_planner_decision_trace",
            "record_simulation_step_trace",
            "observation_mode",
            "observation_level",
            "observation_noise_profile",
            "observation_noise_hash",
            "tracking_precision",
            "tracking_precision_hash",
            "synthetic_actuation_profile",
            "latency_stress_profile",
            "safety_wrapper",
            "cbf_safety_filter",
            "benchmark_track",
            "track_schema_version",
        )
        metadata_config = effective
        if algo in {"ppo", "guarded_ppo"}:
            # Reproduce the producer's typed PPO metadata without loading a model
            # or stepping a policy/environment. The full guard/adapter config
            # remains bound separately by scenario.algo_config_hash.
            deferred = PPOPlanner(_ppo_planner_config(effective), defer_model_loading=True)
            metadata_config = deferred.get_metadata()["config"]
        runtime_rows.append(
            {
                "slot": slot,
                "algo": algo,
                "config": effective,
                "config_hash": _config_hash(effective),
                "metadata_algorithm": metadata["algorithm"],
                "metadata_config": metadata_config,
                "metadata_config_hash": _config_hash(metadata_config),
                "scenario_config_hash": _config_hash(controls),
                "scenario": seeded_scenario,
                "path": planner["path"],
                "controls": {key: controls[key] for key in control_fields if key in controls},
                "observation_noise": normalize_observation_noise_spec(cfg.observation_noise),
                "observation_noise_hash": observation_noise_hash(
                    normalize_observation_noise_spec(cfg.observation_noise)
                ),
            }
        )
    json.dump(
        {
            "config_hash": config_hash,
            "scenario_hash": scenario_hash,
            "rows": runtime_rows,
            "expected_slots": sorted(expected_slots),
            "scoped_hashes": scoped_hashes,
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
