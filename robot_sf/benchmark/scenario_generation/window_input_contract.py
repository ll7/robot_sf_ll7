"""Reconstruct the complete input contract from a configuration-owning checkpoint.

The checkpoint is the authority. Candidate inputs are reconstructed through the
production builders; native carriers are compared as complete dictionaries.
No scenario/run projection or asserted input hash establishes admission.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.scenario_generation.continuation import (
    INPUT_SCHEMA,
    effective_configuration,
    input_digest,
)


def reconstruct_configuration(
    scenario: dict[str, Any],
    scenario_path: Path,
    map_path: Path,
    planner_path: Path,
    inputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use production config/policy construction without creating or resetting an env.

    Returns:
        Complete configuration and resolved policy input.
    """
    import yaml  # noqa: PLC0415

    from robot_sf.benchmark.map_runner.map_runner import build_map_policy  # noqa: PLC0415
    from robot_sf.benchmark.map_runner.map_runner_env import (  # noqa: PLC0415
        apply_active_observation_mode_to_env_config,
        apply_policy_env_observation_overrides,
        build_env_config,
    )
    from robot_sf.benchmark.map_runner.map_runner_identity import (  # noqa: PLC0415
        _scenario_with_episode_seed_defaults,
    )
    from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (  # noqa: PLC0415
        resolve_episode_policy_runtime,
    )

    selected = _scenario_with_episode_seed_defaults(deepcopy(scenario), seed=inputs["seed"])
    selected["map_file"] = str(map_path)
    settings = inputs["run"]["simulator_settings"]
    config = build_env_config(selected, scenario_path=scenario_path)
    config.sim_config.time_per_step_in_secs = settings["dt"]
    algo, policy_config = resolve_episode_policy_runtime(
        default_algo="social_force",
        algo_config_path=str(planner_path),
        algo_config=yaml.safe_load(planner_path.read_text()),
        scenario=selected,
        seed=inputs["seed"],
    )
    apply_active_observation_mode_to_env_config(
        config, active_observation_mode=settings["observation_mode"]
    )
    apply_policy_env_observation_overrides(config, policy_config)
    policy, _ = build_map_policy(
        algo, policy_config, robot_kinematics="differential_drive", robot_command_mode="vx_vy"
    )
    adapter = cast("Any", policy)._planner_adapter
    adapter.bind_env(SimpleNamespace(config=config))
    return effective_configuration(config, adapter), policy_config


def expected_scenario_params(
    scenario: dict[str, Any], policy: dict[str, Any], inputs: dict[str, Any]
) -> dict[str, Any]:
    """Recompute every recorded scenario/run argument using the producer's identity builder.

    Returns:
        Complete native scenario/run identity.
    """
    from robot_sf.benchmark.map_runner.map_runner_identity import (  # noqa: PLC0415
        _scenario_with_episode_seed_defaults,
        scenario_identity_payload,
    )

    run = inputs["run"]
    settings = run["simulator_settings"]
    return scenario_identity_payload(
        _scenario_with_episode_seed_defaults(scenario, seed=inputs["seed"]),
        algo="social_force",
        algo_config=policy,
        horizon=settings["horizon"],
        dt=settings["dt"],
        record_forces=settings["record_forces"],
        observation_mode=settings["observation_mode"],
        observation_level=settings["observation_level"],
        observation_noise=run["observation_noise"],
        tracking_precision=run["tracking_precision"],
        record_planner_decision_trace=run["record_planner_decision_trace"],
        record_simulation_step_trace=run["record_simulation_step_trace"],
    )


def native_input_envelope(row: dict[str, Any], *, window: bool) -> dict[str, Any]:
    """Adapt existing native inputs to the versioned complete comparison object.

    Returns:
        Complete native input envelope.
    """
    meta = row["algorithm_metadata"]
    trace = meta["simulation_step_trace"]
    timing = [{"step": s["step"], "time_s": s["time_s"]} for s in trace["steps"]]
    envelope = {
        "scenario_params": row["scenario_params"],
        "simulator_settings": row["result_provenance"]["simulator_settings"],
        "seed": row["seed"],
        "horizon": row["horizon"],
        "observation_mode": row["observation_mode"],
        "observation_level": row["observation_level"],
        "observation_noise": row["observation_noise"],
        "observation_noise_hash": row["observation_noise_hash"],
        "tracking_precision": row["tracking_precision"],
        "tracking_precision_hash": row["tracking_precision_hash"],
        "trace": {"dt": trace["dt"], "timing": timing},
        "map_sha256": sorted(
            r["sha256"] for r in row["runtime_input_records"] if r.get("role") == "map_file"
        ),
    }
    if window:
        envelope["continuation_inputs"] = meta["generated_window_continuation"].get(
            "effective_inputs"
        )
        envelope["trace"]["window_effective_inputs"] = trace.get("window_effective_inputs")
    return envelope


def expected_native_envelope(
    params: dict[str, Any], inputs: dict[str, Any], *, window: bool
) -> dict[str, Any]:
    """Derive all redundant native input declarations and trace timing from the checkpoint.

    Returns:
        The independently recomputed expected native input envelope.
    """
    run = inputs["run"]
    settings = deepcopy(run["simulator_settings"])
    bounds = inputs["window"]
    start = bounds["start_step"] if window else 0
    count = bounds["steps"] if window else settings["horizon"]
    envelope = {
        "scenario_params": params,
        "simulator_settings": settings,
        "seed": inputs["seed"],
        "horizon": settings["horizon"],
        "observation_mode": settings["observation_mode"],
        "observation_level": settings["observation_level"],
        "observation_noise": run["observation_noise"],
        "observation_noise_hash": settings["observation_noise_hash"],
        "tracking_precision": run["tracking_precision"],
        "tracking_precision_hash": settings["tracking_precision_hash"],
        "trace": {
            "dt": settings["dt"],
            "timing": [
                {"step": i, "time_s": (i + 1) * settings["dt"]} for i in range(start, start + count)
            ],
        },
        "map_sha256": [inputs["map"]["sha256"]],
    }
    if window:
        settings["window_effective_inputs"] = inputs
        envelope["continuation_inputs"] = inputs
        envelope["trace"]["window_effective_inputs"] = inputs
    return envelope


def verify_effective_inputs(
    checkpoint: dict[str, Any],
    *,
    paths: dict[str, Path],
    entry: dict[str, Any],
    source: dict[str, Any],
    replays: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
) -> None:
    """Refuse any complete input object that differs from the checkpoint-derived contract."""
    import os  # noqa: PLC0415

    from robot_sf.benchmark.scenario_generation.replay_adapter import materialize_generated_scenario  # noqa: PLC0415
    from robot_sf.training.scenario_loader import load_scenarios  # noqa: PLC0415

    scenario_path, checkpoint_path = paths["scenario"], paths["checkpoint"]
    map_path, planner_path = paths["map"], paths["planner"]
    if checkpoint["schema_version"] != INPUT_SCHEMA:
        raise ValueError("effective input contract requires configuration-owning checkpoint v2")
    inputs = checkpoint["effective_inputs"]
    if input_digest(inputs["configuration"]) != checkpoint["compatibility"]["config_sha256"]:
        raise ValueError("effective configuration digest mismatch")
    config, policy = reconstruct_configuration(
        checkpoint["source_scenario"], scenario_path, map_path, planner_path, inputs
    )
    if input_digest(config) != input_digest(inputs["configuration"]):
        raise ValueError("effective captured scenario/configuration mismatch")
    source_params = expected_scenario_params(checkpoint["source_scenario"], policy, inputs)
    if input_digest(native_input_envelope(source, window=False)) != input_digest(
        expected_native_envelope(source_params, inputs, window=False)
    ):
        raise ValueError("effective source inputs mismatch")
    materialized = materialize_generated_scenario(
        entry,
        source_scenario=checkpoint["source_scenario"],
        continuation_checkpoint=checkpoint_path,
        max_episode_steps=inputs["window"]["steps"],
    )
    assert materialized.scenario_document is not None
    expected_scenario = materialized.scenario_document["scenarios"][0]
    expected_scenario["map_file"] = os.path.relpath(map_path, scenario_path.parent)
    expected_scenario["generated_replay"]["checkpoint_file"] = os.path.relpath(
        checkpoint_path, scenario_path.parent
    )
    expected_params = expected_scenario_params(expected_scenario, policy, inputs)
    selected = load_scenarios(scenario_path)
    if len(selected) != 1 or input_digest(
        expected_scenario_params(dict(selected[0]), policy, inputs)
    ) != input_digest(expected_params):
        raise ValueError("effective configuration: candidate inputs mismatch")
    selected_config, _ = reconstruct_configuration(
        dict(selected[0]), scenario_path, map_path, planner_path, inputs
    )
    if input_digest(selected_config) != input_digest(inputs["configuration"]):
        raise ValueError("effective candidate configuration mismatch")
    expected = expected_native_envelope(expected_params, inputs, window=True)
    for row, manifest in zip(replays, manifests, strict=True):
        if input_digest(native_input_envelope(row, window=True)) != input_digest(expected):
            raise ValueError("effective replay input envelope mismatch")
        actual_manifest = {
            "seed": manifest["rows"][0]["seed"],
            "simulator_settings": manifest["rows"][0]["simulator_settings"],
        }
        expected_manifest = {
            "seed": inputs["seed"],
            "simulator_settings": expected["simulator_settings"],
        }
        if input_digest(actual_manifest) != input_digest(expected_manifest):
            raise ValueError("effective manifest input envelope mismatch")
