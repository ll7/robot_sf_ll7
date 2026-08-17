#!/usr/bin/env python3
"""Run the deterministic predictive-baseline planner diagnostic for #7319."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from jsonschema import Draft202012Validator

from robot_sf.benchmark.predictive_baseline_contract import (
    PlannerMethodCard,
    PlannerSmokeRecord,
    build_predictive_baseline_report,
)
from robot_sf.planner.cbf_safety_filter import (
    CbfSafetyFilterConfig,
    CbfSafetyFilterPlannerWrapper,
)
from robot_sf.planner.mppi_social import MPPISocialConfig, MPPISocialPlannerAdapter
from robot_sf.planner.nmpc_social import NMPCSocialConfig, NMPCSocialPlannerAdapter
from robot_sf.planner.predictive_human_cost import (
    PredictiveGaussianHumanCostConfig,
)

_METHOD_REFERENCE = "mppi_social_reference_v1"
_METHOD_PGIF = "pgif_mppi_adapted_v1"
_METHOD_CONSTRAINED = "nmpc_cbf_internal_v1"
_UNAVAILABLE_METRICS = {
    "success": "no simulator rollout",
    "collision": "no simulator rollout",
    "near_miss": "no simulator rollout",
    "timeout": "no simulator rollout",
    "path_efficiency": "no route rollout",
    "pedestrian_disruption": "no simulator truth",
    "minimum_distance": "no simulator truth",
    "action_smoothness": "one-step smoke only",
}


def _load_config(path: Path) -> dict[str, Any]:
    """Load a non-empty YAML mapping."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("diagnostic config must be a YAML mapping")
    return payload


def _build_observation(*, seed: int, pedestrian_count: int) -> dict[str, Any]:
    """Build the fixed-seed planar crossing fixture observation."""

    if pedestrian_count < 1:
        raise ValueError("pedestrian_count must be positive")
    rng = np.random.default_rng(int(seed))
    positions = np.column_stack(
        (
            rng.uniform(0.8, 1.8, size=pedestrian_count),
            rng.uniform(-0.55, 0.55, size=pedestrian_count),
        )
    )
    velocities = np.column_stack(
        (
            rng.uniform(-0.05, 0.05, size=pedestrian_count),
            rng.uniform(-0.35, 0.35, size=pedestrian_count),
        )
    )
    return {
        "robot": {
            "position": np.asarray([0.0, 0.0], dtype=float),
            "heading": np.asarray([0.0], dtype=float),
            "speed": np.asarray([0.2], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray([3.0, 0.0], dtype=float),
            "next": np.asarray([3.0, 0.0], dtype=float),
        },
        "pedestrians": {
            "positions": positions,
            "velocities": velocities,
            "count": np.asarray([float(pedestrian_count)], dtype=float),
            "radius": 0.25,
        },
    }


def _mppi_config(
    config: dict[str, Any], *, human_cost: PredictiveGaussianHumanCostConfig
) -> MPPISocialConfig:
    """Build a small deterministic MPPI configuration."""

    return MPPISocialConfig(
        random_seed=int(config["seed"]),
        horizon_steps=int(config["horizon_steps"]),
        rollout_dt=float(config["rollout_dt"]),
        sample_count=int(config["sample_count"]),
        iterations=int(config["iterations"]),
        progress_escape_enabled=False,
        predictive_human_cost=human_cost,
    )


def _constrained_planner(config: dict[str, Any]) -> CbfSafetyFilterPlannerWrapper:
    """Build the existing NMPC plus opt-in CBF composition."""

    nmpc = NMPCSocialPlannerAdapter(
        NMPCSocialConfig(
            horizon_steps=int(config["horizon_steps"]),
            rollout_dt=float(config["rollout_dt"]),
            solver_max_iterations=int(config["solver_max_iterations"]),
            fallback_to_stop=True,
        )
    )
    cbf = CbfSafetyFilterConfig(
        enabled=True,
        variant=str(config["cbf_variant"]),
        max_linear_speed=0.9,
        max_angular_speed=1.1,
    )
    return CbfSafetyFilterPlannerWrapper(nmpc, cbf)


def _build_smoke_planner(
    method_id: str,
    config: dict[str, Any],
    human_cost: PredictiveGaussianHumanCostConfig | None,
) -> Any:
    """Build one fresh planner for a smoke repeat."""

    if method_id == _METHOD_REFERENCE:
        return MPPISocialPlannerAdapter(
            _mppi_config(config, human_cost=PredictiveGaussianHumanCostConfig())
        )
    if method_id == _METHOD_PGIF:
        return MPPISocialPlannerAdapter(
            _mppi_config(config, human_cost=human_cost or PredictiveGaussianHumanCostConfig())
        )
    if method_id == _METHOD_CONSTRAINED:
        return _constrained_planner(config)
    raise ValueError(f"unknown smoke method {method_id!r}")


def _run_smoke(
    *,
    method_id: str,
    observation: dict[str, Any],
    config: dict[str, Any],
    human_cost: PredictiveGaussianHumanCostConfig | None = None,
) -> PlannerSmokeRecord:
    """Run one planner twice on a deep-copied fixture and record its result."""

    try:
        commands: list[tuple[float, float]] = []
        diagnostics: dict[str, Any] = {}
        started = time.perf_counter()
        for _ in range(2):
            planner = _build_smoke_planner(method_id, config, human_cost)
            command = planner.plan(copy.deepcopy(observation))
            command_tuple = (float(command[0]), float(command[1]))
            if not np.all(np.isfinite(command_tuple)):
                raise ValueError("planner emitted non-finite command")
            commands.append(command_tuple)
            diagnostics = dict(planner.diagnostics())
        runtime_ms = (time.perf_counter() - started) * 1000.0
        deterministic = bool(np.allclose(commands[0], commands[1], atol=1e-12, rtol=0.0))
        if not deterministic:
            raise ValueError("same-seed planner smoke was not deterministic")
        return PlannerSmokeRecord(
            method_id=method_id,
            status="smoke_pass",
            command=commands[0],
            repeat_command=commands[1],
            deterministic=deterministic,
            diagnostics=diagnostics,
            unavailable_metrics=dict(_UNAVAILABLE_METRICS),
            runtime_ms=runtime_ms,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic records failure explicitly
        return PlannerSmokeRecord(
            method_id=method_id,
            status="failed",
            command=None,
            repeat_command=None,
            deterministic=False,
            diagnostics={},
            unavailable_metrics=dict(_UNAVAILABLE_METRICS),
            failure_reason=f"{type(exc).__name__}: {exc}",
        )


def _build_method_cards(
    *,
    mppi_config: MPPISocialConfig,
    pgif_config: PredictiveGaussianHumanCostConfig,
    nmpc_config: NMPCSocialConfig,
    cbf_config: CbfSafetyFilterConfig,
) -> tuple[PlannerMethodCard, ...]:
    """Build the reference, PGIF-style, and constrained-MPC method cards."""

    common_observation = "structured SocNav robot, goal, and pedestrian position/velocity state"
    common_action = "unicycle command tuple (linear_speed_mps, angular_rate_rps)"
    return (
        PlannerMethodCard(
            method_id=_METHOD_REFERENCE,
            display_name="Existing social MPPI reference",
            planner_family="sampling_based_mppi",
            adapter_name="MPPISocialPlannerAdapter",
            observation_contract=common_observation,
            action_contract=common_action,
            source_reference="internal Robot SF reference planner",
            license_status="internal implementation",
            implementation_mode="native",
            benchmark_status="diagnostic_only",
            fallback_policy="No predictor fallback; no simulator metrics in this smoke.",
            claim_boundary="reference smoke only; not a benchmark result",
            config=asdict(mppi_config),
        ),
        PlannerMethodCard(
            method_id=_METHOD_PGIF,
            display_name="Predictive Gaussian human-cost MPPI adaptation",
            planner_family="sampling_based_mppi",
            adapter_name="MPPISocialPlannerAdapter + PredictiveGaussianHumanCost",
            observation_contract=common_observation,
            action_contract=common_action,
            source_reference="https://arxiv.org/abs/2608.08323; adaptation, not reproduction",
            license_status="no external code copied; source method reviewed as inspiration",
            implementation_mode="adapter",
            benchmark_status="diagnostic_only",
            fallback_policy="Disabled by default; malformed cost configuration fails closed.",
            claim_boundary="explicit Robot SF cost adaptation; no source-transfer or safety claim",
            config={"mppi": asdict(mppi_config), "predictive_human_cost": asdict(pgif_config)},
        ),
        PlannerMethodCard(
            method_id=_METHOD_CONSTRAINED,
            display_name="Internal NMPC plus collision-cone CBF",
            planner_family="constrained_mpc_with_cbf_filter",
            adapter_name="CbfSafetyFilterPlannerWrapper[NMPCSocialPlannerAdapter]",
            observation_contract=common_observation,
            action_contract=common_action,
            source_reference="internal Robot SF NMPC and CBF primitives; external method is conceptual only",
            license_status="internal implementation; no external code copied",
            implementation_mode="native",
            benchmark_status="diagnostic_only",
            fallback_policy="NMPC retains explicit stop fallback; CBF projection is recorded in diagnostics.",
            claim_boundary="one-step constrained-planner smoke; no safety certificate",
            config={"nmpc": asdict(nmpc_config), "cbf": asdict(cbf_config)},
        ),
    )


def run_diagnostic(config: dict[str, Any]) -> dict[str, Any]:
    """Run the config-first fixture and return a schema-ready report."""

    seed = int(config["seed"])
    scenario_id = str(config["scenario_id"])
    observation = _build_observation(
        seed=seed,
        pedestrian_count=int(config["pedestrian_count"]),
    )
    pgif_config = PredictiveGaussianHumanCostConfig(
        enabled=True,
        weight=float(config["pgif"]["weight"]),
        longitudinal_sigma_m=float(config["pgif"]["longitudinal_sigma_m"]),
        lateral_sigma_m=float(config["pgif"]["lateral_sigma_m"]),
        forward_speed_gain=float(config["pgif"]["forward_speed_gain"]),
    )
    mppi_config = _mppi_config(config, human_cost=PredictiveGaussianHumanCostConfig())
    nmpc_config = NMPCSocialConfig(
        horizon_steps=int(config["horizon_steps"]),
        rollout_dt=float(config["rollout_dt"]),
        solver_max_iterations=int(config["solver_max_iterations"]),
        fallback_to_stop=True,
    )
    cbf_config = CbfSafetyFilterConfig(
        enabled=True,
        variant=str(config["cbf_variant"]),
        max_linear_speed=0.9,
        max_angular_speed=1.1,
    )
    method_cards = _build_method_cards(
        mppi_config=mppi_config,
        pgif_config=pgif_config,
        nmpc_config=nmpc_config,
        cbf_config=cbf_config,
    )
    records = (
        _run_smoke(method_id=_METHOD_REFERENCE, observation=observation, config=config),
        _run_smoke(
            method_id=_METHOD_PGIF,
            observation=observation,
            config=config,
            human_cost=pgif_config,
        ),
        _run_smoke(method_id=_METHOD_CONSTRAINED, observation=observation, config=config),
    )
    return build_predictive_baseline_report(
        config=config,
        seed=seed,
        scenario_id=scenario_id,
        method_cards=method_cards,
        smoke_records=records,
    )


def _schema_path() -> Path:
    """Return the repository schema path from this script location."""

    return (
        Path(__file__).resolve().parents[2]
        / "robot_sf/benchmark/schemas/predictive_baseline_diagnostic.v1.json"
    )


def main() -> int:
    """Run the diagnostic CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _load_config(args.config)
    report = run_diagnostic(config)
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": "ok", "output": str(args.output), "config_digest": report["config_digest"]}
        )
    )
    return 0 if all(record["status"] == "smoke_pass" for record in report["smoke_records"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
