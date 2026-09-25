"""Pre-registered SNQI-v2 anchor derivation from the complete development split.

This module never chooses weights or evaluates planner rankings. It checks the
14-arm by 48-scenario by two-seed grid, selects F by the declared correlation rule,
and computes linear-interpolated episode p95 anchors without zero imputation.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import spearmanr

from robot_sf.benchmark.fallback_policy import (
    resolve_execution_mode,
    summarize_benchmark_availability,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.result_provenance import (
    load_result_provenance_manifest,
    manifest_path_for_result_jsonl,
    validate_result_provenance_manifest,
)
from robot_sf.benchmark.snqi.v2_reports import read_episode_files, validate_episode_execution
from robot_sf.benchmark.snqi.v2_spec import (
    PP_EQUIV_FORCE,
    SIMULATED_FORCE,
    finite_nonnegative,
)
from robot_sf.benchmark.utils import _config_hash
from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def derive_calibration_anchors(
    episodes: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[str],
    scenarios: Sequence[str],
    run_id: str,
    source_commit: str,
    episodes_sha256: str,
    expected_algorithms: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Derive the frozen asset only after complete source and split validation.

    Returns:
        A JSON-serializable anchor document ready for review and commit.
    """
    _validate_provenance(run_id, source_commit, episodes_sha256)
    expected = set(product(arms, scenarios, (101, 102)))
    if (
        len(arms) != 14
        or len(set(arms)) != 14
        or len(scenarios) != 48
        or len(set(scenarios)) != 48
        or len(episodes) != 1344
    ):
        raise ValueError("SNQI-v2 calibration requires 14 arms x48 scenarios x2 seeds =1344")
    observed = set()
    command_modes: dict[str, dict[str, int]] = {arm: {} for arm in arms}
    force, exposure, fractions = [], [], []
    for episode in episodes:
        identity = (episode.get("planner_key"), episode.get("scenario_id"), episode.get("seed"))
        if identity in observed or identity not in expected:
            raise ValueError(f"SNQI-v2 calibration duplicate or out-of-split identity: {identity}")
        observed.add(identity)
        _validate_calibration_episode(
            episode, expected_algorithm=(expected_algorithms or {}).get(identity[0])
        )
        mode = resolve_execution_mode(episode["algorithm_metadata"])
        counts = command_modes[identity[0]]
        counts[mode] = counts.get(mode, 0) + 1
        metrics = episode["metrics"]
        steps = finite_nonnegative(episode.get("steps"), "executed steps")
        near = finite_nonnegative(metrics.get("near_misses"), "near_misses")
        if steps < 1 or not steps.is_integer() or not near.is_integer() or near > steps:
            raise ValueError("SNQI-v2 calibration invalid close-clearance step coverage")
        force.append(finite_nonnegative(metrics.get(SIMULATED_FORCE), SIMULATED_FORCE))
        fractions.append(near / steps)
        exposure.append(min(near / steps / 0.25, 1.0))
    if observed != expected:
        raise ValueError("SNQI-v2 calibration grid incomplete")
    if len(set(force)) < 2 or len(set(exposure)) < 2:
        raise ValueError("SNQI-v2 F/N calibration correlation is undefined")
    rho = float(spearmanr(force, exposure).statistic)
    source = PP_EQUIV_FORCE if abs(rho) >= 0.90 else SIMULATED_FORCE
    anchors = {
        "T": {"lower": 0, "upper": 3, "type": "normative"},
        "N": {"lower": 0, "upper": 0.25, "type": "normative"},
    }
    for term, metric in (("F", source), ("J", "jerk_mean"), ("K", "curvature_mean")):
        values = [finite_nonnegative(ep["metrics"].get(metric), metric) for ep in episodes]
        upper = float(np.percentile(values, 95, method="linear"))
        if upper <= 0:
            raise ValueError(f"SNQI-v2 {term} calibration p95 is not positive")
        anchors[term] = {"lower": 0, "upper": upper, "type": "calibration_p95"}
    grid_hash = hashlib.sha256(
        json.dumps(sorted(expected), separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "version": "SNQI-v2.0",
        "status": "frozen",
        "anchors": anchors,
        "force_decision": {
            "source": source,
            "spearman_rho_F_N": rho,
            "spearman_rho_F_exposure_fraction": float(spearmanr(force, fractions).statistic),
            "threshold_absolute_rho": 0.90,
            "N": "clip(near_misses/steps/0.25)",
            "selected_source_coverage": len(episodes),
        },
        "calibration": {
            "split_id": f"snqi-v2-dev101-102-{grid_hash[:12]}",
            "run_id": run_id,
            "source_commit": source_commit,
            "episodes_sha256": episodes_sha256,
            "grid_sha256": grid_hash,
            "seeds": [101, 102],
            "episode_count": 1344,
            "arms": sorted(arms),
            "scenarios": sorted(scenarios),
            "benchmark_execution": "nonfallback",
            "command_mode_counts": command_modes,
            "quantile_method": "linear",
        },
    }


def freeze_campaign_anchors(
    campaign_root: Path, output_path: Path, *, campaign_config: Any | None = None
) -> dict[str, Any]:
    """Validate a completed development campaign and atomically write its anchor asset.

    Returns:
        Frozen anchor document, including hashes of every source episode file.
    """
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config  # noqa: PLC0415

    campaign_root = campaign_root.resolve()
    config = campaign_config or load_campaign_config(
        get_repository_root() / "configs/benchmarks/snqi_v2/calibration.dev101_102.yaml"
    )
    metadata_paths = [
        campaign_root / name
        for name in (
            "campaign_manifest.json",
            "reports/campaign_summary.json",
            "preflight/preview_scenarios.json",
        )
    ]
    snapshots = _snapshot_calibration_files(metadata_paths, campaign_root)
    manifest, summary, preview = [json.loads(path.read_text()) for path in metadata_paths]
    planners, canonical_scenarios = _bind_calibration_config(config, manifest, preview)
    input_paths = {
        Path(config.scenario_matrix_path),
        get_repository_root() / "robot_sf/benchmark/schemas/episode.schema.v1.json",
    }
    input_paths.update(
        Path(planner.algo_config_path)
        for planner in planners.values()
        if planner.algo_config_path is not None
    )
    snapshots.update({str(path): sha256_file(path) for path in input_paths})
    arms = list(planners)
    scenarios = list(canonical_scenarios)
    expected_algorithms = {key: planner.algo for key, planner in planners.items()}
    records, hashes, sidecar_hashes = [], {}, {}
    observed_arms = set()
    for entry in summary["runs"]:
        arm = entry.get("planner", {}).get("key")
        planner = planners.get(arm)
        if (
            planner is None
            or arm in observed_arms
            or entry["planner"].get("algo") != planner.algo
            or entry["planner"].get("kinematics") != "differential_drive"
        ):
            raise ValueError("SNQI-v2 calibration run arm disagrees with manifest/config")
        observed_arms.add(arm)
        availability = summarize_benchmark_availability(entry.get("summary"))
        if entry.get("status") != "ok" or not availability.benchmark_success:
            raise ValueError("SNQI-v2 calibration run has incomplete/fallback/degraded execution")
        relative = Path("runs") / f"{arm}__differential_drive" / "episodes.jsonl"
        declared = Path(entry["episodes_path"])
        if tuple(declared.parts[-3:]) != relative.parts:
            raise ValueError("SNQI-v2 calibration episode path is not bound to its arm")
        path = campaign_root / relative
        sidecar = manifest_path_for_result_jsonl(path)
        for artifact in (path, sidecar):
            _require_calibration_file(artifact, campaign_root)
        file_hash = sha256_file(path)
        sidecar_hash = sha256_file(sidecar)
        custody = _load_calibration_custody(
            path, planner, config, canonical_scenarios, manifest["git"]["commit"], file_hash
        )
        hashes[str(relative)] = file_hash
        sidecar_hashes[str(sidecar.relative_to(campaign_root))] = sidecar_hash
        snapshots[str(path)] = file_hash
        snapshots[str(sidecar)] = sidecar_hash
        records.extend(
            _read_calibration_arm(
                path, custody, planner, canonical_scenarios, manifest["git"]["commit"], config
            )
        )
    if observed_arms != set(planners):
        raise ValueError("SNQI-v2 calibration run inventory is incomplete")
    digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    document = derive_calibration_anchors(
        records,
        arms=arms,
        scenarios=scenarios,
        run_id=manifest["campaign_id"],
        source_commit=manifest["git"]["commit"],
        episodes_sha256=digest,
        expected_algorithms=expected_algorithms,
    )
    document["calibration"]["episode_files_sha256"] = hashes
    document["calibration"]["producer_sidecars_sha256"] = sidecar_hashes
    document["calibration"]["campaign_config_hash"] = manifest["config_hash"]
    document["calibration"]["episodes_hash_rule"] = (
        "sha256(sorted compact JSON relative-path-to-file-sha256 map)"
    )
    document["calibration"]["campaign_manifest_sha256"] = hashlib.sha256(
        (campaign_root / "campaign_manifest.json").read_bytes()
    ).hexdigest()
    if any(sha256_file(Path(path)) != digest for path, digest in snapshots.items()):
        raise ValueError("SNQI-v2 calibration custody changed during analysis")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output_path)
    return document


def _snapshot_calibration_files(paths: Sequence[Path], root: Path) -> dict[str, str]:
    """Snapshot safe metadata inputs for the post-analysis custody recheck.

    Returns:
        Absolute-path to streamed SHA256 mapping.
    """
    for path in paths:
        _require_calibration_file(path, root)
    return {str(path): sha256_file(path) for path in paths}


def _require_calibration_file(path: Path, root: Path) -> None:
    """Reject missing or symlinked custody inputs before reading or hashing them."""
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root)
        or any(parent.is_symlink() for parent in path.parents if parent != root)
    ):
        raise ValueError("SNQI-v2 calibration custody file is missing or unsafe")


def _bind_calibration_config(
    config: Any, manifest: Mapping[str, Any], preview: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind acquisition metadata to the caller's independent campaign configuration.

    Returns:
        Verified planner specs and canonical resolved scenarios.
    """
    from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios  # noqa: PLC0415
    from robot_sf.benchmark.camera_ready._preflight import _scenario_matrix_hash  # noqa: PLC0415
    from robot_sf.benchmark.camera_ready._util import _config_hash_payload  # noqa: PLC0415

    resolved = _load_campaign_scenarios(config)
    planners = {planner.key: planner for planner in config.planners if planner.enabled}
    declared = {arm["key"]: arm for arm in manifest["planners"] if arm["enabled"]}
    if (
        len(planners) != 14
        or len(resolved) != 48
        or set(declared) != set(planners)
        or len(declared) != len([arm for arm in manifest["planners"] if arm["enabled"]])
        or manifest.get("config_hash") != _config_hash(_config_hash_payload(config))
        or manifest.get("scenario_matrix_hash") != _scenario_matrix_hash(resolved)
        or list(config.kinematics_matrix) != ["differential_drive"]
        or manifest.get("kinematics_matrix") != ["differential_drive"]
        or tuple(config.seed_policy.seeds) != (101, 102)
        or manifest["seed_policy"]["resolved_seeds"] != [101, 102]
        or config.horizon != 600
        or config.dt != 0.1
        or any(declared[key].get("algo") != planner.algo for key, planner in planners.items())
    ):
        raise ValueError("SNQI-v2 calibration manifest/config binding mismatch")
    from robot_sf.benchmark.release_acceptance import _result_provenance_scenarios  # noqa: PLC0415

    effective = _result_provenance_scenarios(config, resolved, kinematics="differential_drive")
    scenarios = {scenario.get("name", scenario.get("id")): scenario for scenario in effective}
    if preview.get("truncated") or {scenario["name"] for scenario in preview["scenarios"]} != set(
        scenarios
    ):
        raise ValueError("SNQI-v2 calibration preview differs from canonical scenarios")
    return planners, scenarios


def _load_calibration_custody(
    path: Path, planner: Any, config: Any, scenarios: Mapping[str, Any], source: str, file_hash: str
) -> dict[str, Any]:
    """Verify producer sidecar schema, source, inputs and exact raw artifact binding.

    Returns:
        Validated sidecar with one row binding per expected development cell.
    """
    from robot_sf.benchmark.release_acceptance import _result_provenance_scenarios  # noqa: PLC0415

    payload = load_result_provenance_manifest(manifest_path_for_result_jsonl(path))
    validate_result_provenance_manifest(payload)
    identity = payload["campaign_identity"]
    inputs = payload["inputs"]
    expected_inputs = {
        "schema_path": get_repository_root() / "robot_sf/benchmark/schemas/episode.schema.v1.json",
        "scenario_matrix": config.scenario_matrix_path,
        "algo_config": planner.algo_config_path,
    }
    for role, expected_path in expected_inputs.items():
        item = inputs[role]
        if expected_path is None:
            if (
                item.get("artifact_status") != "not_provided"
                or item.get("path") is not None
                or item.get("sha256") is not None
            ):
                raise ValueError("SNQI-v2 calibration sidecar declares unexpected config input")
        elif item.get("sha256") != sha256_file(Path(expected_path)):
            raise ValueError("SNQI-v2 calibration sidecar input hash is not config-bound")
    expected_identity = _config_hash(
        {
            "schema_path": inputs["schema_path"]["path"],
            "algo": planner.algo,
            "algo_config_path": inputs["algo_config"].get("path"),
        }
    )
    if (
        not payload.get("input_binding_schema_version")
        or payload["run"].get("repo_commit") != source
        or payload["run"].get("runner") != "map_runner.run_map_batch"
        or payload["completeness"].get("status") != "complete"
        or identity.get("algorithm") != planner.algo
        or identity.get("config_hash") != expected_identity
        or identity.get("scenario_matrix_hash")
        != _config_hash(
            _result_provenance_scenarios(
                config, list(scenarios.values()), kinematics="differential_drive"
            )
        )
        or identity.get("written") != 96
        or identity.get("total_jobs") != 96
        or len(payload["rows"]) != 96
    ):
        raise ValueError("SNQI-v2 calibration producer sidecar identity is not bound")
    artifacts = [item for item in payload["raw_artifacts"] if item.get("kind") == "episodes_jsonl"]
    if (
        len(artifacts) != 1
        or artifacts[0].get("sha256") != file_hash
        or tuple(Path(artifacts[0].get("path", "")).parts[-3:]) != tuple(path.parts[-3:])
    ):
        raise ValueError("SNQI-v2 calibration producer sidecar raw artifact is stale/misrouted")
    return payload


def _read_calibration_arm(
    path: Path,
    custody: Mapping[str, Any],
    planner: Any,
    scenarios: Mapping[str, Any],
    source: str,
    config: Any,
) -> list[dict[str, Any]]:
    """Validate each raw producer row before retaining a compact calibration projection.

    Returns:
        The 96 compact rows for one independently declared arm.
    """
    context = _calibration_identity_context(config, planner, scenarios)
    episode_ids: set[str] = set()
    records = []
    for index, record in enumerate(read_episode_files([path])):
        _validate_calibration_row_custody(
            record, custody, index, planner, scenarios, source, identity_context=context
        )
        if record["episode_id"] in episode_ids:
            raise ValueError("SNQI-v2 calibration duplicate producer episode identity within arm")
        episode_ids.add(record["episode_id"])
        records.append(
            _compact_calibration_record(record, planner.key, expected_algorithm=planner.algo)
        )
        del record
    if len(records) != 96:
        raise ValueError("SNQI-v2 calibration producer arm must contain exactly96 rows")
    return records


def _calibration_identity_context(config: Any, planner: Any, scenarios: Mapping[str, Any]) -> Any:
    """Adapt independently bound campaign inputs to the producer's read-only identity builder.

    Only the resume identity and normalization helpers consume this data-only context. No
    simulator, policy runtime, output writer, or batch preflight is started. Defaults match
    camera-ready's run_map_batch call, including absent tracking/filter/track overrides.

    Returns:
        Normalized inputs for the producer resume identity helper.
    """
    from robot_sf.benchmark.camera_ready._util import (  # noqa: PLC0415
        _latency_stress_metadata,
        _synthetic_actuation_metadata,
    )
    from robot_sf.benchmark.camera_ready.campaign import (  # noqa: PLC0415
        _resolve_arm_safety_wrapper,
    )
    from robot_sf.benchmark.map_runner.map_runner import (  # noqa: PLC0415
        _normalize_batch_specs,
    )
    from robot_sf.benchmark.release_acceptance import (  # noqa: PLC0415
        _full_release_candidate_config,
    )

    path, policy_config, error = _full_release_candidate_config(
        planner_spec=planner,
        source_repository_root=get_repository_root(),
        allowed_scenario_ids=set(scenarios),
    )
    if error:
        raise ValueError(f"SNQI-v2 calibration canonical policy config is invalid: {error}")
    observation_mode = planner.observation_mode or config.observation_mode
    context = SimpleNamespace(
        algo=planner.algo,
        algo_config_path=str(path) if path is not None else None,
        raw_policy_cfg=policy_config,
        horizon=planner.horizon_override or config.horizon,
        dt=planner.dt_override or config.dt,
        record_forces=config.record_forces,
        batch_observation_mode=str(observation_mode).strip()
        if observation_mode is not None
        else None,
        observation_level=None,
        benchmark_track=None,
        track_schema_version=None,
        observation_noise=config.observation_noise,
        tracking_precision=None,
        synthetic_actuation_profile=_synthetic_actuation_metadata(
            config.synthetic_actuation_profile
        ),
        latency_stress_profile=_latency_stress_metadata(
            config.latency_stress_profile, dt=config.dt
        ),
        safety_wrapper=_resolve_arm_safety_wrapper(cfg=config, planner=planner),
        cbf_safety_filter=None,
        record_planner_decision_trace=config.record_planner_decision_trace,
        record_simulation_step_trace=config.record_simulation_step_trace,
    )
    _normalize_batch_specs(context)
    return context


def _validate_calibration_row_custody(
    record: Mapping[str, Any],
    custody: Mapping[str, Any],
    index: int,
    planner: Any,
    scenarios: Mapping[str, Any],
    source: str,
    *,
    identity_context: Any,
) -> None:
    """Reject a mismatched raw identity before assigning its verified containing arm."""
    from robot_sf.benchmark.map_runner.map_runner import (  # noqa: PLC0415
        _compute_resume_identity_payload,
    )
    from robot_sf.benchmark.map_runner.map_runner_identity import (  # noqa: PLC0415
        compute_map_episode_id,
    )
    from robot_sf.benchmark.release_acceptance import (  # noqa: PLC0415
        _full_release_effective_algorithm,
        _full_release_row_contract_blockers,
    )

    scenario = scenarios.get(record.get("scenario_id"))
    if scenario is None or index >= len(custody["rows"]):
        raise ValueError("SNQI-v2 calibration row is outside canonical scenario grid")
    expected_algo, error = _full_release_effective_algorithm(
        planner_spec=planner,
        base_algorithm=planner.algo,
        scenario=scenario,
        allowed_scenario_ids=set(scenarios),
    )
    blockers = _full_release_row_contract_blockers(
        record, prefix="calibration row", expected_algo=expected_algo or planner.algo
    )
    params = record.get("scenario_params")
    if error or blockers or not isinstance(params, dict):
        raise ValueError(f"SNQI-v2 calibration raw row contract mismatch: {error or blockers}")
    if (
        record.get("git_hash") != source
        or record.get("config_hash") != _config_hash(params)
        or ("planner_key" in record and record["planner_key"] != planner.key)
        or ("kinematics" in record and record["kinematics"] != "differential_drive")
    ):
        raise ValueError("SNQI-v2 calibration raw row source/config/arm mismatch")
    seed = record.get("seed")
    if type(seed) is not int or seed not in {101, 102}:
        raise ValueError("SNQI-v2 calibration row seed is outside canonical development split")
    expected_params = _compute_resume_identity_payload(identity_context, dict(scenario), seed)
    # Exact JSON identity rejects extra run-shaping keys and bool/numeric substitutions.
    if json.dumps(params, sort_keys=True) != json.dumps(expected_params, sort_keys=True):
        raise ValueError(
            "SNQI-v2 calibration row scenario config differs from canonical producer identity"
        )
    if record.get("episode_id") != compute_map_episode_id(expected_params, seed):
        raise ValueError(
            "SNQI-v2 calibration row episode identity differs from canonical producer identity"
        )
    bound = custody["rows"][index]
    provenance = record["result_provenance"]
    artifact = next(item for item in custody["raw_artifacts"] if item["kind"] == "episodes_jsonl")
    expected = {
        "episode_id": record.get("episode_id"),
        "scenario_id": record.get("scenario_id"),
        "seed": record.get("seed"),
        "config_hash": record.get("config_hash"),
        "repo_commit": source,
        "jsonl_line": index,
    }
    if (
        any(bound.get(key) != value for key, value in expected.items())
        or any(
            provenance.get(key) != expected[key]
            for key in ("scenario_id", "seed", "config_hash", "repo_commit")
        )
        or any(
            provenance[key] != expected[key]
            for key in ("episode_id", "jsonl_line")
            if key in provenance
        )
        or bound.get("raw_artifact") != artifact["path"]
        or ("raw_artifact" in provenance and provenance["raw_artifact"] != artifact["path"])
        or any(
            bound.get("simulator_settings", {}).get(key) != value
            or provenance.get("simulator_settings", {}).get(key) != value
            for key, value in (("horizon", 600), ("dt", 0.1), ("record_forces", True))
        )
    ):
        raise ValueError("SNQI-v2 calibration producer sidecar row binding mismatch")


def _compact_calibration_record(
    record: Mapping[str, Any], arm: str, *, expected_algorithm: str | None = None
) -> dict[str, Any]:
    """Validate raw execution before retaining only scalar calibration inputs.

    Force samples and simulation traces remain in the hashed source files; their
    decoded payloads must not accumulate over the complete development grid.

    Returns:
        Validated scalar inputs with the arm identity and actual command mode.
    """
    _validate_calibration_episode(record, expected_algorithm=expected_algorithm)
    metrics = record["metrics"]
    return {
        **{key: record.get(key) for key in ("scenario_id", "seed", "status", "horizon", "steps")},
        "planner_key": arm,
        "scenario_params": {
            key: record["scenario_params"][key]
            for key in ("run_horizon", "run_dt", "record_forces")
        },
        "algorithm_metadata": {
            "execution_mode": resolve_execution_mode(record["algorithm_metadata"])
        },
        "metrics": {
            **{
                key: metrics.get(key)
                for key in (
                    SIMULATED_FORCE,
                    PP_EQUIV_FORCE,
                    "near_misses",
                    "jerk_mean",
                    "curvature_mean",
                )
            },
            "robot_force_metadata": {
                "sample_timing": metrics["robot_force_metadata"]["sample_timing"]
            },
        },
    }


def _validate_provenance(run_id: str, source_commit: str, episodes_sha256: str) -> None:
    """Require a named run and exact hexadecimal source/input identities."""
    for label, value, length in (
        ("source_commit", source_commit, 40),
        ("episodes_sha256", episodes_sha256, 64),
    ):
        if len(value) != length or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"SNQI-v2 invalid calibration {label}")
    if not run_id:
        raise ValueError("SNQI-v2 calibration run_id is required")


def _validate_calibration_episode(
    episode: Mapping[str, Any], *, expected_algorithm: str | None = None
) -> None:
    """Require frozen acquisition settings and declared, nonfallback planner execution."""
    validate_episode_execution(episode, expected_algorithm=expected_algorithm)
    if episode.get("status") not in {"success", "collision", "failure"}:
        raise ValueError("SNQI-v2 calibration rejects invalid episode execution status")
    mode = resolve_execution_mode(episode.get("algorithm_metadata"))
    if mode not in {"native", "adapter", "mixed"}:
        raise ValueError(
            "SNQI-v2 calibration requires explicit native, adapter or mixed command mode"
        )
    params = episode.get("scenario_params", {})
    if (
        episode.get("horizon") != 600
        or params.get("run_horizon") != 600
        or params.get("run_dt") != 0.1
        or params.get("record_forces") is not True
    ):
        raise ValueError("SNQI-v2 calibration requires H600/dt0.1 with recorded forces")
    metadata = episode.get("metrics", {}).get("robot_force_metadata", {})
    if metadata.get("sample_timing") != "pre_integration":
        raise ValueError("SNQI-v2 calibration requires pre-integration robot force samples")
