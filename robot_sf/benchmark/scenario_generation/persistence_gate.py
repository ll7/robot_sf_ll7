"""Evidence gate between generated candidates and reusable critical scenarios.

This module adds the ``generated_scenario_persistence.v1`` evidence record that
the stage-1 generation pipeline does not yet produce.  A candidate is promoted
only when three independent status checks all pass:

1. ``exact_replay`` - byte/config-equivalent replay of the source episode;
2. ``critical_event_reproduced`` - the critical event under the source planner;
3. ``perturbation_persistence`` - the event survives a preregistered perturbation grid.

The three statuses remain separate.  A ``pass`` in one does not imply a ``pass``
in another, and the promotion verdict fails closed on any ``fail``/``unknown``
required status, on missing trace fields, on replay divergence, and on an
unregistered (non-frozen) configuration.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

from robot_sf.benchmark.step_trace_comparator import (
    canonical_step_trace_digest,
    find_first_trace_difference,
)
from robot_sf.errors import RobotSfError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PERSISTENCE_SCHEMA_VERSION = "generated_scenario_persistence.v1"
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "generated_scenario_persistence.v1.json"
)
_SCHEMA_VALIDATOR: Draft202012Validator | None = None
_REQUIRED_EPISODE_FIELDS = ("episode_id", "source_seed", "source_map")

PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"

REQUIRED_STATUSES = ("exact_replay", "critical_event_reproduced", "perturbation_persistence")


class ScenarioPersistenceValidationError(RobotSfError, ValueError):
    """Raised when a persistence record is incomplete or internally inconsistent."""


def load_persistence_schema() -> dict[str, Any]:
    """Return the versioned JSON Schema for persistence records."""

    with _SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        return json.load(schema_file)


def _stable_digest(payload: Any) -> str:
    """Return a short, order-independent SHA-256 digest for a JSON-safe payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_schema(record: Mapping[str, Any]) -> None:
    """Validate ``record`` against the cached persistence JSON Schema.

    Lazily builds the validator and raises one combined error listing every
    violation, sorted by path.
    """

    global _SCHEMA_VALIDATOR
    if _SCHEMA_VALIDATOR is None:
        _SCHEMA_VALIDATOR = Draft202012Validator(load_persistence_schema())
    validator = _SCHEMA_VALIDATOR
    if record.get("schema_version") == "generated_scenario_persistence.v2":
        validator = Draft202012Validator(
            json.loads((_SCHEMA_PATH.parent / "generated_scenario_persistence.v2.json").read_text())
        )
    errors = sorted(
        validator.iter_errors(dict(record)),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        formatted = "; ".join(
            f"/{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
            for error in errors
        )
        raise ScenarioPersistenceValidationError(formatted)


def compute_persistence_record(
    *,
    scenario_id: str,
    source_episode: Mapping[str, Any],
    generated_scenario: Mapping[str, Any],
    planner: str,
    seed: int,
    config: Mapping[str, Any],
    commit_hashes: Mapping[str, Any],
    exact_replay: Mapping[str, Any],
    critical_event_reproduced: Mapping[str, Any],
    perturbation_grid: Mapping[str, Any],
    cell_verdicts: Sequence[Mapping[str, Any]],
    missing_cell_reasons: Sequence[Mapping[str, Any]] | None = None,
    replay_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a schema-valid ``generated_scenario_persistence.v1`` record.

    The three status blocks are supplied independently.  This function validates
    the JSON Schema contract, derives the persistence summary from the supplied
    cells, and computes the fail-closed promotion verdict.

    Args:
        scenario_id: Reusable-scenario identifier.
        source_episode: episode_id, source_seed, source_map, replay_digest.
        generated_scenario: catalog_schema_version, scenario_id, optional
            catalog_entry_digest.
        planner: Source planner name used for reproduction.
        seed: Episode/run seed.
        config: Frozen perturbation/promotion configuration (config_id, frozen,
            optional config_hash).
        commit_hashes: code and config commit hashes.
        exact_replay: status, divergence_reason, replay_digest.
        critical_event_reproduced: status, event_type, tolerances, observed deltas.
        perturbation_grid: timing_offsets_s and speed_deltas_m_s lists.
        cell_verdicts: One verdict per preregistered grid cell.
        missing_cell_reasons: Optional per-cell exclusion reasons.
        replay_scope: Explicit verified window scope; emits persistence v2 when supplied.

    Returns:
        A schema-valid persistence record with the promotion verdict filled in.

    Raises:
        ScenarioPersistenceValidationError: If the record cannot be built or
            would violate the fail-closed contract.
    """

    missing_cell_reasons = list(missing_cell_reasons or [])
    if not config.get("frozen"):
        raise ScenarioPersistenceValidationError(
            "config.frozen must be true: unfrozen perturbation config cannot gate promotion"
        )

    pass_count = sum(1 for cell in cell_verdicts if cell.get("verdict") == PASS)
    total_cells = len(cell_verdicts)
    persistence_rate = (pass_count / total_cells) if total_cells else 0.0
    if total_cells:
        rates = [pass_count / total_cells]
    else:
        rates = []
    persistence_interval: list[float] = [min(rates), max(rates)] if rates else [0.0, 0.0]

    record: dict[str, Any] = {
        "schema_version": PERSISTENCE_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "source_episode": dict(source_episode),
        "generated_scenario": dict(generated_scenario),
        "planner": planner,
        "seed": int(seed),
        "config": dict(config),
        "commit_hashes": dict(commit_hashes),
        "exact_replay": dict(exact_replay),
        "critical_event_reproduced": dict(critical_event_reproduced),
        "perturbation_persistence": {
            "grid": dict(perturbation_grid),
            "cells": [dict(cell) for cell in cell_verdicts],
            "persistence_rate": persistence_rate,
            "persistence_interval": persistence_interval,
            "missing_cell_reasons": [dict(reason) for reason in missing_cell_reasons],
        },
        "promotion": _promotion_verdict(
            exact_replay=exact_replay,
            critical_event_reproduced=critical_event_reproduced,
            cell_verdicts=cell_verdicts,
            missing_cell_reasons=missing_cell_reasons,
        ),
    }
    if replay_scope is not None:
        record["schema_version"] = "generated_scenario_persistence.v2"
        record["replay_scope"] = dict(replay_scope)
    _verify_schema(record)
    return record


def _promotion_verdict(
    *,
    exact_replay: Mapping[str, Any],
    critical_event_reproduced: Mapping[str, Any],
    cell_verdicts: Sequence[Mapping[str, Any]],
    missing_cell_reasons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the fail-closed promotion block from the three independent checks."""

    reasons: list[str] = []

    if exact_replay.get("status") != PASS:
        reasons.append(
            f"exact_replay:{exact_replay.get('status')}:{exact_replay.get('divergence_reason')}"
        )
    if critical_event_reproduced.get("status") != PASS:
        reasons.append(
            f"critical_event_reproduced:{critical_event_reproduced.get('status')}:"
            f"event_type={critical_event_reproduced.get('event_type')}"
        )

    cell_verdicts = list(cell_verdicts)
    if not cell_verdicts:
        reasons.append("perturbation_persistence:no_cells:empty preregistered grid")
    else:
        for cell in cell_verdicts:
            if cell.get("verdict") != PASS:
                reasons.append(
                    f"perturbation_cell:{cell.get('timing_offset_s')}:"
                    f"{cell.get('speed_delta_m_s')}:{cell.get('verdict')}"
                )
    if missing_cell_reasons:
        reasons.append(f"perturbation_persistence:missing_cells:{len(missing_cell_reasons)}")

    if reasons:
        return {
            "verdict": "reject",
            "exclusion_reason": "; ".join(reasons),
            "required_statuses": list(REQUIRED_STATUSES),
        }
    return {
        "verdict": "promote",
        "exclusion_reason": "all three independent status checks passed",
        "required_statuses": list(REQUIRED_STATUSES),
    }


def assess_exact_replay(
    source_episode: Mapping[str, Any],
    *,
    replayed_episode: Mapping[str, Any] | None = None,
    replay_error: str | None = None,
) -> dict[str, Any]:
    """Compare a replayed episode to its source and return an exact-replay block.

    ``replayed_episode`` is the byte/config-equivalent replay of
    ``source_episode``.  When it is missing or diverges, the block records the
    concrete reason and a ``fail``/``unknown`` status.  The digest is computed
    over the deterministic identity of the source episode so identical inputs
    produce identical digests.

    Returns:
        An ``exact_replay`` block with status, divergence_reason, and digest.
    """

    source_missing = _missing_episode_fields(source_episode)
    source_digest = _episode_digest(source_episode)
    if source_missing:
        return {
            "status": FAIL,
            "divergence_reason": f"source episode missing required fields: {source_missing}",
            "replay_digest": source_digest,
        }
    if replayed_episode is None:
        status = UNKNOWN if replay_error is None else FAIL
        reason = replay_error or "replay not executed"
        return {
            "status": status,
            "divergence_reason": reason,
            "replay_digest": source_digest,
        }
    replay_missing = _missing_episode_fields(replayed_episode)
    replay_digest = _episode_digest(replayed_episode)
    if replay_missing:
        return {
            "status": FAIL,
            "divergence_reason": f"replayed episode missing required fields: {replay_missing}",
            "replay_digest": source_digest,
        }
    if replay_digest != source_digest:
        return {
            "status": FAIL,
            "divergence_reason": (
                f"replay digest mismatch: source={source_digest} replay={replay_digest}"
            ),
            "replay_digest": source_digest,
        }
    return {
        "status": PASS,
        "divergence_reason": "byte/config-equivalent replay matched source episode",
        "replay_digest": source_digest,
    }


def _missing_episode_fields(episode: Mapping[str, Any]) -> list[str]:
    """Return required episode identity fields that are absent or empty."""
    return [
        field
        for field in _REQUIRED_EPISODE_FIELDS
        if field not in episode
        or episode[field] is None
        or (isinstance(episode[field], str) and not episode[field].strip())
    ]


def _episode_digest(episode: Mapping[str, Any]) -> str:
    """Digest only the exact-replay identity fields.

    Returns:
        The deterministic identity digest.
    """
    return _stable_digest({field: episode.get(field) for field in _REQUIRED_EPISODE_FIELDS})


def extract_episode_steps(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return detailed frames, allowing the native integer episode step count."""
    steps = episode.get("steps")
    if not isinstance(steps, list):
        meta = episode.get("algorithm_metadata")
        trace = meta.get("simulation_step_trace") if isinstance(meta, dict) else None
        steps = trace.get("steps") if isinstance(trace, dict) else None
    return steps if isinstance(steps, list) else []


def resolve_bound_window_artifact(root: Path, binding: Mapping[str, str]) -> Path:
    """Return a verified repository-relative artifact without path escape."""
    relative = Path(binding["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ScenarioPersistenceValidationError(
            "window artifact must be a file under receipt root"
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
        raise ScenarioPersistenceValidationError(f"window artifact digest mismatch: {relative}")
    return path


def _read_bound_json(root: Path, binding: Mapping[str, str]) -> dict[str, Any]:
    """Return one hash-bound JSON or compressed native JSONL episode."""
    path = resolve_bound_window_artifact(root, binding)
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ScenarioPersistenceValidationError("window artifact must contain one object")
    return value


def _window_event(steps: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> dict[str, Any]:
    """Return the recorded near-miss/contact from geometry and source radii."""
    radii = source["effective_physics"]["clearance_definition"]
    radius = radii["robot_radius_m"] + radii["pedestrian_radius_m"]
    threshold = source["metric_parameters"]["threshold_profile"]["near_miss_distance_m"]
    clearances = [
        min(math.dist(s["robot"]["position"], p["position"]) for p in s["pedestrians"]) - radius
        for s in steps
    ]
    near = [s for s, c in zip(steps, clearances, strict=True) if 0 <= c < threshold]
    contacts = [s for s in steps if any(s.get("collision", {}).values())]
    minimum = steps[clearances.index(min(clearances))]
    return {
        "near_miss_samples": len(near),
        "first_near_miss_step": near[0]["step"] if near else None,
        "first_contact_step": contacts[0]["step"] if contacts else None,
        "min_clearance_m": min(clearances),
        "time_s": minimum["time_s"],
        "location": minimum["robot"]["position"],
    }


def _verify_checkpoint_runtime_code(
    root: Path, receipt: dict[str, Any], checkpoint: dict[str, Any]
) -> None:
    """Bind the supported checkpoint inventory to recorded and current runtime code."""
    code_files = [
        "robot_sf/benchmark/scenario_generation/continuation.py",
        "robot_sf/benchmark/map_runner/map_runner_episode.py",
        "robot_sf/planner/socnav_social_force.py",
        "robot_sf/planner/socnav.py",
        "robot_sf/gym_env/robot_env.py",
        "robot_sf/robot/robot_state.py",
    ]
    for revision in receipt["parity_proof"]["replay_commits"]:
        code = {}
        for name in code_files:
            historical = subprocess.run(
                ["git", "show", f"{revision}:{name}"], cwd=root, capture_output=True, check=False
            )
            if historical.returncode:
                raise ScenarioPersistenceValidationError("window replay code revision unavailable")
            code[name] = hashlib.sha256(historical.stdout).hexdigest()
            if code[name] != hashlib.sha256((root / name).read_bytes()).hexdigest():
                raise ScenarioPersistenceValidationError("window runtime code changed since replay")
        fingerprint = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
        if fingerprint != checkpoint["compatibility"]["code_revision"]:
            raise ScenarioPersistenceValidationError(
                "window checkpoint/replay code binding mismatch"
            )


def _verify_model_bindings(root: Path, receipt: dict[str, Any]) -> None:
    """Verify supported model/config/map bytes against their recorded source revision."""
    planner = receipt["planner"]
    bindings = receipt["bindings"]
    model = planner["model"]
    if (
        planner["id"] != "social_force"
        or model["kind"] != "analytic"
        or set(model["files"])
        != {
            "robot_sf/baselines/social_force.py",
            "robot_sf/planner/socnav.py",
            "robot_sf/planner/socnav_social_force.py",
        }
    ):
        raise ScenarioPersistenceValidationError(
            "window receipt v1 supports the analytic social-force model"
        )
    for name, digest in model["files"].items():
        resolve_bound_window_artifact(root, {"path": name, "sha256": digest})
    # Visible input bytes alone do not prove their source-revision provenance.
    for binding in [
        bindings["map"],
        bindings["planner_config"],
        *[{"path": n, "sha256": h} for n, h in model["files"].items()],
    ]:
        historical = subprocess.run(
            ["git", "show", f"{receipt['source_commit']}:{binding['path']}"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if (
            historical.returncode
            or hashlib.sha256(historical.stdout).hexdigest() != binding["sha256"]
        ):
            raise ScenarioPersistenceValidationError(
                "window input is not bound to its source commit"
            )
    model_digest = (
        _stable_digest(model["files"])
        if model["kind"] == "analytic"
        else next(iter(model["files"].values()))
        if len(model["files"]) == 1
        else None
    )
    if model_digest != model["digest"]:
        raise ScenarioPersistenceValidationError("window model digest mismatch")


def _verify_window_runs(
    root: Path, receipt: dict[str, Any], source: dict[str, Any], expected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return both native replays after binding inputs, run identities and every channel."""
    bindings = receipt["bindings"]
    planner = receipt["planner"]
    w = receipt["window"]
    proof = receipt["parity_proof"]
    replays = [_read_bound_json(root, b) for b in bindings["replays"]]
    manifests = [_read_bound_json(root, b) for b in bindings["replay_manifests"]]
    if bindings["replays"][0] == bindings["replays"][1]:
        raise ScenarioPersistenceValidationError("window requires two distinct replay artifacts")
    if manifests[0]["run"]["run_id"] == manifests[1]["run"]["run_id"]:
        raise ScenarioPersistenceValidationError(
            "window requires independent replay run identities"
        )
    for index, row in enumerate([source, *replays]):
        if (
            row.get("algo") != planner["id"]
            or row["algorithm_metadata"].get("config_hash") != planner["effective_config_hash"]
            or row["algorithm_metadata"].get("status") != "ok"
            or row.get("integrity", {}).get("effective_view", {}).get("degraded") is not False
        ):
            raise ScenarioPersistenceValidationError("window planner/config/execution mismatch")
        if index == 0:
            continue
        manifest = manifests[index - 1]
        inputs = manifest["inputs"]
        if (
            inputs["scenario_matrix"]["sha256"] != bindings["scenario"]["sha256"]
            or inputs["algo_config"]["sha256"] != bindings["planner_config"]["sha256"]
            or manifest["run"]["repo_commit"] != row["git_hash"]
            or manifest["rows"][0]["episode_id"] != row["episode_id"]
            or not any(
                r.get("role") == "map_file" and r.get("sha256") == bindings["map"]["sha256"]
                for r in row.get("runtime_input_records", [])
            )
        ):
            raise ScenarioPersistenceValidationError(
                "window replay input-manifest binding mismatch"
            )
        continuation = row["algorithm_metadata"].get("generated_window_continuation", {})
        if (
            row["seed"] != receipt["seed"]
            or row["git_hash"] != proof["replay_commits"][index - 1]
            or continuation.get("checkpoint_sha256") != bindings["checkpoint"]["sha256"]
            or continuation.get("source_episode_id") != source["episode_id"]
            or continuation.get("start_step") != w["start_step"]
            or continuation.get("window_steps") != w["steps"]
        ):
            raise ScenarioPersistenceValidationError("window replay provenance mismatch")
        actual = extract_episode_steps(row)
        difference = find_first_trace_difference(expected, actual)
        if difference is not None:
            raise ScenarioPersistenceValidationError(f"window replay parity failed: {difference}")
    return replays


def verify_window_replay_receipt(receipt_path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Verify a source-bound window from actual artifacts, not an asserted pass.

    Two distinct replay records must reproduce every source-window channel.
    Episode IDs remain distinct; this never asserts full-episode termination or
    perturbation persistence. Missing/bad bindings or parity raise before admission.

    Returns:
        Verified receipt plus independent exact-replay and critical-event blocks.
    """
    import yaml  # noqa: PLC0415 - window materialization dependency only

    from robot_sf.benchmark.scenario_generation.catalog_schema import validate_catalog_entry  # noqa: PLC0415
    from robot_sf.benchmark.scenario_generation.continuation import read_window_checkpoint  # noqa: PLC0415

    root = repo_root.resolve()
    receipt = json.loads(receipt_path.read_text())
    schema = json.loads((_SCHEMA_PATH.parent / "source_bound_window_replay.v1.json").read_text())
    errors = list(Draft202012Validator(schema).iter_errors(receipt))
    if errors:
        raise ScenarioPersistenceValidationError(
            f"window parity proof/schema invalid: {errors[0].message}"
        )
    bindings = receipt["bindings"]
    source = _read_bound_json(root, bindings["source_episode"])
    entry = _read_bound_json(root, bindings["catalog_entry"])
    validate_catalog_entry(entry)
    scenario_path = resolve_bound_window_artifact(root, bindings["scenario"])
    document = yaml.safe_load(scenario_path.read_text())
    if not isinstance(document, dict) or len(document.get("scenarios", [])) != 1:
        raise ScenarioPersistenceValidationError("window scenario must contain exactly one case")
    scenario = document["scenarios"][0]
    checkpoint_path = resolve_bound_window_artifact(root, bindings["checkpoint"])
    checkpoint = read_window_checkpoint(
        checkpoint_path, expected_sha256=bindings["checkpoint"]["sha256"]
    )
    map_path = resolve_bound_window_artifact(root, bindings["map"])
    resolve_bound_window_artifact(root, bindings["planner_config"])
    runtime = scenario.get("generated_replay", {})
    w = receipt["window"]
    source_identity = entry["source_episode"]
    expected_runtime = {
        "schema_version": "robot_sf.generated_replay_runtime.v2",
        "checkpoint_file": runtime.get("checkpoint_file"),
        "checkpoint_sha256": bindings["checkpoint"]["sha256"],
        "start_step": w["start_step"],
        "window_steps": w["steps"],
    }
    checks = [
        source["git_hash"] == receipt["source_commit"],
        source["config_hash"] == receipt["source_config_hash"],
        source["seed"]
        == receipt["seed"]
        == source_identity["source_seed"]
        == checkpoint["source_seed"],
        source["episode_id"] == source_identity["episode_id"] == checkpoint["source_episode_id"],
        bindings["map"]["path"] == source_identity["source_map"],
        checkpoint["compatibility"]["map_sha256"] == bindings["map"]["sha256"],
        source.get("selected_map_identity", {}).get("sha256") == bindings["map"]["sha256"],
        runtime == expected_runtime,
        (scenario_path.parent / str(runtime.get("checkpoint_file"))).resolve() == checkpoint_path,
        (scenario_path.parent / str(scenario.get("map_file"))).resolve() == map_path,
        scenario["seeds"] == [receipt["seed"]],
        checkpoint["start_step"] == w["start_step"],
        checkpoint["window_steps"] == w["steps"],
        checkpoint["compatibility"]["dt_s"] == w["dt_s"],
        math.isclose(w["start_s"], w["start_step"] * w["dt_s"], abs_tol=1e-9),
        math.isclose(w["end_s"], (w["start_step"] + w["steps"]) * w["dt_s"], abs_tol=1e-9),
        math.isclose(entry["segment"]["window_start_s"], w["start_s"], abs_tol=1e-9),
        math.isclose(entry["segment"]["window_end_s"], w["end_s"], abs_tol=1e-9),
        scenario["simulation_config"]["max_episode_steps"] == checkpoint["original_budget_steps"],
    ]
    if not all(checks):
        raise ScenarioPersistenceValidationError(
            "window source/scenario/checkpoint binding mismatch"
        )
    _verify_checkpoint_runtime_code(root, receipt, checkpoint)
    _verify_model_bindings(root, receipt)
    trace = extract_episode_steps(source)
    expected = trace[w["start_step"] : w["start_step"] + w["steps"]]
    proof = receipt["parity_proof"]
    digest = canonical_step_trace_digest(expected)
    if (
        len(expected) != w["steps"]
        or digest != proof["source_window_trace_sha256"]
        or canonical_step_trace_digest(trace) != proof["source_full_trace_sha256"]
    ):
        raise ScenarioPersistenceValidationError("window source trace/bounds mismatch")
    if [s["step"] for s in expected] != list(
        range(w["start_step"], w["start_step"] + w["steps"])
    ) or not math.isclose(expected[-1]["time_s"], w["end_s"], abs_tol=1e-9):
        raise ScenarioPersistenceValidationError("window absolute step/time mismatch")
    replays = _verify_window_runs(root, receipt, source, expected)
    event = _window_event(expected, source)
    if not event["near_miss_samples"] and event["first_contact_step"] is None:
        raise ScenarioPersistenceValidationError("window has no verified critical event")
    critical = assess_critical_event_reproduction(
        event_type="min_clearance",
        source_event_time_s=event["time_s"],
        source_event_location=event["location"],
        replayed_event_time_s=event["time_s"],
        replayed_event_location=event["location"],
    )
    return {
        "receipt": receipt,
        "original_budget_steps": checkpoint["original_budget_steps"],
        "source_scenario_id": source["scenario_id"],
        "scenario_id": entry["scenario_id"],
        "source_episode_id": source["episode_id"],
        "replay_scope": {
            "kind": "source_bound_window",
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "window": w,
            "source_episode_id": source["episode_id"],
            "replay_episode_ids": [r["episode_id"] for r in replays],
        },
        "replay_episode_ids": [r["episode_id"] for r in replays],
        "exact_replay": {
            "status": PASS,
            "divergence_reason": "two source-bound window traces matched",
            "replay_digest": digest,
        },
        "critical_event_reproduced": critical,
        "event": event,
    }


def assess_critical_event_reproduction(
    *,
    event_type: str,
    source_event_time_s: float,
    source_event_location: Sequence[float],
    replayed_event_time_s: float | None = None,
    replayed_event_location: Sequence[float] | None = None,
    time_tolerance_s: float = 0.0,
    location_tolerance_m: float = 0.0,
    not_observed_reason: str | None = None,
) -> dict[str, Any]:
    """Return a critical-event-reproduction block from source vs replayed event.

    The event is reproduced when an event of ``event_type`` is observed in the
    replayed episode within the declared time and location tolerances.  Missing
    replay fields or an out-of-tolerance observation fail closed.

    Returns:
        A ``critical_event_reproduced`` block with status and observed deltas.
    """

    if replayed_event_time_s is None or replayed_event_location is None:
        return {
            "status": UNKNOWN if not_observed_reason is None else FAIL,
            "event_type": event_type,
            "time_tolerance_s": float(time_tolerance_s),
            "location_tolerance_m": float(location_tolerance_m),
            "observed_time_delta_s": None,
            "observed_location_delta_m": None,
        }

    time_delta = abs(float(replayed_event_time_s) - float(source_event_time_s))
    location_delta = float(
        math.dist(
            [float(value) for value in replayed_event_location],
            [float(value) for value in source_event_location],
        )
    )
    reproduced = time_delta <= float(time_tolerance_s) and location_delta <= float(
        location_tolerance_m
    )
    return {
        "status": PASS if reproduced else FAIL,
        "event_type": event_type,
        "time_tolerance_s": float(time_tolerance_s),
        "location_tolerance_m": float(location_tolerance_m),
        "observed_time_delta_s": time_delta,
        "observed_location_delta_m": location_delta,
    }


def evaluate_perturbation_grid(
    *,
    timing_offsets_s: Sequence[float],
    speed_deltas_m_s: Sequence[float],
    cell_verdict_fn: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run a preregistered grid and return per-cell verdicts and missing reasons.

    ``cell_verdict_fn`` receives ``(timing_offset_s, speed_delta_m_s)`` and must
    return either ``{"verdict": "pass"|"fail", "reason": <str>}`` or
    ``None`` when the cell cannot be evaluated (recorded as a missing-cell
    reason rather than a silent pass).

    Returns:
        The cells list and the missing-cell-reasons list.
    """

    cells: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for timing_offset_s in timing_offsets_s:
        for speed_delta_m_s in speed_deltas_m_s:
            result = cell_verdict_fn(
                timing_offset_s=float(timing_offset_s),
                speed_delta_m_s=float(speed_delta_m_s),
            )
            if result is None:
                missing.append(
                    {
                        "timing_offset_s": float(timing_offset_s),
                        "speed_delta_m_s": float(speed_delta_m_s),
                        "reason": "cell_not_evaluable",
                    }
                )
                continue
            verdict = result.get("verdict", FAIL)
            cells.append(
                {
                    "timing_offset_s": float(timing_offset_s),
                    "speed_delta_m_s": float(speed_delta_m_s),
                    "verdict": verdict,
                    "reason": str(result.get("reason", "")),
                }
            )
    return cells, missing


def validate_persistence_record(record: Mapping[str, Any]) -> None:
    """Fail closed unless *record* is a valid, internally consistent persistence entry.

    Raises:
        ScenarioPersistenceValidationError: If JSON Schema or the fail-closed
            promotion invariants reject the record.
    """

    _verify_schema(record)
    if not record["config"].get("frozen"):
        raise ScenarioPersistenceValidationError("config.frozen must be true")
    promotion = record["promotion"]
    expected_statuses = set(REQUIRED_STATUSES)
    if set(promotion.get("required_statuses", [])) != expected_statuses:
        raise ScenarioPersistenceValidationError(
            "promotion.required_statuses must be the three independent checks"
        )

    all_pass = (
        record["exact_replay"]["status"] == PASS
        and record["critical_event_reproduced"]["status"] == PASS
        and all(cell.get("verdict") == PASS for cell in record["perturbation_persistence"]["cells"])
        and not record["perturbation_persistence"]["missing_cell_reasons"]
    )
    if all_pass and promotion["verdict"] != "promote":
        raise ScenarioPersistenceValidationError(
            "all checks passed but promotion verdict is not 'promote'"
        )
    if not all_pass and promotion["verdict"] == "promote":
        raise ScenarioPersistenceValidationError(
            "promotion verdict is 'promote' but a required check did not pass"
        )


__all__ = [
    "FAIL",
    "PASS",
    "PERSISTENCE_SCHEMA_VERSION",
    "REQUIRED_STATUSES",
    "UNKNOWN",
    "ScenarioPersistenceValidationError",
    "assess_critical_event_reproduction",
    "assess_exact_replay",
    "compute_persistence_record",
    "evaluate_perturbation_grid",
    "load_persistence_schema",
    "validate_persistence_record",
]
