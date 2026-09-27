#!/usr/bin/env python3
"""Run the issue #9759 hybrid v3/v4 release-map mirror diagnostic.

The diagnostic begins with the canonical release scenario loader and reflects
the resulting MapDefinition in a scoped map-runner builder override. It never
changes the release manifest or official planner roster. All output is
diagnostic-only and cannot be used as release benchmark evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from robot_sf.benchmark import map_runner_episode
from robot_sf.benchmark.map_runner.map_runner import build_map_policy
from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode
from robot_sf.benchmark.release_map_mirror import MirrorAxis, reflect_map_definition
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.sim.spawn_validation import reset_spawn_clearance
from robot_sf.training.scenario_loader import load_scenarios

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = _REPO_ROOT / "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml"
DEFAULT_OUTPUT_RELATIVE = Path("codex-agent-runs/active/issue-9759/diagnostic")
V3_CONFIG_RELATIVE = Path(
    "configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape_s30_h600_release.yaml"
)
V4_CONFIG_RELATIVE = Path(
    "configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_s30_h600_release.yaml"
)
ALGO = "hybrid_rule_local_planner"
DIAGNOSTIC_STATUS = "diagnostic-only"
SCHEMA_VERSION = "issue_9759_hybrid_mirror_diagnostic.v2"
EpisodeAxis = Literal["base", "x", "y"]
VALID_TERMINAL_STATUSES = frozenset(
    {"success", "collision", "terminated", "truncated", "max_steps"}
)


@dataclass(frozen=True, slots=True)
class ReleaseInputs:
    """Resolved immutable inputs for one diagnostic selection."""

    manifest_path: Path
    scenario_matrix_path: Path
    scenarios: tuple[dict[str, Any], ...]
    seeds: tuple[int, ...]
    horizon: int
    manifest_digest: str
    scenario_matrix_digest: str
    input_digest: str


@dataclass(frozen=True, slots=True)
class EpisodeJob:
    """One independent arm/orientation/seed cell sent to a worker process."""

    scenario: dict[str, Any]
    scenario_id: str
    seed: int
    arm: str
    config_path: str
    config_digest: str
    axis: EpisodeAxis
    map_id: str | None
    map_digest: str | None
    source_map_digest: str | None
    preflight_status: str
    preflight_error: str | None
    scenario_matrix_path: str
    horizon: int
    input_digest: str

    @property
    def identity(self) -> tuple[str, int, str, EpisodeAxis]:
        """Return the stable resume identity for this cell."""

        return self.scenario_id, self.seed, self.arm, self.axis


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _diagnostic_source_paths() -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "map_mirror": (_REPO_ROOT / "robot_sf/benchmark/release_map_mirror.py").resolve(),
        "hybrid_planner": (_REPO_ROOT / "robot_sf/planner/hybrid_rule_local_planner.py").resolve(),
        "grid_route": (_REPO_ROOT / "robot_sf/planner/grid_route.py").resolve(),
        "map_runner": (_REPO_ROOT / "robot_sf/benchmark/map_runner/map_runner.py").resolve(),
        "map_runner_episode": (
            _REPO_ROOT / "robot_sf/benchmark/map_runner/map_runner_episode.py"
        ).resolve(),
    }


def _diagnostic_source_digests() -> dict[str, str]:
    digests: dict[str, str] = {}
    for label, path in _diagnostic_source_paths().items():
        if not path.is_file():
            raise ValueError(f"diagnostic source file does not exist: {path}")
        digests[label] = _sha256_file(path)
    return digests


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=repr,
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load YAML mapping {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return dict(payload)


def _resolve_manifest_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"release manifest does not exist: {resolved}")
    return resolved


def _resolve_scenario_matrix(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    scenario = manifest.get("scenario")
    if not isinstance(scenario, Mapping):
        raise ValueError("release manifest must contain a scenario mapping")
    raw_path = scenario.get("matrix_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("release manifest scenario.matrix_path must be a non-empty string")
    matrix_path = (manifest_path.parent / raw_path).resolve()
    if not matrix_path.is_file():
        raise ValueError(f"resolved scenario matrix does not exist: {matrix_path}")
    return matrix_path


def _resolve_release_seeds(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    policy = manifest.get("seed_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("release manifest must contain seed_policy")
    raw_seeds = policy.get("resolved_seeds")
    if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, (str, bytes)):
        raise ValueError("seed_policy.resolved_seeds must be a sequence")
    try:
        seeds = tuple(int(seed) for seed in raw_seeds)
    except (TypeError, ValueError) as exc:
        raise ValueError("resolved seeds must be integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("resolved seeds must be non-empty and unique")
    return seeds


def _resolve_horizon(manifest: Mapping[str, Any]) -> int:
    matrix = manifest.get("matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("release manifest must contain matrix")
    try:
        horizon = int(matrix["horizon_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("matrix.horizon_steps must be an integer") from exc
    if horizon <= 0:
        raise ValueError("matrix.horizon_steps must be positive")
    return horizon


def _selection_input_digest(
    *,
    manifest_path: Path,
    scenario_matrix_path: Path,
    manifest: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    horizon: int,
) -> str:
    """Hash every release input that can affect selected diagnostic cells."""

    candidate_paths = {
        "v3": (_REPO_ROOT / V3_CONFIG_RELATIVE).resolve(),
        "v4": (_REPO_ROOT / V4_CONFIG_RELATIVE).resolve(),
    }
    candidate_digests: dict[str, str] = {}
    for label, path in candidate_paths.items():
        if not path.is_file():
            raise ValueError(f"pinned {label} candidate config does not exist: {path}")
        candidate_digests[label] = _sha256_file(path)
    source_paths = _diagnostic_source_paths()
    source_digests = _diagnostic_source_digests()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "producing_head_sha": _git_value("rev-parse", "HEAD"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "release_id": manifest.get("release_id"),
        "scenario_matrix_path": str(scenario_matrix_path),
        "scenario_matrix_sha256": _sha256_file(scenario_matrix_path),
        "scenarios": [dict(scenario) for scenario in scenarios],
        "seeds": [int(seed) for seed in seeds],
        "horizon": int(horizon),
        "candidate_paths": {key: str(value) for key, value in candidate_paths.items()},
        "candidate_sha256": candidate_digests,
        "diagnostic_source_paths": {key: str(value) for key, value in source_paths.items()},
        "diagnostic_source_sha256": source_digests,
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def load_release_inputs(manifest_path: Path = DEFAULT_MANIFEST) -> ReleaseInputs:
    """Load the release manifest and resolve its matrix relative to that manifest."""

    resolved_manifest = _resolve_manifest_path(manifest_path)
    manifest = _load_yaml_mapping(resolved_manifest)
    matrix_path = _resolve_scenario_matrix(resolved_manifest, manifest)
    scenarios = tuple(dict(scenario) for scenario in load_scenarios(matrix_path))
    if not scenarios:
        raise ValueError(f"scenario matrix is empty: {matrix_path}")
    seeds = _resolve_release_seeds(manifest)
    horizon = _resolve_horizon(manifest)
    return ReleaseInputs(
        manifest_path=resolved_manifest,
        scenario_matrix_path=matrix_path,
        scenarios=scenarios,
        seeds=seeds,
        horizon=horizon,
        manifest_digest=_sha256_file(resolved_manifest),
        scenario_matrix_digest=_sha256_file(matrix_path),
        input_digest=_selection_input_digest(
            manifest_path=resolved_manifest,
            scenario_matrix_path=matrix_path,
            manifest=manifest,
            scenarios=scenarios,
            seeds=seeds,
            horizon=horizon,
        ),
    )


def common_git_dir(repo_root: Path = _REPO_ROOT) -> Path:
    """Return the shared Git directory, including from a linked worktree."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot resolve common Git directory from {repo_root}: {exc}") from exc
    raw = Path(completed.stdout.strip())
    return (repo_root / raw).resolve() if not raw.is_absolute() else raw.resolve()


def _git_value(*arguments: str, repo_root: Path = _REPO_ROOT) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"cannot resolve git {' '.join(arguments)} from {repo_root}: {exc}"
        ) from exc
    value = completed.stdout.strip()
    if not value:
        raise ValueError(f"git {' '.join(arguments)} returned no value from {repo_root}")
    return value


def _git_worktree_changes(repo_root: Path = _REPO_ROOT) -> tuple[str, ...]:
    """Return tracked and untracked changes that would invalidate a source revision."""

    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot inspect source worktree at {repo_root}: {exc}") from exc
    return tuple(line for line in completed.stdout.splitlines() if line.strip())


def _git_provenance(repo_root: Path = _REPO_ROOT) -> dict[str, str]:
    """Return exact revision identities required for diagnostic reproduction."""

    return {
        "head_sha": _git_value("rev-parse", "HEAD", repo_root=repo_root),
        "base_sha": _git_value("merge-base", "HEAD", "origin/main", repo_root=repo_root),
        "origin_main_sha": _git_value("rev-parse", "origin/main", repo_root=repo_root),
    }


def _environment_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(aliased=True),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "spawn_method": "spawn",
    }


def default_output_dir(repo_root: Path = _REPO_ROOT) -> Path:
    """Return the durable common-Git diagnostic directory."""

    return common_git_dir(repo_root) / DEFAULT_OUTPUT_RELATIVE


def validate_output_dir(path: Path, *, repo_root: Path = _REPO_ROOT) -> Path:
    """Reject output paths outside the common Git diagnostic custody root."""

    requested = path.expanduser().resolve()
    allowed = default_output_dir(repo_root).resolve()
    try:
        requested.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(
            f"diagnostic output must be under common Git directory {allowed}; refusing {requested}"
        ) from exc
    requested.mkdir(parents=True, exist_ok=True)
    return requested


def _accepts_runtime_input_records(builder: object) -> bool:
    try:
        parameters = inspect.signature(builder).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        (
            parameter.name == "runtime_input_records"
            and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
        )
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _map_id_for_config(config: Any) -> str:
    map_pool = getattr(config, "map_pool", None)
    map_defs = getattr(map_pool, "map_defs", None)
    if not isinstance(map_defs, dict) or not map_defs:
        raise ValueError("canonical builder returned an empty map pool")
    configured = getattr(config, "map_id", None)
    if isinstance(configured, str) and configured in map_defs:
        return configured
    if len(map_defs) == 1:
        return next(iter(map_defs))
    raise ValueError("canonical builder returned multiple maps without config.map_id")


def _map_digest(map_definition: object) -> str:
    """Return a compact deterministic digest for a loaded map object."""

    def normalize(value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, bytes):
            return value.hex()
        if hasattr(value, "wkb_hex"):
            return {"wkb_hex": str(value.wkb_hex)}
        if is_dataclass(value):
            return {item.name: normalize(getattr(value, item.name)) for item in fields(value)}
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, set):
            return sorted((normalize(item) for item in value), key=repr)
        return repr(value)

    return _sha256_bytes(_canonical_json(normalize(map_definition)).encode("utf-8"))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class _MapTransformError(ValueError):
    """Retain the loaded source-map identity when a reflection cannot be built."""

    def __init__(self, message: str, *, map_id: str, source_map_digest: str) -> None:
        super().__init__(message)
        self.map_id = map_id
        self.source_map_digest = source_map_digest


def _build_config_with_axis(
    scenario: Mapping[str, Any],
    *,
    scenario_matrix_path: Path,
    axis: EpisodeAxis,
) -> tuple[Any, str, str, str]:
    """Build the canonical config and replace its selected post-loader map."""

    config = map_runner_episode._build_env_config(
        dict(scenario), scenario_path=scenario_matrix_path
    )
    map_id = _map_id_for_config(config)
    map_defs = config.map_pool.map_defs
    map_definition = map_defs[map_id]
    source_map_digest = _map_digest(map_definition)
    if axis != "base":
        try:
            map_definition = reflect_map_definition(map_definition, cast("MirrorAxis", axis))
        except Exception as exc:  # Preserve source-map provenance for every transform failure.
            raise _MapTransformError(
                f"cannot reflect post-loader map {map_id!r} across {axis}: {exc}",
                map_id=map_id,
                source_map_digest=source_map_digest,
            ) from exc
        map_defs[map_id] = map_definition
    return config, map_id, _map_digest(map_definition), source_map_digest


@contextmanager
def scoped_reflected_builder(
    *,
    axis: EpisodeAxis,
    captured_map_digests: list[str] | None = None,
    captured_source_map_digests: list[str] | None = None,
) -> Iterator[None]:
    """Capture or reflect the canonical builder's map and restore it safely."""

    original = map_runner_episode._build_env_config
    digest_sink = captured_map_digests if captured_map_digests is not None else []
    source_digest_sink = (
        captured_source_map_digests if captured_source_map_digests is not None else []
    )

    def _build(
        scenario: dict[str, Any],
        *,
        scenario_path: Path,
        runtime_input_records: list[dict[str, str]] | None = None,
    ) -> Any:
        if runtime_input_records is not None and _accepts_runtime_input_records(original):
            config = original(
                scenario,
                scenario_path=scenario_path,
                runtime_input_records=runtime_input_records,
            )
        else:
            config = original(scenario, scenario_path=scenario_path)
        map_id = _map_id_for_config(config)
        map_definition = config.map_pool.map_defs[map_id]
        source_digest_sink.append(_map_digest(map_definition))
        if axis != "base":
            map_definition = reflect_map_definition(map_definition, cast("MirrorAxis", axis))
            config.map_pool.map_defs[map_id] = map_definition
        digest_sink.append(_map_digest(map_definition))
        return config

    map_runner_episode._build_env_config = _build
    try:
        yield
    finally:
        map_runner_episode._build_env_config = original


def preflight_maps(
    inputs: ReleaseInputs,
    *,
    scenarios: Sequence[Mapping[str, Any]],
    axes: Sequence[EpisodeAxis],
    seed: int,
) -> list[dict[str, Any]]:
    """Validate each post-loader map and perform one seeded environment reset."""

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = str(scenario.get("name") or scenario.get("scenario_id") or "unknown")
        for axis in axes:
            started = time.perf_counter()
            row: dict[str, Any] = {
                "scenario_id": scenario_id,
                "axis": axis,
                "seed": int(seed),
                "status": "ok",
                "map_id": None,
                "map_digest": None,
                "source_map_digest": None,
                "reset_spawn_clearance": None,
                "wall_time_sec": None,
                "error": None,
            }
            env = None
            try:
                config, map_id, digest, source_digest = _build_config_with_axis(
                    scenario, scenario_matrix_path=inputs.scenario_matrix_path, axis=axis
                )
                row["map_id"] = map_id
                row["map_digest"] = digest
                row["source_map_digest"] = source_digest
                env = make_robot_env(config=config, seed=seed, debug=False, recording_enabled=False)
                env.reset(seed=seed)
                clearance = reset_spawn_clearance(env.simulator)
                row["reset_spawn_clearance"] = clearance
                if bool(clearance.get("overlap")):
                    row["status"] = "failed"
                    row["error"] = "reset spawn overlap"
            except Exception as exc:  # noqa: BLE001 - retain every preflight failure.
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                source_map_digest = getattr(exc, "source_map_digest", None)
                map_id = getattr(exc, "map_id", None)
                if isinstance(source_map_digest, str):
                    row["source_map_digest"] = source_map_digest
                if isinstance(map_id, str):
                    row["map_id"] = map_id
            finally:
                if env is not None:
                    env.close()
                row["wall_time_sec"] = round(time.perf_counter() - started, 6)
            rows.append(row)
    return rows


def _outcome_payload(record: Mapping[str, Any]) -> dict[str, bool]:
    outcome = record.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    return {
        "route_complete": bool(outcome.get("route_complete", outcome.get("success", False))),
        "collision_event": bool(outcome.get("collision_event", outcome.get("collision", False))),
        "timeout_event": bool(outcome.get("timeout_event", False)),
    }


def _normalized_status(record: Mapping[str, Any]) -> str:
    status = str(record.get("status", "unknown")).strip()
    termination_reason = str(record.get("termination_reason", "")).strip()
    if status == "failure" and termination_reason in VALID_TERMINAL_STATUSES:
        return termination_reason
    return status


def _outcome_error(record: Mapping[str, Any]) -> str | None:
    outcome = record.get("outcome")
    if not isinstance(outcome, Mapping):
        return "missing or malformed outcome mapping"
    aliases = {
        "route_complete": "success",
        "collision_event": "collision",
        "timeout_event": None,
    }
    for key, alias in aliases.items():
        source_key = key if key in outcome else alias
        if source_key is None or source_key not in outcome:
            return f"missing outcome.{key}"
        if not isinstance(outcome.get(source_key), bool):
            return f"outcome.{key} must be boolean"
    return None


def _compact_record(
    job: EpisodeJob,
    record: Mapping[str, Any],
    elapsed: float,
    *,
    map_digest: str | None = None,
    source_map_digest: str | None = None,
) -> dict[str, Any]:
    """Reduce one canonical episode record to the diagnostic JSONL schema."""

    selected = record.get("selected_map_identity")
    status = _normalized_status(record)
    error = _record_error(record) or _outcome_error(record)
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": DIAGNOSTIC_STATUS,
        "input_digest": job.input_digest,
        "scenario_id": job.scenario_id,
        "seed": int(job.seed),
        "arm": job.arm,
        "axis": job.axis,
        "config_path": job.config_path,
        "config_sha256": job.config_digest,
        "map_sha256": map_digest,
        "source_map_sha256": source_map_digest,
        "preflight_map_sha256": job.map_digest,
        "preflight_source_map_sha256": job.source_map_digest,
        "preflight_status": job.preflight_status,
        "preflight_error": job.preflight_error,
        "status": status,
        "steps": int(record.get("steps", 0) or 0),
        "termination_reason": str(record.get("termination_reason", "unknown")),
        "outcome": _outcome_payload(record),
        "wall_time_sec": round(float(record.get("wall_time_sec", elapsed) or elapsed), 6),
        "map_id": (selected.get("map_id") if isinstance(selected, Mapping) else job.map_id),
        "error": error,
    }


def _record_error(record: Mapping[str, Any]) -> str | None:
    """Preserve a canonical string error or integrity reason, if present."""

    for key in ("error", "integrity_error", "failure_reason"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value.strip()
        if value is not None and not isinstance(value, str):
            return f"{key}={_canonical_json(value)}"
    integrity = record.get("integrity")
    if isinstance(integrity, Mapping):
        contradictions = integrity.get("contradictions")
        if isinstance(contradictions, Sequence) and not isinstance(contradictions, (str, bytes)):
            retained = [str(item) for item in contradictions if str(item).strip()]
            if retained:
                return "; ".join(retained)
    status = _normalized_status(record)
    if status not in VALID_TERMINAL_STATUSES:
        termination_reason = str(record.get("termination_reason", "")).strip()
        if termination_reason and termination_reason != "unknown":
            return f"unsupported status {status or '<missing>'}; termination_reason={termination_reason}"
        return f"unsupported terminal status: {status or '<missing>'}"
    return None


def _record_is_terminal_valid(record: Mapping[str, Any]) -> bool:
    """Return whether a row is safe to compare as a terminal episode."""

    status = _normalized_status(record)
    if status not in VALID_TERMINAL_STATUSES:
        return False
    if _record_error(record) is not None:
        return False
    if str(record.get("termination_reason", "")).strip() != status:
        return False
    if record.get("preflight_status") != "ok" or record.get("preflight_error") is not None:
        return False
    if _outcome_error(record) is not None:
        return False
    outcome = record.get("outcome")
    if not isinstance(outcome, Mapping):
        return False
    return all(
        isinstance(outcome.get(key), bool)
        for key in ("route_complete", "collision_event", "timeout_event")
    ) and all(
        _is_sha256(record.get(key)) for key in ("config_sha256", "source_map_sha256", "map_sha256")
    )


def _failure_record(
    job: EpisodeJob,
    exc: BaseException,
    *,
    map_digest: str | None = None,
    source_map_digest: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": DIAGNOSTIC_STATUS,
        "input_digest": job.input_digest,
        "scenario_id": job.scenario_id,
        "seed": int(job.seed),
        "arm": job.arm,
        "axis": job.axis,
        "config_path": job.config_path,
        "config_sha256": job.config_digest,
        "map_sha256": map_digest,
        "source_map_sha256": source_map_digest,
        "preflight_map_sha256": job.map_digest,
        "preflight_source_map_sha256": job.source_map_digest,
        "map_id": job.map_id,
        "preflight_status": job.preflight_status,
        "preflight_error": job.preflight_error,
        "status": "failed",
        "steps": 0,
        "termination_reason": "error",
        "outcome": {"route_complete": False, "collision_event": False, "timeout_event": False},
        "wall_time_sec": 0.0,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _run_episode_job(job: EpisodeJob) -> dict[str, Any]:
    """Execute one canonical episode in a worker process."""

    started = time.perf_counter()
    captured_map_digests: list[str] = []
    captured_source_map_digests: list[str] = []
    try:
        with scoped_reflected_builder(
            axis=job.axis,
            captured_map_digests=captured_map_digests,
            captured_source_map_digests=captured_source_map_digests,
        ):
            record = run_map_episode(
                dict(job.scenario),
                seed=job.seed,
                horizon=job.horizon,
                dt=None,
                record_forces=False,
                snqi_weights=None,
                snqi_baseline=None,
                algo=ALGO,
                scenario_path=Path(job.scenario_matrix_path),
                algo_config_path=job.config_path,
                policy_builder=build_map_policy,
            )
        if len(captured_map_digests) != 1:
            raise ValueError(
                "canonical episode builder must produce exactly one post-loader map digest; "
                f"observed {len(captured_map_digests)}"
            )
        if len(captured_source_map_digests) != 1:
            raise ValueError(
                "canonical episode builder must produce exactly one source map digest; "
                f"observed {len(captured_source_map_digests)}"
            )
        actual_map_digest = captured_map_digests[0]
        actual_source_map_digest = captured_source_map_digests[0]
        if job.map_digest is not None and actual_map_digest != job.map_digest:
            raise ValueError(
                "post-loader map digest changed between preflight and episode run: "
                f"expected {job.map_digest}, got {actual_map_digest}"
            )
        if job.source_map_digest is not None and actual_source_map_digest != job.source_map_digest:
            raise ValueError(
                "source post-loader map digest changed between preflight and episode run: "
                f"expected {job.source_map_digest}, got {actual_source_map_digest}"
            )
        return _compact_record(
            job,
            record,
            time.perf_counter() - started,
            map_digest=actual_map_digest,
            source_map_digest=actual_source_map_digest,
        )
    except Exception as exc:  # noqa: BLE001 - retain every episode failure as a row.
        return _failure_record(
            job,
            exc,
            map_digest=captured_map_digests[-1] if captured_map_digests else None,
            source_map_digest=(
                captured_source_map_digests[-1] if captured_source_map_digests else None
            ),
        )


def _identity_from_record(record: Mapping[str, Any]) -> tuple[str, int, str, EpisodeAxis]:
    try:
        return (
            str(record["scenario_id"]),
            int(record["seed"]),
            str(record["arm"]),
            cast("EpisodeAxis", str(record["axis"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed episode identity: {record!r}") from exc


def _validate_expected_resume_provenance(
    payload: Mapping[str, Any],
    *,
    identity: tuple[str, int, str, EpisodeAxis],
    path: Path,
    line_number: int,
    expected_provenance: Mapping[tuple[str, int, str, EpisodeAxis], Mapping[str, Any]] | None,
) -> None:
    """Ensure stored row hashes still agree with the current preflight plan."""

    if expected_provenance is None or identity not in expected_provenance:
        return
    expected = expected_provenance[identity]
    for field in (
        "config_sha256",
        "preflight_source_map_sha256",
        "preflight_map_sha256",
        "preflight_status",
        "preflight_error",
    ):
        if payload.get(field) != expected.get(field):
            raise ValueError(f"resume {field} mismatch at {path}:{line_number} for {identity!r}")
    for field in ("source_map_sha256", "map_sha256"):
        observed = payload.get(field)
        expected_field = (
            "preflight_source_map_sha256"
            if field == "source_map_sha256"
            else "preflight_map_sha256"
        )
        if observed is not None and observed != expected.get(expected_field):
            raise ValueError(
                f"resume observed {field} mismatch at {path}:{line_number} for {identity!r}"
            )


def _validate_resume_row_metadata(
    payload: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
) -> None:
    """Validate required config/map and preflight fields on one stored row."""

    if not _is_sha256(payload.get("config_sha256")):
        raise ValueError(f"resume config_sha256 missing or invalid at {path}:{line_number}")
    if payload.get("preflight_status") not in {"ok", "failed"}:
        raise ValueError(f"resume preflight_status missing or invalid at {path}:{line_number}")
    if "preflight_error" not in payload:
        raise ValueError(f"resume preflight_error missing at {path}:{line_number}")
    failed_with_no_error = payload.get("preflight_status") == "failed" and not payload.get(
        "preflight_error"
    )
    ok_with_error = (
        payload.get("preflight_status") == "ok" and payload.get("preflight_error") is not None
    )
    if failed_with_no_error or ok_with_error:
        raise ValueError(f"resume preflight status/error conflict at {path}:{line_number}")
    for field in (
        "source_map_sha256",
        "map_sha256",
        "preflight_source_map_sha256",
        "preflight_map_sha256",
    ):
        if field not in payload:
            raise ValueError(f"resume {field} missing at {path}:{line_number}")
        if payload.get(field) is not None and not _is_sha256(payload.get(field)):
            raise ValueError(f"resume {field} invalid at {path}:{line_number}")


def _validate_resume_payload(
    payload: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
    input_digest: str,
    expected_provenance: Mapping[tuple[str, int, str, EpisodeAxis], Mapping[str, Any]] | None,
) -> tuple[str, int, str, EpisodeAxis]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"resume schema mismatch at {path}:{line_number}")
    if payload.get("input_digest") != input_digest:
        raise ValueError(f"resume input digest mismatch at {path}:{line_number}")
    identity = _identity_from_record(payload)
    _validate_resume_row_metadata(payload, path=path, line_number=line_number)
    _validate_expected_resume_provenance(
        payload,
        identity=identity,
        path=path,
        line_number=line_number,
        expected_provenance=expected_provenance,
    )
    return identity


def load_resume_records(
    path: Path,
    *,
    input_digest: str,
    expected_provenance: Mapping[tuple[str, int, str, EpisodeAxis], Mapping[str, Any]]
    | None = None,
) -> dict[tuple[str, int, str, EpisodeAxis], dict[str, Any]]:
    """Load append-only episode rows and fail closed on stale provenance."""

    if not path.exists():
        return {}
    records: dict[tuple[str, int, str, EpisodeAxis], dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read resume JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in resume file {path}:{line_number}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"resume row {path}:{line_number} must be a JSON mapping")
        identity = _validate_resume_payload(
            payload,
            path=path,
            line_number=line_number,
            input_digest=input_digest,
            expected_provenance=expected_provenance,
        )
        if identity in records:
            raise ValueError(f"duplicate resume episode identity at {path}:{line_number}")
        records[identity] = dict(payload)
    return records


def _missing_record(job: EpisodeJob, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": DIAGNOSTIC_STATUS,
        "input_digest": job.input_digest,
        "scenario_id": job.scenario_id,
        "seed": int(job.seed),
        "arm": job.arm,
        "axis": job.axis,
        "config_path": job.config_path,
        "config_sha256": job.config_digest,
        "map_sha256": None,
        "source_map_sha256": None,
        "preflight_map_sha256": job.map_digest,
        "preflight_source_map_sha256": job.source_map_digest,
        "map_id": job.map_id,
        "preflight_status": job.preflight_status,
        "preflight_error": job.preflight_error,
        "status": "excluded" if job.preflight_status != "ok" else "missing",
        "steps": 0,
        "termination_reason": "not_run",
        "outcome": {"route_complete": False, "collision_event": False, "timeout_event": False},
        "wall_time_sec": 0.0,
        "error": reason,
    }


def _paired_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build base-versus-reflection rows from terminal and failed cells."""

    def provenance(record: Mapping[str, Any] | None) -> dict[str, Any]:
        if record is None:
            return {
                "config_sha256": None,
                "source_map_sha256": None,
                "map_sha256": None,
                "preflight_source_map_sha256": None,
                "preflight_map_sha256": None,
            }
        return {
            "config_sha256": record.get("config_sha256"),
            "source_map_sha256": record.get("source_map_sha256"),
            "map_sha256": record.get("map_sha256"),
            "preflight_source_map_sha256": record.get("preflight_source_map_sha256"),
            "preflight_map_sha256": record.get("preflight_map_sha256"),
        }

    grouped: dict[tuple[str, int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (
            str(record.get("scenario_id")),
            int(record.get("seed", 0)),
            str(record.get("arm")),
        )
        grouped[key][str(record.get("axis"))] = record
    rows: list[dict[str, Any]] = []
    for (scenario_id, seed, arm), by_axis in sorted(grouped.items()):
        base = by_axis.get("base")
        paired: dict[str, Any] = {}
        for axis in ("x", "y"):
            transformed = by_axis.get(axis)
            if base is None or transformed is None:
                paired[axis] = {
                    "outcome_agreement": None,
                    "step_delta": None,
                    "status": "missing",
                }
                continue
            comparable = _record_is_terminal_valid(base) and _record_is_terminal_valid(transformed)
            base_outcome = base.get("outcome")
            transformed_outcome = transformed.get("outcome")
            agreement = None
            if (
                comparable
                and isinstance(base_outcome, Mapping)
                and isinstance(transformed_outcome, Mapping)
            ):
                agreement = dict(base_outcome) == dict(transformed_outcome)
            paired[axis] = {
                "outcome_agreement": agreement,
                "step_delta": (
                    int(transformed.get("steps", 0) or 0) - int(base.get("steps", 0) or 0)
                    if comparable
                    else None
                ),
                "status": "ok" if comparable else "incomplete",
            }
        rows.append(
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "arm": arm,
                "provenance": {
                    "base": provenance(base),
                    "x": provenance(by_axis.get("x")),
                    "y": provenance(by_axis.get("y")),
                },
                "pairs": paired,
            }
        )
    return rows


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_jobs: int,
    input_digest: str,
    eligible_jobs: int | None = None,
    excluded_jobs: int | None = None,
) -> dict[str, Any]:
    """Build compact coverage and paired-outcome summary data."""

    unique_records = {_identity_from_record(record): record for record in records}
    statuses = Counter(str(record.get("status", "unknown")) for record in unique_records.values())
    pairs = _paired_rows(list(unique_records.values()))
    pair_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"pairs": 0, "outcome_agreement": 0, "outcome_disagreement": 0})
    )
    step_deltas: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        arm = str(pair["arm"])
        for axis, details in pair["pairs"].items():
            if details["status"] != "ok":
                continue
            counts = pair_counts[arm][axis]
            counts["pairs"] += 1
            if details["outcome_agreement"] is True:
                counts["outcome_agreement"] += 1
            elif details["outcome_agreement"] is False:
                counts["outcome_disagreement"] += 1
            if details["step_delta"] is not None:
                step_deltas[arm][axis].append(int(details["step_delta"]))
    by_arm: dict[str, Any] = {}
    for arm, axes in sorted(pair_counts.items()):
        by_arm[arm] = {}
        for axis, counts in sorted(axes.items()):
            deltas = step_deltas[arm][axis]
            by_arm[arm][axis] = {
                **dict(counts),
                "step_delta_min": min(deltas) if deltas else None,
                "step_delta_max": max(deltas) if deltas else None,
                "step_delta_mean": sum(deltas) / len(deltas) if deltas else None,
            }
    observed = len(unique_records)
    valid_terminal_jobs = sum(
        _record_is_terminal_valid(record) for record in unique_records.values()
    )
    expected_jobs = int(expected_jobs)
    eligible_jobs = expected_jobs if eligible_jobs is None else int(eligible_jobs)
    excluded_jobs = (
        max(0, expected_jobs - eligible_jobs) if excluded_jobs is None else int(excluded_jobs)
    )
    incomplete_jobs = max(0, expected_jobs - valid_terminal_jobs)
    complete = (
        observed == expected_jobs and valid_terminal_jobs == expected_jobs and excluded_jobs == 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": DIAGNOSTIC_STATUS,
        "claim_boundary": (
            "diagnostic-only; reflected maps are excluded from official release evidence and arm ranking"
        ),
        "input_digest": input_digest,
        "coverage": {
            "expected_jobs": expected_jobs,
            "eligible_jobs": eligible_jobs,
            "excluded_jobs": excluded_jobs,
            "observed_jobs": observed,
            "valid_terminal_jobs": valid_terminal_jobs,
            "missing_jobs": max(0, expected_jobs - observed),
            "incomplete_jobs": incomplete_jobs,
            "status": "complete" if complete else "partial",
        },
        "status_counts": dict(sorted(statuses.items())),
        "paired": {"pair_groups": len(pairs), "by_arm_axis": by_arm},
    }


def _filter_selection(
    inputs: ReleaseInputs,
    *,
    scenario_ids: Sequence[str] | None,
    seeds: Sequence[int] | None,
) -> tuple[dict[str, Any], ...]:
    selected_names = {str(item) for item in scenario_ids or ()}
    selected_seeds = {int(item) for item in seeds or ()}
    scenarios = tuple(
        scenario
        for scenario in inputs.scenarios
        if not selected_names
        or str(scenario.get("name") or scenario.get("scenario_id")) in selected_names
    )
    if not scenarios:
        raise ValueError("scenario selection matched no rows")
    if selected_seeds and not selected_seeds.issubset(set(inputs.seeds)):
        raise ValueError(
            f"seed selection is outside release seed set: {sorted(selected_seeds - set(inputs.seeds))}"
        )
    return scenarios


def _build_jobs(
    inputs: ReleaseInputs,
    *,
    scenarios: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    axes: Sequence[EpisodeAxis],
    preflight: Sequence[Mapping[str, Any]],
) -> list[EpisodeJob]:
    arms = (
        ("v3", (_REPO_ROOT / V3_CONFIG_RELATIVE).resolve()),
        ("v4", (_REPO_ROOT / V4_CONFIG_RELATIVE).resolve()),
    )
    preflight_index = {
        (str(row.get("scenario_id")), str(row.get("axis"))): row for row in preflight
    }
    jobs: list[EpisodeJob] = []
    for scenario in scenarios:
        scenario_id = str(scenario.get("name") or scenario.get("scenario_id") or "unknown")
        for seed in seeds:
            for arm, config_path in arms:
                for axis in axes:
                    preflight_row = preflight_index.get((scenario_id, axis), {})
                    preflight_status = str(preflight_row.get("status", "failed"))
                    preflight_error = preflight_row.get("error")
                    if not isinstance(preflight_error, str) or not preflight_error:
                        preflight_error = (
                            None if preflight_status == "ok" else "missing or failed preflight row"
                        )
                    jobs.append(
                        EpisodeJob(
                            scenario=dict(scenario),
                            scenario_id=scenario_id,
                            seed=int(seed),
                            arm=arm,
                            config_path=str(config_path),
                            config_digest=_sha256_file(config_path),
                            axis=axis,
                            map_id=(
                                str(preflight_row["map_id"])
                                if preflight_row.get("map_id") is not None
                                else None
                            ),
                            map_digest=(
                                str(preflight_row["map_digest"])
                                if preflight_row.get("map_digest") is not None
                                else None
                            ),
                            source_map_digest=(
                                str(preflight_row["source_map_digest"])
                                if preflight_row.get("source_map_digest") is not None
                                else None
                            ),
                            preflight_status=preflight_status,
                            preflight_error=preflight_error,
                            scenario_matrix_path=str(inputs.scenario_matrix_path),
                            horizon=inputs.horizon,
                            input_digest=inputs.input_digest,
                        )
                    )
    return jobs


def _iter_worker_results(jobs: Sequence[EpisodeJob], workers: int) -> Iterator[dict[str, Any]]:
    """Run jobs with a bounded in-flight queue so full matrices stay memory-light."""

    if not jobs:
        return
    context = multiprocessing.get_context("spawn")
    job_iter = iter(jobs)
    pending: dict[Any, EpisodeJob] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        for _ in range(min(len(jobs), max(1, workers * 2))):
            job = next(job_iter, None)
            if job is None:
                break
            pending[executor.submit(_run_episode_job, job)] = job
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    yield future.result()
                except Exception as exc:  # noqa: BLE001 - convert process-future failures into per-job rows.
                    yield _failure_record(job, exc)
                next_job = next(job_iter, None)
                if next_job is not None:
                    pending[executor.submit(_run_episode_job, next_job)] = next_job


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scenario", action="append", dest="scenarios", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--axis", action="append", choices=("base", "x", "y"), default=[])
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected release-map mirror diagnostic and write compact artifacts."""

    selected_argv = sys.argv[1:] if argv is None else argv
    command_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        *(str(item) for item in selected_argv),
    ]
    args = _parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    worktree_changes = _git_worktree_changes(_REPO_ROOT)
    if not args.preflight_only and worktree_changes:
        raise ValueError(
            "episode execution requires a clean source worktree; "
            f"found changes: {worktree_changes!r}"
        )
    git_provenance = _git_provenance(_REPO_ROOT)
    source_paths = _diagnostic_source_paths()
    source_digests = _diagnostic_source_digests()
    inputs = load_release_inputs(args.manifest)
    scenarios = _filter_selection(inputs, scenario_ids=args.scenarios, seeds=args.seed)
    selected_seeds = tuple(dict.fromkeys(args.seed)) if args.seed else inputs.seeds
    selected_axes = dict.fromkeys(args.axis or ["base", "x", "y"])
    axes = tuple(cast("EpisodeAxis", axis) for axis in selected_axes)
    output_dir = validate_output_dir(
        args.output_dir or default_output_dir(_REPO_ROOT), repo_root=_REPO_ROOT
    )
    preflight = preflight_maps(inputs, scenarios=scenarios, axes=axes, seed=int(selected_seeds[0]))
    (output_dir / "preflight.json").write_text(
        _canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "evidence_status": DIAGNOSTIC_STATUS,
                "input_digest": inputs.input_digest,
                "manifest_path": str(inputs.manifest_path),
                "scenario_matrix_path": str(inputs.scenario_matrix_path),
                "rows": preflight,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if args.preflight_only:
        return 0 if all(row["status"] == "ok" for row in preflight) else 2

    jobs = _build_jobs(
        inputs,
        scenarios=scenarios,
        seeds=selected_seeds,
        axes=axes,
        preflight=preflight,
    )
    episode_path = output_dir / "episodes.jsonl"
    if episode_path.exists() and episode_path.stat().st_size and not args.resume:
        raise ValueError(
            f"episode output already exists at {episode_path}; use --resume to continue "
            "with input-digest validation"
        )
    expected_provenance = {
        job.identity: {
            "config_sha256": job.config_digest,
            "preflight_source_map_sha256": job.source_map_digest,
            "preflight_map_sha256": job.map_digest,
            "preflight_status": job.preflight_status,
            "preflight_error": job.preflight_error,
        }
        for job in jobs
    }
    prior = (
        load_resume_records(
            episode_path,
            input_digest=inputs.input_digest,
            expected_provenance=expected_provenance,
        )
        if args.resume
        else {}
    )
    selected_identities = {job.identity for job in jobs}
    records = {
        identity: record for identity, record in prior.items() if identity in selected_identities
    }
    runnable: list[EpisodeJob] = []
    with episode_path.open("a", encoding="utf-8") as handle:
        for job in jobs:
            if job.identity in records:
                continue
            if job.preflight_status != "ok":
                record = _missing_record(
                    job,
                    job.preflight_error or "preflight failed for scenario/axis",
                )
                handle.write(_canonical_json(record) + "\n")
                handle.flush()
                records[job.identity] = record
            else:
                runnable.append(job)
        for record in _iter_worker_results(runnable, args.workers):
            handle.write(_canonical_json(record) + "\n")
            handle.flush()
            records[_identity_from_record(record)] = record

    ordered_records = [records[key] for key in sorted(records)]
    summary = summarize_records(
        ordered_records,
        expected_jobs=len(jobs),
        input_digest=inputs.input_digest,
        eligible_jobs=sum(job.preflight_status == "ok" for job in jobs),
        excluded_jobs=sum(job.preflight_status != "ok" for job in jobs),
    )
    (output_dir / "summary.json").write_text(_canonical_json(summary) + "\n", encoding="utf-8")
    with (output_dir / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for pair in _paired_rows(ordered_records):
            handle.write(_canonical_json(pair) + "\n")
    (output_dir / "run_meta.json").write_text(
        _canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "evidence_status": DIAGNOSTIC_STATUS,
                "input_digest": inputs.input_digest,
                "manifest_digest": inputs.manifest_digest,
                "scenario_matrix_digest": inputs.scenario_matrix_digest,
                "candidate_config_sha256": {
                    arm: next(job.config_digest for job in jobs if job.arm == arm)
                    for arm in sorted({job.arm for job in jobs})
                },
                "diagnostic_source_paths": {
                    label: str(path) for label, path in source_paths.items()
                },
                "diagnostic_source_sha256": source_digests,
                "command_argv": command_argv,
                "worktree_clean": not worktree_changes,
                "worktree_changes": list(worktree_changes),
                **git_provenance,
                "merge_base_sha": git_provenance["base_sha"],
                "environment": _environment_metadata(),
                "scenario_count": len(scenarios),
                "seed_count": len(selected_seeds),
                "axes": list(axes),
                "workers": args.workers,
                "resume": bool(args.resume),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if summary["coverage"]["status"] == "complete" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALGO",
    "DEFAULT_MANIFEST",
    "DIAGNOSTIC_STATUS",
    "SCHEMA_VERSION",
    "EpisodeJob",
    "ReleaseInputs",
    "common_git_dir",
    "default_output_dir",
    "load_release_inputs",
    "load_resume_records",
    "main",
    "preflight_maps",
    "scoped_reflected_builder",
    "summarize_records",
    "validate_output_dir",
]
