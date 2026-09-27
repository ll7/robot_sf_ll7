"""Finite-budget coordinator for planner optimization and adversarial search.

This module owns round identity, immutable inputs, phase caching, regression-case
flow, and conservative stop decisions. It deliberately does not replace the
planner optimizer, adversarial search, replay, feasibility, or corpus owners.
Those owners are connected through :class:`CoevolutionAdapters`; in particular,
the replay/admissibility/corpus adapters remain an explicit integration seam
until their upstream contracts have passed exact review.

The adapter boundary is normalized rather than tied to provisional manifests:

* ``optimize`` wraps ``scripts.validation.planner_optimizer.run_planner_optimization``
  and returns a selected planner, its optimizer selection tuple, and exact budget;
* ``evaluate`` runs the selected planner on frozen held-out and accumulated
  regression case IDs through the canonical evaluation owner;
* ``falsify`` wraps ``robot_sf.adversarial.search.run_adversarial_search`` and
  returns the selected sampler plus every proposed candidate, including invalid
  and failed candidates;
* ``verify_discovery`` adapts replay plus feasibility/admissibility evidence;
* ``admit_case`` adapts the versioned counterexample corpus.

No default adapter silently reconstructs a case or upgrades unknown evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no POSIX flock implementation.
    fcntl = None  # type: ignore[assignment]

import yaml

from robot_sf.adversarial.config import SearchConfig
from robot_sf.adversarial.samplers import SUPPORTED_SAMPLERS, CandidateSampler, build_sampler

CONFIG_SCHEMA = "adversarial_coevolution_config.v1"
RUN_SCHEMA = "adversarial_coevolution_run.v2"
ROUND_SCHEMA = "adversarial_coevolution_round.v2"
PHASE_SCHEMA = "adversarial_coevolution_phase.v1"
MANIFEST_TRANSACTION_SCHEMA = "adversarial_coevolution_manifest_transaction.v1"

_EXECUTION_STATUSES = {"ok", "fallback", "degraded", "failed", "unknown", "not_run"}
_SEARCH_STATUSES = {"evaluated", "invalid", "failed"}
_PLANNER_OUTCOMES = {"confirmed_failure", "no_failure", "unknown", "not_assessed"}
_REPLAY_STATUSES = {"exact_match", "mismatch", "unavailable", "error", "not_run"}
_FEASIBILITY_STATUSES = {"empirically_feasible", "infeasible", "unknown", "not_assessed"}
_ADMISSIBILITY_STATUSES = {"admissible", "invalid", "inadmissible", "unknown", "not_assessed"}
_REGRESSION_STATUSES = {
    "solved",
    "unsolved",
    "invalid",
    "fallback",
    "degraded",
    "unknown",
    "failed",
    "unavailable",
}
_PHASES = ("optimization", "challenge_evaluation", "falsification", "discovery_admission")


class CoevolutionError(RuntimeError):
    """Raised when a stored run cannot be resumed without changing its evidence."""


class CoevolutionInfrastructureError(RuntimeError):
    """Raised internally after a phase fails or produces an invalid contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> str:
    encoded = _canonical_json(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_bytes(encoded)


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes made by atomic replace or unlink."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoevolutionError(f"cannot read persisted JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CoevolutionError(f"persisted JSON artifact must contain an object: {path}")
    return payload


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _detached_json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return a mutable adapter-owned copy of a validated JSON mapping."""
    return json.loads(_canonical_json(_mapping(value, name)))


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identities(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list of case IDs")
    result = tuple(_non_empty_string(item, f"{name}[]") for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicate case IDs")
    return result


def _resolve_file(value: Any, *, anchor: Path, name: str) -> Path:
    raw = Path(_non_empty_string(value, name)).expanduser()
    resolved = (anchor / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is not a file: {resolved}")
    return resolved


def _resolve_output(value: Any, *, anchor: Path) -> Path:
    raw = Path(_non_empty_string(value, "output_dir")).expanduser()
    return (anchor / raw).resolve() if not raw.is_absolute() else raw.resolve()


def _parse_round_stop_policy(
    rounds: Mapping[str, Any], stops: Mapping[str, Any]
) -> tuple[int, int, int | None, bool]:
    minimum = _integer(rounds.get("minimum"), "rounds.minimum", minimum=2)
    maximum = _integer(rounds.get("maximum"), "rounds.maximum", minimum=2)
    if maximum < minimum:
        raise ValueError("rounds.maximum must be >= rounds.minimum")
    plateau_raw = stops.get("optimization_plateau_rounds")
    plateau = (
        None
        if plateau_raw is None
        else _integer(plateau_raw, "stop.optimization_plateau_rounds", minimum=1)
    )
    no_discovery = stops.get("stop_after_minimum_without_new_admission", True)
    if not isinstance(no_discovery, bool):
        raise TypeError("stop.stop_after_minimum_without_new_admission must be boolean")
    return minimum, maximum, plateau, no_discovery


def _parse_search_runtime(search: Mapping[str, Any]) -> dict[str, Any]:
    horizon_raw = search.get("horizon")
    horizon = (
        None if horizon_raw is None else _integer(horizon_raw, "falsification.horizon", minimum=1)
    )
    dt_raw = search.get("dt")
    if dt_raw is None:
        dt = None
    elif isinstance(dt_raw, bool) or not isinstance(dt_raw, int | float):
        raise TypeError("falsification.dt must be a finite positive number or null")
    else:
        dt = float(dt_raw)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("falsification.dt must be a finite positive number")
    record_forces = search.get("record_forces", True)
    require_certification = search.get("require_certification", False)
    sampler = _non_empty_string(search.get("sampler"), "falsification.sampler").lower()
    if not isinstance(record_forces, bool) or not isinstance(require_certification, bool):
        raise TypeError("falsification record_forces and require_certification must be booleans")
    if sampler not in SUPPORTED_SAMPLERS:
        raise ValueError(f"falsification.sampler must be one of: {', '.join(SUPPORTED_SAMPLERS)}")
    return {
        "sampler": sampler,
        "horizon": horizon,
        "dt": dt,
        "workers": _integer(search.get("workers", 1), "falsification.workers", minimum=1),
        "benchmark_profile": _non_empty_string(
            search.get("benchmark_profile", "baseline-safe"), "falsification.benchmark_profile"
        ),
        "record_forces": record_forces,
        "require_certification": require_certification,
    }


@dataclass(frozen=True, slots=True)
class CoevolutionConfig:
    """Validated immutable loop configuration loaded from a versioned YAML file."""

    config_path: Path
    config_sha256: str
    run_id: str
    output_dir: Path
    optimizer_config: Path
    scenario_template: Path
    search_space: Path
    policy: str
    objective: str
    falsification_sampler: str
    horizon: int | None
    dt: float | None
    workers: int
    benchmark_profile: str
    record_forces: bool
    require_certification: bool
    minimum_rounds: int
    maximum_rounds: int
    optimizer_trials_per_method: int
    falsification_candidates_per_round: int
    optimizer_random_seed_base: int
    optimizer_tpe_seed_base: int
    falsification_seed_base: int
    heldout_case_ids: tuple[str, ...]
    initial_regression_case_ids: tuple[str, ...]
    stop_on_no_new_admissible_case: bool
    optimization_plateau_rounds: int | None
    additional_input_files: tuple[Path, ...] = ()

    @property
    def input_files(self) -> tuple[Path, ...]:
        """Return every explicitly named file whose bytes define this run."""
        candidates = (
            self.config_path,
            self.optimizer_config,
            self.scenario_template,
            self.search_space,
            *self.additional_input_files,
        )
        return tuple(dict.fromkeys(path.resolve() for path in candidates))


def load_coevolution_config(path: str | Path) -> CoevolutionConfig:
    """Load the strict config-first co-evolution contract."""
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "config")
    if root.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"schema must equal {CONFIG_SCHEMA!r}")
    run_id = _non_empty_string(root.get("run_id"), "run_id")
    if any(character in run_id for character in "/\\"):
        raise ValueError("run_id must be one path component")
    anchor = config_path.parent
    paths = _mapping(root.get("inputs"), "inputs")
    search = _mapping(root.get("falsification"), "falsification")
    rounds = _mapping(root.get("rounds"), "rounds")
    budgets = _mapping(root.get("budgets"), "budgets")
    seeds = _mapping(root.get("seeds"), "seeds")
    stops = _mapping(root.get("stop"), "stop")

    minimum_rounds, maximum_rounds, plateau_rounds, no_discovery = _parse_round_stop_policy(
        rounds, stops
    )
    search_runtime = _parse_search_runtime(search)
    heldout = _identities(root.get("heldout_case_ids", []), "heldout_case_ids")
    initial_regression = _identities(
        root.get("initial_regression_case_ids", []), "initial_regression_case_ids"
    )
    overlap = sorted(set(heldout) & set(initial_regression))
    if overlap:
        raise ValueError(f"held-out and regression IDs must be disjoint: {overlap}")
    additional_raw = root.get("additional_input_files", [])
    if not isinstance(additional_raw, list):
        raise TypeError("additional_input_files must be a list")
    additional_files = tuple(
        _resolve_file(value, anchor=anchor, name="additional_input_files[]")
        for value in additional_raw
    )
    return CoevolutionConfig(
        config_path=config_path,
        config_sha256=_sha256_file(config_path),
        run_id=run_id,
        output_dir=_resolve_output(root.get("output_dir"), anchor=anchor),
        optimizer_config=_resolve_file(
            paths.get("optimizer_config"), anchor=anchor, name="inputs.optimizer_config"
        ),
        scenario_template=_resolve_file(
            search.get("scenario_template"), anchor=anchor, name="falsification.scenario_template"
        ),
        search_space=_resolve_file(
            search.get("search_space"), anchor=anchor, name="falsification.search_space"
        ),
        policy=_non_empty_string(search.get("policy"), "falsification.policy"),
        objective=_non_empty_string(search.get("objective"), "falsification.objective"),
        falsification_sampler=search_runtime["sampler"],
        horizon=search_runtime["horizon"],
        dt=search_runtime["dt"],
        workers=search_runtime["workers"],
        benchmark_profile=search_runtime["benchmark_profile"],
        record_forces=search_runtime["record_forces"],
        require_certification=search_runtime["require_certification"],
        minimum_rounds=minimum_rounds,
        maximum_rounds=maximum_rounds,
        optimizer_trials_per_method=_integer(
            budgets.get("optimizer_trials_per_method"),
            "budgets.optimizer_trials_per_method",
            minimum=1,
        ),
        falsification_candidates_per_round=_integer(
            budgets.get("falsification_candidates_per_round"),
            "budgets.falsification_candidates_per_round",
            minimum=1,
        ),
        optimizer_random_seed_base=_integer(
            seeds.get("optimizer_random_base"), "seeds.optimizer_random_base"
        ),
        optimizer_tpe_seed_base=_integer(
            seeds.get("optimizer_tpe_base"), "seeds.optimizer_tpe_base"
        ),
        falsification_seed_base=_integer(
            seeds.get("falsification_base"), "seeds.falsification_base"
        ),
        heldout_case_ids=heldout,
        initial_regression_case_ids=initial_regression,
        stop_on_no_new_admissible_case=no_discovery,
        optimization_plateau_rounds=plateau_rounds,
        additional_input_files=additional_files,
    )


@dataclass(frozen=True, slots=True)
class RegressionCase:
    """A corpus case passed forward as immutable regression memory."""

    case_id: str
    origin_round: int | None
    candidate_id: str | None
    scenario: Mapping[str, Any] | None = None
    source_evidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Copy and recursively freeze JSON case payloads against adapter mutation."""
        for field_name in ("scenario", "source_evidence"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise TypeError(f"RegressionCase.{field_name} must be a mapping or null")
            detached = json.loads(_canonical_json(_deep_thaw(value)))
            object.__setattr__(self, field_name, _deep_freeze(detached))

    def to_json(self) -> dict[str, Any]:
        """Return a detached JSON-safe record for the next round manifest."""
        return {
            "case_id": self.case_id,
            "origin_round": self.origin_round,
            "candidate_id": self.candidate_id,
            "scenario": _deep_thaw(self.scenario) if self.scenario is not None else None,
            "source_evidence": _deep_thaw(self.source_evidence)
            if self.source_evidence is not None
            else None,
        }


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RoundRequest:
    """Frozen inputs shared by owner adapters for one round."""

    run_id: str
    round_number: int
    round_dir: Path
    optimizer_trials_per_method: int
    falsification_candidates: int
    optimizer_random_seed: int
    optimizer_tpe_seed: int
    falsification_seed: int
    falsification_sampler: str
    heldout_case_ids: tuple[str, ...]
    regression_cases: tuple[RegressionCase, ...]
    previous_selected_planner: Mapping[str, Any] | None
    source_revision: str
    config: CoevolutionConfig

    def __post_init__(self) -> None:
        """Pin the prior planner identity shared with every adapter in this round."""
        planner = self.previous_selected_planner
        if planner is None:
            return
        detached = json.loads(_canonical_json(_deep_thaw(planner)))
        object.__setattr__(self, "previous_selected_planner", _deep_freeze(detached))

    @property
    def regression_case_ids(self) -> tuple[str, ...]:
        """Return the exact case IDs in this round's frozen regression input."""
        return tuple(case.case_id for case in self.regression_cases)

    def to_json(self) -> dict[str, Any]:
        """Serialize every budget, seed, case identity, and named source input."""
        return {
            "run_id": self.run_id,
            "round_number": self.round_number,
            "source_revision": self.source_revision,
            "budgets": {
                "optimizer_trials_per_method": self.optimizer_trials_per_method,
                "falsification_candidates": self.falsification_candidates,
            },
            "seeds": {
                "optimizer_random": self.optimizer_random_seed,
                "optimizer_tpe": self.optimizer_tpe_seed,
                "falsification": self.falsification_seed,
            },
            "heldout_case_ids": list(self.heldout_case_ids),
            "regression_cases": [case.to_json() for case in self.regression_cases],
            "previous_selected_planner": (
                _deep_thaw(self.previous_selected_planner)
                if self.previous_selected_planner is not None
                else None
            ),
            "optimizer_config_path": str(self.config.optimizer_config),
            "scenario_template_path": str(self.config.scenario_template),
            "search_space_path": str(self.config.search_space),
            "policy": self.config.policy,
            "objective": self.config.objective,
            "search_execution": {
                "sampler": self.falsification_sampler,
                "horizon": self.config.horizon,
                "dt": self.config.dt,
                "workers": self.config.workers,
                "benchmark_profile": self.config.benchmark_profile,
                "record_forces": self.config.record_forces,
                "require_certification": self.config.require_certification,
            },
        }


class OptimizerAdapter(Protocol):
    """Adapter over the canonical bounded planner optimizer."""

    def __call__(self, request: RoundRequest, output_dir: Path) -> Mapping[str, Any]:
        """Run a fixed-budget optimizer and return its normalized result."""


class ChallengeEvaluatorAdapter(Protocol):
    """Adapter over the canonical held-out/current-challenge evaluator."""

    def __call__(
        self, request: RoundRequest, selected_planner: Mapping[str, Any], output_dir: Path
    ) -> Mapping[str, Any]:
        """Return one result row for each held-out and regression case ID."""


class FalsificationAdapter(Protocol):
    """Adapter over ``robot_sf.adversarial.search.run_adversarial_search``."""

    def __call__(
        self, request: RoundRequest, selected_planner: Mapping[str, Any], output_dir: Path
    ) -> Mapping[str, Any]:
        """Return every search candidate, including invalid and failed rows."""


@dataclass(frozen=True, slots=True)
class PreparedFalsificationSearch:
    """Round-owned #9645 input prepared from a verified #9650 optimizer result.

    The prepared object is intentionally separate from search execution. Callers can pass
    ``config`` and ``sampler`` to ``run_adversarial_search``; fixture tests can inspect the
    resulting ``algo_config_path`` without starting a simulator or candidate search.
    """

    config: SearchConfig
    sampler: CandidateSampler
    planner_binding: Mapping[str, Any]
    artifacts: tuple[Mapping[str, str], ...]

    def phase_provenance(self) -> dict[str, Any]:
        """Return JSON-safe provenance for the co-evolution falsification phase."""
        return {
            "planner_config_binding": _deep_thaw(self.planner_binding),
            "search_config": self.config.to_json(),
            "artifacts": [_deep_thaw(artifact) for artifact in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class _OptimizerSearchSource:
    """Digest-bound optimizer artifacts selected for one falsification round."""

    planner: Mapping[str, Any]
    config_path: Path
    registry_path: Path
    manifest_path: Path
    config_sha256: str
    registry_sha256: str
    manifest_sha256: str


def _validate_optimizer_search_source(
    request: RoundRequest, selected_planner: Mapping[str, Any]
) -> _OptimizerSearchSource:
    """Validate that the selected candidate is the completed current-round optimizer output."""
    planner, source_config_path = _validate_selected_planner(selected_planner, request)
    optimization_dir = (request.round_dir / "optimization").resolve()
    expected_source_path = (optimization_dir / "best_candidate.yaml").resolve()
    if source_config_path.resolve() != expected_source_path:
        raise ValueError(
            "selected optimizer config must be the current round's best_candidate.yaml"
        )

    registry_path = optimization_dir / "candidate_registry.yaml"
    optimizer_manifest_path = optimization_dir / "run_manifest.json"
    if not registry_path.is_file() or not optimizer_manifest_path.is_file():
        raise FileNotFoundError(
            "the optimizer output must include candidate_registry.yaml and run_manifest.json"
        )
    source_digest = _sha256_file(source_config_path)
    if source_digest != planner["config_sha256"]:
        raise CoevolutionError("selected optimizer config changed during search preparation")
    registry_digest = _sha256_file(registry_path)
    optimizer_manifest_digest = _sha256_file(optimizer_manifest_path)
    optimizer_manifest = _read_json(optimizer_manifest_path)
    if optimizer_manifest.get("schema") != "planner_optimizer_run.v1":
        raise ValueError("optimizer run manifest schema must be planner_optimizer_run.v1")
    if optimizer_manifest.get("status") != "complete":
        raise ValueError("optimizer run manifest must be complete before search preparation")

    planner_id = planner["planner_id"]
    selected_record = _mapping(optimizer_manifest.get("selected"), "optimizer.selected")
    if selected_record.get("candidate_name") != planner_id:
        raise ValueError("selected planner ID differs from the optimizer manifest candidate name")
    if selected_record.get("candidate_config_path") != source_config_path.name:
        raise ValueError("optimizer manifest does not select the bound candidate config file")
    if optimizer_manifest.get("selected_candidate_config") != source_config_path.name:
        raise ValueError("optimizer manifest selected_candidate_config does not match its output")
    return _OptimizerSearchSource(
        planner=planner,
        config_path=source_config_path.resolve(),
        registry_path=registry_path.resolve(),
        manifest_path=optimizer_manifest_path.resolve(),
        config_sha256=source_digest,
        registry_sha256=registry_digest,
        manifest_sha256=optimizer_manifest_digest,
    )


def _load_optimizer_runtime_config(
    request: RoundRequest, source: _OptimizerSearchSource
) -> tuple[str, Mapping[str, Any]]:
    """Use the canonical policy-search loader to resolve wrapper plus overrides."""
    # This is the canonical owner loader for the policy-search candidate wrapper format.
    loader_module = import_module("scripts.validation.run_policy_search_candidate")
    load_candidate_definition = loader_module.load_candidate_definition
    planner_id = source.planner["planner_id"]
    _entry, candidate_payload, effective_config, loaded_candidate_path = load_candidate_definition(
        source.registry_path, planner_id
    )
    if loaded_candidate_path.resolve() != source.config_path:
        raise ValueError("optimizer candidate registry resolves to a different selected config")
    if candidate_payload.get("name") != planner_id:
        raise ValueError("optimizer candidate wrapper name differs from the selected planner ID")
    candidate_policy = _non_empty_string(candidate_payload.get("algo"), "candidate.algo").lower()
    if candidate_policy != request.config.policy.lower():
        raise ValueError(
            "optimizer candidate policy differs from the configured falsification policy: "
            f"{candidate_policy!r} != {request.config.policy.lower()!r}"
        )
    if not isinstance(effective_config, Mapping):
        raise TypeError("canonical optimizer candidate loader must return a mapping config")
    return candidate_policy, effective_config


def _capture_previous_planner(
    request: RoundRequest,
) -> tuple[dict[str, Any] | None, Path | None, bytes | None]:
    """Record the previous planner identity and retain its exact config bytes."""
    previous_binding: dict[str, Any] | None = None
    previous_path: Path | None = None
    previous_bytes: bytes | None = None
    if request.previous_selected_planner is not None:
        previous = _deep_thaw(request.previous_selected_planner)
        previous_path = Path(
            _non_empty_string(previous.get("config_path"), "previous_selected_planner.config_path")
        ).resolve()
        previous_digest = _non_empty_string(
            previous.get("config_sha256"), "previous_selected_planner.config_sha256"
        ).lower()
        previous_bytes = previous_path.read_bytes()
        if _sha256_bytes(previous_bytes) != previous_digest:
            raise CoevolutionError("previous-round selected planner config digest mismatch")
        previous_binding = {
            "planner_id": _non_empty_string(
                previous.get("planner_id"), "previous_selected_planner.planner_id"
            ),
            "config_path": _relative_run_path(previous_path, request.config.output_dir),
            "config_sha256": previous_digest,
        }
    return previous_binding, previous_path, previous_bytes


def _build_search_planner_binding(
    request: RoundRequest,
    source: _OptimizerSearchSource,
    *,
    candidate_policy: str,
    runtime_config_path: Path,
    runtime_config_digest: str,
    runtime_config_json_digest: str,
    previous_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind optimizer selection, runtime config, prior planner, and finite search inputs."""
    planner_id = source.planner["planner_id"]
    return {
        "schema": "coevolution_search_planner_binding.v1",
        "run_id": request.run_id,
        "round_number": request.round_number,
        "source_revision": request.source_revision,
        "previous_selected_planner": dict(previous_binding) if previous_binding else None,
        "selected_planner": {
            "planner_id": planner_id,
            "optimizer_config_path": _relative_run_path(
                source.config_path, request.config.output_dir
            ),
            "optimizer_config_sha256": source.config_sha256,
            "candidate_registry_path": _relative_run_path(
                source.registry_path, request.config.output_dir
            ),
            "candidate_registry_sha256": source.registry_sha256,
            "optimizer_manifest_path": _relative_run_path(
                source.manifest_path, request.config.output_dir
            ),
            "optimizer_manifest_sha256": source.manifest_sha256,
        },
        "runtime_planner_config": {
            "policy": candidate_policy,
            "path": _relative_run_path(runtime_config_path, request.config.output_dir),
            "sha256": runtime_config_digest,
            "canonical_config_sha256": runtime_config_json_digest,
        },
        "search_budget": request.falsification_candidates,
        "search_seed": request.falsification_seed,
        "search_sampler": request.falsification_sampler,
    }


def _verify_falsification_inputs_unchanged(
    source: _OptimizerSearchSource,
    previous_path: Path | None,
    previous_bytes: bytes | None,
) -> None:
    """Reject concurrent edits to optimizer artifacts or the previous planner config."""
    for path, expected, label in (
        (source.config_path, source.config_sha256, "selected optimizer config"),
        (source.registry_path, source.registry_sha256, "optimizer candidate registry"),
        (source.manifest_path, source.manifest_sha256, "optimizer run manifest"),
    ):
        if _sha256_file(path) != expected:
            raise CoevolutionError(f"{label} changed during search preparation: {path}")
    if previous_bytes is not None and previous_path is not None:
        if previous_path.read_bytes() != previous_bytes:
            raise CoevolutionError(
                "previous-round selected planner config changed during preparation"
            )


def _falsification_artifact_claims(
    request: RoundRequest,
    source: _OptimizerSearchSource,
    runtime_config_path: Path,
    provenance_path: Path,
    previous_path: Path | None,
) -> tuple[Mapping[str, str], ...]:
    """Return hashes for every exact source and round-owned config artifact."""
    phase_dir = request.round_dir / "phases"
    artifact_paths = [
        source.config_path,
        source.registry_path,
        source.manifest_path,
        runtime_config_path,
        provenance_path,
    ]
    if previous_path is not None:
        artifact_paths.append(previous_path)
    return tuple(
        {
            "path": os.path.relpath(path.resolve(), start=phase_dir.resolve()),
            "sha256": _sha256_file(path),
        }
        for path in artifact_paths
    )


def prepare_falsification_search(
    request: RoundRequest,
    selected_planner: Mapping[str, Any],
) -> PreparedFalsificationSearch:
    """Resolve optimizer output into #9645's direct per-round planner config.

    The selected optimizer wrapper is loaded through its canonical owner, then written
    with immutable create-or-verify semantics. Search receives the direct config path;
    all selected, previous-round, and runtime config identities remain digest-bound.
    """
    source = _validate_optimizer_search_source(request, selected_planner)
    candidate_policy, effective_config = _load_optimizer_runtime_config(request, source)
    runtime_config_bytes = yaml.safe_dump(
        dict(effective_config), sort_keys=True, allow_unicode=True
    ).encode("utf-8")
    runtime_config_digest = _sha256_bytes(runtime_config_bytes)
    runtime_config_json_digest = _sha256_bytes(_canonical_json(dict(effective_config)))
    falsification_dir = request.round_dir / "falsification"
    algo_config_path = falsification_dir / "planner_algo_config.yaml"
    provenance_path = falsification_dir / "planner_config_provenance.json"
    _write_or_verify_bytes(algo_config_path, runtime_config_bytes)
    previous_binding, previous_path, previous_bytes = _capture_previous_planner(request)
    binding = _build_search_planner_binding(
        request,
        source,
        candidate_policy=candidate_policy,
        runtime_config_path=algo_config_path,
        runtime_config_digest=runtime_config_digest,
        runtime_config_json_digest=runtime_config_json_digest,
        previous_binding=previous_binding,
    )
    _write_or_verify_bytes(provenance_path, _canonical_json(binding) + b"\n")
    _verify_falsification_inputs_unchanged(source, previous_path, previous_bytes)

    search_config = SearchConfig.from_files(
        policy=candidate_policy,
        scenario_template=request.config.scenario_template,
        search_space=request.config.search_space,
        objective=request.config.objective,
        output_dir=falsification_dir / "search",
        budget=request.falsification_candidates,
        seed=request.falsification_seed,
        algo_config_path=algo_config_path,
        horizon=request.config.horizon,
        dt=request.config.dt,
        workers=request.config.workers,
        record_forces=request.config.record_forces,
        require_certification=request.config.require_certification,
        benchmark_profile=request.config.benchmark_profile,
    )
    search_config.validate()
    sampler = build_sampler(
        request.falsification_sampler,
        search_config.search_space,
        seed=request.falsification_seed,
    )
    artifacts = _falsification_artifact_claims(
        request, source, algo_config_path, provenance_path, previous_path
    )
    return PreparedFalsificationSearch(
        config=search_config,
        sampler=sampler,
        planner_binding=binding,
        artifacts=artifacts,
    )


def _search_artifact_path(value: Any, *, search_output_dir: Path, field: str) -> Path | None:
    """Resolve a search-manifest path and keep it inside this round's search output."""
    if value is None:
        return None
    raw = Path(_non_empty_string(value, field))
    path = raw.resolve() if raw.is_absolute() else (search_output_dir / raw).resolve()
    try:
        path.relative_to(search_output_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} points outside the round search output: {path}") from exc
    return path


def _search_scenario_payload(row: Mapping[str, Any], *, search_output_dir: Path) -> dict[str, Any]:
    """Retain sampled controls and the materialized scenario/route input when present."""
    candidate = dict(_mapping(row.get("candidate"), "search candidate"))
    scenario_path = _search_artifact_path(
        row.get("scenario_yaml_path"),
        search_output_dir=search_output_dir,
        field="search candidate scenario_yaml_path",
    )
    bundle_path = _search_artifact_path(
        row.get("bundle_path"),
        search_output_dir=search_output_dir,
        field="search candidate bundle_path",
    )
    if scenario_path is None or not scenario_path.is_file():
        return {"materialized": False, "sampled_candidate": candidate}
    scenario_payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(scenario_payload, Mapping):
        raise TypeError(f"search scenario artifact must contain a mapping: {scenario_path}")
    scenarios = scenario_payload.get("scenarios")
    if (
        not isinstance(scenarios, list)
        or len(scenarios) != 1
        or not isinstance(scenarios[0], Mapping)
    ):
        raise ValueError(
            f"search scenario artifact must contain exactly one scenario: {scenario_path}"
        )
    route_path = (
        _search_artifact_path(
            (bundle_path / "route_overrides.yaml").as_posix(),
            search_output_dir=search_output_dir,
            field="search candidate route_overrides.yaml",
        )
        if bundle_path is not None
        else None
    )
    route_payload: Mapping[str, Any] = {}
    if route_path is not None and route_path.is_file():
        loaded_route = yaml.safe_load(route_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded_route, Mapping):
            raise TypeError(f"search route artifact must contain a mapping: {route_path}")
        route_payload = loaded_route
    return {
        "materialized": True,
        "sampled_candidate": candidate,
        "scenario": dict(scenarios[0]),
        "route_overrides": dict(route_payload),
        "scenario_file": _relative_run_path(scenario_path, search_output_dir.parent.parent),
        "scenario_file_sha256": _sha256_file(scenario_path),
        "route_file": (
            _relative_run_path(route_path, search_output_dir.parent.parent)
            if route_path is not None and route_path.is_file()
            else None
        ),
        "route_file_sha256": _sha256_file(route_path)
        if route_path is not None and route_path.is_file()
        else None,
    }


def _search_execution_status(row: Mapping[str, Any], search_status: str) -> str:
    """Map canonical benchmark availability into the co-evolution status vocabulary."""
    if search_status == "invalid":
        return "not_run"
    if search_status == "failed":
        return "failed"
    attribution = row.get("failure_attribution")
    if not isinstance(attribution, Mapping):
        return "unknown"
    details = attribution.get("details")
    if not isinstance(details, Mapping):
        return "unknown"
    availability = str(details.get("availability_status", "")).strip().lower()
    readiness = str(details.get("readiness_status", "")).strip().lower()
    execution_mode = str(details.get("execution_mode", "")).strip().lower()
    if availability == "fallback" or readiness == "fallback":
        return "fallback"
    if availability == "degraded" or readiness == "degraded":
        return "degraded"
    if availability in {"failed", "partial-failure"}:
        return "failed"
    if availability == "available" and execution_mode in {"native", "adapter", "mixed"}:
        return "ok"
    return "unknown"


def _search_planner_outcome(row: Mapping[str, Any], *, execution_status: str) -> str:
    """Use explicit canonical failure attribution only for normal executions."""
    if execution_status != "ok":
        return "unknown"
    attribution = row.get("failure_attribution")
    if not isinstance(attribution, Mapping):
        return "unknown"
    primary = attribution.get("primary_failure")
    if primary == "success":
        return "no_failure"
    if primary in {"collision", "severe_intrusion", "timeout", "incomplete"}:
        return "confirmed_failure"
    return "unknown"


def _search_candidate_status(row: Mapping[str, Any]) -> str:
    """Distinguish rejected inputs from failed executions and completed evaluations."""
    attribution = row.get("failure_attribution")
    if isinstance(attribution, Mapping) and (
        attribution.get("status") == "not_evaluated"
        and attribution.get("primary_failure") == "invalid_candidate"
    ):
        return "invalid"
    if row.get("error") or (
        isinstance(attribution, Mapping) and attribution.get("status") == "evaluation_failed"
    ):
        return "failed"
    objective_value = row.get("objective_value")
    if objective_value is not None:
        if (
            isinstance(objective_value, int | float)
            and not isinstance(objective_value, bool)
            and math.isfinite(float(objective_value))
        ):
            return "evaluated"
        return "failed"
    # A completed evaluator can produce attribution even when its objective metric is absent.
    # Preserve that as an evaluation, since planner outcome and objective availability differ.
    if isinstance(attribution, Mapping) and attribution.get("status") == "attributed":
        return "evaluated"
    return "failed"


def _normalize_search_manifest_row(
    row: Any,
    *,
    index: int,
    request: RoundRequest,
    search_output_dir: Path,
    manifest_digest: str,
) -> dict[str, Any]:
    """Preserve one owner search row with fail-closed status and stable identity."""
    source = _mapping(row, f"search manifest candidates[{index}]")
    candidate = _mapping(source.get("candidate"), f"search candidates[{index}].candidate")
    objective_value = source.get("objective_value")
    error = source.get("error")
    search_status = _search_candidate_status(source)
    execution_status = _search_execution_status(source, search_status)
    planner_outcome = (
        "not_assessed"
        if search_status != "evaluated"
        else _search_planner_outcome(source, execution_status=execution_status)
    )
    candidate_digest = _sha256_bytes(
        _canonical_json(
            {
                "run_id": request.run_id,
                "round_number": request.round_number,
                "seed": request.falsification_seed,
                "index": index,
                "candidate": dict(candidate),
            }
        )
    )
    return {
        "candidate_id": f"r{request.round_number:03d}-c{index:04d}-{candidate_digest[:12]}",
        "search_status": search_status,
        "execution_status": execution_status,
        "planner_outcome": planner_outcome,
        "scenario": _search_scenario_payload(source, search_output_dir=search_output_dir),
        "objective_value": objective_value,
        "error": error,
        "search_manifest_candidate_index": index,
        "search_manifest_candidate_sha256": _sha256_bytes(_canonical_json(dict(source))),
        "search_manifest_sha256": manifest_digest,
    }


class ProductionFalsificationAdapter:
    """Run the canonical bounded search against the selected optimizer config.

    Evaluator and certifier injection keeps this adapter testable without simulator work.
    The default path delegates to ``run_adversarial_search`` and its canonical evaluator.
    """

    def __init__(self, *, evaluator: Any = None, certifier: Any = None) -> None:
        self._evaluator = evaluator
        self._certifier = certifier

    def __call__(
        self,
        request: RoundRequest,
        selected_planner: Mapping[str, Any],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Execute one configured search and return all candidate dispositions."""
        expected_output = (request.round_dir / "falsification").resolve()
        if output_dir.resolve() != expected_output:
            raise ValueError("falsification adapter output_dir must be the round-owned directory")
        prepared = prepare_falsification_search(request, selected_planner)
        search_module = import_module("robot_sf.adversarial.search")
        run_adversarial_search = search_module.run_adversarial_search
        result = run_adversarial_search(
            prepared.config,
            evaluator=self._evaluator,
            certifier=self._certifier,
            sampler=prepared.sampler,
        )
        search_output_dir = prepared.config.output_dir.resolve()
        manifest_path = Path(result.manifest_path).resolve()
        if manifest_path.parent != search_output_dir or not manifest_path.is_file():
            raise CoevolutionError("search runner returned a manifest outside its round output")
        manifest = _read_json(manifest_path)
        if _canonical_json(manifest.get("config")) != _canonical_json(prepared.config.to_json()):
            raise CoevolutionError("search manifest config differs from the prepared round config")
        rows = manifest.get("candidates")
        if not isinstance(rows, list) or len(rows) != request.falsification_candidates:
            raise CoevolutionError("search manifest candidate rows differ from the frozen budget")
        manifest_digest = _sha256_file(manifest_path)
        candidates = [
            _normalize_search_manifest_row(
                row,
                index=index,
                request=request,
                search_output_dir=search_output_dir,
                manifest_digest=manifest_digest,
            )
            for index, row in enumerate(rows)
        ]
        phase_dir = request.round_dir / "phases"
        output_files = sorted(path for path in search_output_dir.rglob("*") if path.is_file())
        if any(path.is_symlink() for path in output_files):
            raise CoevolutionError("search output contains an untrusted symlink artifact")
        artifacts = [dict(item) for item in prepared.artifacts]
        artifacts.extend(
            {
                "path": os.path.relpath(path.resolve(), start=phase_dir.resolve()),
                "sha256": _sha256_file(path),
            }
            for path in output_files
        )
        summary = _mapping(manifest.get("summary"), "search manifest summary")
        return {
            "status": "complete",
            "candidate_budget": request.falsification_candidates,
            "seed": request.falsification_seed,
            "sampler": request.falsification_sampler,
            "candidates": candidates,
            "invalid_count": summary.get("num_invalid_candidates"),
            "failed_count": summary.get("num_failed_evaluations"),
            "planner_config_binding": _deep_thaw(prepared.planner_binding),
            "search_output": {
                "manifest_path": _relative_run_path(manifest_path, request.config.output_dir),
                "manifest_sha256": manifest_digest,
                "config_sha256": _sha256_bytes(_canonical_json(manifest["config"])),
                "candidate_count": len(candidates),
            },
            "artifacts": artifacts,
        }


class DiscoveryVerifierAdapter(Protocol):
    """Adapter over replay plus reviewed feasibility/admissibility contracts."""

    def __call__(
        self,
        request: RoundRequest,
        selected_planner: Mapping[str, Any],
        candidate: Mapping[str, Any],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Return separately classified replay, feasibility, admissibility, and outcome."""


class CorpusAdmissionAdapter(Protocol):
    """Adapter over the reviewed counterexample corpus owner."""

    def __call__(
        self,
        request: RoundRequest,
        selected_planner: Mapping[str, Any],
        case: Mapping[str, Any],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Return an explicit admitted/duplicate/rejected corpus result."""


@dataclass(frozen=True, slots=True)
class CoevolutionAdapters:
    """Injected owner adapters; fixture tests use deterministic implementations."""

    optimize: OptimizerAdapter
    evaluate_challenges: ChallengeEvaluatorAdapter
    falsify: FalsificationAdapter
    verify_discovery: DiscoveryVerifierAdapter
    admit_case: CorpusAdmissionAdapter


def _git_identity(repository_root: Path, output_dir: Path) -> dict[str, Any]:
    """Capture exact revision plus non-output source dirt for resume validation."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        output_relative = output_dir.resolve().relative_to(repository_root.resolve()).as_posix()
        exclude = [f":(exclude){output_relative}/**", f":(exclude){output_relative}"]
    except ValueError:
        exclude = []
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary", "--", ".", *exclude],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout
    untracked_rows: list[dict[str, str]] = []
    output_prefix = output_relative.rstrip("/") + "/" if exclude else None
    for raw_path in untracked.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        if output_prefix and (relative == output_relative or relative.startswith(output_prefix)):
            continue
        file_path = repository_root / relative
        if file_path.is_file():
            untracked_rows.append({"path": relative, "sha256": _sha256_file(file_path)})
        else:
            untracked_rows.append({"path": relative, "sha256": "non_file"})
    payload = {
        "revision": revision,
        "tracked_diff_sha256": _sha256_bytes(diff),
        "untracked_files": sorted(untracked_rows, key=lambda row: row["path"]),
    }
    payload["identity_sha256"] = _sha256_bytes(_canonical_json(payload))
    return payload


def _input_identities(config: CoevolutionConfig) -> list[dict[str, str]]:
    return [
        {"path": str(path), "sha256": _sha256_file(path)}
        for path in sorted(config.input_files, key=str)
    ]


@contextmanager
def _run_lock(output_dir: Path):
    if fcntl is None:
        raise CoevolutionError("co-evolution runs require POSIX fcntl.flock process locking")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".coevolution.lock"
    with lock_path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CoevolutionError(f"another co-evolution process owns {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _verify_static_identity(
    manifest: Mapping[str, Any],
    config: CoevolutionConfig,
    repository_root: Path,
    *,
    compare_git: bool = True,
) -> None:
    if manifest.get("schema") != RUN_SCHEMA or manifest.get("run_id") != config.run_id:
        raise CoevolutionError("existing run manifest schema or run_id differs from config")
    if manifest.get("config_sha256") != config.config_sha256:
        raise CoevolutionError("config bytes changed since the run was initialized")
    actual_inputs = _input_identities(config)
    if manifest.get("input_files") != actual_inputs:
        raise CoevolutionError(
            "one or more named input files changed since the run was initialized"
        )
    if manifest.get("environment") != _runtime_environment():
        raise CoevolutionError("runtime environment identity changed since the run was initialized")
    if compare_git:
        actual_git = _git_identity(repository_root, config.output_dir)
        if manifest.get("source_identity") != actual_git:
            raise CoevolutionError("source revision or non-output working-tree bytes changed")


def _phase_path(round_dir: Path, phase: str) -> Path:
    return round_dir / "phases" / f"{phase}.json"


def _verify_artifact_claims(output: Mapping[str, Any], *, phase_dir: Path) -> None:
    artifacts = output.get("artifacts", [])
    if artifacts is None:
        return
    if not isinstance(artifacts, list):
        raise TypeError("adapter artifacts must be a list of {path, sha256} records")
    for index, raw in enumerate(artifacts):
        artifact = _mapping(raw, f"artifacts[{index}]")
        path_value = _non_empty_string(artifact.get("path"), f"artifacts[{index}].path")
        expected = _non_empty_string(artifact.get("sha256"), f"artifacts[{index}].sha256")
        artifact_path = Path(path_value)
        if not artifact_path.is_absolute():
            artifact_path = phase_dir / artifact_path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"adapter-declared artifact is missing: {artifact_path}")
        actual = _sha256_file(artifact_path)
        if actual != expected:
            raise ValueError(f"adapter artifact digest mismatch: {artifact_path}")


def _verify_selected_planner_artifact(output: Mapping[str, Any], *, artifact_dir: Path) -> None:
    planner = _mapping(output.get("selected_planner"), "optimizer.selected_planner")
    path_value = _non_empty_string(planner.get("config_path"), "selected_planner.config_path")
    expected = _non_empty_string(planner.get("config_sha256"), "selected_planner.config_sha256")
    artifact_path = Path(path_value)
    if not artifact_path.is_absolute():
        artifact_path = artifact_dir / artifact_path
    if not artifact_path.is_file():
        raise CoevolutionError(f"selected planner config is missing: {artifact_path}")
    if _sha256_file(artifact_path) != expected:
        raise CoevolutionError(f"selected planner config digest mismatch: {artifact_path}")


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int) -> None:
    """Atomically restore selected planner bytes without changing file permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_verify_bytes(path: Path, payload: bytes) -> None:
    """Create an immutable round artifact or accept an exact-byte resume match."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise CoevolutionError(f"round artifact must not be a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise CoevolutionError(f"round artifact already exists with different bytes: {path}")
        return
    _atomic_write_bytes(path, payload, mode=0o644)


def _relative_run_path(path: Path, run_dir: Path) -> str:
    """Return a stable run-relative artifact path, refusing paths outside the run."""
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"round artifact is outside the co-evolution output directory: {path}"
        ) from exc


def _call_with_selected_planner_integrity(
    selected_planner: Mapping[str, Any],
    boundary: str,
    call: Callable[[], Any],
) -> Any:
    """Run an adapter only while its selected planner config stays digest-bound.

    Adapters receive detached planner descriptors, but the referenced file remains
    shared state. Snapshot and verify its bytes at the filesystem boundary, restore
    any mutation, and fail the owning phase so stale config hashes cannot propagate.
    """
    config_path = Path(
        _non_empty_string(selected_planner.get("config_path"), "selected_planner.config_path")
    ).resolve()
    expected = _non_empty_string(
        selected_planner.get("config_sha256"), "selected_planner.config_sha256"
    ).lower()
    try:
        original_bytes = config_path.read_bytes()
        original_mode = stat.S_IMODE(config_path.stat().st_mode)
    except OSError as exc:
        raise CoevolutionError(
            f"selected planner config is unavailable before {boundary}: {config_path}"
        ) from exc
    if _sha256_bytes(original_bytes) != expected:
        raise CoevolutionError(
            f"selected planner config digest mismatch before {boundary}: {config_path}"
        )

    try:
        return call()
    finally:
        try:
            current_bytes = config_path.read_bytes()
        except OSError:
            current_bytes = None
        if current_bytes is None or _sha256_bytes(current_bytes) != expected:
            try:
                _atomic_write_bytes(config_path, original_bytes, mode=original_mode)
            except OSError as exc:
                raise CoevolutionError(
                    f"selected planner config changed during {boundary} and could not be restored: "
                    f"{config_path}"
                ) from exc
            if _sha256_file(config_path) != expected:
                raise CoevolutionError(
                    f"selected planner config changed during {boundary}; restoration did not "
                    f"restore its declared digest: {config_path}"
                )
            raise CoevolutionError(
                f"selected planner config changed during {boundary}; original bytes were restored: "
                f"{config_path}"
            )


def _verify_phase_file(
    round_dir: Path, phase: str, phase_record: Mapping[str, Any]
) -> dict[str, Any]:
    path = _phase_path(round_dir, phase)
    expected_file_sha = phase_record.get("file_sha256")
    if not path.is_file() or not isinstance(expected_file_sha, str):
        raise CoevolutionError(f"completed phase artifact is missing: {path}")
    if _sha256_file(path) != expected_file_sha:
        raise CoevolutionError(f"completed phase artifact digest changed: {path}")
    payload = _read_json(path)
    if payload.get("schema") != PHASE_SCHEMA or payload.get("phase") != phase:
        raise CoevolutionError(f"phase artifact contract mismatch: {path}")
    if payload.get("status") != phase_record.get("status"):
        raise CoevolutionError(f"phase status differs from round manifest: {path}")
    if payload.get("output_sha256") != _sha256_bytes(_canonical_json(payload.get("output"))):
        raise CoevolutionError(f"phase output digest changed: {path}")
    _verify_artifact_claims(payload.get("output", {}), phase_dir=path.parent)
    if phase == "optimization":
        _verify_selected_planner_artifact(
            payload.get("output", {}), artifact_dir=round_dir / "optimization"
        )
    return payload


def _execute_phase(
    *,
    phase: str,
    request: RoundRequest,
    round_state: dict[str, Any],
    round_dir: Path,
    run_manifest: dict[str, Any],
    execute: Callable[[], Mapping[str, Any]],
    validate: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    phase_records = round_state.setdefault("phases", {})
    phase_input_sha = _sha256_bytes(_canonical_json(request.to_json()))
    existing = phase_records.get(phase)
    if existing is not None:
        status = existing.get("status")
        if existing.get("input_sha256") != phase_input_sha:
            raise CoevolutionError(
                f"phase {phase} input digest differs from the current frozen round request"
            )
        if status == "complete":
            return _verify_phase_file(round_dir, phase, existing)["output"]
        if status in {"running", "failed"}:
            raise CoevolutionInfrastructureError(
                f"phase {phase} in round {request.round_number} is {status}; automatic rerun is refused"
            )
        raise CoevolutionError(f"unrecognized persisted phase status for {phase}: {status!r}")

    phase_records[phase] = {
        "status": "running",
        "input_sha256": phase_input_sha,
        "artifact": str(_phase_path(round_dir, phase).relative_to(request.config.output_dir)),
    }
    round_state["status"] = "running"
    _write_manifests(request.config, run_manifest, round_state)
    raw_output: dict[str, Any] | None = None
    try:
        raw_value = execute()
        raw_mapping = _mapping(raw_value, f"{phase} adapter result")
        raw_output = json.loads(_canonical_json(raw_mapping))
        normalized = validate(raw_output)
        _verify_artifact_claims(normalized, phase_dir=_phase_path(round_dir, phase).parent)
        output_sha = _sha256_bytes(_canonical_json(normalized))
        payload = {
            "schema": PHASE_SCHEMA,
            "phase": phase,
            "status": "complete",
            "round_number": request.round_number,
            "input_sha256": phase_input_sha,
            "output_sha256": output_sha,
            "output": normalized,
        }
        file_sha = _atomic_write_json(_phase_path(round_dir, phase), payload)
        phase_records[phase] = {
            "status": "complete",
            "input_sha256": phase_input_sha,
            "output_sha256": output_sha,
            "file_sha256": file_sha,
            "artifact": str(_phase_path(round_dir, phase).relative_to(request.config.output_dir)),
        }
        _write_manifests(request.config, run_manifest, round_state)
        return normalized
    except Exception as exc:
        failure_output: dict[str, Any] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if raw_output is not None:
            failure_output["raw_output"] = raw_output
        failure_payload = {
            "schema": PHASE_SCHEMA,
            "phase": phase,
            "status": "failed",
            "round_number": request.round_number,
            "input_sha256": phase_input_sha,
            "output_sha256": _sha256_bytes(_canonical_json(failure_output)),
            "output": failure_output,
        }
        file_sha = _atomic_write_json(_phase_path(round_dir, phase), failure_payload)
        phase_records[phase] = {
            "status": "failed",
            "input_sha256": phase_input_sha,
            "output_sha256": failure_payload["output_sha256"],
            "file_sha256": file_sha,
            "artifact": str(_phase_path(round_dir, phase).relative_to(request.config.output_dir)),
        }
        round_state["status"] = "diagnostic"
        round_state["failure"] = {"phase": phase, **failure_output}
        run_manifest["status"] = "diagnostic"
        run_manifest["stop"] = {
            "decision": "stop",
            "reason": "infrastructure_failure",
            "failed_phase": phase,
            "round_number": request.round_number,
            "claim_boundary": "No scientific conclusion is drawn from an incomplete round.",
        }
        _write_manifests(request.config, run_manifest, round_state)
        raise CoevolutionInfrastructureError(
            f"phase {phase} failed in round {request.round_number}: {exc}"
        ) from exc


def _validate_selected_planner(value: Any, request: RoundRequest) -> tuple[dict[str, Any], Path]:
    planner = dict(_mapping(value, "optimizer.selected_planner"))
    planner_id = _non_empty_string(planner.get("planner_id"), "selected_planner.planner_id")
    config_digest = planner.get("config_sha256")
    if (
        not isinstance(config_digest, str)
        or len(config_digest) != 64
        or any(character not in "0123456789abcdef" for character in config_digest.lower())
    ):
        raise ValueError("selected_planner.config_sha256 must be a SHA-256 hex digest")
    planner_config_path = _non_empty_string(
        planner.get("config_path", planner.get("artifact_path")),
        "selected_planner.config_path",
    )
    artifact_path = Path(planner_config_path)
    if not artifact_path.is_absolute():
        artifact_path = request.round_dir / "optimization" / artifact_path
    if not artifact_path.is_file():
        raise FileNotFoundError(f"selected planner config is missing: {artifact_path}")
    if _sha256_file(artifact_path) != config_digest.lower():
        raise ValueError(f"selected planner config digest mismatch: {artifact_path}")
    return (
        {
            **planner,
            "planner_id": planner_id,
            "config_path": str(artifact_path.resolve()),
            "config_sha256": config_digest.lower(),
        },
        artifact_path,
    )


def _validate_optimizer_score(raw: Mapping[str, Any]) -> list[Any]:
    score = raw.get("selection_tuple")
    if not isinstance(score, list) or not score:
        raise ValueError("optimizer.selection_tuple must be a non-empty list")
    for index, value in enumerate(score):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"optimizer.selection_tuple[{index}] must be finite or null")
    return list(score)


def _validate_optimizer_budget(raw: Mapping[str, Any], request: RoundRequest) -> None:
    budget = _mapping(raw.get("budget"), "optimizer.budget")
    expected_budget = request.optimizer_trials_per_method
    if budget.get("trials_per_method") != expected_budget:
        raise ValueError("optimizer trial budget differs from frozen round input")
    if budget.get("random_seed") != request.optimizer_random_seed:
        raise ValueError("optimizer Random seed differs from frozen round input")
    if budget.get("tpe_seed") != request.optimizer_tpe_seed:
        raise ValueError("optimizer TPE seed differs from frozen round input")


def _validate_optimizer_output(raw: dict[str, Any], request: RoundRequest) -> dict[str, Any]:
    if raw.get("status") != "complete":
        raise ValueError(f"optimizer status must be complete, got {raw.get('status')!r}")
    selected_planner, _ = _validate_selected_planner(raw.get("selected_planner"), request)
    score = _validate_optimizer_score(raw)
    improves = raw.get("improves_over_baseline")
    if not isinstance(improves, bool):
        raise TypeError("optimizer.improves_over_baseline must be boolean")
    baseline = raw.get("baseline_planner")
    if request.previous_selected_planner is None:
        if baseline is not None:
            raise ValueError("round-one optimizer must not declare a prior-round baseline planner")
    else:
        baseline = _mapping(baseline, "optimizer.baseline_planner")
        prior = request.previous_selected_planner
        for key in ("planner_id", "config_sha256"):
            if baseline.get(key) != prior.get(key):
                raise ValueError(
                    f"optimizer baseline {key} does not match the prior round's selected planner"
                )
    _validate_optimizer_budget(raw, request)
    return {
        **raw,
        "selected_planner": selected_planner,
        "baseline_planner": dict(baseline) if baseline is not None else None,
        "selection_tuple": score,
        "improves_over_baseline": improves,
    }


def _validate_challenge_evaluations(raw: dict[str, Any], request: RoundRequest) -> dict[str, Any]:
    if raw.get("status") != "complete":
        raise ValueError(f"challenge evaluation status must be complete, got {raw.get('status')!r}")
    for key, expected in (
        ("heldout_rows", request.heldout_case_ids),
        ("regression_rows", request.regression_case_ids),
    ):
        rows = raw.get(key)
        if not isinstance(rows, list):
            raise TypeError(f"challenge evaluation {key} must be a list")
        ids: list[str] = []
        for index, row in enumerate(rows):
            item = _mapping(row, f"{key}[{index}]")
            case_id = _non_empty_string(item.get("case_id"), f"{key}[{index}].case_id")
            status = _non_empty_string(item.get("status"), f"{key}[{index}].status")
            if status not in _REGRESSION_STATUSES:
                raise ValueError(f"unsupported challenge row status: {status!r}")
            ids.append(case_id)
        if ids != list(expected):
            raise ValueError(f"{key} identities do not exactly match the frozen round input")
    return raw


def _normalize_falsification_candidate(raw: Any, index: int, seen_ids: set[str]) -> dict[str, Any]:
    candidate = dict(_mapping(raw, f"falsification.candidates[{index}]"))
    candidate_id = _non_empty_string(
        candidate.get("candidate_id"), f"falsification.candidates[{index}].candidate_id"
    )
    if candidate_id in seen_ids:
        raise ValueError(f"duplicate falsification candidate_id: {candidate_id}")
    seen_ids.add(candidate_id)
    for key, domain in (
        ("search_status", _SEARCH_STATUSES),
        ("execution_status", _EXECUTION_STATUSES),
        ("planner_outcome", _PLANNER_OUTCOMES),
    ):
        value = _non_empty_string(candidate.get(key), f"{candidate_id}.{key}")
        if value not in domain:
            raise ValueError(f"unsupported {key} for {candidate_id}: {value!r}")
        candidate[key] = value
    if not isinstance(candidate.get("scenario"), Mapping):
        raise TypeError(f"{candidate_id}.scenario must preserve the sampled scenario mapping")
    candidate["scenario"] = dict(candidate["scenario"])
    return candidate


def _validate_falsification_output(raw: dict[str, Any], request: RoundRequest) -> dict[str, Any]:
    if raw.get("status") != "complete":
        raise ValueError(f"falsification status must be complete, got {raw.get('status')!r}")
    if raw.get("candidate_budget") != request.falsification_candidates:
        raise ValueError("falsification candidate budget differs from frozen round input")
    if raw.get("seed") != request.falsification_seed:
        raise ValueError("falsification seed differs from frozen round input")
    candidates = raw.get("candidates")
    if not isinstance(candidates, list):
        raise TypeError("falsification.candidates must be a list")
    sampler = _non_empty_string(raw.get("sampler"), "falsification.sampler").lower()
    if sampler != request.falsification_sampler:
        raise ValueError(
            "falsification sampler differs from the frozen round input: "
            f"expected {request.falsification_sampler!r}, got {sampler!r}"
        )
    if len(candidates) != request.falsification_candidates:
        raise ValueError("falsification did not preserve exactly one row per candidate budget slot")
    seen_ids: set[str] = set()
    normalized = [
        _normalize_falsification_candidate(candidate, index, seen_ids)
        for index, candidate in enumerate(candidates)
    ]
    return {**raw, "candidates": normalized}


def _case_id(request: RoundRequest, candidate: Mapping[str, Any]) -> str:
    identity = {
        "run_id": request.run_id,
        "origin_round": request.round_number,
        "candidate_id": candidate["candidate_id"],
        "selected_planner": candidate.get("target_planner_id"),
        "scenario": candidate["scenario"],
    }
    return f"ce-{_sha256_bytes(_canonical_json(identity))[:20]}"


def _validate_gate_output(raw: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    values: dict[str, str] = {}
    for key, allowed in (
        ("replay_status", _REPLAY_STATUSES),
        ("feasibility_status", _FEASIBILITY_STATUSES),
        ("admissibility_status", _ADMISSIBILITY_STATUSES),
        ("planner_outcome", _PLANNER_OUTCOMES),
    ):
        value = _non_empty_string(raw.get(key), f"{candidate_id}.{key}")
        if value not in allowed:
            raise ValueError(f"unsupported {key} for {candidate_id}: {value!r}")
        values[key] = value
    return {**dict(raw), **values}


def _admission_eligible(candidate: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    """Require exact replay, admissibility, and a confirmed target failure.

    Empirical feasibility is useful corroboration, but it is not required to retain an
    otherwise admissible counterexample. The corpus owns the explicit
    ``admissible_feasibility_unknown`` state; this coordinator must preserve it rather
    than dropping the case or upgrading it to feasible.
    """
    return all(
        (
            candidate.get("search_status") == "evaluated",
            candidate.get("execution_status") == "ok",
            candidate.get("planner_outcome") == "confirmed_failure",
            gate.get("replay_status") == "exact_match",
            gate.get("feasibility_status") in {"empirically_feasible", "unknown"},
            gate.get("admissibility_status") == "admissible",
            gate.get("planner_outcome") == "confirmed_failure",
        )
    )


def _candidate_gate(
    request: RoundRequest,
    planner: Mapping[str, Any],
    candidate: Mapping[str, Any],
    index: int,
    adapters: CoevolutionAdapters,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    if candidate["search_status"] != "evaluated":
        admissibility = "invalid" if candidate["search_status"] == "invalid" else "not_assessed"
        return (
            {
                "replay_status": "not_run",
                "feasibility_status": "not_assessed",
                "admissibility_status": admissibility,
                "planner_outcome": candidate["planner_outcome"],
            },
            None,
        )
    if candidate["execution_status"] != "ok":
        return (
            {
                "replay_status": "not_run",
                "feasibility_status": "not_assessed",
                "admissibility_status": "not_assessed",
                "planner_outcome": candidate["planner_outcome"],
            },
            None,
        )
    if candidate["planner_outcome"] != "confirmed_failure":
        return (
            {
                "replay_status": "not_run",
                "feasibility_status": "not_assessed",
                "admissibility_status": "not_assessed",
                "planner_outcome": candidate["planner_outcome"],
            },
            None,
        )
    try:
        result = _call_with_selected_planner_integrity(
            planner,
            "discovery verification",
            lambda: adapters.verify_discovery(
                request,
                _detached_json_mapping(planner, "selected planner for discovery verification"),
                _detached_json_mapping(candidate, "discovery candidate for verification"),
                output_dir / f"candidate_{index:04d}",
            ),
        )
        return _validate_gate_output(result, str(candidate["candidate_id"])), None
    except Exception as exc:  # noqa: BLE001 - preserve the row and stop below
        gate = {
            "replay_status": "error",
            "feasibility_status": "unknown",
            "admissibility_status": "unknown",
            "planner_outcome": "unknown",
            "processing_status": "infrastructure_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return gate, {"candidate_id": str(candidate["candidate_id"]), "error": str(exc)}


def _corpus_admission(
    request: RoundRequest,
    planner: Mapping[str, Any],
    candidate: Mapping[str, Any],
    gate: Mapping[str, Any],
    discovery_id: str,
    index: int,
    adapters: CoevolutionAdapters,
    output_dir: Path,
) -> tuple[dict[str, Any], RegressionCase | None, dict[str, str] | None]:
    case = {
        "discovery_id": discovery_id,
        "origin_round": request.round_number,
        "candidate_id": candidate["candidate_id"],
        "scenario": candidate["scenario"],
        "source_evidence": {
            "run_id": request.run_id,
            "falsification_round": request.round_number,
            "target_planner_id": planner["planner_id"],
            "candidate_id": candidate["candidate_id"],
            "gate": gate,
        },
    }
    try:
        raw = _call_with_selected_planner_integrity(
            planner,
            "corpus admission",
            lambda: adapters.admit_case(
                request,
                _detached_json_mapping(planner, "selected planner for corpus admission"),
                _detached_json_mapping(case, "counterexample for corpus admission"),
                output_dir / f"candidate_{index:04d}",
            ),
        )
        admission = json.loads(
            _canonical_json(_mapping(raw, f"corpus admission for {discovery_id}"))
        )
        status = admission.get("status")
        admitted = admission.get("admitted")
        if status not in {"admitted", "duplicate", "rejected"}:
            raise ValueError(f"unsupported corpus admission status: {status!r}")
        if not isinstance(admitted, bool):
            raise TypeError("corpus admission admitted field must be boolean")
        if (status == "admitted") != admitted:
            raise ValueError("corpus admission status and admitted flag disagree")
        corpus_case_id = admission.get("case_id")
        if status in {"admitted", "duplicate"}:
            corpus_case_id = _non_empty_string(corpus_case_id, "corpus admission canonical case_id")
        elif corpus_case_id is not None:
            corpus_case_id = _non_empty_string(corpus_case_id, "rejected corpus admission case_id")
        if status == "rejected":
            return admission, None, None
        regression_case = _parse_regression_case_record(
            admission.get("corpus_case"),
            field_name="corpus admission corpus_case",
            expected_case_id=str(corpus_case_id),
            expected_scenario=candidate["scenario"],
        )
        return admission, regression_case, None
    except Exception as exc:  # noqa: BLE001 - admission errors remain in the ledger
        admission = {
            "status": "infrastructure_failure",
            "admitted": False,
            "case_id": None,
            "discovery_id": discovery_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        error = {"candidate_id": str(candidate["candidate_id"]), "error": str(exc)}
        return admission, None, error


def _process_discovery_candidate(
    *,
    request: RoundRequest,
    planner: Mapping[str, Any],
    raw_candidate: Mapping[str, Any],
    index: int,
    adapters: CoevolutionAdapters,
    output_dir: Path,
) -> tuple[
    dict[str, Any],
    RegressionCase | None,
    RegressionCase | None,
    list[dict[str, str]],
]:
    candidate = dict(raw_candidate)
    candidate["target_planner_id"] = planner["planner_id"]
    discovery_id = _case_id(request, candidate)
    candidate["discovery_id"] = discovery_id
    gate, gate_error = _candidate_gate(request, planner, candidate, index, adapters, output_dir)
    admission: dict[str, Any] = {"status": "not_eligible", "admitted": False}
    regression_case = None
    newly_admitted_case = None
    errors = [gate_error] if gate_error else []
    if _admission_eligible(candidate, gate):
        admission, regression_case, admission_error = _corpus_admission(
            request, planner, candidate, gate, discovery_id, index, adapters, output_dir
        )
        if admission.get("status") == "admitted":
            newly_admitted_case = regression_case
        if admission_error:
            errors.append(admission_error)
    row = {
        "candidate_id": candidate["candidate_id"],
        "discovery_id": discovery_id,
        "case_id": regression_case.case_id if regression_case is not None else None,
        "search_status": candidate["search_status"],
        "execution_status": candidate["execution_status"],
        "source_planner_outcome": candidate["planner_outcome"],
        "scenario": candidate["scenario"],
        "gate": gate,
        "admission": admission,
        "raw_candidate": candidate,
        "corpus_case": regression_case.to_json() if regression_case is not None else None,
    }
    return row, regression_case, newly_admitted_case, errors


def _build_discovery_admission(
    *,
    request: RoundRequest,
    selected_planner: Mapping[str, Any],
    falsification: Mapping[str, Any],
    adapters: CoevolutionAdapters,
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    regression_cases: dict[str, RegressionCase] = {}
    newly_admitted: dict[str, RegressionCase] = {}
    infra_errors: list[dict[str, str]] = []
    for index, candidate in enumerate(falsification["candidates"]):
        row, regression_case, newly_admitted_case, errors = _process_discovery_candidate(
            request=request,
            planner=selected_planner,
            raw_candidate=candidate,
            index=index,
            adapters=adapters,
            output_dir=output_dir,
        )
        rows.append(row)
        if regression_case is not None:
            regression_cases.setdefault(regression_case.case_id, regression_case)
        if newly_admitted_case is not None:
            newly_admitted.setdefault(newly_admitted_case.case_id, newly_admitted_case)
        infra_errors.extend(errors)
    return {
        "status": "infrastructure_failure" if infra_errors else "complete",
        "candidate_count": len(rows),
        "corpus_regression_cases": [case.to_json() for case in regression_cases.values()],
        "newly_admitted_cases": [case.to_json() for case in newly_admitted.values()],
        "rows": rows,
        "infrastructure_errors": infra_errors,
    }


def _validate_discovery_rows(rows: list[Any]) -> tuple[set[str], set[str]]:
    """Validate candidate, discovery, and canonical corpus identities in the ledger."""
    row_ids: list[str] = []
    discovery_ids: list[str] = []
    regression_case_ids: set[str] = set()
    newly_admitted_case_ids: set[str] = set()
    for index, item in enumerate(rows):
        row = _mapping(item, f"discovery-admission.rows[{index}]")
        row_ids.append(_non_empty_string(row.get("candidate_id"), "discovery row candidate_id"))
        discovery_ids.append(
            _non_empty_string(row.get("discovery_id"), "discovery row discovery_id")
        )
        admission = _mapping(row.get("admission"), "discovery row admission")
        admission_status = admission.get("status")
        if admission_status not in {
            "admitted",
            "duplicate",
            "rejected",
            "not_eligible",
            "infrastructure_failure",
        }:
            raise ValueError(f"unsupported discovery row admission status: {admission_status!r}")
        if admission_status in {"admitted", "duplicate"}:
            case_id = _non_empty_string(row.get("case_id"), "discovery row corpus case_id")
            receipt_case_id = _non_empty_string(
                admission.get("case_id"), "discovery row admission case_id"
            )
            if case_id != receipt_case_id:
                raise ValueError("discovery row corpus case ID differs from its admission receipt")
            corpus_case = _parse_regression_case_record(
                row.get("corpus_case"),
                field_name=f"discovery-admission.rows[{index}].corpus_case",
                expected_case_id=case_id,
            )
            if admission_status == "admitted":
                newly_admitted_case_ids.add(case_id)
            regression_case_ids.add(corpus_case.case_id)
        elif row.get("case_id") is not None or row.get("corpus_case") is not None:
            raise ValueError("non-admitted discovery rows cannot carry a corpus case")
        _mapping(row.get("gate"), "discovery row gate")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("discovery-admission candidate IDs must be unique")
    if len(discovery_ids) != len(set(discovery_ids)):
        raise ValueError("discovery-admission IDs must be unique")
    return regression_case_ids, newly_admitted_case_ids


def _parse_regression_case_record(
    raw: Any,
    *,
    field_name: str,
    expected_case_id: str | None = None,
    expected_scenario: Mapping[str, Any] | None = None,
) -> RegressionCase:
    """Validate a canonical corpus projection without replacing its provenance."""
    case = _mapping(raw, field_name)
    case_id = _non_empty_string(case.get("case_id"), f"{field_name}.case_id")
    if expected_case_id is not None and case_id != expected_case_id:
        raise ValueError(f"{field_name}.case_id differs from the corpus admission receipt")
    origin_round_raw = case.get("origin_round")
    origin_round = (
        None
        if origin_round_raw is None
        else _integer(origin_round_raw, f"{field_name}.origin_round", minimum=1)
    )
    candidate_id_raw = case.get("candidate_id")
    candidate_id = (
        None
        if candidate_id_raw is None
        else _non_empty_string(candidate_id_raw, f"{field_name}.candidate_id")
    )
    scenario = _mapping(case.get("scenario"), f"{field_name}.scenario")
    if expected_scenario is not None and _canonical_json(scenario) != _canonical_json(
        expected_scenario
    ):
        raise ValueError(f"{field_name}.scenario differs from the verified discovery scenario")
    source_evidence = _mapping(case.get("source_evidence"), f"{field_name}.source_evidence")
    return RegressionCase(
        case_id=case_id,
        origin_round=origin_round,
        candidate_id=candidate_id,
        scenario=scenario,
        source_evidence=source_evidence,
    )


def _validate_regression_case_records(cases: list[Any], *, field_name: str) -> set[str]:
    """Validate stable corpus-case records and return their canonical identities."""
    case_ids: set[str] = set()
    for index, item in enumerate(cases):
        case = _parse_regression_case_record(item, field_name=f"{field_name}[{index}]")
        case_ids.add(case.case_id)
    if len(case_ids) != len(cases):
        raise ValueError(f"{field_name} must not contain duplicate corpus case IDs")
    return case_ids


def _validate_discovery_admission(raw: dict[str, Any]) -> dict[str, Any]:
    status = raw.get("status")
    if status not in {"complete", "infrastructure_failure"}:
        raise ValueError(f"unsupported discovery-admission result status: {status!r}")
    rows = raw.get("rows")
    regression_cases = raw.get("corpus_regression_cases")
    admitted = raw.get("newly_admitted_cases")
    errors = raw.get("infrastructure_errors")
    if not all(isinstance(value, list) for value in (rows, regression_cases, admitted, errors)):
        raise TypeError("discovery-admission rows, new cases, and errors must be lists")
    if raw.get("candidate_count") != len(rows):
        raise ValueError("discovery-admission candidate_count does not match its rows")
    row_case_ids, row_new_case_ids = _validate_discovery_rows(rows)
    regression_case_ids = _validate_regression_case_records(
        regression_cases, field_name="corpus_regression_cases"
    )
    admitted_case_ids = _validate_regression_case_records(
        admitted, field_name="newly_admitted_cases"
    )
    if not admitted_case_ids.issubset(regression_case_ids):
        raise ValueError("newly admitted cases must also be available for regression")
    if regression_case_ids != row_case_ids:
        raise ValueError("corpus regression cases must match admitted and duplicate discovery rows")
    if admitted_case_ids != row_new_case_ids:
        raise ValueError("newly admitted cases must match newly admitted discovery rows")
    if (status == "complete") != (not errors):
        raise ValueError("discovery-admission status and infrastructure_errors disagree")
    return raw


def _score_key(values: Sequence[Any]) -> tuple[float, ...]:
    """Match the optimizer's lexicographic max semantics, preserving null as unavailable."""
    return tuple(float("-inf") if value is None else float(value) for value in values)


def _round_request(
    config: CoevolutionConfig,
    round_number: int,
    round_dir: Path,
    source_revision: str,
    regression_cases: Sequence[RegressionCase],
    previous_selected_planner: Mapping[str, Any] | None,
) -> RoundRequest:
    offset = round_number - 1
    return RoundRequest(
        run_id=config.run_id,
        round_number=round_number,
        round_dir=round_dir,
        optimizer_trials_per_method=config.optimizer_trials_per_method,
        falsification_candidates=config.falsification_candidates_per_round,
        optimizer_random_seed=config.optimizer_random_seed_base + offset,
        optimizer_tpe_seed=config.optimizer_tpe_seed_base + offset,
        falsification_seed=config.falsification_seed_base + offset,
        falsification_sampler=config.falsification_sampler,
        heldout_case_ids=config.heldout_case_ids,
        regression_cases=tuple(regression_cases),
        previous_selected_planner=previous_selected_planner,
        source_revision=source_revision,
        config=config,
    )


def _verify_round_phase_artifacts(
    round_dir: Path,
    round_number: int,
    phases: Mapping[str, Any],
    expected_input_sha: str,
) -> None:
    for phase, phase_record in phases.items():
        if phase not in _PHASES or not isinstance(phase_record, Mapping):
            raise CoevolutionError(f"round {round_number} has an unknown phase record")
        if phase_record.get("input_sha256") != expected_input_sha:
            raise CoevolutionError(
                f"round {round_number} phase {phase} has a different input digest"
            )
        status = phase_record.get("status")
        if status in {"complete", "failed"}:
            _verify_phase_file(round_dir, phase, phase_record)
        elif status != "running":
            raise CoevolutionError(
                f"round {round_number} phase {phase} has invalid status {status!r}"
            )


def _verify_round_artifacts(config: CoevolutionConfig, round_row: Mapping[str, Any]) -> None:
    round_number = _integer(round_row.get("round_number"), "round_number", minimum=1)
    round_dir = config.output_dir / f"round_{round_number:03d}"
    round_manifest_path = round_dir / "round_manifest.json"
    if not round_manifest_path.is_file():
        raise CoevolutionError(f"round {round_number} manifest is missing")
    if _canonical_json(_read_json(round_manifest_path)) != _canonical_json(round_row):
        raise CoevolutionError(f"round {round_number} manifest differs from the run manifest")
    input_path = round_dir / "round_input.json"
    input_record = round_row.get("input")
    if not isinstance(input_record, Mapping):
        raise CoevolutionError(f"round {round_number} has no frozen input record")
    if not input_path.is_file() or _sha256_file(input_path) != input_record.get("file_sha256"):
        raise CoevolutionError(f"round {round_number} input digest changed or is missing")
    expected_input_sha = _sha256_bytes(_canonical_json(_read_json(input_path)))
    if expected_input_sha != input_record.get("input_sha256"):
        raise CoevolutionError(f"round {round_number} input payload digest changed")
    phases = _mapping(round_row.get("phases", {}), f"round {round_number} phases")
    _verify_round_phase_artifacts(round_dir, round_number, phases, expected_input_sha)


def _verify_all_persisted_artifacts(config: CoevolutionConfig, manifest: Mapping[str, Any]) -> None:
    rounds = manifest.get("rounds")
    if not isinstance(rounds, list):
        raise CoevolutionError("run manifest rounds must be a list")
    for round_row in rounds:
        _verify_round_artifacts(config, _mapping(round_row, "run manifest round"))


def _new_manifest(
    config: CoevolutionConfig,
    repository_root: Path,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "status": "running",
        "run_id": config.run_id,
        "config_path": str(config.config_path),
        "config_sha256": config.config_sha256,
        "input_files": _input_identities(config),
        "source_identity": dict(source_identity),
        "environment": _runtime_environment(),
        "repository_root": str(repository_root),
        "round_policy": {
            "minimum_rounds": config.minimum_rounds,
            "maximum_rounds": config.maximum_rounds,
        },
        "budgets_per_round": {
            "optimizer_trials_per_method": config.optimizer_trials_per_method,
            "falsification_candidates": config.falsification_candidates_per_round,
        },
        "rounds": [],
        "stop": None,
    }


def _runtime_environment() -> dict[str, str]:
    """Return non-secret runtime identity fields for resume and artifact provenance."""
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _manifest_file_digest(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(payload) + b"\n")


def _recover_pending_manifest_transaction(config: CoevolutionConfig) -> None:
    """Finish an interrupted paired manifest update from its durable intent."""
    transaction_path = config.output_dir / ".manifest_transaction.json"
    if not transaction_path.is_file():
        return
    transaction = _read_json(transaction_path)
    if transaction.get("schema") != MANIFEST_TRANSACTION_SCHEMA:
        raise CoevolutionError("pending manifest transaction has an unsupported schema")
    round_manifest = _mapping(transaction.get("round_manifest"), "transaction.round_manifest")
    run_manifest = _mapping(transaction.get("run_manifest"), "transaction.run_manifest")
    round_number = _integer(round_manifest.get("round_number"), "round_number", minimum=1)
    expected_round_path = config.output_dir / f"round_{round_number:03d}" / "round_manifest.json"
    if (
        transaction.get("round_path")
        != expected_round_path.relative_to(config.output_dir).as_posix()
    ):
        raise CoevolutionError("pending manifest transaction points outside its expected round")
    if transaction.get("run_path") != "run_manifest.json":
        raise CoevolutionError("pending manifest transaction has an unexpected run manifest path")
    if (
        round_manifest.get("schema") != ROUND_SCHEMA
        or run_manifest.get("schema") != RUN_SCHEMA
        or run_manifest.get("run_id") != config.run_id
        or run_manifest.get("config_sha256") != config.config_sha256
    ):
        raise CoevolutionError("pending manifest transaction belongs to a different run")
    rounds = run_manifest.get("rounds")
    if not isinstance(rounds, list):
        raise CoevolutionError("pending transaction run manifest rounds must be a list")
    matches = [
        row
        for row in rounds
        if isinstance(row, Mapping) and row.get("round_number") == round_number
    ]
    if len(matches) != 1 or _canonical_json(matches[0]) != _canonical_json(round_manifest):
        raise CoevolutionError("pending transaction manifests do not contain the same round state")
    if transaction.get("round_sha256") != _manifest_file_digest(round_manifest):
        raise CoevolutionError("pending transaction round manifest digest is invalid")
    if transaction.get("run_sha256") != _manifest_file_digest(run_manifest):
        raise CoevolutionError("pending transaction run manifest digest is invalid")

    _atomic_write_json(expected_round_path, round_manifest)
    _atomic_write_json(config.output_dir / "run_manifest.json", run_manifest)
    transaction_path.unlink()
    _fsync_directory(config.output_dir)


def _write_manifests(
    config: CoevolutionConfig, manifest: dict[str, Any], round_state: dict[str, Any]
) -> None:
    _recover_pending_manifest_transaction(config)
    round_dir = config.output_dir / f"round_{round_state['round_number']:03d}"
    round_number = round_state["round_number"]
    rounds = manifest.get("rounds")
    if not isinstance(rounds, list):
        raise CoevolutionError("run manifest rounds must be a list")
    matching_indices = [
        index
        for index, row in enumerate(rounds)
        if isinstance(row, Mapping) and row.get("round_number") == round_number
    ]
    if len(matching_indices) != 1:
        raise CoevolutionError(f"run manifest must contain round {round_number} exactly once")
    round_payload = json.loads(_canonical_json(round_state))
    rounds[matching_indices[0]] = round_state
    run_payload = json.loads(_canonical_json(manifest))
    transaction_path = config.output_dir / ".manifest_transaction.json"
    journal = {
        "schema": MANIFEST_TRANSACTION_SCHEMA,
        "round_path": (round_dir / "round_manifest.json").relative_to(config.output_dir).as_posix(),
        "run_path": "run_manifest.json",
        "round_sha256": _manifest_file_digest(round_payload),
        "run_sha256": _manifest_file_digest(run_payload),
        "round_manifest": round_payload,
        "run_manifest": run_payload,
    }
    _atomic_write_json(transaction_path, journal)
    try:
        _atomic_write_json(round_dir / "round_manifest.json", round_payload)
        _atomic_write_json(config.output_dir / "run_manifest.json", run_payload)
    except Exception:
        _recover_pending_manifest_transaction(config)
        raise
    transaction_path.unlink()
    _fsync_directory(config.output_dir)


def _open_or_create_run(
    config: CoevolutionConfig,
    repository_root: Path,
    *,
    resume: bool,
) -> tuple[dict[str, Any], bool]:
    if _sha256_file(config.config_path) != config.config_sha256:
        raise CoevolutionError("config bytes changed after load; reload before starting a run")
    _recover_pending_manifest_transaction(config)
    manifest_path = config.output_dir / "run_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise CoevolutionError(
                f"run output already exists; use resume after verifying the exact config: {config.output_dir}"
            )
        manifest = _read_json(manifest_path)
        _verify_static_identity(manifest, config, repository_root)
        _verify_all_persisted_artifacts(config, manifest)
        terminal = manifest.get("status") in {"complete", "budget_exhausted", "diagnostic"}
        return manifest, terminal
    if resume:
        raise CoevolutionError(f"cannot resume; run manifest is missing: {manifest_path}")
    other_files = [item for item in config.output_dir.iterdir() if item.name != ".coevolution.lock"]
    if other_files:
        raise CoevolutionError(f"run output directory is not empty: {config.output_dir}")
    source_identity = _git_identity(repository_root, config.output_dir)
    manifest = _new_manifest(config, repository_root, source_identity)
    _atomic_write_json(manifest_path, manifest)
    return manifest, False


def _prepare_round(
    config: CoevolutionConfig,
    manifest: dict[str, Any],
    round_number: int,
    regression_cases: Sequence[RegressionCase],
    previous_selected_planner: Mapping[str, Any] | None,
) -> tuple[RoundRequest, dict[str, Any]]:
    round_dir = config.output_dir / f"round_{round_number:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    source_revision = manifest["source_identity"]["revision"]
    request = _round_request(
        config,
        round_number,
        round_dir,
        source_revision,
        regression_cases,
        previous_selected_planner,
    )
    input_payload = request.to_json()
    input_sha = _sha256_bytes(_canonical_json(input_payload))
    input_path = round_dir / "round_input.json"
    if input_path.exists():
        old_payload = _read_json(input_path)
        if _sha256_bytes(_canonical_json(old_payload)) != input_sha:
            raise CoevolutionError(
                f"round {round_number} inputs changed while resuming; refusing to continue"
            )
        input_file_sha = _sha256_file(input_path)
    else:
        input_file_sha = _atomic_write_json(input_path, input_payload)
    prior_rows = {row["round_number"]: row for row in manifest.get("rounds", [])}
    round_state = prior_rows.get(
        round_number,
        {
            "schema": ROUND_SCHEMA,
            "round_number": round_number,
            "status": "running",
            "input": {"input_sha256": input_sha, "file_sha256": input_file_sha},
            "phases": {},
        },
    )
    if round_state.get("input", {}).get("input_sha256") != input_sha:
        raise CoevolutionError(f"round {round_number} manifest input digest differs")
    if round_number not in prior_rows:
        manifest["rounds"].append(round_state)
    _write_manifests(config, manifest, round_state)
    return request, round_state


def _run_round_phases(
    *,
    config: CoevolutionConfig,
    manifest: dict[str, Any],
    request: RoundRequest,
    round_state: dict[str, Any],
    adapters: CoevolutionAdapters,
) -> dict[str, dict[str, Any]]:
    round_dir = request.round_dir

    def phase(
        name: str,
        execute: Callable[[], Mapping[str, Any]],
        validate: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        return _execute_phase(
            phase=name,
            request=request,
            round_state=round_state,
            round_dir=round_dir,
            run_manifest=manifest,
            execute=execute,
            validate=validate,
        )

    def optimize() -> Mapping[str, Any]:
        previous_selected = request.previous_selected_planner

        def call() -> Mapping[str, Any]:
            return adapters.optimize(request, round_dir / "optimization")

        if previous_selected is None:
            return call()
        return _call_with_selected_planner_integrity(
            previous_selected,
            "optimization using the prior selected planner",
            call,
        )

    optimizer = phase(
        "optimization",
        optimize,
        lambda payload: _validate_optimizer_output(payload, request),
    )
    selected = optimizer["selected_planner"]
    challenge_evaluation = phase(
        "challenge_evaluation",
        lambda: _call_with_selected_planner_integrity(
            selected,
            "challenge evaluation",
            lambda: adapters.evaluate_challenges(
                request,
                _detached_json_mapping(selected, "selected planner for challenge evaluation"),
                round_dir / "challenge_evaluation",
            ),
        ),
        lambda payload: _validate_challenge_evaluations(payload, request),
    )
    falsification = phase(
        "falsification",
        lambda: _call_with_selected_planner_integrity(
            selected,
            "falsification",
            lambda: adapters.falsify(
                request,
                _detached_json_mapping(selected, "selected planner for falsification"),
                round_dir / "falsification",
            ),
        ),
        lambda payload: _validate_falsification_output(payload, request),
    )
    discovery = phase(
        "discovery_admission",
        lambda: _build_discovery_admission(
            request=request,
            selected_planner=selected,
            falsification=falsification,
            adapters=adapters,
            output_dir=round_dir / "discovery_admission",
        ),
        _validate_discovery_admission,
    )
    return {
        "optimization": optimizer,
        "challenge_evaluation": challenge_evaluation,
        "falsification": falsification,
        "discovery_admission": discovery,
    }


@dataclass(slots=True)
class _LoopProgress:
    regression_cases: list[RegressionCase]
    selected_planner: Mapping[str, Any] | None = None
    best_score: tuple[float, ...] | None = None
    plateau_rounds: int = 0
    last_round_new_admissions: int | None = None


def _record_completed_round(
    config: CoevolutionConfig,
    manifest: dict[str, Any],
    request: RoundRequest,
    round_state: dict[str, Any],
    phases: Mapping[str, Mapping[str, Any]],
    progress: _LoopProgress,
) -> list[RegressionCase]:
    optimizer = phases["optimization"]
    challenge = phases["challenge_evaluation"]
    falsification = phases["falsification"]
    discovery = phases["discovery_admission"]
    progress.selected_planner = _detached_json_mapping(
        optimizer["selected_planner"], "selected planner for coevolution progress"
    )
    selection_score = _score_key(optimizer["selection_tuple"])
    prior_best = progress.best_score
    if prior_best is None or selection_score > prior_best:
        progress.best_score = selection_score
        progress.plateau_rounds = 0
    else:
        progress.plateau_rounds += 1
    corpus_regression_cases = [
        _parse_regression_case_record(
            row,
            field_name="completed round corpus regression case",
        )
        for row in discovery["corpus_regression_cases"]
    ]
    newly_admitted_case_ids = {str(case["case_id"]) for case in discovery["newly_admitted_cases"]}
    newly_admitted = [
        case for case in corpus_regression_cases if case.case_id in newly_admitted_case_ids
    ]
    progress.last_round_new_admissions = len(newly_admitted)
    round_state["status"] = "complete"
    round_state["summary"] = {
        "optimizer_improves_over_baseline": optimizer["improves_over_baseline"],
        "selection_tuple": optimizer["selection_tuple"],
        "selection_improves_over_prior_best": (
            None if prior_best is None else selection_score > prior_best
        ),
        "heldout_case_count": len(challenge["heldout_rows"]),
        "regression_case_count": len(challenge["regression_rows"]),
        "falsification_candidate_count": falsification["candidate_budget"],
        "admissible_new_case_count": len(newly_admitted),
        "total_invalid_candidate_count": sum(
            row["search_status"] == "invalid" for row in falsification["candidates"]
        ),
        "total_failed_candidate_count": sum(
            row["search_status"] == "failed" for row in falsification["candidates"]
        ),
        "evaluated_regression_case_ids": list(request.regression_case_ids),
    }
    round_state["stop_decision"] = {"decision": "continue", "reason": "round_complete"}
    known_case_positions = {
        case.case_id: index for index, case in enumerate(progress.regression_cases)
    }
    for case in corpus_regression_cases:
        position = known_case_positions.get(case.case_id)
        if position is None:
            known_case_positions[case.case_id] = len(progress.regression_cases)
            progress.regression_cases.append(case)
        else:
            progress.regression_cases[position] = _merge_regression_case_provenance(
                progress.regression_cases[position], case
            )
    _write_manifests(config, manifest, round_state)
    return newly_admitted


def _merge_regression_case_provenance(
    existing: RegressionCase, canonical: RegressionCase
) -> RegressionCase:
    """Hydrate ID-only seeds from the corpus and reject conflicting identity metadata."""
    if existing.case_id != canonical.case_id:
        raise ValueError("cannot merge regression cases with different canonical IDs")
    for field_name in ("origin_round", "candidate_id", "scenario"):
        current = getattr(existing, field_name)
        incoming = getattr(canonical, field_name)
        if current is not None and incoming is not None:
            same = (
                _canonical_json(_deep_thaw(current)) == _canonical_json(_deep_thaw(incoming))
                if field_name == "scenario"
                else current == incoming
            )
            if not same:
                raise ValueError(f"corpus regression {field_name} conflicts with existing memory")
    return RegressionCase(
        case_id=existing.case_id,
        origin_round=(
            existing.origin_round if existing.origin_round is not None else canonical.origin_round
        ),
        candidate_id=(
            existing.candidate_id if existing.candidate_id is not None else canonical.candidate_id
        ),
        scenario=existing.scenario if existing.scenario is not None else canonical.scenario,
        source_evidence=(
            existing.source_evidence
            if existing.source_evidence is not None
            else canonical.source_evidence
        ),
    )


def _round_stop_decision(
    config: CoevolutionConfig,
    round_number: int,
    newly_admitted: Sequence[RegressionCase],
    plateau_rounds: int,
) -> dict[str, Any] | None:
    if (
        config.stop_on_no_new_admissible_case
        and round_number >= config.minimum_rounds
        and not newly_admitted
    ):
        return {
            "decision": "stop",
            "reason": "no_new_admissible_counterexample_under_budget",
            "after_round": round_number,
            "minimum_rounds_satisfied": True,
            "claim_boundary": (
                "No new admissible counterexample was found within the configured finite "
                "search budget; this does not establish that none exists."
            ),
        }
    threshold = config.optimization_plateau_rounds
    if (
        threshold is not None
        and plateau_rounds >= threshold
        and round_number >= config.minimum_rounds
    ):
        return {
            "decision": "stop",
            "reason": "planner_optimization_plateau",
            "after_round": round_number,
            "consecutive_rounds_without_new_best": plateau_rounds,
            "claim_boundary": "Finite-budget optimization stopped improving under the recorded objective.",
        }
    return None


def _record_terminal_stop(
    config: CoevolutionConfig,
    manifest: dict[str, Any],
    round_state: dict[str, Any],
    stop: Mapping[str, Any],
) -> None:
    manifest["status"] = "complete"
    manifest["stop"] = dict(stop)
    round_state["stop_decision"] = dict(stop)
    _write_manifests(config, manifest, round_state)


def _run_rounds(
    config: CoevolutionConfig,
    manifest: dict[str, Any],
    adapters: CoevolutionAdapters,
) -> dict[str, Any]:
    progress = _LoopProgress(
        regression_cases=[
            RegressionCase(case_id=case_id, origin_round=None, candidate_id=None)
            for case_id in config.initial_regression_case_ids
        ]
    )
    for round_number in range(1, config.maximum_rounds + 1):
        request, round_state = _prepare_round(
            config,
            manifest,
            round_number,
            progress.regression_cases,
            progress.selected_planner,
        )
        try:
            phases = _run_round_phases(
                config=config,
                manifest=manifest,
                request=request,
                round_state=round_state,
                adapters=adapters,
            )
        except CoevolutionInfrastructureError as exc:
            round_state["status"] = "diagnostic"
            round_state.setdefault("failure", {"phase": "recovery", "error": str(exc)})
            manifest["status"] = "diagnostic"
            manifest["stop"] = {
                "decision": "stop",
                "reason": "infrastructure_failure",
                "round_number": round_number,
                "error": str(exc),
                "claim_boundary": "No scientific conclusion is drawn from an incomplete round.",
            }
            _write_manifests(config, manifest, round_state)
            return manifest
        discovery = phases["discovery_admission"]
        if discovery["status"] == "infrastructure_failure":
            round_state["status"] = "diagnostic"
            round_state["failure"] = {
                "phase": "discovery_admission",
                "errors": discovery["infrastructure_errors"],
            }
            manifest["status"] = "diagnostic"
            manifest["stop"] = {
                "decision": "stop",
                "reason": "infrastructure_failure",
                "failed_phase": "discovery_admission",
                "round_number": round_number,
                "claim_boundary": "No scientific conclusion is drawn from an incomplete round.",
            }
            _write_manifests(config, manifest, round_state)
            return manifest
        try:
            new_cases = _record_completed_round(
                config, manifest, request, round_state, phases, progress
            )
        except (TypeError, ValueError) as exc:
            round_state["status"] = "diagnostic"
            round_state.pop("summary", None)
            round_state["failure"] = {
                "phase": "regression_case_carry",
                "error": str(exc),
            }
            manifest["status"] = "diagnostic"
            diagnostic_stop = {
                "decision": "stop",
                "reason": "infrastructure_failure",
                "failed_phase": "regression_case_carry",
                "round_number": round_number,
                "claim_boundary": "No scientific conclusion is drawn from a conflicted regression-case identity.",
            }
            manifest["stop"] = diagnostic_stop
            round_state["stop_decision"] = diagnostic_stop
            _write_manifests(config, manifest, round_state)
            return manifest
        stop = _round_stop_decision(config, round_number, new_cases, progress.plateau_rounds)
        if stop:
            _record_terminal_stop(config, manifest, round_state, stop)
            return manifest

    manifest["status"] = "budget_exhausted"
    manifest["stop"] = {
        "decision": "stop",
        "reason": "maximum_round_budget_exhausted",
        "after_round": config.maximum_rounds,
        "minimum_rounds_satisfied": config.maximum_rounds >= config.minimum_rounds,
        "last_round_new_admissions": progress.last_round_new_admissions,
        "claim_boundary": "The configured maximum round budget was exhausted; no global conclusion is implied.",
    }
    _atomic_write_json(config.output_dir / "run_manifest.json", manifest)
    return manifest


def run_coevolution(
    config: CoevolutionConfig | str | Path,
    adapters: CoevolutionAdapters,
    *,
    resume: bool = False,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run or resume the configured finite-budget planner↔falsifier loop.

    Completed phase outputs are digest-verified and never called again. A phase
    left ``running`` by process interruption or marked ``failed`` is not retried
    automatically because doing so could duplicate simulator work. The complete
    failure and any adapter-returned candidate/evaluation rows remain durable.
    """
    resolved = config if isinstance(config, CoevolutionConfig) else load_coevolution_config(config)
    repo_root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    with _run_lock(resolved.output_dir):
        manifest, terminal = _open_or_create_run(resolved, repo_root, resume=resume)
        if terminal:
            return manifest
        return _run_rounds(resolved, manifest, adapters)


__all__ = [
    "CONFIG_SCHEMA",
    "RUN_SCHEMA",
    "CoevolutionAdapters",
    "CoevolutionConfig",
    "CoevolutionError",
    "CoevolutionInfrastructureError",
    "RegressionCase",
    "RoundRequest",
    "load_coevolution_config",
    "run_coevolution",
]
