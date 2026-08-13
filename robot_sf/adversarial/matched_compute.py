"""Runtime accounting probes for matched-compute adversarial arms.

This module records diagnostic-only preflight evidence for issue #4360/#6921.
It proves the reactive and open-loop arms can touch their existing runtime seams
without making benchmark, planner-ranking, safety, or paper-facing claims.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.adversarial import search as adversarial_search
from robot_sf.adversarial.config import SearchConfig, SearchRunResult
from robot_sf.ped_npc.residual_adversary import BoundedResidualAdversary, ResidualAdversaryConfig
from robot_sf.ped_npc.residual_search import FiniteGridSearchPolicy, ResidualSearchConfig

MATCHED_COMPUTE_RUNTIME_SCHEMA_VERSION = "matched_compute_trace.v1"
MATCHED_COMPUTE_EVIDENCE_STATUSES = (
    "diagnostic_only_preflight",
    "production_observed",
    "unavailable",
)
MATCHED_COMPUTE_SIMULATOR_STEP_SOURCES = (
    "observed_episode_record",
    "observed_simulator",
    "synthetic_episode_fixture",
    "controller_snapshot",
    "unavailable",
)

MATCHED_COMPUTE_RUNTIME_SCHEMA: dict[str, Any] = {
    "schema_version": MATCHED_COMPUTE_RUNTIME_SCHEMA_VERSION,
    "required": [
        "schema_version",
        "arm",
        "scenario_seed",
        "search_seed",
        "execution_mode",
        "simulator_physics_steps",
        "macro_actions",
        "candidate_evaluations",
        "accepted",
        "rejected",
        "invalid",
        "status",
        "evidence_status",
        "adapter",
        "runtime_status",
        "native_path",
        "candidate_budget",
        "simulator_steps",
        "simulator_steps_source",
        "fallback",
        "degraded",
        "unavailability_reason",
        "metadata",
    ],
}

OpenLoopRunner = Callable[..., SearchRunResult]


@dataclass(frozen=True)
class ReactiveRuntimeSnapshot:
    """One simulator snapshot for a reactive matched-compute probe."""

    dt_s: float
    positions: Sequence[Sequence[float]]
    velocities: Sequence[Sequence[float]]
    max_speeds: Sequence[float]
    robot_pose: tuple[tuple[float, float], float]
    scenario_seed: int
    route_polylines: list[np.ndarray] | dict[int, np.ndarray] | None = None
    obstacle_segments: np.ndarray | list[Any] | None = None
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    ped_radius: float = 0.4


@dataclass(frozen=True)
class MatchedComputeRuntimeTrace:
    """Shared JSON-serializable runtime accounting for both matched-compute arms."""

    arm: str
    scenario_seed: int
    search_seed: int
    execution_mode: str
    simulator_physics_steps: int | None
    macro_actions: int
    candidate_evaluations: int
    accepted: int
    rejected: int
    invalid: int
    status: str
    adapter: str
    native_path: str
    candidate_budget: int
    evidence_status: str = "diagnostic_only_preflight"
    simulator_steps_source: str = "unavailable"
    fallback: bool = False
    degraded: bool = False
    unavailability_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = MATCHED_COMPUTE_RUNTIME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the shared accounting contract."""
        _validate_trace_identity(self)
        _validate_trace_accounting(self)
        _validate_trace_status(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible mapping."""
        return {
            "accepted": self.accepted,
            "adapter": self.adapter,
            "arm": self.arm,
            "candidate_budget": self.candidate_budget,
            "candidate_evaluations": self.candidate_evaluations,
            "degraded": self.degraded,
            "evidence_status": self.evidence_status,
            "execution_mode": self.execution_mode,
            "fallback": self.fallback,
            "invalid": self.invalid,
            "macro_actions": self.macro_actions,
            "metadata": dict(self.metadata),
            "native_path": self.native_path,
            "rejected": self.rejected,
            "runtime_status": self.status,
            "scenario_seed": self.scenario_seed,
            "schema_version": self.schema_version,
            "search_seed": self.search_seed,
            "simulator_physics_steps": self.simulator_physics_steps,
            "simulator_steps": self.simulator_physics_steps,
            "simulator_steps_source": self.simulator_steps_source,
            "status": self.status,
            "unavailability_reason": self.unavailability_reason,
        }

    @property
    def runtime_status(self) -> str:
        """Backward-compatible alias for the packet ``status`` field."""
        return self.status

    @property
    def simulator_steps(self) -> int | None:
        """Backward-compatible alias for ``simulator_physics_steps``."""
        return self.simulator_physics_steps

    def to_json(self, *, indent: int | None = None) -> str:
        """Return deterministic JSON for preflight packets or test fixtures."""
        separators = (",", ":") if indent is None else (",", ": ")
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent, separators=separators)


def _validate_trace_identity(trace: MatchedComputeRuntimeTrace) -> None:
    """Validate schema and seam identity fields."""
    if trace.schema_version != MATCHED_COMPUTE_RUNTIME_SCHEMA_VERSION:
        raise ValueError("unsupported matched-compute runtime schema_version")
    if trace.arm not in {"reactive", "open_loop"}:
        raise ValueError("arm must be 'reactive' or 'open_loop'")
    if not trace.adapter:
        raise ValueError("adapter must be non-empty")
    if not trace.native_path:
        raise ValueError("native_path must be non-empty")


def _validate_trace_accounting(trace: MatchedComputeRuntimeTrace) -> None:
    """Validate non-negative packet accounting and its disjoint partition."""
    _require_nonnegative_int(trace.scenario_seed, "scenario_seed")
    _require_nonnegative_int(trace.search_seed, "search_seed")
    _require_nonnegative_int(trace.candidate_budget, "candidate_budget")
    _require_nonnegative_int(trace.candidate_evaluations, "candidate_evaluations")
    if trace.simulator_physics_steps is not None:
        _require_nonnegative_int(trace.simulator_physics_steps, "simulator_physics_steps")
    _require_nonnegative_int(trace.macro_actions, "macro_actions")
    _require_nonnegative_int(trace.accepted, "accepted")
    _require_nonnegative_int(trace.rejected, "rejected")
    _require_nonnegative_int(trace.invalid, "invalid")
    if trace.accepted + trace.rejected + trace.invalid != trace.candidate_evaluations:
        raise ValueError("accepted + rejected + invalid must equal candidate_evaluations")


def _validate_trace_status(trace: MatchedComputeRuntimeTrace) -> None:
    """Validate execution status, fallback, and unavailability fields."""
    if trace.execution_mode not in {"native", "unavailable"}:
        raise ValueError("execution_mode must be 'native' or 'unavailable'")
    if trace.status not in {"native", "unavailable"}:
        raise ValueError("status must be 'native' or 'unavailable'")
    if trace.status == "native" and trace.execution_mode != "native":
        raise ValueError("native status requires native execution_mode")
    if trace.status == "native" and trace.fallback:
        raise ValueError("native matched-compute traces cannot be fallback")
    if trace.status == "native" and trace.degraded:
        raise ValueError("native matched-compute traces cannot be degraded")
    if trace.status == "unavailable" and not trace.unavailability_reason:
        raise ValueError("unavailable traces require an unavailability_reason")
    _validate_trace_evidence(trace)


def _validate_trace_evidence(trace: MatchedComputeRuntimeTrace) -> None:
    """Validate evidence tier and simulator-step provenance."""
    if trace.evidence_status not in MATCHED_COMPUTE_EVIDENCE_STATUSES:
        raise ValueError("unsupported matched-compute evidence_status")
    if trace.simulator_steps_source not in MATCHED_COMPUTE_SIMULATOR_STEP_SOURCES:
        raise ValueError("unsupported matched-compute simulator_steps_source")
    _validate_trace_evidence_status(trace)
    _validate_trace_step_provenance(trace)


def _validate_trace_evidence_status(trace: MatchedComputeRuntimeTrace) -> None:
    """Validate consistency between runtime and evidence status."""
    if trace.status == "native" and trace.evidence_status == "unavailable":
        raise ValueError("native matched-compute traces cannot be unavailable evidence")
    if trace.status == "unavailable" and trace.evidence_status != "unavailable":
        raise ValueError("unavailable matched-compute traces require unavailable evidence_status")
    if trace.evidence_status == "production_observed":
        if trace.status != "native":
            raise ValueError("production-observed traces require native status")
        if trace.simulator_physics_steps is None:
            raise ValueError("production-observed traces require simulator physics steps")
        if trace.simulator_steps_source not in {
            "observed_episode_record",
            "observed_simulator",
        }:
            raise ValueError("production-observed traces require an observed step source")


def _validate_trace_step_provenance(trace: MatchedComputeRuntimeTrace) -> None:
    """Reject observed-step claims from synthetic or controller-only inputs."""
    if trace.evidence_status == "diagnostic_only_preflight":
        if trace.simulator_steps_source in {
            "observed_episode_record",
            "observed_simulator",
        }:
            raise ValueError("diagnostic-only preflight cannot claim observed simulator steps")
        if (
            trace.simulator_steps_source
            in {
                "synthetic_episode_fixture",
                "controller_snapshot",
            }
            and trace.simulator_physics_steps is not None
        ):
            raise ValueError(
                "synthetic/controller preflight cannot populate observed simulator physics steps"
            )


def probe_reactive_runtime(
    search_config: ResidualSearchConfig,
    residual_config: ResidualAdversaryConfig,
    *,
    snapshot: ReactiveRuntimeSnapshot,
) -> MatchedComputeRuntimeTrace:
    """Run one reactive macro-boundary step through the native residual seam."""
    positions_array = _finite_array(snapshot.positions, "positions")
    velocities_array = _finite_array(snapshot.velocities, "velocities")
    max_speeds_array = _finite_array(snapshot.max_speeds, "max_speeds")
    if positions_array.ndim != 2 or positions_array.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    num_peds = int(positions_array.shape[0])

    policy = FiniteGridSearchPolicy(
        search_config,
        residual_config,
        snapshot.dt_s,
        num_peds,
        route_polylines=snapshot.route_polylines,
        obstacle_segments=snapshot.obstacle_segments,
        bounds=snapshot.bounds,
        ped_radius=snapshot.ped_radius,
    )
    adversary = BoundedResidualAdversary(
        config=residual_config,
        policy=policy,
        dt_s=snapshot.dt_s,
        num_peds=num_peds,
        route_polylines=snapshot.route_polylines,
        obstacle_segments=snapshot.obstacle_segments,
        bounds=snapshot.bounds,
        ped_radius=snapshot.ped_radius,
    )
    adversary.step_residual(
        positions_array, velocities_array, max_speeds_array, snapshot.robot_pose
    )

    record = policy.last_record
    if not hasattr(record, "to_dict"):
        raise ValueError("reactive SearchDiagnosticRecord accounting is missing")
    record_payload = record.to_dict()
    candidate_budget = _require_nonnegative_int(record_payload.get("budget"), "record.budget")
    candidate_evaluations = _require_nonnegative_int(
        record_payload.get("total_evaluated"), "record.total_evaluated"
    )
    if candidate_evaluations > candidate_budget:
        raise ValueError("reactive candidate evaluations exceed declared budget")

    controller_steps = _require_nonnegative_int(adversary.step_index, "adversary.step_index")
    macro_actions = _require_nonnegative_int(
        adversary.macro_action_index, "adversary.macro_action_index"
    )
    return MatchedComputeRuntimeTrace(
        arm="reactive",
        scenario_seed=_require_nonnegative_int(snapshot.scenario_seed, "snapshot.scenario_seed"),
        search_seed=_require_nonnegative_int(search_config.seed, "search_config.seed"),
        execution_mode="native",
        simulator_physics_steps=None,
        macro_actions=macro_actions,
        candidate_evaluations=candidate_evaluations,
        accepted=_require_nonnegative_int(record_payload.get("accepted"), "record.accepted"),
        rejected=_require_nonnegative_int(record_payload.get("rejected"), "record.rejected"),
        invalid=_require_nonnegative_int(record_payload.get("invalid"), "record.invalid"),
        status="native",
        adapter="finite_grid_residual_adversary",
        native_path=(
            "robot_sf.ped_npc.residual_search.FiniteGridSearchPolicy+"
            "robot_sf.ped_npc.residual_adversary.BoundedResidualAdversary"
        ),
        candidate_budget=candidate_budget,
        evidence_status="diagnostic_only_preflight",
        simulator_steps_source="controller_snapshot",
        metadata={
            "controller_steps": controller_steps,
            "simulator_physics_steps_observed": False,
            "preflight_reason": "one-step controller snapshot; no simulator tick was observed",
            "search_diagnostic_record": record_payload,
        },
    )


def probe_open_loop_runtime(
    config: SearchConfig,
    *,
    macro_actions: int,
    runner: OpenLoopRunner | None = None,
    production_evaluator_factory: Callable[[], Callable[..., Any]] | None = None,
) -> MatchedComputeRuntimeTrace:
    """Run the canonical open-loop search seam and account returned records.

    Tests should inject ``runner`` and monkeypatch the canonical evaluator factory
    to avoid launching a campaign. Production callers that omit ``runner`` use
    :func:`robot_sf.adversarial.search.run_adversarial_search` directly.
    """
    factory = production_evaluator_factory or adversarial_search.production_candidate_evaluator
    production_evaluator = factory()
    if not callable(production_evaluator):
        raise ValueError("production_candidate_evaluator did not return a callable")

    def _production_candidate_bridge(
        search_config: SearchConfig,
        candidate: Any,
        _scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> Any:
        index = _candidate_index_from_dir(candidate_dir)
        return production_evaluator(search_config, candidate, index)

    active_runner = runner or adversarial_search.run_adversarial_search
    result = active_runner(config, evaluator=_production_candidate_bridge)
    candidate_budget = _require_nonnegative_int(config.budget, "config.budget")
    candidate_evaluations = _require_nonnegative_int(result.num_candidates, "result.num_candidates")
    if candidate_evaluations > candidate_budget:
        raise ValueError("open-loop candidate evaluations exceed declared budget")

    simulator_steps, status, reason = _simulator_steps_from_manifest(result)
    preflight_only = runner is not None or production_evaluator_factory is not None
    if status == "unavailable":
        evidence_status = "unavailable"
        simulator_steps_source = "unavailable"
        reported_simulator_steps = None
    elif preflight_only:
        evidence_status = "diagnostic_only_preflight"
        simulator_steps_source = "synthetic_episode_fixture"
        reported_simulator_steps = None
    else:
        evidence_status = "diagnostic_only_preflight"
        simulator_steps_source = "unavailable"
        reported_simulator_steps = None
    invalid = _require_nonnegative_int(
        result.num_invalid_candidates, "result.num_invalid_candidates"
    )
    rejected = _require_nonnegative_int(
        result.num_failed_evaluations, "result.num_failed_evaluations"
    )
    reported_accepted = _require_nonnegative_int(
        result.num_valid_candidates, "result.num_valid_candidates"
    )
    if invalid + rejected > candidate_evaluations:
        raise ValueError("open-loop invalid and failed evaluations exceed candidate_evaluations")
    accepted = candidate_evaluations - invalid - rejected
    if reported_accepted != accepted:
        raise ValueError(
            "open-loop SearchRunResult accounting is inconsistent: "
            "num_valid_candidates must equal candidate_evaluations - "
            "num_invalid_candidates - num_failed_evaluations"
        )
    return MatchedComputeRuntimeTrace(
        arm="open_loop",
        scenario_seed=_scenario_seed_from_config(config),
        search_seed=_require_nonnegative_int(config.seed, "config.seed"),
        execution_mode=status,
        simulator_physics_steps=reported_simulator_steps,
        macro_actions=_require_nonnegative_int(macro_actions, "macro_actions"),
        candidate_evaluations=candidate_evaluations,
        accepted=accepted,
        rejected=rejected,
        invalid=invalid,
        status=status,
        adapter="adversarial_search_production_candidate",
        native_path="robot_sf.adversarial.search.run_adversarial_search",
        candidate_budget=candidate_budget,
        evidence_status=evidence_status,
        simulator_steps_source=simulator_steps_source,
        unavailability_reason=reason,
        metadata={
            "production_candidate_evaluator": (
                "robot_sf.adversarial.search.production_candidate_evaluator"
            ),
            "best_bundle_path": (
                str(result.best_bundle_path) if result.best_bundle_path is not None else None
            ),
            "num_failed_evaluations": rejected,
            "num_invalid_candidates": invalid,
            "num_valid_candidates": accepted,
            "preflight_reason": (
                "injected runner/evaluator; episode steps are synthetic fixtures"
                if preflight_only
                else (
                    "production-observed admission is reserved for a successor canary; "
                    "omitted test hooks do not prove per-candidate provenance"
                )
            ),
            "preflight_simulator_physics_steps": simulator_steps if preflight_only else None,
            "production_observed_admission": "reserved_for_successor_canary",
            "production_provenance_validated": False,
        },
    )


def _simulator_steps_from_manifest(
    result: SearchRunResult,
) -> tuple[int | None, str, str | None]:
    """Extract simulator-step accounting from all candidate episode records."""
    manifest_steps = _simulator_steps_from_search_manifest(result.manifest_path)
    if manifest_steps is not None:
        return manifest_steps, "native", None

    if result.best_candidate is None:
        return None, "unavailable", "search returned no best candidate"
    episode_path = result.best_candidate.episode_record_path
    if episode_path is None:
        return None, "unavailable", "best candidate did not expose an episode record path"
    record = _read_first_jsonl_mapping(episode_path)
    return _extract_simulator_steps(record), "native", None


def _simulator_steps_from_search_manifest(manifest_path: Path) -> int | None:
    """Sum simulator steps for every candidate episode listed in a search manifest."""
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"search manifest is malformed JSON: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("search manifest must be a JSON object")
    candidates = manifest.get("candidates")
    if candidates is None:
        return None
    if not isinstance(candidates, list):
        raise ValueError("search manifest candidates must be a list")
    if not candidates:
        return None
    total_steps = 0
    for index, item in enumerate(candidates):
        if not isinstance(item, Mapping):
            raise ValueError(f"search manifest candidates[{index}] must be an object")
        raw_path = item.get("episode_record_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"search manifest candidates[{index}] is missing episode_record_path")
        record_path = _resolve_manifest_path(manifest_path.parent, raw_path)
        total_steps += _extract_simulator_steps(_read_first_jsonl_mapping(record_path))
    return total_steps


def _resolve_manifest_path(manifest_dir: Path, raw_path: str) -> Path:
    """Resolve a manifest path using as-written or manifest-relative location."""
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest_dir / path
    if candidate.exists():
        return candidate
    return path


def _candidate_index_from_dir(candidate_dir: Path) -> int:
    """Extract ``candidate_XXXX`` index for the production candidate seam."""
    name = candidate_dir.name
    prefix = "candidate_"
    if not name.startswith(prefix):
        raise ValueError(f"candidate_dir must be named candidate_XXXX, got {candidate_dir}")
    try:
        return _require_nonnegative_int(int(name.removeprefix(prefix)), "candidate index")
    except ValueError as exc:
        raise ValueError(
            f"candidate_dir must be named candidate_XXXX, got {candidate_dir}"
        ) from exc


def _read_first_jsonl_mapping(path: Path) -> Mapping[str, Any]:
    """Read the first non-empty JSONL object or fail closed."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"episode record path is unavailable: {path}") from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"episode record is malformed JSON: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("episode record must be a JSON object")
        return payload
    raise ValueError(f"episode record is empty: {path}")


def _extract_simulator_steps(record: Mapping[str, Any]) -> int:
    """Return simulator-step count from a known episode-record shape."""
    raw_steps = record.get("steps")
    if isinstance(raw_steps, Sequence) and not isinstance(raw_steps, str | bytes | bytearray):
        return _require_nonnegative_int(len(raw_steps), "record.steps")
    if raw_steps is not None:
        return _require_nonnegative_int(raw_steps, "record.steps")
    trace = record.get("trace")
    if isinstance(trace, Mapping):
        trace_steps = trace.get("steps")
        if isinstance(trace_steps, Sequence) and not isinstance(
            trace_steps, str | bytes | bytearray
        ):
            return _require_nonnegative_int(len(trace_steps), "record.trace.steps")
    raise ValueError("episode record does not contain simulator-step accounting")


def _scenario_seed_from_config(config: SearchConfig) -> int:
    """Return the frozen scenario seed from a SearchConfig or fail closed."""
    seed_range = config.search_space.scenario_seed
    if seed_range.min != seed_range.max or not float(seed_range.min).is_integer():
        raise ValueError("search_space.scenario_seed must be a frozen integer")
    return _require_nonnegative_int(int(seed_range.min), "search_space.scenario_seed")


def _finite_array(value: Any, name: str) -> np.ndarray:
    """Return a finite float array."""
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_nonnegative_int(value: Any, name: str) -> int:
    """Return ``value`` as a non-negative integer, rejecting booleans and NaN."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0 or not math.isfinite(float(result)):
        raise ValueError(f"{name} must be a non-negative integer")
    return result


__all__ = [
    "MATCHED_COMPUTE_RUNTIME_SCHEMA",
    "MATCHED_COMPUTE_RUNTIME_SCHEMA_VERSION",
    "MatchedComputeRuntimeTrace",
    "ReactiveRuntimeSnapshot",
    "probe_open_loop_runtime",
    "probe_reactive_runtime",
]
