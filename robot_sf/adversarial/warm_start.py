"""Extract knife-edge warm-start candidates for the adversarial optimizer samplers.

Knife-edge seeds (#5816/#5817) sit on scenario decision boundaries: a small
per-context perturbation flips the episode outcome. Seed-sensitivity analysis
already surfaces them as near-boundary episodes whose ``|outcome margin|`` is
small. This module converts those reports into :class:`WarmStartCandidate` tuples
the optimizer-backed samplers accept as initial candidates (Optuna enqueued
trials, CMA-ES x0), so an active search starts from known high-value points
instead of cold random sampling.

The warm starts are CAPABILITY artifacts, not validated evidence (design #1433):
the extractor never writes into a release evidence store and the pilot report
lands only under the adversarial archive path.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    Pose2D,
    SearchConfig,
    SearchSpaceConfig,
    WarmStartCandidate,
)
from robot_sf.adversarial.samplers import build_sampler
from robot_sf.adversarial.search import run_adversarial_search

WARM_START_SCHEMA_VERSION = "adversarial-warm-start.v1"

_FLIP_ENTRY_KEYS = (
    "scenario",
    "planner",
    "outcome_margin",
    "candidate",
)


@dataclass(frozen=True)
class WarmStartExtraction:
    """Result of extracting warm starts from a knife-edge flip report."""

    schema_version: str
    source: str
    margin_threshold: float
    num_near_boundary: int
    num_selected: int
    warm_starts: tuple[WarmStartCandidate, ...]
    rejected: tuple[dict[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable extraction payload with provenance."""
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "margin_threshold": float(self.margin_threshold),
            "num_near_boundary": int(self.num_near_boundary),
            "num_selected": int(self.num_selected),
            "warm_starts": [warm.to_json() for warm in self.warm_starts],
            "rejected": [dict(entry) for entry in self.rejected],
        }


def load_flip_report(path: str | Path) -> dict[str, Any]:
    """Load a knife-edge flip report (JSON) describing near-boundary seeds."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Flip report must be a JSON object: {path}")
    return payload


def extract_warm_starts(
    report: dict[str, Any] | list[dict[str, Any]],
    *,
    search_space: SearchSpaceConfig,
    margin_threshold: float = 0.5,
    source: str = "",
) -> WarmStartExtraction:
    """Extract warm-start candidates from a knife-edge flip report.

    A flip-report entry is a mapping with ``scenario``, ``planner``,
    ``outcome_margin`` (signed margin; sign encodes which side of the boundary
    the original outcome landed), and ``candidate`` geometry. Entries whose
    ``|outcome_margin|`` is at or below ``margin_threshold`` are selected as warm
    starts, provided their candidate geometry stays inside ``search_space``.

    The report may be a single mapping with an ``entries`` list (the #5817-class
    format) or a bare list of entries.

    Args:
        report: Flip report mapping or entry list.
        search_space: Search space the warm starts must lie inside.
        margin_threshold: Maximum ``|outcome_margin|`` for selection.
        source: Provenance label (file path or campaign id).

    Returns:
        WarmStartExtraction with selected warm starts and provenance.
    """
    if not math.isfinite(margin_threshold) or margin_threshold < 0.0:
        raise ValueError("margin_threshold must be finite and >= 0")
    entries = _coerce_entries(report)
    warm_starts: list[WarmStartCandidate] = []
    rejected: list[dict[str, Any]] = []
    near_boundary = 0

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            rejected.append({"index": index, "reason": "entry must be a mapping"})
            continue
        margin = entry.get("outcome_margin")
        margin_value = _finite_float(margin)
        if margin_value is None:
            rejected.append({"index": index, "reason": "missing/outcome_margin"})
            continue
        if abs(margin_value) > margin_threshold:
            rejected.append(
                {
                    "index": index,
                    "reason": "margin_beyond_threshold",
                    "outcome_margin": margin_value,
                }
            )
            continue
        near_boundary += 1
        candidate = _candidate_from_entry(entry, search_space=search_space)
        if candidate is None:
            rejected.append({"index": index, "reason": "candidate_geometry_unparseable"})
            continue
        errors = search_space.validate_candidate(candidate)
        if errors:
            rejected.append({"index": index, "reason": "outside_search_space", "errors": errors})
            continue
        warm_starts.append(
            WarmStartCandidate(
                candidate=candidate,
                scenario=str(entry.get("scenario", "")),
                planner=str(entry.get("planner", "")),
                outcome_margin=float(margin_value),
            )
        )

    return WarmStartExtraction(
        schema_version=WARM_START_SCHEMA_VERSION,
        source=source,
        margin_threshold=float(margin_threshold),
        num_near_boundary=near_boundary,
        num_selected=len(warm_starts),
        warm_starts=tuple(warm_starts),
        rejected=tuple(rejected),
    )


def _coerce_entries(report: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the entry list from a report mapping or bare list."""
    if isinstance(report, list):
        return [entry for entry in report if isinstance(entry, dict)]
    if isinstance(report, dict):
        entries = report.get("entries")
        if entries is None:
            return [report] if _looks_like_entry(report) else []
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)]
    return []


def _looks_like_entry(payload: dict[str, Any]) -> bool:
    """Return whether a mapping resembles a flip-report entry."""
    return any(key in payload for key in _FLIP_ENTRY_KEYS)


def _candidate_from_entry(
    entry: dict[str, Any], *, search_space: SearchSpaceConfig
) -> CandidateSpec | None:
    """Build a CandidateSpec from a flip-report entry geometry."""
    raw = entry.get("candidate")
    if not isinstance(raw, dict):
        return None
    start = _pose_from_mapping(raw.get("start"), name="candidate.start")
    goal = _pose_from_mapping(raw.get("goal"), name="candidate.goal")
    if start is None or goal is None:
        return None

    def _scalar(name: str, default: float) -> float:
        value = raw.get(name)
        parsed = _finite_float(value)
        return float(parsed) if parsed is not None else default

    seed = raw.get("scenario_seed", raw.get("seed"))
    seed_value = (
        int(seed)
        if isinstance(seed, int)
        else (int(seed) if isinstance(seed, float) and seed.is_integer() else None)
    )
    if seed_value is None:
        seed_value = int(search_space.scenario_seed.min)
    return CandidateSpec(
        start=start,
        goal=goal,
        spawn_time_s=_scalar("spawn_time_s", float(search_space.spawn_time_s.min)),
        pedestrian_speed_mps=_scalar(
            "pedestrian_speed_mps", float(search_space.pedestrian_speed_mps.min)
        ),
        pedestrian_delay_s=_scalar(
            "pedestrian_delay_s", float(search_space.pedestrian_delay_s.min)
        ),
        scenario_seed=seed_value,
    )


def _pose_from_mapping(payload: object, *, name: str) -> Pose2D | None:
    """Build a pose from a mapping with x/y fields."""
    if not isinstance(payload, dict) or "x" not in payload or "y" not in payload:
        return None
    theta_raw = payload.get("theta")
    theta = float(theta_raw) if _finite_float(theta_raw) is not None else 0.0
    return Pose2D(float(payload["x"]), float(payload["y"]), theta)


def _finite_float(value: Any) -> float | None:
    """Return a finite float or None."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class WarmVsColdPilotReport:
    """Certification-grade pilot comparing warm-started vs cold-started search.

    The report is a CAPABILITY artifact (design #1433): it demonstrates whether
    warm starts let an optimizer-backed search find stable failures faster at a
    fixed budget. It never touches a release evidence store.
    """

    schema_version: str
    budget: int
    search_space: str
    warm_first_failure_iteration: int | None
    cold_first_failure_iteration: int | None
    warm_best_objective: float | None
    cold_best_objective: float | None
    warm_num_candidates: int
    cold_num_candidates: int
    num_warm_starts: int
    warm_start_seeds: tuple[int, ...]
    faster: bool
    provenance: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable pilot report payload."""
        return {
            "schema_version": self.schema_version,
            "budget": int(self.budget),
            "search_space": self.search_space,
            "warm_first_failure_iteration": self.warm_first_failure_iteration,
            "cold_first_failure_iteration": self.cold_first_failure_iteration,
            "warm_best_objective": self.warm_best_objective,
            "cold_best_objective": self.cold_best_objective,
            "warm_num_candidates": int(self.warm_num_candidates),
            "cold_num_candidates": int(self.cold_num_candidates),
            "num_warm_starts": int(self.num_warm_starts),
            "warm_start_seeds": list(self.warm_start_seeds),
            "faster": bool(self.faster),
            "provenance": dict(self.provenance),
        }


def warm_vs_cold_pilot(
    config: SearchConfig,
    *,
    warm_start: tuple[WarmStartCandidate, ...],
    objective: str,
    evaluator: Callable[..., CandidateEvaluation],
    certifier: Callable[..., Any] | None = None,
    output_dir: Path,
    sampler_name: str = "optuna",
    collapse_predicate: Callable[[CandidateSpec], bool] | None = None,
) -> WarmVsColdPilotReport:
    """Run a warm-started vs cold-started search at a fixed budget and report efficiency.

    Both arms share ``config`` except for the warm-start list, so the only
    difference is whether the optimizer begins from knife-edge candidates. The
    report records the first iteration at which each arm surfaces a stable
    failure (per ``collapse_predicate``) and whether the warm arm was at least
    as fast. The artifact is written under ``output_dir`` (an adversarial archive
    path), never into a release evidence store.

    Args:
        config: Base search configuration (budget is shared by both arms).
        warm_start: Knife-edge warm-start candidates for the warm arm.
        objective: Objective name to score candidates.
        evaluator: Injected candidate evaluator (synthetic for CPU pilots).
        certifier: Optional injected certifier.
        output_dir: Adversarial archive directory for the report artifact.
        sampler_name: Optimizer sampler family to drive both arms.
        collapse_predicate: Returns True for a candidate that counts as a stable failure.

    Returns:
        WarmVsColdPilotReport written to ``output_dir/warm_vs_cold_pilot.json``.
    """
    budget = int(config.budget)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if not warm_start:
        raise ValueError("warm_vs_cold_pilot requires at least one warm start")

    def _collapse(candidate: CandidateSpec) -> bool:
        return bool(collapse_predicate(candidate)) if collapse_predicate else False

    warm_config = _replace_search_config(
        config, objective=objective, output_dir=Path(output_dir) / "warm", warm_start=warm_start
    )
    cold_config = _replace_search_config(
        config, objective=objective, output_dir=Path(output_dir) / "cold", warm_start=()
    )

    warm_result = run_adversarial_search(
        warm_config,
        sampler=build_sampler(
            sampler_name,
            config.search_space,
            seed=config.seed,
            warm_start=warm_start,
        ),
        evaluator=evaluator,
        certifier=certifier,
    )
    cold_result = run_adversarial_search(
        cold_config,
        sampler=build_sampler(sampler_name, config.search_space, seed=config.seed),
        evaluator=evaluator,
        certifier=certifier,
    )

    warm_first = _first_failure_iteration(warm_config.output_dir)
    cold_first = _first_failure_iteration(cold_config.output_dir)
    if collapse_predicate is not None:
        warm_first = _first_collapse_iteration(warm_config.output_dir, _collapse)
        cold_first = _first_collapse_iteration(cold_config.output_dir, _collapse)

    report = WarmVsColdPilotReport(
        schema_version="adversarial-warm-vs-cold-pilot.v1",
        budget=budget,
        search_space=(config.search_space_path.as_posix() if config.search_space_path else ""),
        warm_first_failure_iteration=warm_first,
        cold_first_failure_iteration=cold_first,
        warm_best_objective=_best_objective(warm_config.output_dir),
        cold_best_objective=_best_objective(cold_config.output_dir),
        warm_num_candidates=warm_result.num_candidates,
        cold_num_candidates=cold_result.num_candidates,
        num_warm_starts=len(warm_start),
        warm_start_seeds=tuple(warm.candidate.scenario_seed for warm in warm_start),
        faster=bool(warm_first is not None and (cold_first is None or warm_first <= cold_first)),
        provenance={
            "source": "issue #5833 warm-start pilot",
            "design_note": "#1433",
            "sampler": sampler_name,
            "objective": objective,
            "warm_manifest": warm_result.manifest_path.as_posix(),
            "cold_manifest": cold_result.manifest_path.as_posix(),
            "evidence_tier": "capability_artifact_not_release_evidence",
        },
    )
    report_path = Path(output_dir) / "warm_vs_cold_pilot.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _replace_search_config(
    config: SearchConfig,
    *,
    objective: str,
    output_dir: Path,
    warm_start: tuple[WarmStartCandidate, ...],
) -> SearchConfig:
    """Return a search config variant with the pilot objective/output/warm starts."""
    return replace(
        config,
        objective=objective,
        output_dir=output_dir,
        warm_start=warm_start,
    )


def _first_failure_iteration(output_dir: Path) -> int | None:
    """Return the first candidate index whose attribution names a failure."""
    manifest = _load_manifest(output_dir)
    if manifest is None:
        return None
    candidates = manifest.get("candidates") if isinstance(manifest, dict) else []
    if not isinstance(candidates, list):
        return None
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue
        attribution = item.get("failure_attribution")
        if not isinstance(attribution, dict):
            continue
        primary = attribution.get("primary_failure")
        if primary and str(primary).strip().lower() not in {"", "success", "invalid_candidate"}:
            return index
    return None


def _first_collapse_iteration(
    output_dir: Path, predicate: Callable[[CandidateSpec], bool]
) -> int | None:
    """Return the first candidate index satisfying the collapse predicate."""
    manifest = _load_manifest(output_dir)
    if manifest is None:
        return None
    candidates = manifest.get("candidates") if isinstance(manifest, dict) else []
    if not isinstance(candidates, list):
        return None
    for index, item in enumerate(candidates):
        candidate = _candidate_from_manifest(item)
        if candidate is not None and predicate(candidate):
            return index
    return None


def _candidate_from_manifest(item: dict[str, Any]) -> CandidateSpec | None:
    """Build a CandidateSpec from a manifest candidate payload."""
    raw = item.get("candidate") if isinstance(item, dict) else None
    if not isinstance(raw, dict):
        return None
    space = SearchSpaceConfig.from_mapping(
        {
            "variables": {
                "start_x": {"min": 0.0, "max": 100.0},
                "start_y": {"min": 0.0, "max": 100.0},
                "goal_x": {"min": 0.0, "max": 100.0},
                "goal_y": {"min": 0.0, "max": 100.0},
                "spawn_time_s": {"min": 0.0, "max": 0.0},
                "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                "scenario_seed": {"min": 0, "max": 1_000_000},
            },
            "constraints": {"min_start_goal_distance_m": 0.0},
        }
    )
    return _candidate_from_entry({"candidate": raw}, search_space=space)


def _best_objective(output_dir: Path) -> float | None:
    """Return the best (max) objective value recorded in a search manifest."""
    manifest = _load_manifest(output_dir)
    if manifest is None:
        return None
    candidates = manifest.get("candidates") if isinstance(manifest, dict) else []
    if not isinstance(candidates, list):
        return None
    best: float | None = None
    for item in candidates:
        if not isinstance(item, dict):
            continue
        value = item.get("objective_value")
        parsed = _finite_float(value)
        if parsed is not None and (best is None or parsed > best):
            best = parsed
    return best


def _load_manifest(output_dir: Path) -> dict[str, Any] | None:
    """Load a search manifest if present."""
    manifest_path = Path(output_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


__all__ = [
    "WARM_START_SCHEMA_VERSION",
    "WarmStartCandidate",
    "WarmStartExtraction",
    "WarmVsColdPilotReport",
    "extract_warm_starts",
    "load_flip_report",
    "warm_vs_cold_pilot",
]
