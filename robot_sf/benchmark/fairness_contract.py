"""Matched-capability fairness contract for cross-planner comparison.

Generates a machine-readable planner capability matrix from the algorithm
registry and emits explicit mismatch flags when cross-planner rows mix
planners with different observation privilege, adapter type, or tuning
provenance.  Only rows that pass the fair-comparison gate may feed
algorithm-ranking claims; mismatched rows remain case-study evidence.

Report-layer only — no metric-semantics changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robot_sf.benchmark.algorithm_metadata import (
    _BASELINE_CATEGORY_BY_CANONICAL,
    _DEFAULT_OBSERVATION_SPEC,
    _KINEMATICS_PROFILE_BY_CANONICAL,
    _OBSERVATION_SPEC_BY_CANONICAL,
    canonical_algorithm_name,
)
from robot_sf.benchmark.algorithm_readiness import (
    get_algorithm_readiness,
)
from robot_sf.benchmark.observation_levels import (
    observation_level_for_mode,
)

# ──────────────────────────────────────────────────────────────────────
# Observation privilege tiers (ordered most-to-least privileged).
# ──────────────────────────────────────────────────────────────────────

OBSERVATION_PRIVILEGE_TIERS: dict[str, int] = {
    "oracle_full_state": 0,
    "tracked_agents_no_noise": 1,
    "tracked_agents_with_noise": 2,
    "lidar_2d": 3,
    "occluded_partial_state": 4,
}

_PRIVILEGED_INPUTS = frozenset({"pedestrians", "humans", "history", "social_groups"})
_LIDAR_FAMILY_MODES = frozenset({"sensor_fusion_state", "lidar_human_state"})
_GROUND_TRUTH_FAMILY_MODES = frozenset(
    {"socnav_state", "headed_socnav_state", "gst_human_state", "goal_state"}
)


def _observation_privilege_level(observation_mode: str) -> int:
    """Return a numeric privilege tier for an observation mode.

    Lower numbers mean more privileged access.  Falls back to the
    observation-level registry when available, otherwise assigns a
    conservative high tier.
    """
    spec = observation_level_for_mode(observation_mode)
    return OBSERVATION_PRIVILEGE_TIERS.get(spec.key, 3)


def _observation_privilege_label(observation_mode: str) -> str:
    """Return the observation-level key for an observation mode."""
    return observation_level_for_mode(observation_mode).key


def _has_privileged_inputs(required_inputs: tuple[str, ...]) -> bool:
    """Return whether the input set includes privileged ground-truth fields."""
    return bool(_PRIVILEGED_INPUTS & frozenset(required_inputs))


# ──────────────────────────────────────────────────────────────────────
# Capability entry (one per canonical planner)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlannerCapabilityEntry:
    """Machine-generated capability snapshot for one canonical planner.

    Populated entirely from the algorithm registry — no hand-written rows.
    """

    canonical_name: str
    baseline_category: str
    readiness_tier: str
    observation_mode: str
    observation_privilege_level: int
    observation_privilege_label: str
    has_privileged_inputs: bool
    required_inputs: tuple[str, ...]
    action_command_space: str
    default_execution_mode: str
    default_adapter_name: str
    adapter_active: bool
    projection_policy: str
    projection_documented: bool
    upstream_command_space: str
    diagnostic_reference_only: bool
    testing_only_adapter: bool
    prototype_only: bool
    tuning_budget_runs: int | None
    tuning_source: str
    compute_budget_class: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable capability payload."""
        return {
            "canonical_name": self.canonical_name,
            "baseline_category": self.baseline_category,
            "readiness_tier": self.readiness_tier,
            "observation_mode": self.observation_mode,
            "observation_privilege_level": self.observation_privilege_level,
            "observation_privilege_label": self.observation_privilege_label,
            "has_privileged_inputs": self.has_privileged_inputs,
            "required_inputs": list(self.required_inputs),
            "action_command_space": self.action_command_space,
            "default_execution_mode": self.default_execution_mode,
            "default_adapter_name": self.default_adapter_name,
            "adapter_active": self.adapter_active,
            "projection_policy": self.projection_policy,
            "projection_documented": self.projection_documented,
            "upstream_command_space": self.upstream_command_space,
            "diagnostic_reference_only": self.diagnostic_reference_only,
            "testing_only_adapter": self.testing_only_adapter,
            "prototype_only": self.prototype_only,
            "tuning_budget_runs": self.tuning_budget_runs,
            "tuning_source": self.tuning_source,
            "compute_budget_class": self.compute_budget_class,
        }


def build_capability_entry(
    canonical_name: str,
    *,
    observation_mode: str | None = None,
    tuning_budget_runs: int | None = None,
    tuning_source: str = "unknown",
) -> PlannerCapabilityEntry:
    """Build a capability entry for one canonical planner from the registry.

    All fields except the optional overrides are machine-generated from the
    algorithm metadata tables.

    Returns:
        PlannerCapabilityEntry with registry-derived capability snapshot.
    """
    obs_spec = _OBSERVATION_SPEC_BY_CANONICAL.get(canonical_name, _DEFAULT_OBSERVATION_SPEC)
    default_obs_mode = str(obs_spec.get("default_mode", "socnav_state"))
    active_obs_mode = observation_mode or default_obs_mode
    required_inputs = tuple(str(v) for v in obs_spec.get("inputs", ()))

    kin = _KINEMATICS_PROFILE_BY_CANONICAL.get(canonical_name, {})
    readiness = get_algorithm_readiness(canonical_name)
    tier = readiness.tier if readiness is not None else "unknown"
    category = _BASELINE_CATEGORY_BY_CANONICAL.get(canonical_name, "unknown")

    exec_mode = str(kin.get("default_execution_mode", "unknown"))
    adapter_name = str(kin.get("default_adapter_name", "none"))
    adapter_active = exec_mode in {"adapter", "mixed"}

    return PlannerCapabilityEntry(
        canonical_name=canonical_name,
        baseline_category=category,
        readiness_tier=tier,
        observation_mode=active_obs_mode,
        observation_privilege_level=_observation_privilege_level(active_obs_mode),
        observation_privilege_label=_observation_privilege_label(active_obs_mode),
        has_privileged_inputs=_has_privileged_inputs(required_inputs),
        required_inputs=required_inputs,
        action_command_space=str(kin.get("planner_command_space", "unknown")),
        default_execution_mode=exec_mode,
        default_adapter_name=adapter_name,
        adapter_active=adapter_active,
        projection_policy=str(kin.get("projection_policy", "")),
        projection_documented=bool(kin.get("projection_documented", False)),
        upstream_command_space=str(kin.get("upstream_command_space", "")),
        diagnostic_reference_only=bool(kin.get("diagnostic_reference_only", False)),
        testing_only_adapter=bool(kin.get("testing_only_adapter", False)),
        prototype_only=bool(kin.get("prototype_only", False)),
        tuning_budget_runs=tuning_budget_runs,
        tuning_source=tuning_source,
        compute_budget_class=_classify_compute_budget(canonical_name, kin),
    )


# ──────────────────────────────────────────────────────────────────────
# Capability matrix (all planners in a campaign)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlannerCapabilityMatrix:
    """Machine-generated capability matrix for a set of planners.

    Each entry is populated from the algorithm registry, not hand-written.
    """

    entries: dict[str, PlannerCapabilityEntry]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable capability matrix payload."""
        return {name: entry.to_dict() for name, entry in self.entries.items()}

    def entry(self, canonical_name: str) -> PlannerCapabilityEntry | None:
        """Look up a capability entry by canonical name.

        Returns:
            PlannerCapabilityEntry or None when not found.
        """
        return self.entries.get(canonical_name)

    def planner_names(self) -> tuple[str, ...]:
        """Return the canonical names of all planners in the matrix."""
        return tuple(self.entries.keys())


def build_capability_matrix(
    planner_configs: list[dict[str, Any]],
) -> PlannerCapabilityMatrix:
    """Build a capability matrix from campaign planner configs.

    Each config dict should contain at minimum ``algo`` (the algorithm key).
    Optional keys: ``observation_mode``, ``tuning`` (dict with
    ``budget_runs`` and ``source``).

    Returns:
        PlannerCapabilityMatrix with one entry per unique canonical planner.
    """
    entries: dict[str, PlannerCapabilityEntry] = {}
    for cfg in planner_configs:
        algo = str(cfg.get("algo", "")).strip()
        if not algo:
            continue
        canonical = canonical_algorithm_name(algo)
        if canonical in entries:
            continue
        obs_mode = cfg.get("observation_mode")
        tuning = cfg.get("tuning") or {}
        budget_runs = tuning.get("budget_runs")
        source = str(tuning.get("source", "unknown"))
        entries[canonical] = build_capability_entry(
            canonical,
            observation_mode=obs_mode,
            tuning_budget_runs=budget_runs if isinstance(budget_runs, int) else None,
            tuning_source=source,
        )
    return PlannerCapabilityMatrix(entries=entries)


# ──────────────────────────────────────────────────────────────────────
# Mismatch detection
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CapabilityMismatch:
    """One detected capability mismatch between two planners."""

    dimension: str
    planner_a: str
    planner_b: str
    value_a: str
    value_b: str
    severity: str  # "hard" | "soft"
    description: str


@dataclass(frozen=True)
class FairnessReport:
    """Result of evaluating a set of planners against the fairness contract."""

    matrix: PlannerCapabilityMatrix
    mismatches: tuple[CapabilityMismatch, ...]
    fair_subset: tuple[str, ...]
    excluded_planners: tuple[str, ...]
    ranking_claim_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable fairness report payload."""
        return {
            "matrix": self.matrix.to_dict(),
            "mismatches": [
                {
                    "dimension": m.dimension,
                    "planner_a": m.planner_a,
                    "planner_b": m.planner_b,
                    "value_a": m.value_a,
                    "value_b": m.value_b,
                    "severity": m.severity,
                    "description": m.description,
                }
                for m in self.mismatches
            ],
            "fair_subset": list(self.fair_subset),
            "excluded_planners": list(self.excluded_planners),
            "ranking_claim_allowed": self.ranking_claim_allowed,
        }


def detect_mismatches(
    matrix: PlannerCapabilityMatrix,
) -> tuple[CapabilityMismatch, ...]:
    """Detect capability mismatches across all planner pairs in the matrix.

    Checks three dimensions:
    1. Adapter mismatch — different adapter types or native vs adapter execution
    2. Observation privilege — different perception assumption tiers
    3. Tuning asymmetry — different tuning budgets or tuning sources

    Returns:
        Tuple of CapabilityMismatch instances.
    """
    mismatches: list[CapabilityMismatch] = []
    names = sorted(matrix.entries.keys())
    for i, name_a in enumerate(names):
        entry_a = matrix.entries[name_a]
        for name_b in names[i + 1 :]:
            entry_b = matrix.entries[name_b]
            mismatches.extend(_compare_entries(entry_a, entry_b))
    return tuple(mismatches)


def _compare_entries(
    a: PlannerCapabilityEntry,
    b: PlannerCapabilityEntry,
) -> list[CapabilityMismatch]:
    """Compare two capability entries and return detected mismatches.

    Returns:
        List of CapabilityMismatch instances for this pair.
    """
    result: list[CapabilityMismatch] = []

    # 1. Adapter mismatch
    if a.default_execution_mode != b.default_execution_mode:
        result.append(
            CapabilityMismatch(
                dimension="adapter",
                planner_a=a.canonical_name,
                planner_b=b.canonical_name,
                value_a=a.default_execution_mode,
                value_b=b.default_execution_mode,
                severity="hard",
                description=(
                    f"Execution mode differs: {a.canonical_name}={a.default_execution_mode} "
                    f"vs {b.canonical_name}={b.default_execution_mode}."
                ),
            )
        )
    elif a.adapter_active and b.adapter_active and a.default_adapter_name != b.default_adapter_name:
        result.append(
            CapabilityMismatch(
                dimension="adapter",
                planner_a=a.canonical_name,
                planner_b=b.canonical_name,
                value_a=a.default_adapter_name,
                value_b=b.default_adapter_name,
                severity="soft",
                description=(
                    f"Adapter name differs: {a.canonical_name}={a.default_adapter_name} "
                    f"vs {b.canonical_name}={b.default_adapter_name}."
                ),
            )
        )

    # 2. Observation privilege mismatch
    if a.observation_privilege_level != b.observation_privilege_level:
        result.append(
            CapabilityMismatch(
                dimension="observation_privilege",
                planner_a=a.canonical_name,
                planner_b=b.canonical_name,
                value_a=f"{a.observation_privilege_label} (tier {a.observation_privilege_level})",
                value_b=f"{b.observation_privilege_label} (tier {b.observation_privilege_level})",
                severity="hard",
                description=(
                    f"Observation privilege differs: {a.canonical_name} uses "
                    f"{a.observation_privilege_label} (privileged_inputs="
                    f"{a.has_privileged_inputs}) vs {b.canonical_name} uses "
                    f"{b.observation_privilege_label} (privileged_inputs="
                    f"{b.has_privileged_inputs})."
                ),
            )
        )
    elif a.has_privileged_inputs != b.has_privileged_inputs:
        result.append(
            CapabilityMismatch(
                dimension="observation_privilege",
                planner_a=a.canonical_name,
                planner_b=b.canonical_name,
                value_a=f"privileged_inputs={a.has_privileged_inputs}",
                value_b=f"privileged_inputs={b.has_privileged_inputs}",
                severity="hard",
                description=(
                    f"Privileged input access differs: {a.canonical_name} "
                    f"has_privileged_inputs={a.has_privileged_inputs} "
                    f"vs {b.canonical_name} has_privileged_inputs={b.has_privileged_inputs}."
                ),
            )
        )

    # 3. Tuning asymmetry
    if (
        a.tuning_source != b.tuning_source
        and a.tuning_source not in {"unknown"}
        and b.tuning_source not in {"unknown"}
    ):
        result.append(
            CapabilityMismatch(
                dimension="tuning_asymmetry",
                planner_a=a.canonical_name,
                planner_b=b.canonical_name,
                value_a=f"source={a.tuning_source}, budget_runs={a.tuning_budget_runs}",
                value_b=f"source={b.tuning_source}, budget_runs={b.tuning_budget_runs}",
                severity="soft",
                description=(
                    f"Tuning source differs: {a.canonical_name}={a.tuning_source} "
                    f"vs {b.canonical_name}={b.tuning_source}."
                ),
            )
        )
    if (
        a.tuning_budget_runs is not None
        and b.tuning_budget_runs is not None
        and a.tuning_budget_runs != b.tuning_budget_runs
    ):
        result.append(
            CapabilityMismatch(
                dimension="tuning_asymmetry",
                planner_a=a.canonical_name,
                planner_b=b.canonical_name,
                value_a=f"budget_runs={a.tuning_budget_runs}",
                value_b=f"budget_runs={b.tuning_budget_runs}",
                severity="soft",
                description=(
                    f"Tuning budget differs: {a.canonical_name}={a.tuning_budget_runs} runs "
                    f"vs {b.canonical_name}={b.tuning_budget_runs} runs."
                ),
            )
        )

    return result


# ──────────────────────────────────────────────────────────────────────
# Fair-comparison subset
# ──────────────────────────────────────────────────────────────────────


def fair_comparison_subset(
    matrix: PlannerCapabilityMatrix,
    mismatches: tuple[CapabilityMismatch, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition planners into fair-subset and excluded groups.

    A planner is excluded from the fair subset when it has any hard-severity
    mismatch with another planner in the matrix.  Soft mismatches (adapter
    name, tuning budget) are recorded but do not exclude.

    Returns:
        ``(fair_subset, excluded_planners)`` — both tuples of canonical names.
    """
    hard_excluded: set[str] = set()
    for mismatch in mismatches:
        if mismatch.severity == "hard":
            hard_excluded.add(mismatch.planner_a)
            hard_excluded.add(mismatch.planner_b)

    all_names = set(matrix.entries.keys())
    fair = sorted(all_names - hard_excluded)
    excluded = sorted(hard_excluded)
    return tuple(fair), tuple(excluded)


# ──────────────────────────────────────────────────────────────────────
# Ranking-claim gate
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RankingClaimVerdict:
    """Verdict from the ranking-claim gate."""

    allowed: bool
    reason: str
    fair_subset: tuple[str, ...]
    excluded: tuple[str, ...]
    hard_mismatch_count: int
    soft_mismatch_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable verdict payload."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "fair_subset": list(self.fair_subset),
            "excluded": list(self.excluded),
            "hard_mismatch_count": self.hard_mismatch_count,
            "soft_mismatch_count": self.soft_mismatch_count,
        }


def ranking_claim_gate(
    matrix: PlannerCapabilityMatrix,
) -> RankingClaimVerdict:
    """Evaluate whether the planner set may feed algorithm-ranking claims.

    Ranking claims are allowed only when every planner pair in the matrix
    has matched capabilities (no hard mismatches).  Soft mismatches are
    recorded as caveats but do not block.

    Returns:
        RankingClaimVerdict with allowed flag and detailed reason.
    """
    mismatches = detect_mismatches(matrix)
    fair, excluded = fair_comparison_subset(matrix, mismatches)
    hard_count = sum(1 for m in mismatches if m.severity == "hard")
    soft_count = sum(1 for m in mismatches if m.severity == "soft")

    if hard_count == 0:
        reason = (
            f"All {len(matrix.entries)} planners have matched capabilities; "
            f"{soft_count} soft mismatch(es) recorded as caveats."
        )
        if soft_count > 0:
            caveat_dims = sorted({m.dimension for m in mismatches if m.severity == "soft"})
            reason += f" Soft dimensions: {', '.join(caveat_dims)}."
        return RankingClaimVerdict(
            allowed=True,
            reason=reason,
            fair_subset=fair,
            excluded=excluded,
            hard_mismatch_count=hard_count,
            soft_mismatch_count=soft_count,
        )

    excluded_dims = sorted({m.dimension for m in mismatches if m.severity == "hard"})
    reason = (
        f"Ranking claim blocked: {hard_count} hard mismatch(es) across "
        f"dimension(s) [{', '.join(excluded_dims)}]. "
        f"Fair subset: {fair}. Excluded: {excluded}. "
        f"Only fair-subset rows may feed ranking claims; mismatched rows "
        f"remain case-study evidence."
    )
    return RankingClaimVerdict(
        allowed=False,
        reason=reason,
        fair_subset=fair,
        excluded=excluded,
        hard_mismatch_count=hard_count,
        soft_mismatch_count=soft_count,
    )


# ──────────────────────────────────────────────────────────────────────
# Report integration helpers
# ──────────────────────────────────────────────────────────────────────


def emit_mismatch_flags(
    matrix: PlannerCapabilityMatrix,
    report_row: dict[str, Any],
    canonical_name: str,
) -> dict[str, Any]:
    """Augment a campaign report row with mismatch flags relative to the matrix.

    Adds ``fairness_mismatch_flags`` and ``fairness_in_ranking_subset`` keys
    to the row.  Does not modify existing keys.

    Returns:
        The augmented report row (mutated in place and returned).
    """
    mismatches = detect_mismatches(matrix)
    fair, _ = fair_comparison_subset(matrix, mismatches)
    planner_mismatches = [
        {
            "dimension": m.dimension,
            "peer": m.planner_b if m.planner_a == canonical_name else m.planner_a,
            "severity": m.severity,
            "description": m.description,
        }
        for m in mismatches
        if canonical_name in (m.planner_a, m.planner_b)
    ]
    report_row["fairness_mismatch_flags"] = planner_mismatches
    report_row["fairness_in_ranking_subset"] = canonical_name in fair
    return report_row


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────


def _classify_compute_budget(canonical_name: str, kin: dict[str, Any]) -> str:
    """Classify the compute budget class for a planner.

    Returns:
        One of ``realtime_reactive``, ``short_horizon_optimization``,
        ``learned_inference``, or ``diagnostic_reference``.
    """
    category = _BASELINE_CATEGORY_BY_CANONICAL.get(canonical_name, "unknown")
    if category == "diagnostic":
        return "diagnostic_reference"
    if category == "learning":
        return "learned_inference"
    if "mpc" in canonical_name.lower() or "nmpc" in canonical_name.lower():
        return "short_horizon_optimization"
    if "mppi" in canonical_name.lower() or "sampling" in canonical_name.lower():
        return "short_horizon_optimization"
    if "prediction" in canonical_name.lower():
        return "short_horizon_optimization"
    return "realtime_reactive"


__all__ = [
    "CapabilityMismatch",
    "FairnessReport",
    "PlannerCapabilityEntry",
    "PlannerCapabilityMatrix",
    "RankingClaimVerdict",
    "build_capability_entry",
    "build_capability_matrix",
    "detect_mismatches",
    "emit_mismatch_flags",
    "fair_comparison_subset",
    "ranking_claim_gate",
]
