"""Cross-planner adversarial transfer matrix and minimax robustness ranking.

Slice 1 of issue #5303 (cheap-lane, capability-only). This module *measures*
transfer structure: it reuses the existing adversarial archive of certified
worst-case configs found against ONE target planner and builds the K x N
transfer matrix — does a discovered weak point transfer to the other planners,
or is it policy-specific?

It deliberately does NOT run the minimax search game (that is a later slice,
only if slice 1 shows meaningful transfer structure). It also does NOT run any
bench re-evaluations itself: those run on the ops queue (the issue pins
"compute via ops queue"). Instead it consumes a per-planner evaluation table
whose rows are produced by replaying each certified config against each planner
at the standard seed protocol and summarising the episode with
:func:`robot_sf.adversarial.robustness.compute_robustness_report` (so the
transfer metric uses the same signed-robustness semantics as the search
objectives).

Capability-not-evidence boundary: the matrix is built only from archive paths
and pinned configs/seeds. No benchmark or paper-facing claim is made here; the
report is explicitly labelled capability-only.

Status: research/exploratory. These artifacts are transfer measurements, not
reported benchmark metrics.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.adversarial.archive import (
    ARCHIVE_SCHEMA_VERSION,
    SEARCH_MANIFEST_SCHEMA_VERSION,
)

_TRANSFER_MATRIX_SCHEMA_VERSION = "adversarial_transfer_matrix.v1"

# Benchmark eligibility tiers that count as "certified / repliable" for slice 1.
_CERTIFIED_ELIGIBILITY = frozenset({"eligible", "stress_only"})

# Default 3-planner mechanism-stratified roster for slice 1: the issue asks for
# the target planner plus 2 other planners. The roster mirrors the engineered
# candidates called out in the issue's evidence-grade promotion plan.
DEFAULT_TRANSFER_ROSTER: tuple[str, ...] = (
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "ppo",
)


@dataclass(frozen=True)
class CertifiedConfig:
    """One certified worst-case config selected from a target-planner archive.

    Attributes
    ----------
    config_id : str
        Stable id for the config within the transfer matrix.
    target_planner : str
        Planner the config was optimized / certified against.
    candidate : dict[str, Any]
        The perturbable scenario candidate (start/goal/seed/speed/...).
    objective_value : float
        Worst-case composite objective (signed-robustness based) against the
        target planner. Larger = worse (more negative robustness).
    source_manifest : str
        Origin manifest path (provenance; archive path only).
    source_candidate_index : int
        Candidate index within the source manifest.
    certification_tier : str
        Benchmark eligibility tier from certification (eligible / stress_only).
    scenario_seed : int | None
        Pinned scenario seed for replay reproducibility.
    """

    config_id: str
    target_planner: str
    candidate: dict[str, Any]
    objective_value: float
    source_manifest: str
    source_candidate_index: int
    certification_tier: str
    scenario_seed: int | None


@dataclass(frozen=True)
class PlannerEval:
    """One per-planner re-evaluation result for a certified config.

    Attributes
    ----------
    config_id : str
        Config this result belongs to (matches :class:`CertifiedConfig`).
    planner : str
        Evaluated planner.
    robustness : float
        Overall signed robustness against this planner (negative = violated).
    failed : bool
        Whether the planner reproduced a failure (robustness < 0).
    seed : int | None
        Pinned evaluation seed (standard seed protocol).
    """

    config_id: str
    planner: str
    robustness: float
    failed: bool
    seed: int | None


@dataclass(frozen=True)
class TransferCell:
    """One cell of the K x N transfer matrix."""

    config_id: str
    planner: str
    robustness: float
    failed: bool
    transferred: bool


@dataclass(frozen=True)
class PlannerRanking:
    """Minimax (worst-case regret) ranking for one planner."""

    planner: str
    worst_case_robustness: float
    transfer_failure_rate: float
    minimax_regret: float
    rank: int


@dataclass(frozen=True)
class TransferMatrix:
    """The full K x N transfer measurement plus summary statistics."""

    schema_version: str = _TRANSFER_MATRIX_SCHEMA_VERSION
    target_planner: str = ""
    config_ids: tuple[str, ...] = ()
    planners: tuple[str, ...] = ()
    cells: tuple[TransferCell, ...] = ()
    ranking: tuple[PlannerRanking, ...] = ()
    overall_transfer_rate: float = 0.0
    transfer_rate_ci: tuple[float, float] = (0.0, 0.0)
    transfer_rate_bootstrap_n: int = 0

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""
        return {
            "schema_version": self.schema_version,
            "target_planner": self.target_planner,
            "config_ids": list(self.config_ids),
            "planners": list(self.planners),
            "cells": [c.__dict__ for c in self.cells],
            "ranking": [r.__dict__ for r in self.ranking],
            "overall_transfer_rate": self.overall_transfer_rate,
            "transfer_rate_ci": list(self.transfer_rate_ci),
            "transfer_rate_bootstrap_n": self.transfer_rate_bootstrap_n,
        }


def _candidate_certification_tier(candidate: dict[str, Any]) -> str | None:
    """Extract the benchmark eligibility tier from a candidate payload."""
    cert = candidate.get("certification_status")
    if not isinstance(cert, dict):
        return None
    certificates = (
        cert.get("details", {}).get("certificates")
        if isinstance(cert.get("details"), dict)
        else None
    )
    if isinstance(certificates, list) and certificates:
        first = certificates[0]
        if isinstance(first, dict):
            return first.get("benchmark_eligibility")
    # Fall back to a single top-level status if present.
    return cert.get("status")


def _candidate_scenario_seed(candidate: dict[str, Any]) -> int | None:
    """Extract the pinned scenario seed from a candidate payload."""
    seed = candidate.get("scenario_seed")
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None


def _is_certified(candidate: dict[str, Any]) -> bool:
    """Return whether a candidate is certified / repliable for slice 1."""
    tier = _candidate_certification_tier(candidate)
    return tier in _CERTIFIED_ELIGIBILITY


def _objective_value(candidate: dict[str, Any]) -> float:
    """Return the worst-case objective value for a candidate."""
    value = candidate.get("objective_value")
    if value is None:
        return float("-inf")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return parsed if __import__("math").isfinite(parsed) else float("-inf")


def select_certified_configs(
    manifest_paths: list[str | Path],
    *,
    target_planner: str,
    K: int,
    scenario_template: str | None = None,
) -> list[CertifiedConfig]:
    """Select the top-K certified worst-case configs against ONE planner.

    Reads adversarial search manifests (real ``adversarial-search-manifest.v1``
    schema), keeps only certified, repliable candidates (``eligible`` /
    ``stress_only`` benchmark eligibility), optionally filters by scenario
    template, and returns the K configs with the worst (largest) objective
    value, i.e. the strongest discovered weak points.

    Parameters
    ----------
    manifest_paths : list[str | Path]
        Search manifests to read configs from.
    target_planner : str
        Planner the configs were optimized / certified against.
    K : int
        Maximum number of configs to return (>= 5 required by the issue for the
        transfer measurement).
    scenario_template : str | None
        Optional scenario-template filter (exact match on manifest config).

    Returns
    -------
    list[CertifiedConfig]
        Up to K certified worst-case configs, sorted worst-first.
    """
    if K < 1:
        raise ValueError("K must be >= 1")

    configs: list[CertifiedConfig] = []
    for manifest_path in sorted(Path(p) for p in manifest_paths):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Search manifest must be a JSON object: {manifest_path}")
        schema = payload.get("schema_version")
        if schema != SEARCH_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported search manifest schema for {manifest_path}: {schema!r}; "
                f"expected {SEARCH_MANIFEST_SCHEMA_VERSION!r}"
            )
        manifest_config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        if (
            scenario_template is not None
            and manifest_config.get("scenario_template") != scenario_template
        ):
            continue
        candidates = payload.get("candidates") or []
        if not isinstance(candidates, list):
            continue
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            cand = (
                candidate.get("candidate") if isinstance(candidate.get("candidate"), dict) else {}
            )
            if not _is_certified(candidate):
                continue
            tier = _candidate_certification_tier(candidate) or "unknown"
            configs.append(
                CertifiedConfig(
                    config_id=f"{manifest_path.stem}#{index}",
                    target_planner=target_planner,
                    candidate=cand,
                    objective_value=_objective_value(candidate),
                    source_manifest=manifest_path.as_posix(),
                    source_candidate_index=index,
                    certification_tier=tier,
                    scenario_seed=_candidate_scenario_seed(cand),
                )
            )

    configs.sort(key=lambda c: (-c.objective_value, c.config_id))
    return configs[:K]


def _bootstrap_transfer_rate(
    failures: list[int],
    evaluations: list[int],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap a normal-approximation CI for the transfer (failure) rate.

    The transfer rate is the fraction of (config, planner) pairs in which a
    config that failed against the target planner *also* failed against the
    evaluated planner. This measures whether discovered weak points are
    structural (transfer) or policy-specific.

    Returns
        (point_estimate, ci_low, ci_high).
    """
    total_fails = sum(failures)
    total_evals = sum(evaluations)
    if total_evals == 0:
        return 0.0, 0.0, 0.0
    point = total_fails / total_evals
    if n_resamples <= 0:
        return point, point, point
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        acc = 0
        for f, n in zip(failures, evaluations, strict=True):
            if n == 0:
                continue
            acc += sum(1 for _ in range(n) if rng.random() < (f / n))
        means.append(acc / total_evals)
    means.sort()
    low = means[max(0, int(0.025 * len(means)))]
    high = means[min(len(means) - 1, int(0.975 * len(means)))]
    return point, low, high


def _build_cells(
    config_ids: tuple[str, ...],
    planners: tuple[str, ...],
    eval_by_key: dict[tuple[str, str], PlannerEval],
) -> list[TransferCell]:
    """Assemble K x N transfer cells from certified configs and eval results."""
    cells: list[TransferCell] = []
    for cfg_id in config_ids:
        for planner in planners:
            ev = eval_by_key.get((cfg_id, planner))
            if ev is None:
                cells.append(
                    TransferCell(
                        config_id=cfg_id,
                        planner=planner,
                        robustness=float("nan"),
                        failed=False,
                        transferred=False,
                    )
                )
                continue
            # A weak point transfers when the target config's failure also
            # reproduces against the evaluated planner.
            cells.append(
                TransferCell(
                    config_id=cfg_id,
                    planner=planner,
                    robustness=ev.robustness,
                    failed=ev.failed,
                    transferred=ev.failed,
                )
            )
    return cells


def _build_ranking(
    cells: list[TransferCell],
    planners: tuple[str, ...],
) -> list[PlannerRanking]:
    """Compute per-planner minimax (worst-case robustness) ranking rows."""
    per_planner = _group_cells_by_planner(cells, planners)

    ranking_rows: list[PlannerRanking] = []
    for planner in planners:
        planner_cells = per_planner[planner]
        finite = [c.robustness for c in planner_cells if not math.isnan(c.robustness)]
        worst = min(finite) if finite else float("nan")
        failures = [c for c in planner_cells if c.failed]
        transfer_rate = len(failures) / len(planner_cells) if planner_cells else 0.0
        ranking_rows.append(
            PlannerRanking(
                planner=planner,
                worst_case_robustness=worst,
                transfer_failure_rate=transfer_rate,
                minimax_regret=-worst if not math.isnan(worst) else float("nan"),
                rank=0,
            )
        )
    return ranking_rows


def _group_cells_by_planner(
    cells: list[TransferCell],
    planners: tuple[str, ...],
) -> dict[str, list[TransferCell]]:
    """Group transfer cells by planner column, in planner order."""
    per_planner: dict[str, list[TransferCell]] = {p: [] for p in planners}
    for cell in cells:
        if cell.planner in per_planner:
            per_planner[cell.planner].append(cell)
    return per_planner


def build_transfer_matrix(
    configs: list[CertifiedConfig],
    evaluations: list[PlannerEval],
    *,
    planners: tuple[str, ...] | None = None,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 0,
) -> TransferMatrix:
    """Build the K x N transfer matrix from certified configs + eval results.

    Parameters
    ----------
    configs : list[CertifiedConfig]
        Certified worst-case configs (typically against the target planner).
    evaluations : list[PlannerEval]
        Per-planner re-evaluation results, keyed by config_id + planner.
    planners : tuple[str, ...] | None
        Planner order for the matrix columns. Defaults to the union of planners
        seen in ``evaluations`` preserving first-seen order.
    bootstrap_n : int
        Number of bootstrap resamples for the transfer-rate CI.
    bootstrap_seed : int
        Deterministic seed for the bootstrap resampling.

    Returns
    -------
    TransferMatrix
        The transfer measurement, per-planner minimax ranking, and bootstrap CI.
    """
    if not configs:
        raise ValueError("Cannot build a transfer matrix from zero certified configs")
    if len(configs) < 5:
        raise ValueError(
            f"Issue #5303 slice 1 requires K >= 5 certified configs; got {len(configs)}"
        )

    config_ids = tuple(c.config_id for c in configs)
    target_planner = configs[0].target_planner

    if planners is None:
        seen: list[str] = []
        for ev in evaluations:
            if ev.planner not in seen:
                seen.append(ev.planner)
        planners = tuple(seen)

    # Map (config_id, planner) -> eval.
    eval_by_key: dict[tuple[str, str], PlannerEval] = {
        (ev.config_id, ev.planner): ev for ev in evaluations
    }

    cells = _build_cells(config_ids, planners, eval_by_key)
    ranking_rows = _build_ranking(cells, planners)
    ranking_rows.sort(key=lambda r: (r.worst_case_robustness, r.planner))
    for rank, row in enumerate(ranking_rows, start=1):
        ranking_rows[rank - 1] = PlannerRanking(
            planner=row.planner,
            worst_case_robustness=row.worst_case_robustness,
            transfer_failure_rate=row.transfer_failure_rate,
            minimax_regret=row.minimax_regret,
            rank=rank,
        )

    # Overall transfer rate + bootstrap CI across evaluated planners (excluding
    # the target planner column, which is by construction the source of failure).
    other_planners = [p for p in planners if p != target_planner]
    if other_planners:
        grouped = _group_cells_by_planner(cells, planners)
        failures_per_planner: list[int] = []
        evals_per_planner: list[int] = []
        for planner in other_planners:
            planner_cells = grouped[planner]
            failures_per_planner.append(sum(1 for c in planner_cells if c.failed))
            evals_per_planner.append(len(planner_cells))
        rate, ci_low, ci_high = _bootstrap_transfer_rate(
            failures_per_planner,
            evals_per_planner,
            n_resamples=bootstrap_n,
            seed=bootstrap_seed,
        )
        overall_rate = rate
        ci = (ci_low, ci_high)
    else:
        overall_rate = 0.0
        ci = (0.0, 0.0)

    return TransferMatrix(
        target_planner=target_planner,
        config_ids=config_ids,
        planners=planners,
        cells=tuple(cells),
        ranking=tuple(ranking_rows),
        overall_transfer_rate=overall_rate,
        transfer_rate_ci=ci,
        transfer_rate_bootstrap_n=bootstrap_n if other_planners else 0,
    )


def render_transfer_report(matrix: TransferMatrix, *, configs: list[CertifiedConfig]) -> str:
    """Render a one-page transfer-measurement report (capability-only)."""
    lines: list[str] = []
    lines.append("# Cross-planner adversarial transfer matrix (slice 1, capability-only)")
    lines.append("")
    lines.append(
        "> Capability-not-evidence boundary: built only from archive paths and "
        "pinned configs/seeds. Not a benchmark or paper-facing claim."
    )
    lines.append("")
    lines.append(f"- Target planner (weak points discovered against): `{matrix.target_planner}`")
    lines.append(f"- Certified configs (K): {len(matrix.config_ids)}")
    lines.append(f"- Evaluated planners (N): {len(matrix.planners)}")
    lines.append(f"- Overall transfer rate (excl. target): {matrix.overall_transfer_rate:.3f}")
    lines.append(
        f"- Transfer-rate 95% CI: [{matrix.transfer_rate_ci[0]:.3f}, "
        f"{matrix.transfer_rate_ci[1]:.3f}] (bootstrap n={matrix.transfer_rate_bootstrap_n})"
    )
    lines.append("")
    lines.append("## Minimax (worst-case regret) ranking")
    lines.append("")
    lines.append("| rank | planner | worst-case robustness | transfer-failure rate |")
    lines.append("|---|---|---|---|")
    for row in matrix.ranking:
        wc = (
            f"{row.worst_case_robustness:.3f}"
            if row.worst_case_robustness == row.worst_case_robustness
            else "n/a"
        )
        lines.append(f"| {row.rank} | `{row.planner}` | {wc} | {row.transfer_failure_rate:.3f} |")
    lines.append("")
    lines.append(
        "## Transfer matrix (rows=configs, cols=planners; X=transferred failure, .=ok, ?=untested)"
    )
    lines.append("")
    header = (
        "| config | "
        + " | ".join(p.replace("scenario_adaptive_hybrid_orca", "orca") for p in matrix.planners)
        + " |"
    )
    sep = "|---|" + "|".join(["---"] * len(matrix.planners)) + "|"
    lines.append(header)
    lines.append(sep)
    by_config: dict[str, dict[str, TransferCell]] = {}
    for cell in matrix.cells:
        by_config.setdefault(cell.config_id, {})[cell.planner] = cell
    for cfg_id in matrix.config_ids:
        mark = []
        for planner in matrix.planners:
            cell = by_config.get(cfg_id, {}).get(planner)
            if cell is None or cell.robustness != cell.robustness:
                mark.append("?")
            elif cell.transferred:
                mark.append("X")
            else:
                mark.append(".")
        lines.append(f"| `{cfg_id}` | " + " | ".join(mark) + " |")
    lines.append("")
    lines.append("## Certified config provenance")
    lines.append("")
    lines.append("| config | scenario_seed | objective | tier | source manifest |")
    lines.append("|---|---|---|---|---|")
    for cfg in configs:
        lines.append(
            f"| `{cfg.config_id}` | {cfg.scenario_seed} | {cfg.objective_value:.3f} | "
            f"{cfg.certification_tier} | `{cfg.source_manifest}` |"
        )
    lines.append("")
    return "\n".join(lines)


def write_transfer_artifact(matrix: TransferMatrix, *, out_dir: str | Path) -> Path:
    """Write the transfer matrix JSON + one-page report to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "transfer_matrix.json"
    report_path = out_dir / "transfer_report.md"
    matrix_path.write_text(
        json.dumps(matrix.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(_render_matrix_only_report(matrix), encoding="utf-8")
    return matrix_path


def _render_matrix_only_report(matrix: TransferMatrix) -> str:
    """Render report without full config provenance (used by write_transfer_artifact)."""
    lines: list[str] = []
    lines.append("# Cross-planner adversarial transfer matrix (slice 1, capability-only)")
    lines.append("")
    lines.append("> Capability-not-evidence boundary: archive paths and pinned configs/seeds only.")
    lines.append("")
    lines.append(f"- Target planner: `{matrix.target_planner}`")
    lines.append(f"- Configs (K): {len(matrix.config_ids)}")
    lines.append(f"- Planners (N): {len(matrix.planners)}")
    lines.append(f"- Overall transfer rate: {matrix.overall_transfer_rate:.3f}")
    lines.append(f"- Transfer-rate 95% CI: {tuple(round(x, 3) for x in matrix.transfer_rate_ci)}")
    lines.append("")
    lines.append("## Minimax ranking")
    lines.append("")
    lines.append("| rank | planner | worst-case robustness | transfer-failure rate |")
    lines.append("|---|---|---|---|")
    for row in matrix.ranking:
        wc = (
            f"{row.worst_case_robustness:.3f}"
            if row.worst_case_robustness == row.worst_case_robustness
            else "n/a"
        )
        lines.append(f"| {row.rank} | `{row.planner}` | {wc} | {row.transfer_failure_rate:.3f} |")
    return "\n".join(lines) + "\n"


__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "DEFAULT_TRANSFER_ROSTER",
    "CertifiedConfig",
    "PlannerEval",
    "PlannerRanking",
    "TransferCell",
    "TransferMatrix",
    "build_transfer_matrix",
    "render_transfer_report",
    "select_certified_configs",
    "write_transfer_artifact",
]
