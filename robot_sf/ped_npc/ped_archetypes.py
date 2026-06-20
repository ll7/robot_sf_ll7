"""Heterogeneous pedestrian behavior archetypes (speed-based MVP).

Issue #3230 (child of #3206). The Social Force backend already derives each
pedestrian's desired/maximum speed from its *initial* velocity magnitude per
agent (``fast-pysf`` ``DesiredForce`` uses ``self.peds.max_speeds``), so speed
heterogeneity needs no physics change: assigning a per-pedestrian initial-speed
multiplier is sufficient.

This module is pure (no simulation state): it loads an archetype registry and
deterministically assigns a per-pedestrian ``desired_speed_factor`` given a
population composition. The factor multiplies the spawn config's base
``initial_speed`` in :func:`robot_sf.ped_npc.ped_population.populate_ped_routes`.

The archetypes are clearly-labeled *modeling choices*, not dataset-calibrated
values; this is mechanism evidence, not benchmark or realism evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Composition fractions must sum to one within this absolute tolerance.
_COMPOSITION_SUM_TOL = 1e-6


def load_archetypes(path: str | Path) -> dict[str, float]:
    """Load an archetype registry (``name -> desired_speed_factor``) from YAML.

    The YAML shape is::

        archetypes:
          cautious:
            desired_speed_factor: 0.7
          ...

    Returns:
        Mapping of archetype name to its ``desired_speed_factor`` multiplier.
    """
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "archetypes" not in payload:
        raise ValueError(f"archetype registry must be a mapping with 'archetypes': {path}")
    archetypes = payload["archetypes"]
    if not isinstance(archetypes, dict) or not archetypes:
        raise ValueError(f"'archetypes' must be a non-empty mapping: {path}")
    factors: dict[str, float] = {}
    for name, spec in archetypes.items():
        if not isinstance(spec, dict) or "desired_speed_factor" not in spec:
            raise ValueError(f"archetype '{name}' must define 'desired_speed_factor'")
        factor = float(spec["desired_speed_factor"])
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(
                f"archetype '{name}' desired_speed_factor must be finite and > 0 (got {factor})"
            )
        factors[str(name)] = factor
    return factors


def validate_composition(
    composition: dict[str, float],
    speed_factors: dict[str, float],
) -> None:
    """Validate a population composition against an archetype registry.

    Raises ``ValueError`` when fractions are non-positive, do not sum to one, or
    reference an archetype absent from ``speed_factors``.
    """
    if not composition:
        raise ValueError("archetype_composition must be a non-empty mapping")
    missing = [name for name in composition if name not in speed_factors]
    if missing:
        raise ValueError(f"composition references unknown archetypes: {sorted(missing)}")
    fractions = {name: float(frac) for name, frac in composition.items()}
    for name, frac in fractions.items():
        if not math.isfinite(frac) or frac <= 0:
            raise ValueError(
                f"composition fraction for '{name}' must be finite and > 0 (got {frac})"
            )
    total = float(sum(fractions.values()))
    if abs(total - 1.0) > _COMPOSITION_SUM_TOL:
        raise ValueError(f"composition fractions must sum to 1.0 (got {total})")


def allocate_archetype_counts(n: int, composition: dict[str, float]) -> dict[str, int]:
    """Allocate ``n`` pedestrians across archetypes using largest-remainder rounding.

    Deterministic and exact: the returned counts always sum to ``n``. Ties in the
    fractional remainder are broken by archetype name for stable output.

    Returns:
        Mapping of archetype name to integer pedestrian count (summing to ``n``).
    """
    if n <= 0:
        return dict.fromkeys(composition, 0)
    if not composition:
        raise ValueError("archetype_composition must be non-empty when allocating pedestrians")
    raw = {name: frac * n for name, frac in composition.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = n - sum(counts.values())
    # Distribute the remaining seats to the largest fractional parts (name-stable ties).
    order = sorted(composition, key=lambda name: (-(raw[name] - counts[name]), name))
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


def assign_archetype_speed_factors(
    n: int,
    composition: dict[str, float],
    speed_factors: dict[str, float],
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Return a length-``n`` array of per-pedestrian desired-speed factors.

    Counts per archetype follow :func:`allocate_archetype_counts`; the resulting
    factors are shuffled with a seeded RNG so archetypes are not correlated with
    spawn order (and thus not spatially clustered). Deterministic for a fixed
    ``seed``.

    Returns:
        Float array of shape ``(n,)`` with each pedestrian's desired-speed factor.
    """
    labels = assign_archetype_labels(n, composition, seed=seed)
    return np.asarray([speed_factors[label] for label in labels], dtype=float)


def assign_archetype_labels(
    n: int,
    composition: dict[str, float],
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Return a length-``n`` array of sampled archetype labels.

    The label order matches :func:`assign_archetype_speed_factors` for the same
    ``n``, composition, and seed, so downstream reports can join per-pedestrian
    labels to speed factors without re-sampling.

    Returns:
        String array of shape ``(n,)`` with each pedestrian's archetype label.
    """
    if n <= 0:
        return np.empty(0, dtype=str)
    counts = allocate_archetype_counts(n, composition)
    labels = np.concatenate([np.full(count, name, dtype=object) for name, count in counts.items()])
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    return labels.astype(str)


def build_archetype_population_report(
    *,
    n: int,
    composition: dict[str, float],
    speed_factors: dict[str, float],
    initial_speed: float,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic, no-result report for one population composition.

    Returns:
        JSON-serializable summary of intended and realized composition counts,
        desired-speed factors, and initial-speed assumptions.
    """
    validate_composition(composition, speed_factors)
    labels = assign_archetype_labels(n, composition, seed=seed)
    factors = np.asarray([speed_factors[label] for label in labels], dtype=float)
    counts = {name: int(np.count_nonzero(labels == name)) for name in composition}
    realized_total = max(1, int(n))
    return {
        "schema_version": "pedestrian-archetype-population-report.v1",
        "status": "composition_report_only",
        "claim_boundary": (
            "No benchmark, realism, or planner-ranking claim. This report only "
            "records deterministic population composition and speed-factor "
            "assumptions for later smoke/campaign runs."
        ),
        "population_size": int(n),
        "seed": seed,
        "initial_speed_m_s": float(initial_speed),
        "archetypes": {
            name: {
                "intended_fraction": float(composition[name]),
                "realized_count": counts[name],
                "realized_fraction": float(counts[name] / realized_total) if n > 0 else 0.0,
                "desired_speed_factor": float(speed_factors[name]),
                "initial_speed_m_s": float(initial_speed * speed_factors[name]),
            }
            for name in sorted(composition)
        },
        "assignment_order_sha1": _assignment_digest(labels.tolist()),
        "speed_factor_mean": float(np.mean(factors)) if factors.size else 0.0,
        "speed_factor_min": float(np.min(factors)) if factors.size else 0.0,
        "speed_factor_max": float(np.max(factors)) if factors.size else 0.0,
    }


def _assignment_digest(labels: list[str]) -> str:
    """Return a stable digest for the sampled label order."""
    encoded = json.dumps(labels, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]
