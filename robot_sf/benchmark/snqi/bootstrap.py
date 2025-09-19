"""Bootstrap stability scaffold for SNQI (placeholder).

This module provides a placeholder interface for future bootstrap-based
stability estimation. The intent is to replace / augment the existing
heuristic ranking stability metric with a statistically grounded
measure derived from repeated resampling of the episodic dataset.

Planned algorithm (design doc outlines details):
    1. Sample B bootstrap subsets of episodes (with optional stratified
       sampling by algorithm / policy label when available).
    2. For each bootstrap sample, compute per-algorithm mean SNQI and
       derive a ranking (or full score vector).
    3. Compute average pairwise Spearman correlation across bootstrap
       rankings (or equivalently average correlation to a consensus
       ranking) to yield a stability score in [0,1].
    4. (Future) Provide confidence intervals for the stability metric
       plus optional per-weight sensitivity under resampling.

Current state:
    - The public function ``bootstrap_stability`` is a non-operational
      stub returning a structured placeholder result. Integration is
      intentionally deferred until performance budget and CI strategy
      for the additional runtime are finalized.

Usage expectations once implemented:
    stability_info = bootstrap_stability(episodes, weights, rng=np.random.default_rng(seed))
    stability_score = stability_info["stability"]

Schema / contract (planned):
    {
        "stability": float | None,         # Final bootstrap stability score
        "samples": int,                    # B
        "method": "bootstrap_spearman",   # Identifier for chosen method
        "status": "placeholder" | "ok",  # Placeholder vs computed
        "details": { ... }                 # Optional distributional stats
    }

Until implemented, callers SHOULD NOT rely on this function for
production. The presence of this module only signals impending
integration and serves as a stable import path for early experimentation.

"""

from __future__ import annotations

from typing import Any, Dict, Iterable

__all__ = ["bootstrap_stability"]


def bootstrap_stability(
    _episodes: Iterable[dict],
    _weights: Dict[str, float],
    *,
    _rng,  # numpy.random.Generator expected
    samples: int = 30,
    group_key: str | None = "algo",
) -> Dict[str, Any]:
    """Return a placeholder bootstrap stability result.

    Parameters
    ----------
    _episodes:
        (Unused placeholder) Iterable of episode dictionaries.
    _weights:
        (Unused placeholder) Mapping of weight names to positive floats.
    _rng:
        (Unused placeholder) Numpy RNG used for deterministic resampling once implemented.
    samples:
        Planned number of bootstrap resamples (B). Ignored in placeholder.
    group_key:
        Optional key in episode dict for stratified sampling. Ignored in placeholder.

    Returns
    -------
    dict
        Structured placeholder result with ``stability`` set to ``None`` and status ``"placeholder"``.

    Notes
    -----
    - This stub intentionally does not inspect inputs to avoid implying operational status.
    - Downstream callers MUST feature-detect by checking ``result["status"]`` before use.
    - Once implemented, backward compatibility will be maintained (fields only added, not removed).
    """
    return {
        "stability": None,
        "samples": samples,
        "method": "bootstrap_spearman",
        "status": "placeholder",
        "details": {
            "message": "Bootstrap stability not yet implemented; this is a scaffold.",
            "group_key": group_key,
        },
    }
