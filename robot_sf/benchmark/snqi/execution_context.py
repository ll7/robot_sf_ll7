"""Fail-closed numerical-context admission for source-bound SNQI-v2 releases."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf._execution_context import LEARNED_POLICY_CONTEXT_FIELDS
from robot_sf.benchmark._runtime_smoke_planner_keys import _RUNTIME_SMOKE_CHECKPOINT_PLANNER_KEYS
from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness
from robot_sf.benchmark.result_provenance import build_execution_context_provenance

if TYPE_CHECKING:
    from collections.abc import Iterator

CONTEXT_ENV = "ROBOT_SF_SNQI_V2_CALIBRATION_CONTEXT"
_LEARNED_FAMILIES = _RUNTIME_SMOKE_CHECKPOINT_PLANNER_KEYS | frozenset(
    {
        "crowdnav_height",
        "drl_vo",
        "sonic_crowdnav",
        "sac",
        "distributional_rl",
        "socnav_sampling",
        "dr_mpc",
        "learned_prediction_mpc",
        "hybrid_global_rl",
        "gensafenav_ours_gst",
        "gensafenav_ours_gst_guarded",
        "gensafenav_gst_predictor_rand",
        "gensafenav_gst_predictor_rand_guarded",
        "hybrid_portfolio",
        "gap_prediction",
    }
)


def _learned_algorithm_names() -> frozenset[str]:
    """Expand explicit learned/checkpoint families through the authoritative alias catalog.

    Returns:
        Canonical learned-family names, registered aliases and legacy compatibility names.
    """
    names = {"drl", "sonic"}  # Retain the pre-catalog compatibility names.
    for family in sorted(_LEARNED_FAMILIES):
        readiness = get_algorithm_readiness(family)
        if readiness is None:
            raise ValueError(f"SNQI-v2 learned family absent from readiness catalog: {family}")
        names.update((readiness.canonical_name, *readiness.aliases))
    return frozenset(names)


LEARNED_ALGORITHMS = _learned_algorithm_names()
REQUIRED_FIELDS = (
    "cpu_model",
    "platform",
    "python_version",
    "numpy_version",
    "numba_version",
    "thread_env",
)
OPTIONAL_FIELDS = ("kernel", "glibc", *LEARNED_POLICY_CONTEXT_FIELDS)


def assert_context_equal(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Compare numerical fields, including kernel/glibc in platform; never trust node identity."""
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        raise ValueError("SNQI-v2 execution context must be a mapping")
    for key in (*REQUIRED_FIELDS, *OPTIONAL_FIELDS):
        if key in OPTIONAL_FIELDS and key not in expected:
            continue
        value = expected.get(key)
        if value in (None, "", "unknown", "Unknown CPU"):
            raise ValueError(f"SNQI-v2 calibration execution context missing {key}")
        if key == "thread_env" and (
            not isinstance(value, dict) or not value or any(v in (None, "") for v in value.values())
        ):
            raise ValueError("SNQI-v2 calibration execution context missing thread_env")
        if observed.get(key) != value:
            raise ValueError(f"SNQI-v2 execution context mismatch: {key}")


def load_calibration_context(
    anchors_path: Path, asset_binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Load the original context from the paired, hash-bound acquisition/determinism proof.

    Hash binding is integrity, not scientific authority. Independent mint must still authenticate
    these actual source blobs; caller-generated files cannot fill that trust set.

    Returns:
        The original calibration numerical context.
    """
    proof = json.loads((anchors_path.parent / "acquisition-proof.json").read_bytes())
    if proof["anchors_sha256"] != hashlib.sha256(anchors_path.read_bytes()).hexdigest():
        raise ValueError("SNQI-v2 execution context anchor binding mismatch")
    binding = proof["determinism_receipt"]
    # The freeze need not contain its own acquisition output. The independently
    # bound spec pins the delivered bytes; rehashing custody alone cannot change it.
    receipt_path = asset_binding.get("determinism_receipt_path")
    pinned_digest = asset_binding.get("determinism_receipt_sha256")
    if receipt_path is None or not pinned_digest:
        raise ValueError("SNQI-v2 calibration context requires pinned determinism receipt")
    receipt_path = Path(receipt_path)
    if Path(binding["path"]).name != receipt_path.name:
        raise ValueError("SNQI-v2 determinism receipt path mismatch")
    raw = receipt_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != pinned_digest:
        raise ValueError("SNQI-v2 determinism receipt differs from pinned calibration reference")
    if digest != binding["sha256"]:
        raise ValueError("SNQI-v2 determinism receipt digest mismatch")
    receipt = json.loads(raw)
    if receipt["classification"] != "a" or receipt["original_vs_repeat"]["different_rows"] != 0:
        raise ValueError("SNQI-v2 fixed-environment calibration repeat not admitted")
    if (
        receipt["source_commit"] != proof["source_commit"]
        or receipt["scheduler_job_ids"]["original"] != proof["scheduler_job_id"]
        or not receipt["same_recorded_environment"]
        or receipt["original_vs_repeat"]["identical_rows"] != 1344
        or receipt["original_vs_repeat"]["rows"] != 1344
    ):
        raise ValueError("SNQI-v2 calibration repeat/source binding mismatch")
    expected = receipt["execution_contexts"]["original"]
    for key in LEARNED_POLICY_CONTEXT_FIELDS:
        if not isinstance(expected, Mapping) or key not in expected:
            raise ValueError(f"SNQI-v2 calibration execution context missing {key}")
    assert_context_equal(receipt["execution_contexts"]["repeat"], expected)
    assert_context_equal(expected, expected)
    return expected


@contextmanager
def episode_context_guard(expected: dict[str, Any] | None) -> Iterator[None]:
    """Carry the checked reference to local subprocess workers; restore prior state on exit."""
    previous = os.environ.get(CONTEXT_ENV)
    try:
        if expected is None:
            os.environ.pop(CONTEXT_ENV, None)
        else:
            os.environ[CONTEXT_ENV] = json.dumps(expected, sort_keys=True, allow_nan=False)
        yield
    finally:
        if previous is None:
            os.environ.pop(CONTEXT_ENV, None)
        else:
            os.environ[CONTEXT_ENV] = previous


def require_calibrated_algorithm(algo: str) -> None:
    """Refuse composite selectors whose children have no calibrated per-child contract."""
    if algo.strip().lower() in {"planner_selector_v2", "planner_selector_v2_diagnostic"}:
        raise ValueError(
            "SNQI-v2 calibrated execution refuses planner_selector_v2: "
            "selector children require independent context admission"
        )


def admit_episode_context(algo: str) -> dict[str, Any] | None:
    """Check the actual learned-policy worker before environment/planner construction or reset.

    Returns:
        The checked worker context, or None outside a gated learned-policy episode.
    """
    raw = os.environ.get(CONTEXT_ENV)
    if raw is None:
        return None
    require_calibrated_algorithm(algo)
    if algo.strip().lower() not in LEARNED_ALGORITHMS:
        return None
    expected = json.loads(raw)
    observed = build_execution_context_provenance()
    assert_context_equal(observed, expected)
    return {k: v for k, v in observed.items() if k != "hostname"}


def verify_episode_contexts(
    campaign_root: Path, expected: dict[str, Any], planner_keys: tuple[str, ...]
) -> int:
    """Check learned workers and require every learned arm named by the manifest.

    Returns:
        The number of admitted learned-policy rows; full acceptance checks cell completeness.
    """
    for key in planner_keys:
        require_calibrated_algorithm(key)
    required = {
        key.strip().lower() for key in planner_keys if key.strip().lower() in LEARNED_ALGORITHMS
    }
    seen: set[str] = set()
    count = 0
    for path in sorted(campaign_root.glob("runs/*/episodes.jsonl")):
        arm = path.parent.name.split("__", 1)[0].strip().lower()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            require_calibrated_algorithm(row["algo"])
            if row["algo"].strip().lower() not in LEARNED_ALGORITHMS and arm not in required:
                continue
            context = row.get("algorithm_metadata", {}).get("execution_context")
            if not isinstance(context, dict):
                raise ValueError("SNQI-v2 learned episode execution context missing")
            assert_context_equal(context, expected)
            seen.add(arm)
            count += 1
    missing = required - seen
    if missing:
        raise ValueError(
            "SNQI-v2 learned episode execution context census missing arms: "
            + ", ".join(sorted(missing))
        )
    return count
