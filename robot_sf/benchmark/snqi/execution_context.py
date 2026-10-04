"""Fail-closed numerical-context admission for source-bound SNQI-v2 releases."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.result_provenance import build_execution_context_provenance
from robot_sf.benchmark.runtime_smoke_admission import _RUNTIME_SMOKE_CHECKPOINT_PLANNER_KEYS

if TYPE_CHECKING:
    from collections.abc import Iterator

CONTEXT_ENV = "ROBOT_SF_SNQI_V2_CALIBRATION_CONTEXT"
LEARNED_ALGORITHMS = _RUNTIME_SMOKE_CHECKPOINT_PLANNER_KEYS | frozenset(
    {"sa_cadrl", "drl", "sonic", "socnav_sampling"}
)
REQUIRED_FIELDS = (
    "cpu_model",
    "platform",
    "python_version",
    "numpy_version",
    "numba_version",
    "thread_env",
)
OPTIONAL_FIELDS = ("kernel", "glibc", "torch_version", "stable_baselines3_version")


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


def admit_episode_context(algo: str) -> dict[str, Any] | None:
    """Check the actual learned-policy worker before environment/planner construction or reset.

    Returns:
        The checked worker context, or None outside a gated learned-policy episode.
    """
    raw = os.environ.get(CONTEXT_ENV)
    if raw is None or algo.strip().lower() not in LEARNED_ALGORITHMS:
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
    required = {
        key.strip().lower() for key in planner_keys if key.strip().lower() in LEARNED_ALGORITHMS
    }
    seen: set[str] = set()
    count = 0
    for path in sorted(campaign_root.glob("runs/*/episodes.jsonl")):
        arm = path.parent.name.split("__", 1)[0].strip().lower()
        for line in path.read_text().splitlines():
            row = json.loads(line)
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
