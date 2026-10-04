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

if TYPE_CHECKING:
    from collections.abc import Iterator

CONTEXT_ENV = "ROBOT_SF_SNQI_V2_CALIBRATION_CONTEXT"
LEARNED_ALGORITHMS = frozenset({"ppo", "guarded_ppo", "sacadrl", "drl", "sonic"})
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


def load_calibration_context(anchors_path: Path) -> dict[str, Any]:
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
    # The delivered portable receipt is in the same evidence directory. Do not follow a caller
    # absolute path or traversal from proof metadata.
    receipt_path = anchors_path.parent / "determinism-receipt.json"
    if Path(binding["path"]).name != receipt_path.name:
        raise ValueError("SNQI-v2 determinism receipt path mismatch")
    raw = receipt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
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
    if raw is None or algo not in LEARNED_ALGORITHMS:
        return None
    expected = json.loads(raw)
    observed = build_execution_context_provenance()
    assert_context_equal(observed, expected)
    return {k: v for k, v in observed.items() if k != "hostname"}


def verify_episode_contexts(campaign_root: Path, expected: dict[str, Any]) -> int:
    """Require each learned-policy row's worker context before accepting/scoring release output.

    Returns:
        The number of admitted learned-policy rows. Release acceptance checks the roster census.
    """
    count = 0
    for path in sorted(campaign_root.glob("runs/*/episodes.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["algo"] not in LEARNED_ALGORITHMS:
                continue
            context = row.get("algorithm_metadata", {}).get("execution_context")
            if not isinstance(context, dict):
                raise ValueError("SNQI-v2 learned episode execution context missing")
            assert_context_equal(context, expected)
            count += 1
    if not count:
        raise ValueError("SNQI-v2 learned episode execution context census empty")
    return count
