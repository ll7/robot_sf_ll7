"""Environment predicates for tests with host-specific assumptions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping


def is_github_actions(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the test is running on GitHub Actions."""
    env = os.environ if env is None else env
    return env.get("GITHUB_ACTIONS", "").lower() == "true"


def is_licca_or_shared_hpc(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the test is running on LiCCA or a shared Slurm/HPC node."""
    env = os.environ if env is None else env
    if is_github_actions(env):
        return False

    explicit_env = env.get("ROBOT_SF_TEST_ENV", "").lower()
    if explicit_env in {"licca", "shared-hpc", "shared_hpc", "hpc"}:
        return True

    cluster_name = env.get("SLURM_CLUSTER_NAME", "").lower()
    if "licca" in cluster_name:
        return True

    if env.get("SLURM_JOB_ID"):
        return True

    cwd_markers = (
        env.get("PWD", ""),
        env.get("TMPDIR", ""),
        env.get("SCRATCH", ""),
    )
    return any("/hpc/" in marker or "/gpfs" in marker for marker in cwd_markers)


def should_enforce_wallclock_budget(env: Mapping[str, str] | None = None) -> bool:
    """Return whether hard per-test wall-clock budgets should be asserted."""
    env = os.environ if env is None else env
    if is_github_actions(env):
        return True
    return env.get("ROBOT_SF_PERF_ENFORCE", "") == "1"


def skip_on_licca_shared_hpc(reason: str) -> None:
    """Skip when LiCCA/shared-HPC host assumptions invalidate a test contract."""
    if not reason:
        raise ValueError("skip reason must name the invalid environment assumption")
    if is_licca_or_shared_hpc():
        pytest.skip(reason)
