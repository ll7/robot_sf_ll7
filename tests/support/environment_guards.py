"""Environment predicates for tests with host-specific assumptions."""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

# Deterministic identity used by test-only temporary Git repositories so
# fixtures never depend on ambient developer or CI-runner Git configuration.
GIT_TEST_IDENTITY_NAME = "Robot SF test"
GIT_TEST_IDENTITY_EMAIL = "robot-sf-test@example.invalid"


def git_identity_environment(
    env: Mapping[str, str] | None = None,
    *,
    name: str = GIT_TEST_IDENTITY_NAME,
    email: str = GIT_TEST_IDENTITY_EMAIL,
) -> dict[str, str]:
    """Return a hermetic environment for Git subprocess calls in tests.

    The returned environment carries a deterministic author/committer identity
    and disables global and system Git configuration so temporary repositories
    never depend on (or mutate) ambient developer configuration.

    Args:
        env: Base environment to extend (defaults to ``os.environ``).
        name: Author/committer name for the temporary repositories.
        email: Author/committer email for the temporary repositories.

    Returns:
        A fresh dict with the identity and config-isolation variables set.
    """
    base = dict(os.environ if env is None else env)
    base.update(
        {
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
    )
    return base


def configure_git_identity(
    repo: Path,
    *,
    name: str = GIT_TEST_IDENTITY_NAME,
    email: str = GIT_TEST_IDENTITY_EMAIL,
) -> None:
    """Configure deterministic local Git identity in a temporary repository.

    Runs ``git config`` for the repository-local identity so subsequent
    ``git commit`` calls inside ``repo`` do not depend on ambient identity.

    Args:
        repo: Temporary repository root that already has ``git init`` applied.
        name: Author/committer name for the temporary repository.
        email: Author/committer email for the temporary repository.
    """
    subprocess.run(
        ["git", "config", "user.name", name],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", email],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


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

    # PWD may be unset in non-interactive shells / direct subprocess launches;
    # fall back to the live cwd when reading the real process environment.
    pwd = env.get("PWD", "")
    if not pwd and env is os.environ:
        pwd = os.getcwd()
    cwd_markers = (
        pwd,
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
