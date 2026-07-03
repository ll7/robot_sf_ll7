"""Tests for host-specific test guard predicates."""

from __future__ import annotations

import pytest

from tests.support.environment_guards import (
    is_github_actions,
    is_licca_or_shared_hpc,
    should_enforce_wallclock_budget,
    skip_on_licca_shared_hpc,
)


def test_simulated_licca_env_is_shared_hpc_without_github_actions() -> None:
    """Explicit LiCCA marker should trigger shared-HPC guard behavior."""
    assert is_licca_or_shared_hpc({"ROBOT_SF_TEST_ENV": "licca"})
    assert is_licca_or_shared_hpc({"SLURM_JOB_ID": "12345"})
    assert is_licca_or_shared_hpc({"SLURM_CLUSTER_NAME": "licca"})


def test_github_actions_takes_precedence_over_licca_markers() -> None:
    """GitHub CI must not lose coverage when a LiCCA marker is also present."""
    env = {"GITHUB_ACTIONS": "true", "ROBOT_SF_TEST_ENV": "licca", "SLURM_JOB_ID": "12345"}
    assert is_github_actions(env)
    assert not is_licca_or_shared_hpc(env)
    assert should_enforce_wallclock_budget(env)


def test_unknown_local_env_does_not_skip_or_enforce_perf_budget() -> None:
    """Plain developer shells should keep functional tests running without perf enforcement."""
    assert not is_github_actions({})
    assert not is_licca_or_shared_hpc({})
    assert not should_enforce_wallclock_budget({})


def test_perf_enforce_override_enables_wallclock_budget_on_any_host() -> None:
    """Maintainers can explicitly opt into hard performance assertions off GitHub."""
    assert should_enforce_wallclock_budget({"ROBOT_SF_PERF_ENFORCE": "1"})


def test_skip_helper_uses_explicit_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip helper should require a reason and skip only on shared-HPC markers."""
    with pytest.raises(ValueError, match="skip reason"):
        skip_on_licca_shared_hpc("")

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("ROBOT_SF_TEST_ENV", "licca")
    with pytest.raises(pytest.skip.Exception, match="requires stable wall-clock budget"):
        skip_on_licca_shared_hpc("requires stable wall-clock budget")
