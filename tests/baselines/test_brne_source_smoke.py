"""BRNE source-side smoke test (issue #5311).

Exercises the REAL upstream BRNE core algorithm
(``MurpheyLab/brne`` at the pinned commit) when it is staged locally. The test
skips cleanly when the staged checkout or the numpy/numba/scipy stack is absent
(CI default), so it does not regress environments without the external clone.

This is source-side smoke only: no robot_sf planner is registered, no benchmark
claim is made, and no BRNE source is vendored (GPL-3.0, local-only reference).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BRNE_STAGE_PATH = REPO_ROOT / "third_party" / "external_repos" / "brne"
BRNE_CORE_REL = "brne_nav/brne_py/brne_py/brne.py"


def _brne_dependency_stack_available() -> bool:
    """Return True only when the BRNE core dependency stack is importable."""
    return all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "numba"))


def _load_upstream_brne(stage_path: Path):
    """Import the real upstream brne.py core module from the staged clone."""
    core_file = stage_path / BRNE_CORE_REL
    spec = importlib.util.spec_from_file_location("brne_upstream", core_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {core_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["brne_upstream"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_load_upstream_brne_fails_closed_when_import_spec_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unimportable staged module produces a clear, fail-closed error."""
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *_args: None)
    with pytest.raises(ImportError, match="Could not build import spec"):
        _load_upstream_brne(tmp_path)


@pytest.fixture(scope="module")
def upstream_brne():
    """Provide the staged upstream BRNE core module, or skip if unavailable."""
    if not BRNE_STAGE_PATH.exists():
        pytest.skip(f"BRNE external repo is not staged at {BRNE_STAGE_PATH}")
    if not (BRNE_STAGE_PATH / BRNE_CORE_REL).is_file():
        pytest.skip(f"BRNE core module missing at {BRNE_STAGE_PATH / BRNE_CORE_REL}")
    if not _brne_dependency_stack_available():
        pytest.skip("BRNE dependency stack (numpy/scipy/numba) is not installed")
    return _load_upstream_brne(BRNE_STAGE_PATH)


def test_brne_skip_without_external_repo() -> None:
    """Marker skip test used by the manage_external_repos validation command.

    This test always passes (or skips) so the registry validation command has a
    stable, fast check that the smoke file exists and is collected. The real
    upstream exercise happens in the staged-repo tests below.
    """
    if not BRNE_STAGE_PATH.exists():
        pytest.skip(f"BRNE external repo is not staged at {BRNE_STAGE_PATH}")


def test_brne_core_module_loads_and_exposes_contract(upstream_brne) -> None:
    """The staged upstream brne.py exposes the documented BRNE core contract."""
    for name in (
        "get_Lmat_nb",
        "mvn_sample_normal",
        "get_ulist_essemble",
        "traj_sim_essemble",
        "brne_nav",
        "dyn_step",
    ):
        assert hasattr(upstream_brne, name), f"upstream brne.py missing expected symbol: {name}"


def test_brne_nav_returns_finite_weights_for_two_agents(upstream_brne) -> None:
    """One full BRNE solve produces a valid mixed-strategy weight distribution.

    The robot is index 0; brne_nav must return finite, normalized weights over the
    robot trajectory samples (this is the trajectory-distribution output of the
    contract). A None return is the upstream 'corridor out-of-bounds' sentinel and
    is not expected for this in-corridor scene.
    """
    import numpy as np

    brne = upstream_brne
    num_samples = 49  # must be a perfect square (meshgrid); smaller than upstream 196 for speed
    plan_steps = 15
    dt = 0.1
    tlist = np.arange(plan_steps) * dt
    train_ts = np.array([tlist[0]])
    train_noise = np.array([1e-04])
    lmat, _ = brne.get_Lmat_nb(train_ts, tlist, train_noise, 0.2, 0.2)

    num_agents = 2  # robot + 1 pedestrian
    xtraj = np.zeros((num_agents * num_samples, plan_steps))
    ytraj = np.zeros((num_agents * num_samples, plan_steps))
    # Robot: straight rollout along +x.
    xtraj[:num_samples] = np.arange(plan_steps) * dt * 0.4
    ytraj[:num_samples] = brne.mvn_sample_normal(num_samples, plan_steps, lmat) * 0.05
    # Pedestrian: samples around a constant-velocity mean heading toward the robot.
    xp = brne.mvn_sample_normal(num_samples, plan_steps, lmat)
    yp = brne.mvn_sample_normal(num_samples, plan_steps, lmat)
    xmean = 3.0 + np.arange(plan_steps) * dt * (-0.4)
    ymean = np.zeros(plan_steps)
    xtraj[num_samples:] = xp * 0.1 + xmean
    ytraj[num_samples:] = yp * 0.1 + ymean

    weights = brne.brne_nav(
        xtraj,
        ytraj,
        num_agents,
        plan_steps,
        num_samples,
        4.0,
        1.0,
        80.0,
        0.1,
        -0.65,
        0.65,
    )
    assert weights is not None, "brne_nav returned the corridor out-of-bounds sentinel unexpectedly"
    assert weights.shape == (num_agents, num_samples)
    robot_weights = weights[0]
    assert np.all(np.isfinite(robot_weights)), "robot weights contain non-finite values"
    # Upstream normalizes so mean(weights[0]) == 1.0; the distribution is non-degenerate.
    assert robot_weights.mean() == pytest.approx(1.0, rel=1e-6)


def test_brne_unicycle_dynamics_match_differential_drive_command(upstream_brne) -> None:
    """BRNE integrates native unicycle [v*cos, v*sin, omega] dynamics.

    This directly answers the differential/Ackermann command-mapping question: a
    constant [v, omega] command produces a circular arc whose heading rate matches
    the analytic unicycle solution. Robot SF unicycle_vw is therefore directly
    compatible with no projection required.

    Note: upstream ``traj_sim`` (the single-trajectory helper) is buggy -- it calls
    ``dyn_step(st, u)`` without the required ``dt`` argument -- so this test uses
    ``traj_sim_essemble``, the ensemble rollout that the actual BRNE navigation
    loop uses and that correctly threads ``dt``. The ``traj_sim`` bug is a real
    upstream contract quirk worth recording (the nav loop never calls it).
    """
    import numpy as np

    brne = upstream_brne
    dt = 0.1
    v, omega = 0.4, 0.5
    n = 5
    st0 = np.zeros((3, 1))  # (3, 1) single sample
    ulist = np.full((n, 1, 2), [v, omega])  # (tsteps, num_samples=1, 2)
    # traj_sim_essemble returns (tsteps, 3, num_samples); squeeze the sample axis.
    traj = brne.traj_sim_essemble(st0, ulist, dt)[:, :, 0]
    assert traj.shape == (n, 3)
    assert np.all(np.isfinite(traj))
    # Heading after n steps of omega: theta = n * omega * dt (analytic unicycle).
    assert traj[-1, 2] == pytest.approx(omega * n * dt, abs=1e-6)
    # Arc length over the horizon ~ v * total_time for this short arc.
    dist = float(np.linalg.norm(traj[-1, :2]))
    assert dist == pytest.approx(v * n * dt, rel=0.05)


def test_brne_solve_completes_within_control_budget_for_small_crowd(upstream_brne) -> None:
    """The steady-state BRNE solve for a small crowd (<=5 agents) is under budget.

    This is a soft control-budget guard, not a hard CI assertion on a large crowd:
    it only checks the small-crowd case that BRNE handles comfortably, so it does
    not flake on slow CI. The full neighbor-count sweep (including the borderline
    8-agent case) is measured by scripts/tools/probe_brne_source_harness.py and
    recorded in the contract-mapping note, not asserted here.
    """
    import time

    import numpy as np

    brne = upstream_brne
    num_samples = 49
    plan_steps = 15
    dt = 0.1
    tlist = np.arange(plan_steps) * dt
    train_ts = np.array([tlist[0]])
    train_noise = np.array([1e-04])
    lmat, _ = brne.get_Lmat_nb(train_ts, tlist, train_noise, 0.2, 0.2)

    num_peds = 3  # 4 agents total
    num_agents = num_peds + 1
    xtraj = np.zeros((num_agents * num_samples, plan_steps))
    ytraj = np.zeros((num_agents * num_samples, plan_steps))
    xtraj[:num_samples] = np.arange(plan_steps) * dt * 0.4
    ytraj[:num_samples] = brne.mvn_sample_normal(num_samples, plan_steps, lmat) * 0.05
    for i in range(num_peds):
        xp = brne.mvn_sample_normal(num_samples, plan_steps, lmat)
        yp = brne.mvn_sample_normal(num_samples, plan_steps, lmat)
        xmean = (3.0 - 0.5 * i) + np.arange(plan_steps) * dt * (-0.4)
        ymean = np.full(plan_steps, (-1) ** i * (0.4 + 0.15 * i))
        xtraj[(i + 1) * num_samples : (i + 2) * num_samples] = xp * 0.1 + xmean
        ytraj[(i + 1) * num_samples : (i + 2) * num_samples] = yp * 0.1 + ymean

    # Warm up JIT compilation outside the timed region.
    brne.brne_nav(
        xtraj, ytraj, num_agents, plan_steps, num_samples, 4.0, 1.0, 80.0, 0.1, -0.65, 0.65
    )
    t0 = time.perf_counter()
    w = brne.brne_nav(
        xtraj, ytraj, num_agents, plan_steps, num_samples, 4.0, 1.0, 80.0, 0.1, -0.65, 0.65
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert w is not None
    # 4-agent steady solve at num_samples=49 is far under the 100 ms budget; use a
    # generous 2x headroom over the observed ~3-5 ms to avoid CI flakiness.
    assert elapsed_ms < 100.0, f"4-agent BRNE solve took {elapsed_ms:.1f} ms (expected < 100 ms)"
