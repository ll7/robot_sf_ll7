"""Feasibility-smoke test for the CrowdNav_Prediction_AttnGraph learned baseline adapter.

Issue #4871 asked for a go/no-go smoke that the shipped checkpoint loads and
produces an action on synthetic Robot SF observations, plus the per-step
wall-clock measurement. The staged-checkout smoke is skipped unless the external
repo has been staged with
``uv run python scripts/tools/manage_external_repos.py stage crowdnav_pred_attng``,
mirroring the SICNav external-repo smoke convention.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from robot_sf.planner.crowdnav_pred_attng import (
    CrowdNavPredAttnGraphAdapter,
    CrowdNavPredAttnGraphConfig,
    build_crowdnav_pred_attng_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_PATH = REPO_ROOT / "third_party" / "external_repos" / "crowdnav_pred_attng"
# Training cadence of the shipped checkpoint (env.time_step). The recurrent
# policy was trained at 0.25s steps; the adapter enforces this contract.
TIME_STEP = 0.25
V_PREF = 1.0


def _synthetic_obs(agent_count: int) -> dict[str, object]:
    """Return a world-frame observation with ``agent_count`` nearby pedestrians.

    The robot starts at the origin heading toward a goal on +x with pedestrians
    placed beside/ahead so the open-space optimum is to move toward the goal.

    Returns:
        Benchmark-style observation payload accepted by the adapter.
    """
    agents = [
        {
            "position": [1.5 + 0.45 * (i % 6), 0.6 * ((i % 3) - 1)],
            "velocity": [0.1 * ((-1) ** i), 0.0],
        }
        for i in range(agent_count)
    ]
    return {
        "dt": TIME_STEP,
        "robot": {
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [5.0, 0.0],
            "radius": 0.3,
        },
        "agents": agents,
        "obstacles": [],
    }


def test_config_builder_defaults_point_at_staged_repo() -> None:
    """Config defaults should point at the license-staged external repo path."""
    cfg = build_crowdnav_pred_attng_config({})
    assert cfg.repo_root == Path("third_party/external_repos/crowdnav_pred_attng")
    assert cfg.checkpoint_name == "41200.pt"
    assert cfg.human_num == 20
    assert cfg.time_step == pytest.approx(TIME_STEP)
    assert cfg.predict_steps == 5


def test_config_builder_rejects_invalid_values() -> None:
    """Invalid human_num / time_step / v_pref should raise immediately."""
    with pytest.raises(ValueError, match="human_num"):
        build_crowdnav_pred_attng_config({"human_num": 0})
    with pytest.raises(ValueError, match="time_step"):
        build_crowdnav_pred_attng_config({"time_step": 0.0})
    with pytest.raises(ValueError, match="v_pref"):
        build_crowdnav_pred_attng_config({"v_pref": -1.0})


def test_crowdnav_pred_attng_skip_without_external_repo() -> None:
    """Smoke the staged CrowdNav_Prediction_AttnGraph checkpoint when present.

    Asserts the checkpoint loads as the attention-graph SRNN policy, produces a
    finite holonomic command clipped to the preferred-speed envelope, and runs
    well under any benchmark step budget. The per-step wall-clock table is
    printed as smoke evidence. A documented negative (repo not staged, or load /
    action failure) is itself a valid smoke outcome.
    """
    if not STAGE_PATH.exists():
        pytest.skip(f"CrowdNav_Prediction_AttnGraph repo is not staged at {STAGE_PATH}")

    adapter = CrowdNavPredAttnGraphAdapter(CrowdNavPredAttnGraphConfig(device="cpu"))

    # The shipped "Ours" checkpoint is the attention-graph SRNN policy.
    base = adapter._policy.base
    assert type(base).__name__ == "selfAttn_merge_SRNN"
    assert int(base.human_num) == 20
    assert int(base.nenv) == 1

    obs = _synthetic_obs(agent_count=5)
    vx, vy, meta = adapter.act(obs, time_step=TIME_STEP)

    # Hard smoke contract: a finite command clipped to the v_pref envelope.
    assert math.isfinite(vx) and math.isfinite(vy)
    assert math.hypot(vx, vy) <= V_PREF + 1e-6
    assert meta["detected_human_num"] == 5
    assert "raw_action_xy" in meta and "clipped_action_xy" in meta

    # Goal-direction evidence (soft): in open space the policy should make
    # progress toward the +x goal. Reported, not hard-asserted, to avoid flakiness.
    print(f"\n[smoke] action toward goal (+x): vx={vx:.4f} vy={vy:.4f}")

    # Per-step wall-clock across neighbor counts (issue #4871 decisive metric).
    # human_num is fixed at 20 so the cost is flat in observed neighbor count;
    # the bound is generous to stay machine-tolerant while catching regressions.
    print("[smoke] per-step wall-clock (30-step mean after 3 warmup steps):")
    per_step: dict[int, float] = {}
    for agent_count in (2, 5, 10, 20):
        adapter.reset()
        step_obs = _synthetic_obs(agent_count=agent_count)
        for _ in range(3):
            adapter.act(step_obs, time_step=TIME_STEP)
        start = time.perf_counter()
        for _ in range(30):
            adapter.act(step_obs, time_step=TIME_STEP)
        elapsed_ms = (time.perf_counter() - start) / 30.0 * 1000.0
        per_step[agent_count] = elapsed_ms
        print(f"[smoke]   neighbors={agent_count:2d}: {elapsed_ms:.3f} ms/step")
    assert max(per_step.values()) < 100.0  # trivially fast vs any step budget

    # Reset must clear recurrent state so a fresh episode is deterministic.
    adapter.reset()
    vx_a, vy_a, _ = adapter.act(obs, time_step=TIME_STEP)
    adapter.reset()
    vx_b, vy_b, _ = adapter.act(obs, time_step=TIME_STEP)
    assert vx_a == pytest.approx(vx_b, abs=1e-6)
    assert vy_a == pytest.approx(vy_b, abs=1e-6)


def test_crowdnav_pred_attng_skip_time_step_guard() -> None:
    """The adapter must reject a dt that breaks the 0.25s training cadence."""
    if not STAGE_PATH.exists():
        pytest.skip(f"CrowdNav_Prediction_AttnGraph repo is not staged at {STAGE_PATH}")
    adapter = CrowdNavPredAttnGraphAdapter(CrowdNavPredAttnGraphConfig(device="cpu"))
    with pytest.raises(ValueError, match="fixed time_step"):
        adapter.act(_synthetic_obs(agent_count=3), time_step=0.1)
