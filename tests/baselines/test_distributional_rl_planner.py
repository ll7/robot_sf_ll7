"""Tests for the issue #4016 distributional RL runtime adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from robot_sf.baselines.distributional_rl import DistributionalRLPlanner
from robot_sf.training.discrete_action_lattice import DiscreteUnicycleActionLattice
from robot_sf.training.distributional_rl import QuantileQNetwork, save_quantile_checkpoint

if TYPE_CHECKING:
    from pathlib import Path


def _write_checkpoint(tmp_path: Path) -> Path:
    lattice = DiscreteUnicycleActionLattice(
        linear_values=(0.0, 0.5),
        angular_values=(0.0,),
        max_linear_speed=0.5,
        max_angular_speed=1.0,
    )
    model = QuantileQNetwork(
        observation_dim=2,
        action_count=lattice.action_count,
        num_quantiles=4,
        hidden_sizes=(4,),
    )
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        final = model.net[-1]
        final.bias[:] = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, 3.0, 3.0, 3.0])
    path = tmp_path / "qr_dqn.pt"
    save_quantile_checkpoint(path, model=model, action_lattice=lattice)
    return path


def test_distributional_rl_planner_selects_lower_cvar_action(tmp_path: Path) -> None:
    """CVaR mode prefers the action with the safer lower tail."""

    planner = DistributionalRLPlanner(
        {
            "checkpoint_path": str(_write_checkpoint(tmp_path)),
            "risk_objective": "cvar_lower",
            "risk_alpha": 0.25,
        }
    )

    action = planner.step({"robot_position": [0.0, 0.0]})

    assert action == {"v": 0.0, "omega": 0.0}
    assert planner.diagnostics()["last_decision"]["selected_action_index"] == 0
    assert planner.get_metadata()["evidence_tier"] == "diagnostic-only"


def test_distributional_rl_planner_selects_mean_return_action(tmp_path: Path) -> None:
    """Mean mode can choose a different action from lower-CVaR mode."""

    planner = DistributionalRLPlanner(
        {
            "checkpoint_path": str(_write_checkpoint(tmp_path)),
            "risk_objective": "mean",
            "risk_alpha": 0.25,
        }
    )

    action = planner.step({"robot_position": [0.0, 0.0]})

    assert action == {"v": 0.5, "omega": 0.0}
    assert planner.diagnostics()["last_decision"]["selected_action_index"] == 1


def test_distributional_rl_planner_missing_checkpoint_fails_closed(tmp_path: Path) -> None:
    """Fallback is disabled by default for benchmark-facing runs."""

    with pytest.raises(FileNotFoundError):
        DistributionalRLPlanner({"checkpoint_path": str(tmp_path / "missing.pt")})
