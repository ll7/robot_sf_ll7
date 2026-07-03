"""Map-runner registry tests for issue #4016 distributional RL adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from robot_sf.benchmark.map_runner import _build_policy
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
    model = QuantileQNetwork(observation_dim=2, action_count=2, num_quantiles=4, hidden_sizes=(4,))
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.net[-1].bias[:] = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, 3.0, 3.0, 3.0])
    path = tmp_path / "qr_dqn.pt"
    save_quantile_checkpoint(path, model=model, action_lattice=lattice)
    return path


def test_map_runner_builds_distributional_rl_policy(tmp_path: Path) -> None:
    """_build_policy wires distributional_rl through the registered builder."""

    policy, meta = _build_policy(
        "distributional_rl",
        {
            "checkpoint_path": str(_write_checkpoint(tmp_path)),
            "risk_objective": "cvar_lower",
            "risk_alpha": 0.25,
            "profile": "experimental",
        },
    )

    assert policy({"robot_position": [0.0, 0.0]}) == (0.0, 0.0)
    assert meta["algorithm"] == "distributional_rl"
    assert meta["policy_semantics"] == "qr_dqn_discrete_unicycle_lattice_risk_aware_selector"
    assert meta["planner_kinematics"]["native_env_action"] is False
    assert meta["diagnostics"]["last_decision"]["candidate_count"] == 2
