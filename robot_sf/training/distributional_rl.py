"""QR-DQN-style primitives for issue #4016 distributional RL slices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path

    from robot_sf.training.discrete_action_lattice import DiscreteUnicycleActionLattice


def fixed_quantile_fractions(
    num_quantiles: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """Return QR-DQN midpoint quantile fractions.

    Returns:
        One-dimensional tensor of fixed quantile midpoint fractions.
    """

    if num_quantiles < 2:
        raise ValueError("num_quantiles must be at least 2")
    return (torch.arange(num_quantiles, device=device, dtype=torch.float32) + 0.5) / num_quantiles


def quantile_huber_loss(
    predicted_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    *,
    taus: torch.Tensor | None = None,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Return the QR-DQN quantile Huber regression loss.

    ``predicted_quantiles`` and ``target_quantiles`` must share batch dimensions
    and end with the quantile axis. Pairwise target-prediction deltas are used as
    in quantile regression DQN.

    Returns:
        Scalar loss tensor.
    """

    if predicted_quantiles.shape[:-1] != target_quantiles.shape[:-1]:
        raise ValueError("predicted and target quantiles must share non-quantile dimensions")
    if predicted_quantiles.shape[-1] < 2 or target_quantiles.shape[-1] < 2:
        raise ValueError("predicted and target tensors must contain at least two quantiles")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if not torch.is_floating_point(predicted_quantiles) or not torch.is_floating_point(
        target_quantiles
    ):
        raise TypeError("quantile tensors must be floating point")

    num_predicted = predicted_quantiles.shape[-1]
    if taus is None:
        taus = fixed_quantile_fractions(num_predicted, device=predicted_quantiles.device)
    taus = taus.to(device=predicted_quantiles.device, dtype=predicted_quantiles.dtype)
    if taus.shape != (num_predicted,):
        raise ValueError("taus must have shape [num_predicted_quantiles]")

    deltas = target_quantiles.unsqueeze(-2) - predicted_quantiles.unsqueeze(-1)
    abs_deltas = deltas.abs()
    huber = torch.where(
        abs_deltas <= kappa,
        0.5 * deltas.pow(2),
        kappa * (abs_deltas - 0.5 * kappa),
    )
    leading_dims = (1,) * (deltas.ndim - 2)
    indicator = (deltas < 0).to(predicted_quantiles.dtype)
    quantile_weights = (taus.view(*leading_dims, num_predicted, 1) - indicator).abs()
    return (quantile_weights * huber / kappa).sum(dim=-2).mean()


class QuantileQNetwork(nn.Module):
    """Small MLP producing ordered quantile estimates per discrete action."""

    def __init__(
        self,
        observation_dim: int,
        action_count: int,
        num_quantiles: int,
        *,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        """Initialize the quantile value head."""

        super().__init__()
        if observation_dim < 1:
            raise ValueError("observation_dim must be positive")
        if action_count < 1:
            raise ValueError("action_count must be positive")
        if num_quantiles < 2:
            raise ValueError("num_quantiles must be at least 2")

        self.observation_dim = int(observation_dim)
        self.action_count = int(action_count)
        self.num_quantiles = int(num_quantiles)

        layers: list[nn.Module] = []
        input_dim = self.observation_dim
        for hidden_size in hidden_sizes:
            if hidden_size < 1:
                raise ValueError("hidden_sizes must contain positive layer widths")
            layers.extend([nn.Linear(input_dim, hidden_size), nn.ReLU()])
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, self.action_count * self.num_quantiles))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return quantiles with shape ``[batch, action_count, num_quantiles]``.

        Returns:
            Batched per-action return quantile estimates.
        """

        if observations.ndim != 2 or observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"observations must have shape [batch, {self.observation_dim}], "
                f"got {tuple(observations.shape)}"
            )
        output = self.net(observations)
        return output.view(-1, self.action_count, self.num_quantiles)

    def metadata(self) -> dict[str, Any]:
        """Return checkpoint metadata needed by the next implementation slice.

        Returns:
            Non-evidence QR-DQN metadata contract.
        """

        return {
            "algorithm": "qr_dqn",
            "observation_dim": self.observation_dim,
            "action_count": self.action_count,
            "num_quantiles": self.num_quantiles,
            "claim_boundary": "primitive-only; not benchmark evidence",
        }


@dataclass(frozen=True)
class QRDQNTargetBatch:
    """Bellman target quantiles selected by double-Q action indices."""

    target_quantiles: torch.Tensor
    next_action_indices: torch.Tensor


def select_action_quantiles(quantiles: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
    """Gather per-batch action distributions from ``[B, A, N]`` quantiles.

    Returns:
        Tensor of selected action quantiles with shape ``[batch, num_quantiles]``.
    """

    if quantiles.ndim != 3:
        raise ValueError("quantiles must have shape [batch, action_count, num_quantiles]")
    if action_indices.ndim != 1 or action_indices.shape[0] != quantiles.shape[0]:
        raise ValueError("action_indices must have shape [batch]")
    gather_index = action_indices.to(device=quantiles.device, dtype=torch.long).view(-1, 1, 1)
    gather_index = gather_index.expand(-1, 1, quantiles.shape[-1])
    return quantiles.gather(dim=1, index=gather_index).squeeze(dim=1)


def build_qr_dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    target_next_quantiles: torch.Tensor,
    next_action_indices: torch.Tensor,
    *,
    gamma: float,
) -> QRDQNTargetBatch:
    """Construct distributional Bellman targets for selected next actions.

    Returns:
        Target quantile batch and action indices used to select next-action distributions.
    """

    if rewards.ndim != 1 or dones.ndim != 1 or rewards.shape != dones.shape:
        raise ValueError("rewards and dones must have matching shape [batch]")
    if target_next_quantiles.ndim != 3 or target_next_quantiles.shape[0] != rewards.shape[0]:
        raise ValueError(
            "target_next_quantiles must have shape [batch, action_count, num_quantiles]"
        )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in the interval [0, 1]")

    selected_next = select_action_quantiles(target_next_quantiles, next_action_indices)
    not_done = 1.0 - dones.to(dtype=selected_next.dtype, device=selected_next.device)
    target = rewards.to(dtype=selected_next.dtype, device=selected_next.device).unsqueeze(-1)
    target = target + gamma * not_done.unsqueeze(-1) * selected_next
    return QRDQNTargetBatch(target_quantiles=target, next_action_indices=next_action_indices)


def hard_update_target_network(source: nn.Module, target: nn.Module) -> None:
    """Copy online network weights into the target network."""

    target.load_state_dict(source.state_dict())


def save_quantile_checkpoint(
    path: Path,
    *,
    model: QuantileQNetwork,
    action_lattice: DiscreteUnicycleActionLattice,
) -> None:
    """Save a primitive QR-DQN checkpoint with explicit non-evidence metadata."""

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_metadata": model.metadata(),
            "action_lattice": action_lattice.to_dict(),
        },
        path,
    )


def load_quantile_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Load metadata from a primitive checkpoint without instantiating adapters.

    Returns:
        Checkpoint metadata and action-lattice contract.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "model_metadata": checkpoint["model_metadata"],
        "action_lattice": checkpoint["action_lattice"],
    }
