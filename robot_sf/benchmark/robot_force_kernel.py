"""The SocialForce pair kernel and its versioned comparison reference."""

from __future__ import annotations

import numpy as np

ROBOT_FORCE_REFERENCE_RULE = "social_force_head_on_contact_relative_speed_1m_s_v1"


def pedestrian_pair_force(
    delta: np.ndarray, relative_velocity: np.ndarray, cfg: dict
) -> np.ndarray:
    """Evaluate the repository's SocialForce pair kernel including its lateral term.

    Returns:
        Model acceleration vectors for the supplied pairwise geometry.
    """
    distance = np.linalg.norm(delta, axis=-1)
    direction = np.divide(
        delta, distance[..., None], out=np.zeros_like(delta), where=distance[..., None] > 0
    )
    interaction = cfg["lambda_importance"] * relative_velocity + direction
    length = np.linalg.norm(interaction, axis=-1)
    unit = np.divide(
        interaction, length[..., None], out=np.zeros_like(delta), where=length[..., None] > 0
    )
    theta = np.arctan2(unit[..., 1], unit[..., 0]) - np.arctan2(
        direction[..., 1], direction[..., 0]
    )
    scale = cfg["gamma"] * length + 1e-8
    along = np.exp(-distance / scale - (cfg["n_prime"] * scale * theta) ** 2)
    lateral = -np.where(theta >= 0, 1, -1) * np.exp(
        -distance / scale - (cfg["n"] * scale * theta) ** 2
    )
    normal = np.stack((-unit[..., 1], unit[..., 0]), axis=-1)
    result = cfg["factor"] * (unit * along[..., None] + normal * lateral[..., None])
    return np.where((distance <= cfg["activation_threshold"])[..., None], result, 0.0)


def robot_force_reference(cfg: dict, ped_radius_m: float) -> float:
    """Resolve full SocialForce magnitude at contact and 1 m/s head-on closing speed.

    Returns:
        Reference acceleration in m/s² for the declared pedestrian radius/configuration.
    """
    return float(
        np.linalg.norm(
            pedestrian_pair_force(np.array([2 * ped_radius_m, 0.0]), np.array([1.0, 0.0]), cfg)
        )
    )
