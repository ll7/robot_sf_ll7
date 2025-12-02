"""Demo: per-pedestrian vs aggregated force quantiles.

Run with:
    uv run python examples/benchmarks/per_ped_force_quantiles_demo.py

The script constructs a toy episode with three pedestrians and compares the
aggregated force quantiles (`force_q*`) against the per-pedestrian averaged
quantiles (`ped_force_q*`). It highlights scenarios where high forces are
concentrated on one pedestrian.
"""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics


def _make_episode(T: int, K: int) -> EpisodeData:
    """Make episode.

    Args:
        T: Auto-generated placeholder description.
        K: Auto-generated placeholder description.

    Returns:
        EpisodeData: Auto-generated placeholder description.
    """
    robot_pos = np.zeros((T, 2))
    robot_vel = np.zeros((T, 2))
    robot_acc = np.zeros((T, 2))
    peds_pos = np.zeros((T, K, 2))
    ped_forces = np.zeros((T, K, 2))
    goal = np.array([5.0, 0.0])
    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        goal=goal,
        dt=0.1,
        reached_goal_step=None,
    )


def main() -> None:
    """Main.

    Returns:
        None: Auto-generated placeholder description.
    """
    T, K = 5, 3
    ep = _make_episode(T=T, K=K)

    # Ped 0: high forces; ped 1/2: low forces
    ep.ped_forces[:, 0] = np.array([10.0, 0.0])  # consistently high
    ep.ped_forces[:, 1] = np.array([1.0, 0.0])
    ep.ped_forces[:, 2] = np.array([1.0, 0.0])

    metrics = compute_all_metrics(ep, horizon=10)
    agg = {k: v for k, v in metrics.items() if k.startswith("force_q")}
    per_ped = {k: v for k, v in metrics.items() if k.startswith("ped_force_q")}

    print("Aggregated force quantiles (flattened across all peds/timesteps):")
    for k, v in sorted(agg.items()):
        print(f"  {k}: {v:.2f}")

    print("\nPer-pedestrian mean quantiles (average of per-ped quantiles):")
    for k, v in sorted(per_ped.items()):
        print(f"  {k}: {v:.2f}")

    print(
        "\nObservation: per-pedestrian metrics surface that one pedestrian experiences "
        "higher forces, while aggregated quantiles are dominated by the majority of low-force samples."
    )


if __name__ == "__main__":
    main()
