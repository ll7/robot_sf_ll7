
"""SocNavBench personal-space objective."""

import numpy as np
from dotmap import DotMap
from metrics.cost_functions import asym_gauss_from_vel
from objectives.objective_function import Objective
from simulators.sim_state import AgentState, SimState
from trajectory.trajectory import Trajectory


class PersonalSpaceCost(Objective):
    """
    Compute the cost of being in non ego gen_agents' path.
    """

    def __init__(self, params: DotMap):
        """Initialize the personal-space objective."""
        self.p: DotMap = params
        self.tag: str = "personal_space_cost_per_nonego_agent"

    def _agent_velocity_from_config(
        self, agent_vals: AgentState, theta: float
    ) -> tuple[float, float]:
        """Return the velocity vector used to orient and scale personal space."""
        heading_velocity = (float(np.cos(theta)), float(np.sin(theta)))
        if not self.p.get("use_agent_velocity", False):
            return heading_velocity

        current_config = agent_vals.get_current_config()
        if current_config is None:
            return heading_velocity

        speed_nk1 = current_config.speed_nk1()
        if speed_nk1 is None:
            return heading_velocity

        speed_values = np.asarray(speed_nk1, dtype=float).reshape(-1)
        if speed_values.size == 0:
            return heading_velocity

        speed = float(speed_values[-1])
        min_speed = float(self.p.get("min_agent_speed", 1e-3))
        if not np.isfinite(speed) or abs(speed) <= min_speed:
            return heading_velocity

        velocity_scale = float(self.p.get("agent_velocity_scale", 1.0))
        return (
            float(velocity_scale * speed * np.cos(theta)),
            float(velocity_scale * speed * np.sin(theta)),
        )

    def evaluate_objective(
        self, trajectory: Trajectory, sim_state_hist: dict[int, SimState]
    ) -> np.ndarray:
        """Evaluate personal-space cost along a candidate ego trajectory."""
        # get ego agent trajectory
        ego_traj = trajectory.position_and_heading_nk3()

        # get the last sim_state if it exists
        if len(sim_state_hist) == 0:
            return np.zeros((1, ego_traj.shape[1]))
        sim_state: SimState = sim_state_hist[max(sim_state_hist.keys())]
        assert isinstance(sim_state, SimState)
        # loop through each trajectory point
        _, k, _ = ego_traj.shape
        personal_space_cost = np.zeros((1, k))
        for i in range(k):
            ego_pos3 = ego_traj[0, i]  # (x,y,th)_self latest timestep

            # iterate through every non ego agent
            agents: dict[str, AgentState] = sim_state.get_all_agents()

            for agent_vals in agents.values():
                agent_pos3 = agent_vals.get_pos3()  # (x,y,th)
                theta = agent_pos3[2]
                velx, vely = self._agent_velocity_from_config(agent_vals, theta)
                # gaussian centered around the non ego agent
                personal_space_cost[0, i] += asym_gauss_from_vel(
                    x=ego_pos3[0],
                    y=ego_pos3[1],
                    velx=velx,
                    vely=vely,
                    xc=agent_pos3[0],
                    yc=agent_pos3[1],
                )

        return self.p.psc_scale * personal_space_cost
