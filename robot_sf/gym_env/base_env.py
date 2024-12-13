#!/usr/bin/env python3

"""
Base environment for the robot simulation environment.
"""

from loguru import logger
from gymnasium import Env
from typing import List

from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.render.sim_view import VisualizableSimState, SimulationView
from robot_sf.sim.simulator import init_simulators


class BaseEnv(Env):
    """
    Represents the base class of robot_sf environment.
    It should be possible to simulate this environment without any action.
    So this could be a standalone pedestrian simulation.
    """

    def __init__(
        self,
        env_config: EnvSettings = EnvSettings(),
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: str = None,
        video_fps: float = None,
        peds_have_obstacle_forces: bool = False,
    ):
        """
        Initializes the base environment.
        """
        super().__init__()
        self.env_config = env_config

        self.debug = debug

        self.recording_enabled = recording_enabled
        if self.recording_enabled:
            self.recorded_states: List[VisualizableSimState] = []
            self.record_video = record_video
            if self.record_video:
                self.video_path = video_path
                self.video_fps = video_fps
                self._set_video_fps()

        self.map_def = env_config.map_pool.choose_random_map()

        # Initialize simulator with a random start position
        self.simulator = init_simulators(
            env_config,
            self.map_def,
            random_start_pos=True,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
        )[0]

        if debug or record_video:
            self.sim_ui = SimulationView(
                scaling=10,
                map_def=self.map_def,
                obstacles=self.map_def.obstacles,
                robot_radius=env_config.robot_config.radius,
                ped_radius=env_config.sim_config.ped_radius,
                goal_radius=env_config.sim_config.goal_radius,
                record_video=record_video,
                video_path=video_path,
                video_fps=video_fps,
            )

    def _set_video_fps(self):
        """
        This methods sets a default value for the video_fps attribute if it was not provided.
        """

        if video_fps is None:
            video_fps = 1 / self.env_config.sim_config.time_per_step_in_secs
            logger.info(f"Video FPS not provided, setting to {video_fps}")

    def step(self, action=None):
        """
        Advances the simulation by one step.
        Does not take any action.
        Returns only dummy values.
        """
        # todo: the simulator in this configuration requires a robot action
        # Instead the robot should be redesigned to not require a pedestrian.
        self.simulator.step_once()

        obs = None
        reward = None
        terminal = None
        truncated = None
        info = None

        return (obs, reward, terminal, truncated, info)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        """
        super().reset(seed=seed, options=options)

        # Reset internal simulator state
        self.simulator.reset_state()

        if self.recording_enabled:
            self.save_recording()
        return None

    def _prepare_visualizable_state(self):
        # Prepare action visualization, if any action was executed
        action = (
            None
            if not self.last_action
            else VisualizableAction(
                self.simulator.robot_poses[0],
                self.last_action,
                self.simulator.goal_pos[0],
            )
        )

        # Robot position and LIDAR scanning visualization preparation
        robot_pos = self.simulator.robot_poses[0][0]
        distances, directions = lidar_ray_scan(
            self.simulator.robot_poses[0],
            self.state.occupancy,
            self.env_config.lidar_config,
        )

        # Construct ray vectors for visualization
        ray_vecs = zip(np.cos(directions) * distances, np.sin(directions) * distances)
        ray_vecs_np = np.array(
            [
                [[robot_pos[0], robot_pos[1]], [robot_pos[0] + x, robot_pos[1] + y]]
                for x, y in ray_vecs
            ]
        )

        # Prepare pedestrian action visualization
        ped_actions = zip(
            self.simulator.pysf_sim.peds.pos(),
            self.simulator.pysf_sim.peds.pos() + self.simulator.pysf_sim.peds.vel() * 2,
        )
        ped_actions_np = np.array([[pos, vel] for pos, vel in ped_actions])

        # Package the state for visualization
        state = VisualizableSimState(
            self.state.timestep,
            action,
            self.simulator.robot_poses[0],
            deepcopy(self.simulator.ped_pos),
            ray_vecs_np,
            ped_actions_np,
        )

    def render(self):
        """
        Render the environment visually if in debug mode.

        Raises RuntimeError if debug mode is not enabled.
        """
        if not self.sim_ui:
            raise RuntimeError(
                "Debug mode is not activated! Consider setting " "debug=True!"
            )

        state = self._prepare_visualizable_state()

        # Execute rendering of the state through the simulation UI
        self.sim_ui.render(state)

    def record(self):
        """
        Records the current state as visualizable state and stores it in the list.
        """
        state = self._prepare_visualizable_state()
        self.recorded_states.append(state)

    def save_recording(self, filename: str = None):
        """
        save the recorded states to a file
        filname: str, must end with *.pkl
        resets the recorded states list at the end
        """
        if filename is None:
            now = datetime.datetime.now()
            # get current working directory
            cwd = os.getcwd()
            filename = f'{cwd}/recordings/{now.strftime("%Y-%m-%d_%H-%M-%S")}.pkl'

        # only save if there are recorded states
        if len(self.recorded_states) == 0:
            logger.warning("No states recorded, skipping save")
            # TODO: First env.reset will always have no recorded states
            return

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "wb") as f:  # write binary
            pickle.dump((self.recorded_states, self.map_def), f)
            logger.info(f"Recording saved to {filename}")
            logger.info("Reset state list")
            self.recorded_states = []

    def exit(self):
        """
        Clean up and exit the simulation UI, if it exists.
        """
        if self.sim_ui:
            self.sim_ui.exit_simulation()
