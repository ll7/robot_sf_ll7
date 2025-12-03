"""Module sim_view auto-generated docstring."""

import os
import sys
from dataclasses import dataclass, field
from math import cos, sin

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import numpy as np
import pygame
from loguru import logger

from robot_sf.common.types import DifferentialDriveAction, PedPose, RgbColor, RobotPose, Vec2D
from robot_sf.nav.map_config import MapDefinition, Obstacle
from robot_sf.ped_ego.unicycle_drive import UnicycleAction
from robot_sf.robot.bicycle_drive import BicycleAction
from robot_sf.sensor.range_sensor import euclid_dist

# Make moviepy optional
try:
    from moviepy import ImageSequenceClip

    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning(
        "MoviePy is not available. Video recording is disabled. Have you installed ffmpeg?",
    )

## Note: PYGAME_HIDE_SUPPORT_PROMPT is set before importing pygame above


BACKGROUND_COLOR = (255, 255, 255)
BACKGROUND_COLOR_TRANSP = (255, 255, 255, 128)
OBSTACLE_COLOR = (20, 30, 20, 128)
PED_SPAWN_COLOR = (255, 204, 203)
PED_GOAL_COLOR = (144, 238, 144)
PED_COLOR = (255, 50, 50)
EGO_PED_COLOR = (108, 70, 117)
EGO_PED_ACTION_COLOR = (108, 70, 117)
PED_ROUTE_COLOR = (0, 0, 255)
ROBOT_ROUTE_COLOR = (30, 30, 255)
ROBOT_COLOR = (0, 0, 200)
COLLISION_COLOR = (200, 0, 0)
ROBOT_ACTION_COLOR = (65, 105, 225)
PED_ACTION_COLOR = (255, 50, 50)
ROBOT_GOAL_COLOR = (0, 204, 102)
ROBOT_LIDAR_COLOR = (238, 160, 238, 128)
TEXT_COLOR = (255, 255, 255)  # White text
TEXT_BACKGROUND = (0, 0, 0, 180)  # Semi-transparent black background
TEXT_OUTLINE_COLOR = (0, 0, 0)  # Black outline


def _empty_map_definition() -> MapDefinition:
    """Return a minimal valid MapDefinition used as a safe default.

    This avoids needing callers to always supply a map and keeps SimulationView
    construction lightweight for utility or test contexts. The map is a 1x1 square
    with a single spawn/goal triangle; routes are empty.
    """
    rect = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    bounds = [
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 0.0),
    ]
    return MapDefinition(
        width=1.0,
        height=1.0,
        obstacles=[],
        robot_spawn_zones=[rect],
        ped_spawn_zones=[rect],
        robot_goal_zones=[rect],
        bounds=bounds,  # type: ignore[arg-type]  # bounds format compatible at runtime
        robot_routes=[],
        ped_goal_zones=[rect],
        ped_crowded_zones=[],
        ped_routes=[],
    )


@dataclass
class VisualizableAction:
    """VisualizableAction class."""

    pose: RobotPose
    action: DifferentialDriveAction | BicycleAction | UnicycleAction
    goal: Vec2D


@dataclass
class VisualizableSimState:
    """
    VisualizableSimState represents a collection of properties to display
    the simulator's state at a discrete timestep.
    """

    timestep: int
    """The discrete timestep of the simulation."""

    robot_action: VisualizableAction | None
    """The action taken by the robot at this timestep."""

    robot_pose: RobotPose
    """The pose of the robot at this timestep."""

    pedestrian_positions: np.ndarray
    """The positions of pedestrians at this timestep."""

    ray_vecs: np.ndarray
    """The ray vectors associated with the robot's sensors."""

    ped_actions: np.ndarray
    """The actions taken by pedestrians at this timestep."""

    ego_ped_pose: PedPose | None = None
    """The pose of the ego pedestrian at this timestep. Defaults to None."""

    ego_ped_ray_vecs: np.ndarray | None = None
    """The ray vectors associated with the ego pedestrian's sensors. Defaults to None."""

    ego_ped_action: VisualizableAction | None = None
    """The action taken by the ego pedestrian at this timestep. Defaults to None."""

    time_per_step_in_secs: float | None = None
    """The time taken for each step in seconds. Defaults to None. Usually 0.1 seconds."""

    def __post_init__(self):
        """validate the visualizable state"""
        if self.time_per_step_in_secs is None:
            logger.warning("time_per_step_in_secs is None, defaulting to 0.1s.")
            self.time_per_step_in_secs = 0.1


@dataclass
class SimulationView:
    """
    SimulationView class for rendering the simulation using PyGame.

    Attributes:
        width (float): Width of the simulation window.
        height (float): Height of the simulation window.
        scaling (float): Scaling factor for rendering.
        robot_radius (float): Radius of the robot.
        ego_ped_radius (float): Radius of the ego pedestrian.
        ped_radius (float): Radius of the pedestrian.
        goal_radius (float): Radius of the goal.
        map_def (MapDefinition): Definition of the map.
        obstacles (List[Obstacle]): List of obstacles in the simulation.
        caption (str): Caption of the simulation window.
        focus_on_robot (bool): Whether to focus the camera on the robot.
        focus_on_ego_ped (bool): Whether to focus the camera on the ego pedestrian.
        record_video (bool): Whether to record the simulation as a video.
        video_path (str): Path to save the recorded video.
        video_fps (float): Frames per second for the recorded video.
        frames (List[np.ndarray]): List of frames recorded for the video.
        clock (pygame.time.Clock): PyGame clock for controlling frame rate.
        screen (pygame.surface.Surface): PyGame surface for rendering.
        font (pygame.font.Font): PyGame font for rendering text.
        size_changed (bool): Whether the window size has changed.
        redraw_needed (bool): Whether a redraw is needed.
        is_exit_requested (bool): Whether an exit is requested.
        is_abortion_requested (bool): Whether an abortion is requested.
        offset (np.ndarray): Offset for the camera.
        display_robot_info (int): Level of robot information to display.
        display_help (bool): Whether to display help text.
        current_target_fps (float): Current target frames per second for rendering.
        display_text: bool = False: Whether to display text on the screen.

    Methods:
        __post_init__(): Initialize PyGame components.
        render(state: VisualizableSimState, target_fps: float = 100):
            Render one frame and handle events.
        exit_simulation(return_frames: bool = False): Exit the simulation.
        clear(): Clears the screen and updates the display.
    """

    width: float = 1280
    height: float = 720
    scaling: float = 10
    robot_radius: float = 1.0
    ego_ped_radius: float = 0.4
    ped_radius: float = 0.4
    goal_radius: float = 1.0
    # Provide a minimal valid default map definition (see _empty_map_definition)
    map_def: MapDefinition = field(default_factory=_empty_map_definition)
    obstacles: list[Obstacle] = field(default_factory=list)
    caption: str = "RobotSF Simulation"
    focus_on_robot: bool = True
    focus_on_ego_ped: bool = False
    record_video: bool = False
    video_path: str | None = None
    video_fps: float = 10.0
    frames: list[np.ndarray] = field(default_factory=list)
    clock: pygame.time.Clock = field(init=False)

    # Add UI state fields
    screen: pygame.surface.Surface = field(init=False)
    font: pygame.font.Font = field(init=False)
    size_changed: bool = field(init=False, default=False)
    redraw_needed: bool = field(init=False, default=False)
    is_exit_requested: bool = field(init=False, default=False)
    is_abortion_requested: bool = field(init=False, default=False)
    offset: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    display_robot_info: int = field(default=0)  # Add this line
    display_help: bool = field(default=False)  # Also add this for help text
    current_target_fps: float = field(default=60.0)  # Add field for current_target_fps
    display_text: bool = field(default=False)  # Add this field to control text visibility
    ped_velocity_scale: float = field(default=1.0)  # Velocity visualization scaling factor
    # Internal flag: True when a display window is created via pygame.display.set_mode
    _use_display: bool = field(init=False, default=False)
    # Maximum number of frames to retain in memory when recording. None means no hard cap.
    # Default chosen to balance typical 720p usage (<~4.4GB for 1200 frames) vs runaway memory.
    max_frames: int | None = field(default=2000)
    _frame_cap_warned: bool = field(init=False, default=False)

    def __post_init__(self):
        """Initialize PyGame components."""
        logger.debug("Initializing the simulation view.")
        # Environment variable override for max_frames (e.g., runtime tuning / CI scenarios)
        env_cap = os.environ.get("ROBOT_SF_MAX_VIDEO_FRAMES")
        if env_cap is not None:
            raw = env_cap.strip().lower()
            if raw in {"none", "", "-1"}:
                self.max_frames = None
                logger.debug(
                    "ROBOT_SF_MAX_VIDEO_FRAMES set to '%s' -> disabling frame cap (max_frames=None)",
                    env_cap,
                )
            else:
                try:
                    parsed = int(raw)
                    if parsed <= 0:
                        raise ValueError
                    self.max_frames = parsed
                    logger.debug(
                        "ROBOT_SF_MAX_VIDEO_FRAMES override applied: max_frames=%d",
                        parsed,
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid ROBOT_SF_MAX_VIDEO_FRAMES value '%s' (expected positive int or 'none'). Using default %s.",
                        env_cap,
                        self.max_frames,
                    )
        pygame.init()
        pygame.font.init()
        self.clock = pygame.time.Clock()

        if self.record_video and not self.video_path:
            logger.warning(
                "record_video=True but no video_path provided; frames will be buffered but no file will be written.",
            )

        # Check if we're running in a headless environment
        is_headless = self._is_headless_environment()

        if self.record_video or is_headless:
            # Create offscreen surface for recording or headless mode
            self._use_display = False
            self.screen = pygame.Surface((int(self.width), int(self.height)))
            if self.record_video:
                logger.debug("Created offscreen surface for video recording")
            else:
                logger.debug("Created offscreen surface for headless mode")
        else:
            # Create window for display
            self._use_display = True
            self.screen = pygame.display.set_mode(
                (int(self.width), int(self.height)),
                pygame.RESIZABLE,
            )
            pygame.display.set_caption(self.caption)
        self.font = pygame.font.Font(None, 36)

    def _is_headless_environment(self) -> bool:
        """Return True if the runtime should be treated as headless.

        Rules:
        1) Always consider headless when SDL_VIDEODRIVER == "dummy" (cross-platform).
        2) On Linux, consider headless only when both DISPLAY and WAYLAND_DISPLAY
           are missing or empty (covers X11 and Wayland).
        3) Do not use MPLBACKEND as a signal for pygame headless decisions.
        """
        sdl_driver = os.environ.get("SDL_VIDEODRIVER", "")
        display = os.environ.get("DISPLAY", "")
        wayland = os.environ.get("WAYLAND_DISPLAY", "")

        # Universal dummy video driver implies headless
        if sdl_driver == "dummy":
            logger.debug(
                "Headless environment detected: "
                f"DISPLAY='{display}', WAYLAND_DISPLAY='{wayland}', SDL_VIDEODRIVER='{sdl_driver}'",
            )
            return True

        # Platform-specific handling
        if sys.platform.startswith("linux"):
            is_headless = display == "" and wayland == ""
            if is_headless:
                logger.debug(
                    "Headless environment detected: "
                    f"DISPLAY='{display}', WAYLAND_DISPLAY='{wayland}', SDL_VIDEODRIVER='{sdl_driver}'",
                )
            return is_headless

        # On non-Linux platforms, rely on SDL driver only (do not treat missing DISPLAY as headless)
        return False

    def render(self, state: VisualizableSimState, target_fps: float = 60):
        """
        Render one frame and handle events.

        Args:
            state (VisualizableSimState): The current state of the simulation to be visualized.
            target_fps (float, optional): Target frames per second for displaying the simulation.
                Defaults to 60 fps.
        """
        # Handle events on main thread
        self._process_events()

        if self.is_exit_requested:
            self._handle_exit()
            return

        # Render the frame
        self._prepare_frame(state)

        # Capture or display the frame
        self._finalize_frame(target_fps)

    def _process_events(self):
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._handle_quit()
            elif event.type == pygame.VIDEORESIZE:
                self._handle_video_resize(event)
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

    def _handle_exit(self):
        """Handle the exit state."""
        pygame.quit()
        if self.is_abortion_requested:
            sys.exit()

    def _prepare_frame(self, state: VisualizableSimState):
        """Prepare a new frame with the given state."""
        # Adjust the view based on the focus
        self._move_camera(state)
        self.screen.fill(BACKGROUND_COLOR)

        # Draw scene components in order
        self._draw_static_elements()
        self._draw_dynamic_elements(state)
        self._draw_information(state)

    def _draw_static_elements(self):
        """Draw static elements like obstacles and grid."""
        if self.map_def.obstacles:
            self._draw_obstacles()
        self._draw_grid()

    def _draw_dynamic_elements(self, state: VisualizableSimState):
        """Draw dynamic elements based on the simulation state."""
        self._draw_sensor_data(state)
        self._draw_actions(state)
        self._draw_entities(state)

    def _draw_sensor_data(self, state: VisualizableSimState):
        """Draw sensor data like lidar rays."""
        if hasattr(state, "ray_vecs") and state.ray_vecs is not None:
            self._augment_lidar(state.ray_vecs)
        if (
            hasattr(state, "ego_ped_pose")
            and state.ego_ped_pose
            and hasattr(state, "ego_ped_ray_vecs")
        ):
            # ego_ped_ray_vecs is Optional; skip if None to avoid TypeError
            if state.ego_ped_ray_vecs is not None:
                self._augment_lidar(state.ego_ped_ray_vecs)

    def _draw_actions(self, state: VisualizableSimState):
        """Draw action indicators for all entities."""
        if hasattr(state, "ped_actions"):
            self._augment_ped_actions(state.ped_actions)

        if hasattr(state, "robot_action") and state.robot_action:
            self._augment_action(state.robot_action, ROBOT_ACTION_COLOR)
            if hasattr(state.robot_action, "goal"):
                self._augment_goal_position(state.robot_action.goal)

        if (
            hasattr(state, "ego_ped_pose")
            and hasattr(state, "ego_ped_action")
            and state.ego_ped_action
        ):
            self._augment_action(state.ego_ped_action, EGO_PED_ACTION_COLOR)

    def _draw_entities(self, state: VisualizableSimState):
        """Draw all entities (robot, pedestrians, etc.)."""
        if hasattr(state, "pedestrian_positions"):
            self._draw_pedestrians(state.pedestrian_positions)

        if hasattr(state, "robot_pose"):
            self._draw_robot(state.robot_pose)

        if hasattr(state, "ego_ped_pose") and state.ego_ped_pose:
            self._draw_ego_ped(state.ego_ped_pose)

    def _draw_information(self, state: VisualizableSimState):
        """Draw UI information elements."""
        if self.display_text:
            # Full text display
            self._add_text(state.timestep, state)
            if self.display_help:
                self._add_help_text()
        else:
            # Minimal hint when text is disabled
            self._add_minimal_hint()

    def _finalize_frame(self, target_fps: float):
        """Capture or display the completed frame."""
        if self.record_video:
            self._capture_frame()
        else:
            # Store the current target FPS for display
            self.current_target_fps = target_fps
            if self._use_display:
                pygame.display.update()
            # Control frame rate with pygame's clock
            self.clock.tick(target_fps)

    def _capture_frame(self):
        """Capture the current frame for video recording."""
        # Enforce frame cap (if configured) before capturing new frame
        if self.max_frames is not None and len(self.frames) >= self.max_frames:
            if not self._frame_cap_warned:
                est_bytes = int(self.width * self.height * 3 * len(self.frames))
                est_gb = est_bytes / (1024**3)
                logger.warning(
                    "Max video frame buffer reached (max_frames=%d, ~%.2f GiB est). "
                    "Halting further frame capture to prevent excessive memory use. "
                    "You can raise this via SimulationView(max_frames=...) or disable via max_frames=None.",
                    self.max_frames,
                    est_gb,
                )
                self._frame_cap_warned = True
            return

        # Use a view for speed, then transpose to (H, W, C) and copy to detach from the Surface.
        # pixels3d() avoids an immediate copy like array3d() does, improving per-frame performance.
        surf_view = pygame.surfarray.pixels3d(self.screen)
        frame_data = np.transpose(surf_view, (1, 0, 2)).copy()
        # Ensure the surface is unlocked before returning (explicitly drop the view)
        del surf_view
        self.frames.append(frame_data)

    @property
    def _timestep_text_pos(self) -> Vec2D:
        """Timestep text pos.

        Returns:
            Vec2D: 2D vector (x, y).
        """
        return (16, 16)

    def _scale_tuple(self, tup: tuple[float, float]) -> tuple[float, float]:
        """scales a tuple of floats by the scaling factor and adds the offset."""
        x = tup[0] * self.scaling + self.offset[0]
        y = tup[1] * self.scaling + self.offset[1]
        return (x, y)

    def exit_simulation(self, return_frames: bool = False):
        """Exit the simulation."""
        logger.debug("Exiting the simulation.")
        self.is_exit_requested = True
        # Diagnostic guard: warn if recording requested but no frames captured
        if self.record_video:
            if not self.frames:
                logger.warning(
                    "record_video=True but zero frames were captured; video file will not be written. "
                    "Likely causes: (1) render() was never called; (2) early exit before any frame finalized. "
                    "Call render() each step (or enable debug mode) to populate frames.",
                )
            else:
                # Heuristic: sample up to first 5 frames; if all sums are zero, content may be blank
                try:  # pragma: no cover - defensive
                    sample = self.frames[:5]
                    if sample and all(np.array(f).sum() == 0 for f in sample):
                        logger.warning(
                            "record_video=True but captured frames appear empty (all-zero pixel data). "
                            "Ensure drawing code executed before frame capture; verify entities are rendered.",
                        )
                except (TypeError, ValueError) as exc:
                    # Defensive: conversion/summation failed for unexpected frame data
                    logger.debug("Frame sample check failed: %s", exc)
        if return_frames:
            intermediate_frames = self.frames
        self._handle_quit()
        if return_frames:
            logger.debug("Returning intermediate frames.")
            return intermediate_frames

    def _handle_quit(self):
        """Handle the quit event of the pygame window."""
        self.is_exit_requested = True
        self.is_abortion_requested = True
        if self.record_video and self.frames and MOVIEPY_AVAILABLE and self.video_path:
            logger.debug("Writing video file.")
            # TODO: get the correct fps from the simulation
            clip = ImageSequenceClip(self.frames, fps=self.video_fps)
            clip.write_videofile(self.video_path)
            self.frames = []
        elif self.record_video and self.frames and MOVIEPY_AVAILABLE and not self.video_path:
            logger.warning(
                "record_video=True but video_path is None; cannot write video file. Skipping write.",
            )
        elif self.record_video and self.frames and not MOVIEPY_AVAILABLE:
            logger.warning("MoviePy is not available. Cannot write video file.")

    def _handle_video_resize(self, e):
        """Handle the resize event of the pygame window."""
        self.size_changed = True
        self.width, self.height = e.w, e.h

    def _handle_keydown(self, e):
        """Handle key presses for the simulation view."""
        new_offset = 100
        new_scaling = 1
        if pygame.key.get_mods() & pygame.KMOD_CTRL:
            new_offset = 250
            new_scaling = 10

        if pygame.key.get_mods() & pygame.KMOD_ALT:
            new_offset = 10

        key_action_map = {
            # scale the view
            pygame.K_PLUS: lambda: setattr(self, "scaling", self.scaling + new_scaling),
            pygame.K_MINUS: lambda: setattr(self, "scaling", max(self.scaling - new_scaling, 1)),
            # move the view
            pygame.K_LEFT: lambda: self.offset.__setitem__(0, self.offset[0] + new_offset),
            pygame.K_RIGHT: lambda: self.offset.__setitem__(0, self.offset[0] - new_offset),
            pygame.K_UP: lambda: self.offset.__setitem__(1, self.offset[1] + new_offset),
            pygame.K_DOWN: lambda: self.offset.__setitem__(1, self.offset[1] - new_offset),
            # reset the view
            pygame.K_r: lambda: self.offset.__setitem__(slice(None), (0, 0)),
            # focus on the robot or ped
            pygame.K_f: lambda: setattr(self, "focus_on_robot", not self.focus_on_robot),
            pygame.K_p: lambda: setattr(self, "focus_on_ego_ped", not self.focus_on_ego_ped),
            # display help
            pygame.K_h: lambda: setattr(self, "display_help", not self.display_help),
            # display robotinfo
            pygame.K_q: lambda: setattr(
                self,
                "display_robot_info",
                (self.display_robot_info + 1) % 3,
            ),
            # toggle text display (add this line)
            pygame.K_t: lambda: setattr(self, "display_text", not self.display_text),
        }

        if e.key in key_action_map:
            key_action_map[e.key]()
            if e.key in (pygame.K_PLUS, pygame.K_MINUS):
                # for the scaling, we need to redraw the obstacles
                # this is not necessary for a fixed offset
                self.redraw_needed = True

    def _process_event_queue(self):
        """Process the event queue with better timing control."""
        event_handler_map = {
            pygame.QUIT: self._handle_quit,
            pygame.VIDEORESIZE: self._handle_video_resize,
            pygame.KEYDOWN: self._handle_keydown,
        }

        # Use pygame's clock for consistent timing
        while not self.is_exit_requested:
            for e in pygame.event.get():
                handler = event_handler_map.get(e.type)
                if handler:
                    handler(e)  # type: ignore[call-arg]  # handler signature varies

            # Limit this loop to 30 event checks per second (sufficient for UI interaction)
            self.clock.tick(30)

    def clear(self):
        """
        Clears the screen and updates the display.

        This method fills the screen with the background color,
        adds text at position 0, and updates the display.
        """
        self.screen.fill(BACKGROUND_COLOR)
        if self._use_display:
            pygame.display.update()

    def _resize_window(self):
        """Resize window.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        logger.debug("Resizing the window.")
        old_surface = self.screen
        if self._use_display:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        else:
            # In offscreen mode, just recreate the Surface at the new size
            self.screen = pygame.Surface((int(self.width), int(self.height)))
        self.screen.blit(old_surface, (0, 0))

    def _move_camera(self, state: VisualizableSimState):
        """Moves the camera based on the focused object."""
        if self.focus_on_robot:
            r_x, r_y = state.robot_pose[0]
            self.offset[0] = int(r_x * self.scaling - self.width / 2) * -1
            self.offset[1] = int(r_y * self.scaling - self.height / 2) * -1
        if self.focus_on_ego_ped and state.ego_ped_pose:
            r_x, r_y = state.ego_ped_pose[0]
            self.offset[0] = int(r_x * self.scaling - self.width / 2) * -1
            self.offset[1] = int(r_y * self.scaling - self.height / 2) * -1

    def _draw_robot(self, pose: RobotPose):
        """Draw robot.

        Args:
            pose: Pose tuple (position + heading).

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        # TODO(#252): display robot with an image instead of a circle
        # See: https://github.com/ll7/robot_sf_ll7/issues/252
        pygame.draw.circle(
            self.screen,
            ROBOT_COLOR,
            self._scale_tuple(pose[0]),
            self.robot_radius * self.scaling,
        )

    def _draw_ego_ped(self, pose: PedPose):
        """Draw ego ped.

        Args:
            pose: Pose tuple (position + heading).

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        # TODO(#252): display ego ped with an image instead of a circle
        # See: https://github.com/ll7/robot_sf_ll7/issues/252
        pygame.draw.circle(
            self.screen,
            EGO_PED_COLOR,
            self._scale_tuple(pose[0]),
            self.ego_ped_radius * self.scaling,
        )

    def _draw_pedestrians(self, ped_pos: np.ndarray):
        """Draw pedestrians.

        Args:
            ped_pos: position for pedestrian.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        # TODO(#252): display pedestrians with an image instead of a circle
        # See: https://github.com/ll7/robot_sf_ll7/issues/252
        for ped_x, ped_y in ped_pos:
            pygame.draw.circle(
                self.screen,
                PED_COLOR,
                self._scale_tuple((ped_x, ped_y)),
                self.ped_radius * self.scaling,
            )

    def _draw_obstacles(self):
        """Draw obstacles.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        # Iterate over each obstacle in the list of obstacles
        for obstacle in self.map_def.obstacles:
            # Scale and offset the vertices of the obstacle
            scaled_vertices = [(self._scale_tuple((x, y))) for x, y in obstacle.vertices_np]
            # Draw the obstacle as a polygon on the screen
            pygame.draw.polygon(self.screen, OBSTACLE_COLOR, scaled_vertices)

    def _draw_spawn_zones(self):
        """Draw spawn zones.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        # Iterate over each spawn_zone in the list of spawn_zones
        for spawn_zone in self.map_def.ped_spawn_zones:
            # Scale and offset the vertices of the zones
            vertices_np = np.array(spawn_zone)
            scaled_vertices = [(self._scale_tuple((x, y))) for x, y in vertices_np]
            # Draw the spawn zone as a polygon on the screen
            pygame.draw.polygon(self.screen, PED_SPAWN_COLOR, scaled_vertices)

    def _draw_goal_zones(self):
        """Draw goal zones.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        # Iterate over each goal_zone in the list of goal_zones
        for goal_zone in self.map_def.ped_goal_zones:
            # Scale and offset the vertices of the goal zones
            vertices_np = np.array(goal_zone)
            scaled_vertices = [(self._scale_tuple((x, y))) for x, y in vertices_np]
            # Draw the goal_zone as a polygon on the screen
            pygame.draw.polygon(self.screen, PED_GOAL_COLOR, scaled_vertices)

    def _augment_goal_position(self, robot_goal: Vec2D):
        """Augment goal position.

        Args:
            robot_goal: goal for robot.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        pygame.draw.circle(
            self.screen,
            ROBOT_GOAL_COLOR,
            self._scale_tuple(robot_goal),
            self.goal_radius * self.scaling,
        )

    def _augment_lidar(self, ray_vecs: np.ndarray):
        """Draw lidar rays given an array of point pairs.

        Accepts an empty array and returns early. If None is provided, does nothing.
        """
        if ray_vecs is None:
            return
        # Handle empty arrays/lists gracefully
        try:
            if len(ray_vecs) == 0:  # works for np.ndarray and list-like
                return
        except TypeError:
            # Not iterable or no length; nothing to draw
            return

        for p1, p2 in ray_vecs:
            pygame.draw.line(
                self.screen,
                ROBOT_LIDAR_COLOR,
                self._scale_tuple(p1),
                self._scale_tuple(p2),
            )

    def _augment_action(self, action: VisualizableAction, color):
        """Augment action.

        Args:
            action: Action applied to the environment.
            color: Color specification for plotting.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        r_x, r_y = action.pose[0]
        # scale vector length to be always visible
        vec_length = action.action[0] * self.scaling
        vec_orient = action.pose[1]

        def from_polar(length: float, orient: float) -> Vec2D:
            """From polar.

            Args:
                length: Length value.
                orient: Orientation value.

            Returns:
                Vec2D: 2D vector (x, y).
            """
            return cos(orient) * length, sin(orient) * length

        def add_vec(v_1: Vec2D, v_2: Vec2D) -> Vec2D:
            """Add vec.

            Args:
                v_1: v 1.
                v_2: v 2.

            Returns:
                Vec2D: 2D vector (x, y).
            """
            return v_1[0] + v_2[0], v_1[1] + v_2[1]

        vec_x, vec_y = add_vec((r_x, r_y), from_polar(vec_length, vec_orient))
        pygame.draw.line(
            self.screen,
            color,
            self._scale_tuple((r_x, r_y)),
            self._scale_tuple((vec_x, vec_y)),
            width=3,
        )

    def _augment_ped_actions(self, ped_actions: np.ndarray):
        """Draw the actions of the pedestrians as lines with optional velocity scaling."""
        for p1, p2 in ped_actions:
            # Apply velocity scaling for visualization if different from 1.0
            if self.ped_velocity_scale != 1.0:
                velocity_vector = np.array(p2) - np.array(p1)
                scaled_p2 = np.array(p1) + velocity_vector * self.ped_velocity_scale
                p2_display = tuple(scaled_p2)
            else:
                p2_display = p2

            pygame.draw.line(
                self.screen,
                PED_ACTION_COLOR,
                self._scale_tuple(p1),
                self._scale_tuple(p2_display),
                width=3,
            )

    def _draw_pedestrian_routes(self):
        """
        draw the map_def.routes on the screen
        """
        for route in self.map_def.ped_routes:
            pygame.draw.lines(
                self.screen,
                PED_ROUTE_COLOR,
                False,
                [self._scale_tuple((x, y)) for x, y in route.waypoints],
                width=1,
            )

    def _draw_robot_routes(self):
        """
        draw the map_def.routes on the screen
        """
        for route in self.map_def.robot_routes:
            pygame.draw.lines(
                self.screen,
                ROBOT_ROUTE_COLOR,
                False,
                [self._scale_tuple((x, y)) for x, y in route.waypoints],
                width=1,
            )

    def _draw_coordinates(self, x, y):
        """
        Draws the coordinates (x, y) on the screen.
        """
        text = self.font.render(f"({x}, {y})", False, TEXT_COLOR)
        self.screen.blit(text, (x, y))

    def _add_text(self, timestep: int, state: VisualizableSimState):
        """Add text.

        Args:
            timestep: Simulation timestep metadata.
            state: State payload.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        info_lines = self._get_display_info_lines(state)
        text_lines = self._build_text_lines(timestep, state, info_lines)
        self._render_text_display(text_lines)

    def _get_display_info_lines(self, state: VisualizableSimState) -> list[str]:
        """Get lines for robot/pedestrian info display based on display mode."""
        if self.display_robot_info == 1:
            return self._get_robot_info_lines(state)
        elif self.display_robot_info == 2:
            return self._get_pedestrian_info_lines(state)
        return []

    def _get_robot_info_lines(self, state: VisualizableSimState) -> list[str]:
        """Get robot information lines for display."""
        if hasattr(state, "robot_action") and state.robot_action:
            return [
                f"RobotPose: {state.robot_pose}",
                f"RobotAction: {state.robot_action.action if state.robot_action else None}",
                f"RobotGoal: {state.robot_action.goal if state.robot_action else None}",
            ]
        return []

    def _get_pedestrian_info_lines(self, state: VisualizableSimState) -> list[str]:
        """Get pedestrian information lines for display."""
        if self._has_pedestrian_data(state):
            assert state.ego_ped_pose is not None, "ego_ped_pose must be set"
            distance_to_robot = euclid_dist(state.ego_ped_pose[0], state.robot_pose[0])
            assert state.ego_ped_action is not None, "ego_ped_action must be set"
            return [
                f"PedestrianPose: {state.ego_ped_pose}",
                f"PedestrianAction: {state.ego_ped_action.action}",
                f"PedestrianGoal: {state.ego_ped_action.goal}",
                f"DistanceRobot: {distance_to_robot:.2f}",
            ]
        else:
            self.display_robot_info = 0
            return []

    def _has_pedestrian_data(self, state: VisualizableSimState) -> bool:
        """Check if the state has complete pedestrian data."""
        return bool(
            hasattr(state, "ego_ped_pose")
            and state.ego_ped_pose
            and hasattr(state, "ego_ped_action")
            and state.ego_ped_action
        )

    def _build_text_lines(
        self, timestep: int, state: VisualizableSimState, info_lines: list[str]
    ) -> list[str]:
        """Build the complete list of text lines for display."""
        # Calculate speedup factor safely
        actual_fps = self.clock.get_fps()
        time_per_step = getattr(state, "time_per_step_in_secs", 0.1)
        speedup = actual_fps * time_per_step

        text_lines = [
            f"step: {timestep}",
            f"scaling: {self.scaling}",
        ]

        # Add FPS and speedup information if not recording
        if not self.record_video:
            text_lines += [
                f"target fps: {actual_fps:.1f}/{getattr(self, 'current_target_fps', 60):.1f}",
                f"speedup: {speedup:.1f}x",
            ]

        text_lines += [
            f"x-offset: {self.offset[0] / self.scaling:.2f}",
            f"y-offset: {self.offset[1] / self.scaling:.2f}",
        ]

        text_lines += info_lines
        text_lines += ["(Press h for help)"]
        return text_lines

    def _render_text_display(self, text_lines: list[str]):
        """Render the text display on screen."""
        # Create a surface for the text background
        max_width = max(self.font.size(line)[0] for line in text_lines)
        text_height = len(text_lines) * self.font.get_linesize()
        text_surface = pygame.Surface((max_width + 10, text_height + 10), pygame.SRCALPHA)
        text_surface.fill(TEXT_BACKGROUND)

        for i, text in enumerate(text_lines):
            self._render_text_line(text_surface, text, i)

        self.screen.blit(text_surface, self._timestep_text_pos)

    def _render_text_line(self, surface, text: str, line_index: int):
        """Render a single text line with outline effect."""
        text_render = self.font.render(text, True, TEXT_COLOR)
        text_outline = self.font.render(text, True, TEXT_OUTLINE_COLOR)

        pos = (5, line_index * self.font.get_linesize() + 5)

        # Draw text outline
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            surface.blit(text_outline, (pos[0] + dx, pos[1] + dy))

        # Draw main text
        surface.blit(text_render, pos)

    def _add_minimal_hint(self):
        """Show a minimal hint when text display is disabled."""
        hint_text = "Press T to show text"

        # Create a smaller font for the hint
        hint_font = pygame.font.Font(None, 12)  # Reduced from 36 to 24

        # Create a semi-transparent surface for the hint
        text_render = hint_font.render(hint_text, True, TEXT_COLOR)
        text_outline = hint_font.render(hint_text, True, TEXT_OUTLINE_COLOR)

        width, height = hint_font.size(hint_text)
        hint_surface = pygame.Surface((width + 10, height + 10), pygame.SRCALPHA)
        hint_surface.fill((0, 0, 0, 128))  # More transparent than normal

        # Position for the text
        pos = (5, 5)

        # Draw outline
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            hint_surface.blit(text_outline, (pos[0] + dx, pos[1] + dy))

        # Draw main text
        hint_surface.blit(text_render, pos)

        # Display the hint in the corner
        self.screen.blit(hint_surface, (16, 16))

    def _add_help_text(self):
        """Add help text.

        Returns:
            Any: Arbitrary value passed through unchanged.
        """
        text_lines = [
            "Move camera: arrow keys",
            "Move fast: CTRL + arrow keys",
            "Move slow: ALT + arrow keys",
            "Reset view: r",
            "Focus robot: f",
            "Focus ego_ped: p",
            "Scale up: +",
            "Scale down: -",
            "Display robot info: q",
            "Toggle text: t",  # Add this line
            "Help: h",
        ]

        # Determine max width of the text
        text_surface = self.font.render(text_lines[1], False, TEXT_COLOR)

        max_width = max(self.font.size(line)[0] for line in text_lines)
        text_height = len(text_lines) * self.font.get_linesize()
        text_surface = pygame.Surface((max_width + 10, text_height + 10), pygame.SRCALPHA)
        text_surface.fill(TEXT_BACKGROUND)

        for i, text in enumerate(text_lines):
            text_render = self.font.render(text, True, TEXT_COLOR)
            text_outline = self.font.render(text, True, TEXT_OUTLINE_COLOR)

            pos = (5, i * self.font.get_linesize() + 5)

            # Draw text outline
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                text_surface.blit(text_outline, (pos[0] + dx, pos[1] + dy))

            # Draw main text
            text_surface.blit(text_render, pos)

        # Blit and return the rect to allow children to position additional help relative to this
        return self.screen.blit(
            text_surface,
            (self.width - max_width - 10, self._timestep_text_pos[1]),
        )

    def _draw_grid(self, grid_increment: int = 50, grid_color: RgbColor = (200, 200, 200)):
        """
        Draw a grid on the screen.
        :param grid_increment: The increment of the grid in pixels.
        :param grid_color: The color of the grid lines.
        """
        scaled_grid_size = grid_increment * self.scaling
        font = pygame.font.Font(None, 24)
        # draw the vertical lines
        start_x = ((-self.offset[0]) // scaled_grid_size) * scaled_grid_size
        for x in range(int(start_x), int(self.width - self.offset[0]), int(scaled_grid_size)):
            pygame.draw.line(
                self.screen,
                grid_color,
                (x + self.offset[0], 0),
                (x + self.offset[0], self.height),
            )
            label = font.render(str(int(x / self.scaling)), 1, grid_color)
            self.screen.blit(label, (x + self.offset[0], 0))

        # draw the horizontal lines
        start_y = ((-self.offset[1]) // scaled_grid_size) * scaled_grid_size
        for y in range(int(start_y), int(self.height - self.offset[1]), int(scaled_grid_size)):
            pygame.draw.line(
                self.screen,
                grid_color,
                (0, y + self.offset[1]),
                (self.width, y + self.offset[1]),
            )
            label = font.render(str(int(y / self.scaling)), 1, grid_color)
            self.screen.blit(label, (0, y + self.offset[1]))
