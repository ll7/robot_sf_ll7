"""Pygame runner for local manual-control Robot SF sessions."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from robot_sf.common.artifact_paths import resolve_artifact_path
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.manual_control.baseline import BaselineMetric, MetricDirection, PolicyBaseline
from robot_sf.manual_control.config import ManualControlRuntimeConfig
from robot_sf.manual_control.input_mapping import (
    ManualKeyState,
    mapper_for_manual_mode,
)
from robot_sf.manual_control.manifest import ManualSessionManifest, write_manual_session_manifest
from robot_sf.manual_control.recording import (
    ManualControlRecord,
    ManualJsonlRecorder,
    ManualSessionMetadata,
)
from robot_sf.manual_control.session import AttemptKey, ManualSessionController
from robot_sf.robot.differential_drive import DifferentialDriveSettings

EventSource = Callable[[], Sequence[Any]]
EnvFactory = Callable[..., Any]

CONTROL_KEYS = frozenset({"escape", "q", "p", "r", "+", "=", "-", "_"})
_NO_RESET_OBSERVATION = object()


@dataclass(frozen=True)
class ManualPygameRunnerSettings:
    """Configuration for one local manual-control Pygame session."""

    scenario_id: str
    seed: int
    policy_to_beat: str
    policy_to_beat_source: str
    baseline_primary_metric: str = "success"
    baseline_primary_value: float = 0.0
    baseline_direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    baseline_tolerance: float = 0.0
    output_dir: Path = Path("output/manual_control")
    session_id: str | None = None
    max_steps: int = 500
    max_frames: int | None = None
    countdown_steps: int = 3
    stop_between_episodes: bool = True
    speed_multiplier: float = 1.0
    control_mode: str = "keyboard_hold"
    view_mode: str = "fixed_map"
    target_fps: float = 30.0
    headless: bool = False
    render: bool = True

    def __post_init__(self) -> None:
        """Validate simple scalar settings before the runner allocates resources."""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if not self.policy_to_beat:
            raise ValueError("policy_to_beat is required for fail-closed baseline selection")
        if not self.policy_to_beat_source:
            raise ValueError("policy_to_beat_source is required for baseline provenance")

    @property
    def resolved_session_id(self) -> str:
        """Return the explicit session id or a timestamped default."""
        if self.session_id:
            return self.session_id
        stamp = time.strftime("%Y%m%dT%H%M%S")
        return f"manual-{self.scenario_id}-seed{self.seed}-{stamp}"


@dataclass(frozen=True)
class ManualPygameRunnerResult:
    """Summary of one manual-control runner invocation."""

    session_id: str
    records_path: Path
    manifest_path: Path
    steps: int
    success: bool
    beat_baseline: bool
    failure_reason: str | None


class ManualPygameRunner:
    """Orchestrate keyboard input, RobotEnv stepping, recording, and manifest output."""

    def __init__(
        self,
        settings: ManualPygameRunnerSettings,
        *,
        env_factory: EnvFactory = make_robot_env,
        pygame_module: Any | None = None,
        event_source: EventSource | None = None,
    ) -> None:
        """Create a runner with injectable dependencies for headless tests."""
        self.settings = settings
        self.env_factory = env_factory
        self._pygame = pygame_module
        self._event_source = event_source
        self._pressed_keys: set[str] = set()
        self._quit_requested = False
        self._step_credit = 0.0

    def run(self) -> ManualPygameRunnerResult:  # noqa: C901, PLR0915
        """Run one configured manual-control session and return artifact paths.

        Returns
        -------
        ManualPygameRunnerResult
            Artifact paths and terminal outcome for the session.
        """
        pygame = self._load_pygame()
        runtime_config = ManualControlRuntimeConfig.from_strings(
            control_mode=self.settings.control_mode,
            view_mode=self.settings.view_mode,
            robot_action_space="differential_drive",
        )
        baseline = self._build_baseline()
        session_id = self.settings.resolved_session_id
        output_dir = resolve_artifact_path(self.settings.output_dir / session_id)
        records_path = output_dir / "records.jsonl"
        manifest_path = output_dir / "manifest.json"
        controller = ManualSessionController(
            countdown_steps=self.settings.countdown_steps,
            stop_between_episodes=self.settings.stop_between_episodes,
            speed_multiplier=self.settings.speed_multiplier,
        )
        session = ManualSessionMetadata(
            session_id=session_id,
            input_mapping_version=runtime_config.input_mapping_version,
            control_mode=runtime_config.control_mode.value,
            view_mode=runtime_config.view_mode.value,
            policy_to_beat=baseline.policy_id,
            policy_to_beat_source=baseline.source,
            extra={
                "scenario_id": self.settings.scenario_id,
                "seed": self.settings.seed,
                "target_fps": self.settings.target_fps,
            },
        )
        env = self._make_env(runtime_config)
        mapper = mapper_for_manual_mode(env.env_config.robot_config, runtime_config.control_mode)
        if not isinstance(env.env_config.robot_config, DifferentialDriveSettings):
            raise NotImplementedError("manual-control MVP requires a differential-drive robot")

        started_at_utc = _utc_now()
        started_monotonic = time.perf_counter()
        frame_clock = _make_frame_clock(pygame)
        steps = 0
        success = False
        beat_baseline = False
        failure_reason: str | None = None
        observation: Any = None
        try:
            observation, _info = env.reset(seed=self.settings.seed)
            attempt = controller.start_attempt(self.settings.scenario_id, self.settings.seed)
            with ManualJsonlRecorder(records_path) as recorder:
                self._write_event(
                    recorder,
                    event="session_start",
                    attempt_key=attempt.key,
                    attempt_id=attempt.retry_count,
                    step_idx=steps,
                    session=session,
                    metrics={"speed_multiplier": controller.speed_multiplier},
                )
                frames = 0
                while not self._quit_requested:
                    frames += 1
                    reset_observation = self._process_events(
                        pygame,
                        controller,
                        recorder,
                        attempt.key,
                        session,
                        steps,
                        env,
                    )
                    if reset_observation is not _NO_RESET_OBSERVATION:
                        observation = reset_observation
                    if self._quit_requested:
                        break
                    active_attempt = controller.active_attempt or attempt
                    if controller.state.value == "countdown":
                        self._write_event(
                            recorder,
                            event="countdown",
                            attempt_key=active_attempt.key,
                            attempt_id=active_attempt.retry_count,
                            step_idx=steps,
                            session=session,
                            metrics={"countdown_remaining": controller.countdown_remaining},
                        )
                        controller.advance_countdown()
                    elif controller.should_step:
                        self._step_credit += controller.speed_multiplier
                        while self._step_credit >= 1.0 and steps < self.settings.max_steps:
                            action = mapper.map_action(
                                ManualKeyState.from_keys(self._pressed_keys),
                                current_velocity=_current_velocity(env),
                            )
                            observation, reward, terminated, truncated, info = env.step(action)
                            steps += 1
                            self._step_credit -= 1.0
                            metrics = _step_metrics(
                                reward=reward,
                                info=info,
                                terminated=terminated,
                                truncated=truncated,
                                step_idx=steps,
                            )
                            self._write_event(
                                recorder,
                                event="step",
                                attempt_key=active_attempt.key,
                                attempt_id=active_attempt.retry_count,
                                step_idx=steps,
                                session=session,
                                input_keys=sorted(self._pressed_keys),
                                mapped_action=tuple(float(value) for value in action),
                                observation=_manual_state_payload(env, observation),
                                metrics=metrics,
                                training_sample=True,
                            )
                            if terminated or truncated:
                                success = bool(metrics["success"])
                                failure_reason = _failure_reason(metrics)
                                break
                        if steps >= self.settings.max_steps and failure_reason is None:
                            failure_reason = "max_steps"
                    if (
                        self.settings.max_frames is not None
                        and frames >= self.settings.max_frames
                        and failure_reason is None
                    ):
                        failure_reason = "frame_limit"
                    if self.settings.render:
                        self._render(env, pygame, controller, steps, failure_reason)
                    _tick_frame_clock(frame_clock, self.settings.target_fps)
                    if failure_reason is not None:
                        candidate_metrics = _candidate_metrics(
                            success=success,
                            steps=steps,
                            failure_reason=failure_reason,
                        )
                        comparison = baseline.compare(candidate_metrics)
                        beat_baseline = comparison.beat_baseline
                        controller.mark_terminal(
                            success=success,
                            beat_baseline=beat_baseline,
                            failure_reason=failure_reason if not success else None,
                        )
                        self._write_event(
                            recorder,
                            event="terminal",
                            attempt_key=active_attempt.key,
                            attempt_id=active_attempt.retry_count,
                            step_idx=steps,
                            session=session,
                            metrics={
                                **candidate_metrics,
                                "beat_baseline": beat_baseline,
                                "failure_reason": failure_reason,
                            },
                        )
                        break
            finished_at_utc = _utc_now()
            runtime_seconds = max(0.0, time.perf_counter() - started_monotonic)
            manifest_session = replace(
                session,
                extra={
                    **session.extra,
                    "started_at_utc": started_at_utc,
                    "finished_at_utc": finished_at_utc,
                    "runtime_seconds": runtime_seconds,
                },
            )
            manifest = ManualSessionManifest(
                session=manifest_session,
                baseline=baseline,
                completed_attempts=tuple(controller.completed.values()),
                unresolved_attempts=tuple(controller.unresolved.values()),
                artifacts={
                    "records_jsonl": str(records_path),
                    "manifest_json": str(manifest_path),
                },
                notes=(
                    "manual-control MVP runner: keyboard_hold + fixed_map",
                    f"last_observation_type={type(observation).__name__}",
                ),
            )
            write_manual_session_manifest(manifest, manifest_path)
        finally:
            _close_env(env)
        return ManualPygameRunnerResult(
            session_id=session_id,
            records_path=records_path,
            manifest_path=manifest_path,
            steps=steps,
            success=success,
            beat_baseline=beat_baseline,
            failure_reason=failure_reason,
        )

    def _load_pygame(self) -> Any:
        """Import pygame lazily after optional headless environment setup.

        Returns
        -------
        Any
            Imported pygame module or injected pygame-compatible test double.
        """
        if self.settings.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("MPLBACKEND", "Agg")
        if self._pygame is not None:
            return self._pygame
        import pygame  # noqa: PLC0415

        return pygame

    def _make_env(self, runtime_config: ManualControlRuntimeConfig) -> Any:
        """Construct the RobotEnv instance used by the session.

        Returns
        -------
        Any
            Gymnasium-compatible Robot SF environment.
        """
        return self.env_factory(
            seed=self.settings.seed,
            debug=self.settings.render,
            scenario_name=self.settings.scenario_id,
            algorithm_name="manual_control",
            recording_enabled=False,
        )

    def _build_baseline(self) -> PolicyBaseline:
        """Freeze the policy-to-beat metadata for this session.

        Returns
        -------
        PolicyBaseline
            Explicit policy-to-beat baseline used for terminal comparison.
        """
        metric = BaselineMetric(
            name=self.settings.baseline_primary_metric,
            value=float(self.settings.baseline_primary_value),
            direction=self.settings.baseline_direction,
            tolerance=float(self.settings.baseline_tolerance),
        )
        return PolicyBaseline(
            policy_id=self.settings.policy_to_beat,
            source=self.settings.policy_to_beat_source,
            primary_metric=metric.name,
            metrics={metric.name: metric},
            metadata={"selection_reason": "explicit_cli_override"},
        )

    def _process_events(  # noqa: C901
        self,
        pygame: Any,
        controller: ManualSessionController,
        recorder: ManualJsonlRecorder,
        attempt_key: AttemptKey,
        session: ManualSessionMetadata,
        step_idx: int,
        env: Any,
    ) -> object:
        """Convert Pygame events into session controls and held-key state.

        Returns
        -------
        object
            Reset observation when retry was requested, otherwise a sentinel.
        """
        for event in self._read_events(pygame):
            event_type = getattr(event, "type", None)
            if event_type == getattr(pygame, "QUIT", object()):
                self._quit_requested = True
                return _NO_RESET_OBSERVATION
            if event_type not in {
                getattr(pygame, "KEYDOWN", object()),
                getattr(pygame, "KEYUP", object()),
            }:
                continue
            key_name = _pygame_key_name(pygame, getattr(event, "key", ""))
            if event_type == getattr(pygame, "KEYUP", object()):
                self._pressed_keys.discard(key_name)
                continue
            if key_name in {"escape", "q"}:
                self._quit_requested = True
                return _NO_RESET_OBSERVATION
            if key_name == "p":
                controller.toggle_pause()
                self._write_event(
                    recorder,
                    event="pause_toggle",
                    attempt_key=attempt_key,
                    attempt_id=controller.active_attempt.retry_count
                    if controller.active_attempt
                    else 0,
                    step_idx=step_idx,
                    session=session,
                    metrics={"state": controller.state.value},
                )
                continue
            if key_name == "r":
                retry = controller.retry_active()
                self._pressed_keys.clear()
                self._step_credit = 0.0
                observation, _info = env.reset(seed=self.settings.seed)
                self._write_event(
                    recorder,
                    event="retry",
                    attempt_key=retry.key,
                    attempt_id=retry.retry_count,
                    step_idx=step_idx,
                    session=session,
                    metrics={"retry_count": retry.retry_count},
                )
                return observation
            if key_name in {"+", "="}:
                controller.set_speed_multiplier(controller.speed_multiplier + 0.25)
                continue
            if key_name in {"-", "_"}:
                controller.set_speed_multiplier(max(0.25, controller.speed_multiplier - 0.25))
                continue
            if key_name not in CONTROL_KEYS:
                self._pressed_keys.add(key_name)
        return _NO_RESET_OBSERVATION

    def _read_events(self, pygame: Any) -> Sequence[Any]:
        """Return the next Pygame event batch."""
        if self._event_source is not None:
            return self._event_source()
        return pygame.event.get()

    def _render(
        self,
        env: Any,
        pygame: Any,
        controller: ManualSessionController,
        step_idx: int,
        failure_reason: str | None,
    ) -> None:
        """Render the env and draw a compact manual-control overlay."""
        env.render()
        sim_ui = getattr(env, "sim_ui", None)
        if sim_ui is None or not hasattr(sim_ui, "screen"):
            return
        lines = [
            f"manual {self.settings.scenario_id} seed={self.settings.seed}",
            f"state={controller.state.value} step={step_idx}/{self.settings.max_steps}",
            f"keys={','.join(sorted(self._pressed_keys)) or '-'} speed={controller.speed_multiplier:g}x",
            f"policy_to_beat={self.settings.policy_to_beat}",
        ]
        if controller.countdown_remaining:
            lines.append(f"countdown={controller.countdown_remaining}")
        if failure_reason:
            lines.append(f"terminal={failure_reason}")
        _draw_overlay(pygame, sim_ui, lines)

    @staticmethod
    def _write_event(  # noqa: PLR0913
        recorder: ManualJsonlRecorder,
        *,
        event: str,
        attempt_key: AttemptKey,
        attempt_id: int,
        step_idx: int,
        session: ManualSessionMetadata,
        input_keys: list[str] | None = None,
        mapped_action: tuple[float, ...] | None = None,
        observation: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        training_sample: bool = False,
    ) -> None:
        """Append one manual-control record."""
        recorder.write(
            ManualControlRecord.for_attempt(
                event=event,
                attempt_key=attempt_key,
                attempt_id=attempt_id,
                step_idx=step_idx,
                session=session,
                input_keys=input_keys,
                mapped_action=mapped_action,
                observation=observation,
                metrics=metrics,
                training_sample=training_sample,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the manual-control runner CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="manual_control_default")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--policy-to-beat", required=True)
    parser.add_argument("--policy-to-beat-source", required=True)
    parser.add_argument("--baseline-primary-metric", default="success")
    parser.add_argument("--baseline-primary-value", type=float, default=0.0)
    parser.add_argument(
        "--baseline-direction",
        choices=[direction.value for direction in MetricDirection],
        default=MetricDirection.HIGHER_IS_BETTER.value,
    )
    parser.add_argument("--baseline-tolerance", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("output/manual_control"))
    parser.add_argument("--session-id")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--countdown-steps", type=int, default=3)
    parser.add_argument("--speed-multiplier", type=float, default=1.0)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--control-mode", default="keyboard_hold")
    parser.add_argument("--view-mode", default="fixed_map")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-stop-between-episodes", action="store_true")
    return parser


def settings_from_args(args: argparse.Namespace) -> ManualPygameRunnerSettings:
    """Convert parsed CLI args into runner settings.

    Returns
    -------
    ManualPygameRunnerSettings
        Validated runner settings.
    """
    return ManualPygameRunnerSettings(
        scenario_id=args.scenario_id,
        seed=args.seed,
        policy_to_beat=args.policy_to_beat,
        policy_to_beat_source=args.policy_to_beat_source,
        baseline_primary_metric=args.baseline_primary_metric,
        baseline_primary_value=args.baseline_primary_value,
        baseline_direction=MetricDirection(args.baseline_direction),
        baseline_tolerance=args.baseline_tolerance,
        output_dir=args.output_dir,
        session_id=args.session_id,
        max_steps=args.max_steps,
        max_frames=args.max_frames,
        countdown_steps=args.countdown_steps,
        stop_between_episodes=not args.no_stop_between_episodes,
        speed_multiplier=args.speed_multiplier,
        control_mode=args.control_mode,
        view_mode=args.view_mode,
        target_fps=args.target_fps,
        headless=args.headless,
        render=not args.no_render,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the manual-control Pygame session CLI.

    Returns
    -------
    int
        Process exit status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    result = ManualPygameRunner(settings_from_args(args)).run()
    logger.info(
        "Manual-control session {} wrote records={} manifest={} steps={} success={} beat_baseline={}",
        result.session_id,
        result.records_path,
        result.manifest_path,
        result.steps,
        result.success,
        result.beat_baseline,
    )
    return 0


def _current_velocity(env: Any) -> tuple[float, float]:
    """Return the current differential-drive velocity from the live environment."""
    simulator = getattr(env, "simulator", None)
    robots = getattr(simulator, "robots", None) or ()
    if not robots:
        return 0.0, 0.0
    speed = getattr(robots[0], "current_speed", (0.0, 0.0))
    try:
        return float(speed[0]), float(speed[1])
    except (TypeError, ValueError, IndexError):
        return 0.0, 0.0


def _manual_state_payload(env: Any, observation: Any) -> dict[str, Any]:
    """Build a compact JSONL observation payload for manual-control training rows.

    Returns
    -------
    dict[str, Any]
        JSON-compatible state payload for one training sample.
    """
    simulator = getattr(env, "simulator", None)
    robot_pose = _first_or_default(
        getattr(simulator, "robot_poses", None),
        ((0.0, 0.0), 0.0),
    )
    goal = _first_or_default(getattr(simulator, "goal_pos", None), (0.0, 0.0))
    return {
        "robot_pose": robot_pose,
        "goal": goal,
        "current_velocity": _current_velocity(env),
        "observation_type": type(observation).__name__,
    }


def _step_metrics(
    *,
    reward: float,
    info: dict[str, Any],
    terminated: bool,
    truncated: bool,
    step_idx: int,
) -> dict[str, Any]:
    """Extract runner metrics from one Gymnasium step result.

    Returns
    -------
    dict[str, Any]
        Compact step metrics used in records and terminal classification.
    """
    return {
        "step": step_idx,
        "reward_total": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "collision": bool(info.get("collision", False)),
        "success": bool(info.get("success", info.get("is_success", False))),
    }


def _candidate_metrics(
    *, success: bool, steps: int, failure_reason: str | None
) -> dict[str, float]:
    """Return numeric terminal metrics suitable for baseline comparison."""
    return {
        "success": 1.0 if success else 0.0,
        "steps": float(steps),
        "terminal_without_failure": 1.0 if failure_reason is None else 0.0,
    }


def _failure_reason(metrics: dict[str, Any]) -> str:
    """Classify terminal status for the manifest and terminal JSONL record.

    Returns
    -------
    str
        Terminal status label.
    """
    if bool(metrics.get("success")):
        return "success"
    if bool(metrics.get("collision")):
        return "collision"
    if bool(metrics.get("truncated")):
        return "truncated"
    return "terminated"


def _pygame_key_name(pygame: Any, key: Any) -> str:
    """Return a normalized pygame key name."""
    try:
        return str(pygame.key.name(key)).strip().lower()
    except (AttributeError, TypeError, ValueError):
        return str(key).strip().lower()


def _first_or_default(values: Any, default: Any) -> Any:
    """Return the first value from a sequence-like object or a default."""
    if values is None:
        return default
    try:
        if len(values) == 0:
            return default
        return values[0]
    except (TypeError, IndexError):
        return default


def _make_frame_clock(pygame: Any) -> Any | None:
    """Create a Pygame frame clock when the injected module provides one.

    Returns
    -------
    Any | None
        Clock-like object with ``tick`` support, or ``None`` when unavailable.
    """
    time_module = getattr(pygame, "time", None)
    clock_factory = getattr(time_module, "Clock", None)
    if not callable(clock_factory):
        return None
    return clock_factory()


def _tick_frame_clock(clock: Any | None, target_fps: float) -> None:
    """Limit the runner loop to the configured frame rate when a clock is available."""
    if clock is None:
        return
    tick = getattr(clock, "tick", None)
    if callable(tick):
        tick(target_fps)


def _utc_now() -> str:
    """Return a compact UTC timestamp for manual-control provenance metadata."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _draw_overlay(pygame: Any, sim_ui: Any, lines: Iterable[str]) -> None:
    """Draw compact manual-control status text on the SimulationView surface."""
    font = getattr(sim_ui, "font", None)
    screen = getattr(sim_ui, "screen", None)
    if font is None or screen is None:
        return
    x = 12
    y = 12
    for line in lines:
        text = font.render(str(line), True, (255, 255, 255))
        shadow = font.render(str(line), True, (0, 0, 0))
        screen.blit(shadow, (x + 1, y + 1))
        screen.blit(text, (x, y))
        y += 18
    display = getattr(pygame, "display", None)
    update = getattr(display, "update", None)
    if callable(update) and getattr(sim_ui, "_use_display", False):
        update()


def _close_env(env: Any) -> None:
    """Release env and renderer resources when available."""
    close = getattr(env, "close", None)
    if callable(close):
        close()
        return
    exit_env = getattr(env, "exit", None)
    if callable(exit_env):
        exit_env()


if __name__ == "__main__":
    raise SystemExit(main())
