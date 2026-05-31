"""Tests for manual-control Pygame runner orchestration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from robot_sf.manual_control.pygame_runner import (
    ManualPygameRunner,
    ManualPygameRunnerSettings,
    _current_velocity,
    _failure_reason,
    _pygame_key_name,
    build_parser,
    main,
    settings_from_args,
)
from robot_sf.manual_control.recording import ManualSessionMetadata, load_manual_jsonl_records
from robot_sf.manual_control.session import ManualSessionController
from robot_sf.robot.differential_drive import DifferentialDriveSettings


class _FakeRobot:
    def __init__(self) -> None:
        self.current_speed = (0.0, 0.0)


class _FakeEnv:
    def __init__(
        self,
        *,
        terminal_after: int = 2,
        success: bool = True,
        sim_ui=None,
    ) -> None:
        self.env_config = SimpleNamespace(robot_config=DifferentialDriveSettings())
        self.simulator = SimpleNamespace(
            robots=[_FakeRobot()],
            robot_poses=[((0.0, 0.0), 0.0)],
            goal_pos=[(1.0, 0.0)],
        )
        self.terminal_after = terminal_after
        self.success = success
        self.step_count = 0
        self.reset_count = 0
        self.render_count = 0
        self.exit_count = 0
        self.sim_ui = sim_ui

    def reset(self, *, seed: int | None = None):
        self.reset_count += 1
        self.step_count = 0
        self.simulator.robots[0].current_speed = (0.0, 0.0)
        return {"seed": seed}, {}

    def step(self, action):
        self.step_count += 1
        current_linear, current_angular = self.simulator.robots[0].current_speed
        self.simulator.robots[0].current_speed = (
            current_linear + float(action[0]),
            current_angular + float(action[1]),
        )
        terminated = self.step_count >= self.terminal_after
        return (
            {"step": self.step_count},
            1.0,
            terminated,
            False,
            {"success": self.success and terminated, "collision": False},
        )

    def render(self) -> None:
        self.render_count += 1

    def exit(self) -> None:
        self.exit_count += 1


class _FakeManualView:
    """Renderer test double that records manual view mode configuration."""

    def __init__(self) -> None:
        self.configured_mode = None
        self.font = _FakeFont()
        self.screen = _FakeScreen()
        self._use_display = False

    def set_manual_view_mode(self, mode) -> None:
        self.configured_mode = mode


class _FakeKey:
    @staticmethod
    def name(key) -> str:
        return str(key)


class _FakePygame:
    KEYDOWN = 1
    KEYUP = 2
    QUIT = 3
    key = _FakeKey()


def _event(event_type: int, key: str | None = None):
    return SimpleNamespace(type=event_type, key=key)


class _FakeFont:
    def render(self, text, *_args):
        return f"rendered:{text}"


class _FakeScreen:
    def __init__(self) -> None:
        self.blits = []

    def blit(self, surface, position) -> None:
        self.blits.append((surface, position))


class _FakeDisplay:
    updates = 0

    @classmethod
    def update(cls) -> None:
        cls.updates += 1


class _FakeClock:
    ticks: list[float] = []

    def tick(self, target_fps: float) -> None:
        self.ticks.append(target_fps)


class _FakeTime:
    @staticmethod
    def Clock() -> _FakeClock:
        return _FakeClock()


class _OverlayPygame(_FakePygame):
    display = _FakeDisplay
    time = _FakeTime


class _InitPygame(_FakePygame):
    init_called = False

    @classmethod
    def init(cls) -> None:
        cls.init_called = True


def test_pygame_runner_writes_training_records_and_manifest(tmp_path):
    """Runner should write append-only records and a baseline-bearing manifest."""
    env = _FakeEnv(terminal_after=2, success=True)
    settings = ManualPygameRunnerSettings(
        scenario_id="scenario-a",
        seed=7,
        policy_to_beat="policy-a",
        policy_to_beat_source="model/registry.yaml",
        output_dir=tmp_path,
        session_id="session-a",
        countdown_steps=0,
        max_steps=5,
        render=False,
    )
    batches = [[_event(_FakePygame.KEYDOWN, "w")], []]

    result = ManualPygameRunner(
        settings,
        env_factory=lambda **_kwargs: env,
        pygame_module=_FakePygame,
        event_source=lambda: batches.pop(0) if batches else [],
    ).run()

    records = load_manual_jsonl_records(result.records_path)
    step_records = [record for record in records if record.event == "step"]

    assert result.success is True
    assert result.beat_baseline is True
    assert len(step_records) == 2
    assert all(record.training_sample for record in step_records)
    assert step_records[0].input_keys == ["w"]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest_text = json.dumps(manifest, sort_keys=True)
    assert "manual_control_session_manifest_v1" in manifest_text
    assert "policy-a" in manifest_text
    assert manifest["session"]["extra"]["started_at_utc"].endswith("Z")
    assert manifest["session"]["extra"]["finished_at_utc"].endswith("Z")
    assert manifest["session"]["extra"]["runtime_seconds"] >= 0.0
    assert env.exit_count == 1


def test_pygame_runner_processes_pause_retry_and_speed_controls(tmp_path):
    """Runner controls should update session state without marking control events as samples."""
    env = _FakeEnv(terminal_after=2, success=False)
    settings = ManualPygameRunnerSettings(
        scenario_id="scenario-b",
        seed=11,
        policy_to_beat="policy-b",
        policy_to_beat_source="explicit-test",
        output_dir=tmp_path,
        session_id="session-b",
        countdown_steps=0,
        max_steps=3,
        render=False,
    )
    batches = [
        [_event(_FakePygame.KEYDOWN, "p")],
        [_event(_FakePygame.KEYDOWN, "p")],
        [_event(_FakePygame.KEYDOWN, "r")],
        [_event(_FakePygame.KEYDOWN, "=")],
        [],
    ]

    result = ManualPygameRunner(
        settings,
        env_factory=lambda **_kwargs: env,
        pygame_module=_FakePygame,
        event_source=lambda: batches.pop(0) if batches else [],
    ).run()

    records = load_manual_jsonl_records(result.records_path)
    events = [record.event for record in records]

    assert "pause_toggle" in events
    assert "retry" in events
    assert max(record.attempt_id for record in records if record.event == "step") == 1
    assert result.failure_reason == "terminated"
    assert env.reset_count >= 2
    assert all(not record.training_sample for record in records if record.event != "step")


def test_pygame_runner_countdown_render_overlay_and_keyup(tmp_path):
    """Countdown and render mode should draw overlay text and handle key releases."""
    screen = _FakeScreen()
    sim_ui = SimpleNamespace(font=_FakeFont(), screen=screen, _use_display=True)
    env = _FakeEnv(terminal_after=1, success=True, sim_ui=sim_ui)
    settings = ManualPygameRunnerSettings(
        scenario_id="scenario-render",
        seed=12,
        policy_to_beat="policy-render",
        policy_to_beat_source="explicit-test",
        output_dir=tmp_path,
        session_id="session-render",
        countdown_steps=1,
        max_steps=2,
        render=True,
    )
    batches = [
        [_event(_FakePygame.KEYDOWN, "w"), _event(_FakePygame.KEYUP, "w")],
        [SimpleNamespace(type=999, key="ignored")],
    ]

    result = ManualPygameRunner(
        settings,
        env_factory=lambda **_kwargs: env,
        pygame_module=_OverlayPygame,
        event_source=lambda: batches.pop(0) if batches else [],
    ).run()

    records = load_manual_jsonl_records(result.records_path)

    assert "countdown" in [record.event for record in records]
    assert env.render_count >= 1
    assert screen.blits
    assert _FakeDisplay.updates >= 1
    assert _FakeClock.ticks


@pytest.mark.parametrize("view_mode", ["ego_up", "robot_static"])
def test_pygame_runner_configures_camera_transform_renderer_view(tmp_path, view_mode):
    """Camera-transform sessions should configure the env renderer before rendering."""
    _FakeClock.ticks.clear()
    sim_ui = _FakeManualView()
    env = _FakeEnv(terminal_after=1, success=True, sim_ui=sim_ui)
    settings = ManualPygameRunnerSettings(
        scenario_id="scenario-ego-up",
        seed=16,
        policy_to_beat="policy-ego-up",
        policy_to_beat_source="explicit-test",
        output_dir=tmp_path,
        session_id="session-ego-up",
        countdown_steps=0,
        max_steps=1,
        render=True,
        view_mode=view_mode,
    )

    ManualPygameRunner(
        settings,
        env_factory=lambda **_kwargs: env,
        pygame_module=_OverlayPygame,
        event_source=lambda: [],
    ).run()

    assert sim_ui.configured_mode == view_mode
    assert all(target_fps == settings.target_fps for target_fps in _FakeClock.ticks)


def test_pygame_runner_initializes_pygame_for_headless_event_reads(tmp_path):
    """Headless CLI-style runs should initialize Pygame before reading events."""
    _InitPygame.init_called = False
    settings = ManualPygameRunnerSettings(
        scenario_id="scenario-init",
        seed=15,
        policy_to_beat="policy-init",
        policy_to_beat_source="explicit-test",
        output_dir=tmp_path,
        headless=True,
        render=False,
    )

    pygame = ManualPygameRunner(settings, pygame_module=_InitPygame)._load_pygame()

    assert pygame is _InitPygame
    assert _InitPygame.init_called is True


def test_pygame_runner_quit_event_writes_empty_manifest(tmp_path):
    """A quit event should stop cleanly and still write a session manifest."""
    env = _FakeEnv()
    settings = ManualPygameRunnerSettings(
        scenario_id="scenario-quit",
        seed=13,
        policy_to_beat="policy-quit",
        policy_to_beat_source="explicit-test",
        output_dir=tmp_path,
        session_id="session-quit",
        countdown_steps=0,
        render=False,
    )

    result = ManualPygameRunner(
        settings,
        env_factory=lambda **_kwargs: env,
        pygame_module=_FakePygame,
        event_source=lambda: [_event(_FakePygame.QUIT)],
    ).run()

    assert result.steps == 0
    assert result.manifest_path.exists()


def test_pygame_runner_speed_controls_use_quarter_step(tmp_path) -> None:
    """Speed controls should be reversible from the minimum multiplier."""
    settings = ManualPygameRunnerSettings(
        scenario_id="scenario-speed",
        seed=14,
        policy_to_beat="policy-speed",
        policy_to_beat_source="explicit-test",
        output_dir=tmp_path,
    )
    controller = ManualSessionController(countdown_steps=0)
    attempt = controller.start_attempt(settings.scenario_id, settings.seed)
    session = ManualSessionMetadata(
        session_id=settings.resolved_session_id,
        input_mapping_version="manual_keyboard_diff_drive_hold_v1",
    )
    records = []

    runner = ManualPygameRunner(
        settings,
        pygame_module=_FakePygame,
        event_source=lambda: [
            _event(_FakePygame.KEYDOWN, "-"),
            _event(_FakePygame.KEYDOWN, "-"),
            _event(_FakePygame.KEYDOWN, "-"),
            _event(_FakePygame.KEYDOWN, "-"),
            _event(_FakePygame.KEYDOWN, "="),
        ],
    )
    runner._process_events(
        _FakePygame,
        controller,
        SimpleNamespace(write=records.append),
        attempt.key,
        session,
        0,
        SimpleNamespace(),
    )

    assert controller.speed_multiplier == 0.5


def test_pygame_runner_requires_explicit_policy_to_beat() -> None:
    """Policy-to-beat metadata should fail closed instead of using an implicit baseline."""
    with pytest.raises(ValueError, match="policy_to_beat"):
        ManualPygameRunnerSettings(
            scenario_id="scenario-c",
            seed=1,
            policy_to_beat="",
            policy_to_beat_source="model/registry.yaml",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_steps": 0}, "max_steps"),
        ({"max_frames": 0}, "max_frames"),
        ({"target_fps": 0.0}, "target_fps"),
        ({"policy_to_beat_source": ""}, "policy_to_beat_source"),
    ],
)
def test_pygame_runner_settings_validate_scalars(kwargs, message) -> None:
    """Invalid runner settings should fail before env construction."""
    base = {
        "scenario_id": "scenario-invalid",
        "seed": 1,
        "policy_to_beat": "policy",
        "policy_to_beat_source": "source",
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        ManualPygameRunnerSettings(**base)


def test_runner_settings_default_session_id_is_stable_shape(monkeypatch) -> None:
    """Default session ids should include scenario and seed for artifact discovery."""
    monkeypatch.setattr("robot_sf.manual_control.pygame_runner.time.strftime", lambda _fmt: "STAMP")
    settings = ManualPygameRunnerSettings(
        scenario_id="scenario-default",
        seed=21,
        policy_to_beat="policy",
        policy_to_beat_source="source",
    )

    assert settings.resolved_session_id == "manual-scenario-default-seed21-STAMP"


def test_cli_parser_and_main_convert_settings(monkeypatch, tmp_path) -> None:
    """CLI helpers should parse runner settings and report success through main."""
    parsed = build_parser().parse_args(
        [
            "--scenario-id",
            "scenario-cli",
            "--seed",
            "31",
            "--policy-to-beat",
            "policy-cli",
            "--policy-to-beat-source",
            "source-cli",
            "--baseline-direction",
            "lower_is_better",
            "--output-dir",
            str(tmp_path),
            "--headless",
            "--no-render",
        ]
    )
    settings = settings_from_args(parsed)

    assert settings.scenario_id == "scenario-cli"
    assert settings.baseline_direction.value == "lower_is_better"
    assert settings.headless is True
    assert settings.render is False

    class _FakeRunner:
        def __init__(self, settings):
            self.settings = settings

        def run(self):
            return SimpleNamespace(
                session_id=self.settings.resolved_session_id,
                records_path=tmp_path / "records.jsonl",
                manifest_path=tmp_path / "manifest.json",
                steps=0,
                success=False,
                beat_baseline=False,
            )

    monkeypatch.setattr("robot_sf.manual_control.pygame_runner.ManualPygameRunner", _FakeRunner)

    assert (
        main(
            [
                "--scenario-id",
                "scenario-cli",
                "--policy-to-beat",
                "policy-cli",
                "--policy-to-beat-source",
                "source-cli",
                "--no-render",
            ]
        )
        == 0
    )


def test_helper_branches_cover_fallbacks() -> None:
    """Small helper branches should keep terminal classification explicit."""
    assert _failure_reason({"success": True}) == "success"
    assert _failure_reason({"collision": True}) == "collision"
    assert _failure_reason({"truncated": True}) == "truncated"
    assert _failure_reason({}) == "terminated"
    assert _pygame_key_name(SimpleNamespace(key=object()), "raw-key") == "raw-key"
    assert _current_velocity(SimpleNamespace()) == (0.0, 0.0)
