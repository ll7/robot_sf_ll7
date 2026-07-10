"""Tests for the VecEnv worker-mode throughput comparator (issue #5118)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from gymnasium import Env, spaces

_REPO_ROOT = Path(__file__).parent.parent.parent.resolve()

# scripts/ is not a package; insert repo root so the script is importable.
sys.path.insert(0, str(_REPO_ROOT))

from scripts.validation.run_vecenv_worker_mode_throughput import (  # noqa: E402
    _build_env_fns,
    _build_vec_env,
    _env_factory_kwargs,
    _EnvFactory,
    _resolve_num_envs,
    _sha256_file,
    main,
)

_SMOKE_CONFIG = _REPO_ROOT / "configs/training/lidar/lidar_ppo_mlp_smoke_issue_1662.yaml"


# ---------------------------------------------------------------------------
# Unit tests for config helpers
# ---------------------------------------------------------------------------


class TestResolveNumEnvs:
    """Tests for _resolve_num_envs helper."""

    def test_override_wins(self):
        assert _resolve_num_envs({}, 7) == 7

    def test_config_fallback(self):
        assert _resolve_num_envs({"num_envs": 3}, None) == 3

    def test_auto_string_returns_default(self):
        assert _resolve_num_envs({"num_envs": "auto"}, None) == 4

    def test_missing_returns_default(self):
        assert _resolve_num_envs({}, None) == 4

    def test_nonpositive_override_is_rejected(self):
        """A zero environment count is invalid, not a request for one environment."""
        with pytest.raises(ValueError, match="positive integer"):
            _resolve_num_envs({}, 0)


class TestEnvFactoryKwargs:
    """Tests for _env_factory_kwargs helper."""

    def test_empty_config(self):
        assert _env_factory_kwargs({}) == {}

    def test_passthrough(self):
        cfg = {"env_factory_kwargs": {"reward_name": "route_completion_v2"}}
        kw = _env_factory_kwargs(cfg)
        assert kw["reward_name"] == "route_completion_v2"

    def test_none_value_returns_empty(self):
        assert _env_factory_kwargs({"env_factory_kwargs": None}) == {}


class TestSha256File:
    """Tests for _sha256_file helper."""

    def test_deterministic(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_bytes(b"hello")
        assert _sha256_file(f) == _sha256_file(f)

    def test_different_content_differs(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_bytes(b"x")
        b.write_bytes(b"y")
        assert _sha256_file(a) != _sha256_file(b)


# ---------------------------------------------------------------------------
# Unit tests for _EnvFactory picklability
# ---------------------------------------------------------------------------


class TestEnvFactory:
    """Tests for _EnvFactory picklability contract."""

    def test_picklable(self, tmp_path):
        import pickle

        f = _EnvFactory(
            scenario_path=tmp_path / "scenario.yaml",
            env_factory_kwargs={},
            seed=1,
        )
        dumped = pickle.dumps(f)
        restored = pickle.loads(dumped)
        assert restored.seed == 1
        assert restored.scenario_path == tmp_path / "scenario.yaml"


# ---------------------------------------------------------------------------
# Integration contract: _build_env_fns length
# ---------------------------------------------------------------------------


def test_build_env_fns_length():
    """_build_env_fns returns the requested factory count with consecutive seeds."""
    scenario_path = Path("/tmp/scenario.yaml")
    env_fns = _build_env_fns(scenario_path, {}, num_envs=3, base_seed=10)
    assert len(env_fns) == 3
    assert env_fns[0].seed == 10
    assert env_fns[2].seed == 12


class _SyntheticEnv(Env):
    """Deterministic environment for worker-mode transition equivalence tests."""

    metadata = {"render_modes": []}

    def __init__(self, offset: float) -> None:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        self._offset = offset

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.array([self._offset], dtype=np.float32), {}

    def step(self, action: np.ndarray):
        observation = np.asarray(action, dtype=np.float32) + self._offset
        return observation, float(observation[0]), False, False, {}


def test_threaded_lidar_batch_matches_threaded_synthetic_transitions():
    """The comparator's batch mode preserves synthetic threaded transitions exactly."""
    env_fns = [lambda: _SyntheticEnv(0.0), lambda: _SyntheticEnv(1.0)]
    actions = np.array([[0.25], [0.75]], dtype=np.float32)
    threaded = _build_vec_env("threaded", env_fns)
    batched = _build_vec_env("threaded_lidar_batch", env_fns)
    try:
        np.testing.assert_array_equal(batched.reset(), threaded.reset())
        batched_transition = batched.step(actions)
        threaded_transition = threaded.step(actions)
    finally:
        batched.close()
        threaded.close()

    for batched_value, threaded_value in zip(batched_transition, threaded_transition, strict=True):
        if isinstance(batched_value, np.ndarray):
            np.testing.assert_array_equal(batched_value, threaded_value)
        else:
            assert batched_value == threaded_value


# ---------------------------------------------------------------------------
# Smoke test: main() with a fake DummyVecEnv (no real env construction)
# ---------------------------------------------------------------------------


class _FakeActionSpace:
    def sample(self):
        return np.zeros(2, dtype=np.float32)


class _FakeVecEnv:
    """Minimal VecEnv double for CLI smoke tests."""

    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.action_space = _FakeActionSpace()
        self.closed = False

    def reset(self):
        return np.zeros((self.num_envs, 1))

    def step(self, actions):
        obs = np.zeros((self.num_envs, 1))
        rewards = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        return obs, rewards, dones, infos

    def close(self):
        self.closed = True


def _patch_build_vec_env(mode: str, env_fns):
    return _FakeVecEnv(env_fns)


@pytest.mark.skipif(
    not _SMOKE_CONFIG.exists(),
    reason="smoke config not found",
)
def test_main_dummy_mode_writes_json(tmp_path):
    """main() with fake DummyVecEnv writes valid JSON output."""
    out = tmp_path / "result.json"
    with patch(
        "scripts.validation.run_vecenv_worker_mode_throughput._build_vec_env",
        side_effect=_patch_build_vec_env,
    ):
        rc = main(
            [
                "--config",
                str(_SMOKE_CONFIG),
                "--modes",
                "dummy",
                "--num-envs",
                "2",
                "--warmup-steps",
                "2",
                "--measure-steps",
                "5",
                "--output",
                str(out),
            ]
        )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["schema"] == "vecenv_throughput_comparator.v1"
    assert data["num_envs"] == 2
    assert len(data["results"]) == 1
    rec = data["results"][0]
    assert rec["mode"] == "dummy"
    assert rec["status"] == "ok"
    assert rec["transitions_per_second"] > 0
    assert rec["speedup_vs_baseline"] == pytest.approx(1.0)


@pytest.mark.skipif(
    not _SMOKE_CONFIG.exists(),
    reason="smoke config not found",
)
def test_main_threaded_mode_writes_json(tmp_path):
    """main() measures dummy and threaded, both succeed."""
    out = tmp_path / "result.json"
    with patch(
        "scripts.validation.run_vecenv_worker_mode_throughput._build_vec_env",
        side_effect=_patch_build_vec_env,
    ):
        rc = main(
            [
                "--config",
                str(_SMOKE_CONFIG),
                "--modes",
                "dummy",
                "threaded",
                "--num-envs",
                "2",
                "--warmup-steps",
                "2",
                "--measure-steps",
                "5",
                "--output",
                str(out),
            ]
        )
    assert rc == 0
    data = json.loads(out.read_text())
    modes = {r["mode"] for r in data["results"]}
    assert modes == {"dummy", "threaded"}
    for rec in data["results"]:
        assert rec["status"] == "ok"


@pytest.mark.skipif(
    not _SMOKE_CONFIG.exists(),
    reason="smoke config not found",
)
def test_main_threaded_lidar_batch_mode_writes_json(tmp_path):
    """The comparator accepts the batched LiDAR worker mode and records it unchanged."""
    out = tmp_path / "result.json"
    with patch(
        "scripts.validation.run_vecenv_worker_mode_throughput._build_vec_env",
        side_effect=_patch_build_vec_env,
    ):
        rc = main(
            [
                "--config",
                str(_SMOKE_CONFIG),
                "--modes",
                "dummy",
                "threaded_lidar_batch",
                "--num-envs",
                "2",
                "--warmup-steps",
                "2",
                "--measure-steps",
                "5",
                "--output",
                str(out),
            ]
        )
    assert rc == 0
    data = json.loads(out.read_text())
    assert [record["mode"] for record in data["results"]] == [
        "dummy",
        "threaded_lidar_batch",
    ]
    assert all(record["status"] == "ok" for record in data["results"])


@pytest.mark.skipif(
    not _SMOKE_CONFIG.exists(),
    reason="smoke config not found",
)
def test_main_construction_failure_reflected_in_output(tmp_path):
    """Construction failure status is recorded; main returns 2."""
    out = tmp_path / "result.json"

    def _fail_build(mode, env_fns):
        raise RuntimeError("simulated env build failure")

    with patch(
        "scripts.validation.run_vecenv_worker_mode_throughput._build_vec_env",
        side_effect=_fail_build,
    ):
        rc = main(
            [
                "--config",
                str(_SMOKE_CONFIG),
                "--modes",
                "dummy",
                "--num-envs",
                "2",
                "--warmup-steps",
                "1",
                "--measure-steps",
                "2",
                "--output",
                str(out),
            ]
        )
    assert rc == 2
    data = json.loads(out.read_text())
    rec = data["results"][0]
    assert rec["status"] == "construction_failed"
    assert "simulated" in rec["error"]


def test_main_missing_config_returns_1(tmp_path):
    """main() returns 1 for a missing config path (no env construction attempted)."""
    out = tmp_path / "result.json"
    rc = main(["--config", str(tmp_path / "nonexistent.yaml"), "--output", str(out)])
    assert rc == 1


def test_main_missing_scenario_config_returns_1(tmp_path, capsys):
    """A config without its required scenario input must fail before env construction."""
    config = tmp_path / "missing_scenario_config.yaml"
    config.write_text("num_envs: 2\n", encoding="utf-8")

    rc = main(["--config", str(config), "--output", str(tmp_path / "result.json")])

    assert rc == 1
    assert "scenario_config must be a non-empty path string" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value", "expected"),
    [
        ("--num-envs", "0", "num_envs must be a positive integer"),
        ("--modes", "threaded", "--modes must begin with dummy"),
        ("--warmup-steps", "-1", "--warmup-steps must be >= 0"),
        ("--measure-steps", "0", "--measure-steps must be >= 1"),
    ],
)
def test_main_rejects_invalid_measurement_parameters(tmp_path, capsys, flag, value, expected):
    """Invalid inputs must fail instead of being silently normalized into a measurement."""
    rc = main(
        [
            "--config",
            str(_SMOKE_CONFIG),
            flag,
            value,
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert rc == 1
    assert expected in capsys.readouterr().err
