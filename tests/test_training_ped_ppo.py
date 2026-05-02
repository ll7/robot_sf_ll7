"""Tests for the legacy pedestrian PPO training script."""

from __future__ import annotations

from types import SimpleNamespace

from scripts import training_ped_ppo as mod


def test_training_closes_vectorized_env_after_success(monkeypatch) -> None:
    """The SubprocVecEnv wrapper should always be closed after training completes."""
    close_calls: list[str] = []
    learn_calls: list[int] = []
    save_paths: list[str] = []

    class _EnvStub:
        def close(self) -> None:
            close_calls.append("closed")

    class _CallbackListStub(tuple):
        def __new__(cls, callbacks):
            return super().__new__(cls, callbacks)

    class _ModelStub:
        @staticmethod
        def load(*_args, **_kwargs):
            return "robot-model"

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def learn(self, *, total_timesteps: int, progress_bar: bool, callback) -> None:
            assert progress_bar is True
            assert callback is not None
            learn_calls.append(total_timesteps)

        def save(self, path: str) -> None:
            save_paths.append(path)

    monkeypatch.setattr(mod, "convert_map", lambda _path: "map-def")
    monkeypatch.setattr(mod, "make_pedestrian_env", lambda **_kwargs: "ped-env")
    monkeypatch.setattr(mod, "make_vec_env", lambda *args, **kwargs: _EnvStub())
    monkeypatch.setattr(mod, "PPO", _ModelStub)
    monkeypatch.setattr(mod, "CheckpointCallback", lambda *args, **kwargs: "checkpoint")
    monkeypatch.setattr(
        mod,
        "AdversialPedestrianMetricsCallback",
        lambda *args, **kwargs: "metrics",
    )
    monkeypatch.setattr(mod, "CallbackList", _CallbackListStub)
    monkeypatch.setattr(
        mod.datetime,
        "datetime",
        type(
            "_DateTimeStub",
            (),
            {
                "now": staticmethod(
                    lambda: SimpleNamespace(strftime=lambda _fmt: "2026-04-01_12-00-00")
                )
            },
        ),
    )

    mod.training("fake-map.svg")

    assert learn_calls == [1_500_000]
    assert save_paths == ["./model_ped/ppo_2026-04-01_12-00-00"]
    assert close_calls == ["closed"]
