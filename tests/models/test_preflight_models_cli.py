"""Exit-code contract tests for the model-preflight command."""

from __future__ import annotations

from scripts.models import preflight_models as cli


def test_main_returns_ok_after_preflight(monkeypatch, tmp_path) -> None:
    """A resolved asset produces the command's success exit code."""
    monkeypatch.setattr(cli, "_config_model_ids", lambda paths: ["model-a"])
    monkeypatch.setattr(cli, "preflight_models", lambda *args, **kwargs: {"model-a": tmp_path})

    assert cli.main(["--config", "config.yaml"]) == cli.EXIT_OK


def test_main_returns_config_error_for_missing_model_ids() -> None:
    """An invocation without model ids fails before attempting preflight."""
    assert cli.main([]) == cli.EXIT_CONFIG_ERROR


def test_main_returns_config_error_when_config_loading_fails(monkeypatch) -> None:
    """Unreadable config paths retain the distinct configuration exit code."""
    monkeypatch.setattr(
        cli, "_config_model_ids", lambda paths: (_ for _ in ()).throw(FileNotFoundError())
    )

    assert cli.main(["--config", "missing.yaml"]) == cli.EXIT_CONFIG_ERROR


def test_main_returns_blocked_when_preflight_cannot_stage_assets(monkeypatch) -> None:
    """A bounded preflight failure is distinguishable from bad CLI input."""
    monkeypatch.setattr(
        cli,
        "preflight_models",
        lambda *args, **kwargs: (_ for _ in ()).throw(cli.ModelPreflightError("unavailable")),
    )

    assert cli.main(["model-a"]) == cli.EXIT_BLOCKED
