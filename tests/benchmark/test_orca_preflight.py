"""Tests for the ORCA-rvo2 preflight guard."""

from __future__ import annotations

import builtins
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from robot_sf.benchmark.camera_ready_campaign import CampaignConfig, PlannerSpec, SeedPolicy
from robot_sf.benchmark.orca_preflight import (
    OrcaRvo2PreflightError,
    _has_orca_algo,
    _is_orca_dependent_planner,
    check_orca_rvo2_preflight,
    check_orca_rvo2_preflight_from_config,
    check_rvo2_importable,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def _rvo2_missing() -> Generator[None, None, None]:
    """Context manager that temporarily removes rvo2 from sys.modules."""
    saved = sys.modules.get("rvo2")
    sys.modules["rvo2"] = None
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("rvo2", None)
        else:
            sys.modules["rvo2"] = saved


@contextmanager
def _rvo2_import_raises(error: Exception) -> Generator[None, None, None]:
    """Context manager that makes rvo2 import raise a specific exception."""
    original_import = builtins.__import__

    def _blocked_import(
        name: str,
        globals_: object = None,
        locals_: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "rvo2" or name.startswith("rvo2."):
            raise error
        return original_import(name, globals_, locals_, fromlist, level)

    with patch("builtins.__import__", _blocked_import):
        yield


def _make_cfg(
    *,
    planners: tuple[PlannerSpec, ...],
) -> CampaignConfig:
    """Build a minimal CampaignConfig with just planners."""
    return CampaignConfig(
        name="test",
        scenario_matrix_path=Path("/fake/scenarios.yaml"),
        planners=planners,
        seed_policy=SeedPolicy(),
    )


def _planner(key: str, algo: str, *, enabled: bool = True) -> PlannerSpec:
    """Shorthand for a PlannerSpec with sensible defaults."""
    return PlannerSpec(key=key, algo=algo, enabled=enabled)


def _write_config(path: Path, *, algo: str, key: str | None = None) -> None:
    """Write a minimal campaign config with a resolvable scenario matrix path."""
    (path.parent / "scenarios.yaml").write_text("scenarios: []\n", encoding="utf-8")
    path.write_text(
        "name: test\n"
        "scenario_matrix: scenarios.yaml\n"
        "planners:\n"
        f"  - key: {key or algo}\n"
        f"    algo: {algo}\n",
        encoding="utf-8",
    )


class TestHasOrcaAlgo:
    """Unit tests for _has_orca_algo."""

    def test_exact_match(self) -> None:
        """Exact algo string 'orca' is detected."""
        assert _has_orca_algo("orca")

    def test_contains_orca(self) -> None:
        """Algo string containing 'orca' as substring is detected."""
        assert _has_orca_algo("social_navigation_pyenvs_orca")

    def test_case_insensitive(self) -> None:
        """Detection is case-insensitive."""
        assert _has_orca_algo("ORCA")

    def test_non_orca(self) -> None:
        """Non-ORCA algo does not match."""
        assert not _has_orca_algo("ppo")

    def test_not_contained(self) -> None:
        """Algo without 'orca' substring does not match."""
        assert not _has_orca_algo("social_force")


class TestIsOrcaDependentPlanner:
    """Unit tests for _is_orca_dependent_planner."""

    def test_orca_enabled(self) -> None:
        """Enabled ORCA planner spec is detected."""
        assert _is_orca_dependent_planner(_planner("orca", "orca", enabled=True))

    def test_orca_disabled(self) -> None:
        """Disabled ORCA planner spec is not detected."""
        assert not _is_orca_dependent_planner(_planner("orca", "orca", enabled=False))

    def test_non_orca(self) -> None:
        """Non-ORCA planner spec is not detected."""
        assert not _is_orca_dependent_planner(_planner("ppo", "ppo"))

    def test_pyenvs_orca_enabled(self) -> None:
        """Pyenvs ORCA planner with enabled=True is detected."""
        assert _is_orca_dependent_planner(_planner("social_nav", "social_navigation_pyenvs_orca"))

    def test_missing_algo(self) -> None:
        """Planner spec without an algo field is not detected as ORCA."""
        spec = PlannerSpec(key="bad", algo="")  # type: ignore[arg-type]
        assert not _is_orca_dependent_planner(spec)


class TestCheckRvo2Importable:
    """Tests for check_rvo2_importable standalone function."""

    def test_passes_when_rvo2_available(self) -> None:
        """Function does not raise when rvo2 is in sys.modules."""
        with patch.dict(sys.modules, {"rvo2": True}):
            check_rvo2_importable()

    def test_raises_when_rvo2_missing(self) -> None:
        """Function raises a typed preflight error when rvo2 is not importable."""
        with _rvo2_missing():
            with pytest.raises(OrcaRvo2PreflightError) as exc_info:
                check_rvo2_importable()
            assert "rvo2" in str(exc_info.value).lower()
            assert "uv sync --extra orca" in str(exc_info.value)
            assert "uv sync --all-extras" in str(exc_info.value)

    def test_raises_when_native_extension_import_fails(self) -> None:
        """Native-extension import failures are reported with actionable sync commands."""
        with _rvo2_import_raises(OSError("libRVO.so: cannot open shared object file")):
            with pytest.raises(OrcaRvo2PreflightError) as exc_info:
                check_rvo2_importable()
        message = str(exc_info.value)
        assert "rvo2" in message.lower()
        assert "OSError" in message
        assert "uv sync --extra orca" in message
        assert "uv sync --all-extras" in message


class TestCheckOrcaRvo2Preflight:
    """Tests for the campaign-level preflight guard."""

    def test_no_orca_planners_passes(self) -> None:
        """Preflight passes when config has no ORCA planners."""
        cfg = _make_cfg(planners=(_planner("ppo", "ppo"), _planner("goal", "goal")))
        check_orca_rvo2_preflight(cfg)

    def test_orca_planner_with_rvo2_available_passes(self) -> None:
        """Preflight passes when ORCA planners exist and rvo2 is available."""
        cfg = _make_cfg(planners=(_planner("orca", "orca"),))
        with patch.dict(sys.modules, {"rvo2": True}):
            check_orca_rvo2_preflight(cfg)

    def test_orca_planner_with_rvo2_missing_fails(self) -> None:
        """Preflight raises a typed error when ORCA planners exist and rvo2 is missing."""
        cfg = _make_cfg(planners=(_planner("orca", "orca"),))
        with _rvo2_missing():
            with pytest.raises(OrcaRvo2PreflightError) as exc_info:
                check_orca_rvo2_preflight(cfg)
            assert "rvo2" in str(exc_info.value).lower()

    def test_disabled_orca_does_not_fail(self) -> None:
        """Disabled ORCA planners are skipped and the preflight passes."""
        cfg = _make_cfg(planners=(_planner("orca", "orca", enabled=False),))
        check_orca_rvo2_preflight(cfg)

    def test_mixed_orca_and_non_orca_with_rvo2_missing_fails(self) -> None:
        """Preflight fails when any enabled ORCA planner exists with rvo2 missing."""
        cfg = _make_cfg(
            planners=(
                _planner("ppo", "ppo"),
                _planner("orca", "orca"),
                _planner("goal", "goal"),
            )
        )
        with _rvo2_missing():
            with pytest.raises(OrcaRvo2PreflightError) as exc_info:
                check_orca_rvo2_preflight(cfg)
            assert "rvo2" in str(exc_info.value).lower()

    def test_pyenvs_orca_with_rvo2_missing_fails(self) -> None:
        """Preflight fails for social_navigation_pyenvs_orca when rvo2 is missing."""
        cfg = _make_cfg(planners=(_planner("social_nav_orca", "social_navigation_pyenvs_orca"),))
        with _rvo2_missing():
            with pytest.raises(OrcaRvo2PreflightError) as exc_info:
                check_orca_rvo2_preflight(cfg)
            message = str(exc_info.value)
            assert "social_nav_orca" in message
            assert "rvo2" in message.lower()

    def test_error_message_names_orca_keys_and_sync_commands(self) -> None:
        """Error message contains ORCA planner keys and both sync commands."""
        cfg = _make_cfg(
            planners=(
                _planner("orca", "orca"),
                _planner("social_nav", "social_navigation_pyenvs_orca"),
            )
        )
        with _rvo2_missing():
            with pytest.raises(OrcaRvo2PreflightError) as exc_info:
                check_orca_rvo2_preflight(cfg)
        message = str(exc_info.value)
        assert "orca" in message
        assert "social_nav" in message
        assert "rvo2" in message.lower()
        assert "uv sync --extra orca" in message
        assert "uv sync --all-extras" in message

    def test_orca_hybrid_sampler_detected(self) -> None:
        """hybrid_orca_sampler is detected as ORCA-dependent."""
        cfg = _make_cfg(planners=(_planner("hybrid_orca", "hybrid_orca_sampler"),))
        with _rvo2_missing():
            with pytest.raises(OrcaRvo2PreflightError):
                check_orca_rvo2_preflight(cfg)

    def test_orca_native_extension_import_failure_names_sync_commands(self) -> None:
        """Campaign preflight treats native rvo2 load errors as actionable dependency failures."""
        cfg = _make_cfg(planners=(_planner("orca", "orca"),))
        with _rvo2_import_raises(OSError("libRVO.so: cannot open shared object file")):
            with pytest.raises(OrcaRvo2PreflightError) as exc_info:
                check_orca_rvo2_preflight(cfg)
        message = str(exc_info.value)
        assert "orca" in message
        assert "OSError" in message
        assert "uv sync --extra orca" in message
        assert "uv sync --all-extras" in message


class TestCheckOrcaRvo2PreflightFromConfig:
    """Tests for the config-path-based entry point."""

    def test_loads_config_and_checks(self, tmp_path: Path) -> None:
        """Config with ORCA planner triggers a typed preflight failure when rvo2 is missing."""
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, algo="orca")
        with _rvo2_missing():
            with pytest.raises(OrcaRvo2PreflightError) as exc_info:
                check_orca_rvo2_preflight_from_config(config_path)
            assert "rvo2" in str(exc_info.value).lower()

    def test_non_orca_config_passes(self, tmp_path: Path) -> None:
        """Config without ORCA planners passes preflight."""
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, algo="ppo")
        check_orca_rvo2_preflight_from_config(config_path)
