"""Direct unit tests for ``robot_sf/common/matplotlib_utils.py`` backend contracts.

These tests lock the headless-detection rules, the platform-specific backend
ordering, the success short-circuit, and the unavailable-backend handling
without ever opening a GUI or requiring an installed interactive backend.

The built-in ``monkeypatch`` fixture (not ``pytest-mock``) is used for all
patching. Every test sets/restores environment variables and monkeypatches via
``monkeypatch`` teardown so no process-global backend state leaks between tests.
The real ``matplotlib.use``/``matplotlib.get_backend`` are never invoked: the
module-level ``matplotlib`` binding is replaced with a controllable mock.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from robot_sf.common import matplotlib_utils as mu


@pytest.fixture
def fake_mpl(monkeypatch):
    """Replace the module-level matplotlib binding with a controllable mock.

    Replacing ``mu.matplotlib`` (rather than patching the real matplotlib
    package) keeps these tests fully isolated: the real ``matplotlib.use`` is
    never invoked, so no real backend switch can leak across tests.
    """
    mpl = Mock()
    mpl.use = Mock(return_value=None)
    mpl.get_backend = Mock(return_value="agg")
    monkeypatch.setattr(mu, "matplotlib", mpl)
    return mpl


@pytest.fixture
def fake_logger(monkeypatch):
    """Replace the module-level loguru logger so verbose branches are observable."""
    log = Mock()
    monkeypatch.setattr(mu, "logger", log)
    return log


# ---------------------------------------------------------------------------
# is_headless_environment
# ---------------------------------------------------------------------------


def test_is_headless_explicit_mplbackend_agg(monkeypatch, fake_mpl):
    """An explicit MPLBACKEND=Agg must report headless without touching the backend."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.delenv("DISPLAY", raising=False)

    assert mu.is_headless_environment() is True
    fake_mpl.get_backend.assert_not_called()


def test_is_headless_missing_display_on_non_windows(monkeypatch, fake_mpl):
    """A missing DISPLAY on a non-Windows platform must report headless early."""
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(mu.platform, "system", lambda: "Linux")

    assert mu.is_headless_environment() is True
    fake_mpl.get_backend.assert_not_called()


def test_is_headless_when_current_backend_is_agg(monkeypatch, fake_mpl):
    """With MPLBACKEND unset and DISPLAY present, an Agg backend reports headless."""
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(mu.platform, "system", lambda: "Linux")
    fake_mpl.get_backend.return_value = "Agg"

    assert mu.is_headless_environment() is True


def test_is_headless_non_headless_environment(monkeypatch, fake_mpl):
    """DISPLAY present, MPLBACKEND unset, and a non-Agg backend is not headless."""
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(mu.platform, "system", lambda: "Linux")
    fake_mpl.get_backend.return_value = "QtAgg"

    assert mu.is_headless_environment() is False


# ---------------------------------------------------------------------------
# _get_backend_candidates
# ---------------------------------------------------------------------------


def test_get_backend_candidates_darwin(monkeypatch):
    """macOS (Darwin) prefers the native backend first."""
    monkeypatch.setattr(mu.platform, "system", lambda: "Darwin")

    assert mu._get_backend_candidates() == ["MacOSX", "QtAgg", "Qt5Agg"]


def test_get_backend_candidates_non_darwin(monkeypatch):
    """Non-Darwin platforms use the Qt/Tk/WX ordering without MacOSX."""
    monkeypatch.setattr(mu.platform, "system", lambda: "Linux")

    assert mu._get_backend_candidates() == ["QtAgg", "Qt5Agg", "TkAgg", "WXAgg"]


def test_get_backend_candidates_windows_is_non_darwin_order(monkeypatch):
    """Windows shares the non-Darwin backend ordering."""
    monkeypatch.setattr(mu.platform, "system", lambda: "Windows")

    assert mu._get_backend_candidates() == ["QtAgg", "Qt5Agg", "TkAgg", "WXAgg"]


# ---------------------------------------------------------------------------
# _try_set_backend
# ---------------------------------------------------------------------------


def test_try_set_backend_success(fake_mpl):
    """A backend that matplotlib.use accepts is reported as set, with force=True."""
    assert mu._try_set_backend("QtAgg", verbose=False) is True
    fake_mpl.use.assert_called_once_with("QtAgg", force=True)


def test_try_set_backend_success_verbose_logs(fake_mpl, fake_logger):
    """The success path emits a debug log when verbose."""
    assert mu._try_set_backend("QtAgg", verbose=True) is True
    fake_mpl.use.assert_called_once_with("QtAgg", force=True)
    fake_logger.debug.assert_called()


def test_try_set_backend_import_error_returns_false(fake_mpl):
    """An ImportError from matplotlib.use means the backend is unavailable."""
    fake_mpl.use.side_effect = ImportError("no Qt")

    assert mu._try_set_backend("QtAgg", verbose=False) is False


def test_try_set_backend_import_error_verbose_logs(fake_mpl, fake_logger):
    """The ImportError path emits a debug log when verbose."""
    fake_mpl.use.side_effect = ImportError("no Qt")

    assert mu._try_set_backend("QtAgg", verbose=True) is False
    fake_logger.debug.assert_called()


def test_try_set_backend_general_exception_returns_false(fake_mpl):
    """Any non-ImportError exception is swallowed and reported as failure."""
    fake_mpl.use.side_effect = RuntimeError("boom")

    assert mu._try_set_backend("TkAgg", verbose=False) is False


def test_try_set_backend_general_exception_verbose_logs(fake_mpl, fake_logger):
    """The general-exception path emits a debug log including the error when verbose."""
    fake_mpl.use.side_effect = RuntimeError("boom")

    assert mu._try_set_backend("TkAgg", verbose=True) is False
    fake_logger.debug.assert_called()
    args, _ = fake_logger.debug.call_args
    assert any("boom" in str(a) for a in args)


# ---------------------------------------------------------------------------
# ensure_interactive_backend
# ---------------------------------------------------------------------------


def test_ensure_interactive_mplbackend_agg_returns_false(monkeypatch, fake_mpl):
    """MPLBACKEND=Agg requests headless mode; no backend probing must occur."""
    monkeypatch.setenv("MPLBACKEND", "Agg")

    assert mu.ensure_interactive_backend() is False
    fake_mpl.get_backend.assert_not_called()
    fake_mpl.use.assert_not_called()


def test_ensure_interactive_mplbackend_agg_verbose_logs(monkeypatch, fake_mpl, fake_logger):
    """The MPLBACKEND=Agg early return logs a debug message when verbose."""
    monkeypatch.setenv("MPLBACKEND", "Agg")

    assert mu.ensure_interactive_backend(verbose=True) is False
    fake_logger.debug.assert_called()


def test_ensure_interactive_already_interactive_returns_true(monkeypatch, fake_mpl):
    """An already-interactive current backend is kept without trying others."""
    monkeypatch.delenv("MPLBACKEND", raising=False)
    fake_mpl.get_backend.return_value = "Qt5Agg"

    assert mu.ensure_interactive_backend() is True
    fake_mpl.use.assert_not_called()


def test_ensure_interactive_already_interactive_verbose_logs(monkeypatch, fake_mpl, fake_logger):
    """The already-interactive path logs a debug message when verbose."""
    monkeypatch.delenv("MPLBACKEND", raising=False)
    fake_mpl.get_backend.return_value = "Qt5Agg"

    assert mu.ensure_interactive_backend(verbose=True) is True
    fake_logger.debug.assert_called()


def test_ensure_interactive_first_success_short_circuits(monkeypatch, fake_mpl):
    """The first successful candidate wins; later candidates are not tried."""
    monkeypatch.delenv("MPLBACKEND", raising=False)
    fake_mpl.get_backend.return_value = "agg"  # non-interactive -> enter candidate loop
    monkeypatch.setattr(mu, "_get_backend_candidates", lambda: ["First", "Second", "Third"])

    assert mu.ensure_interactive_backend() is True
    fake_mpl.use.assert_called_once_with("First", force=True)


def test_ensure_interactive_pdf_current_falls_through_to_candidates(monkeypatch, fake_mpl):
    """pdf (like agg/ps/svg) is not interactive, so candidates are still tried."""
    monkeypatch.delenv("MPLBACKEND", raising=False)
    fake_mpl.get_backend.return_value = "pdf"
    monkeypatch.setattr(mu, "_get_backend_candidates", lambda: ["OnlyOne"])

    assert mu.ensure_interactive_backend() is True
    fake_mpl.use.assert_called_once_with("OnlyOne", force=True)


def test_ensure_interactive_all_candidates_fail_returns_false(monkeypatch, fake_mpl):
    """When every candidate raises ImportError, no backend is set and False is returned."""
    monkeypatch.delenv("MPLBACKEND", raising=False)
    fake_mpl.get_backend.return_value = "agg"
    monkeypatch.setattr(mu, "_get_backend_candidates", lambda: ["A", "B", "C"])
    fake_mpl.use.side_effect = ImportError("missing")

    assert mu.ensure_interactive_backend() is False
    assert fake_mpl.use.call_count == 3


def test_ensure_interactive_verbose_warns_when_all_candidates_fail(
    monkeypatch, fake_mpl, fake_logger
):
    """The all-fail path emits a warning when verbose."""
    monkeypatch.delenv("MPLBACKEND", raising=False)
    fake_mpl.get_backend.return_value = "agg"
    monkeypatch.setattr(mu, "_get_backend_candidates", lambda: ["A"])
    fake_mpl.use.side_effect = ImportError("missing")

    assert mu.ensure_interactive_backend(verbose=True) is False
    fake_logger.warning.assert_called()
