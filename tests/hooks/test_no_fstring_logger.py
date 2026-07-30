"""Contract tests for the no-fstring-logger pre-commit hook (issue #6468).

The hook AST-scans ``logger.<method>(f"...")`` calls because ruff rule G004
cannot see ``loguru``. These tests lock the detector behavior (including the
allowlist ratchet) and prove it flags a deliberately injected f-string.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_HOOK_PATH = Path(__file__).resolve().parents[2] / "hooks" / "no_fstring_logger.py"
_SPEC = importlib.util.spec_from_file_location("no_fstring_logger", _HOOK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HOOK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HOOK)

find_violations = _HOOK.find_fstring_logger_violations
Violation = _HOOK.Violation


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_flags_injected_fstring_logger_call(tmp_path: Path) -> None:
    """The core negative control: an f-string logger call is detected."""
    source = "from loguru import logger\n\n\ndef f(x):\n    logger.info(f'value is {x}')\n"
    violations = find_violations(source, str(_write(tmp_path, "bad.py", source)))
    assert violations == [Violation(lineno=5, method="info", preview="f'value is {x}'")]


def test_structured_style_is_not_flagged() -> None:
    """The structured {key}+kwargs idiom is accepted (no false positive)."""
    source = "from loguru import logger\n\nlogger.info('Loaded {n} map(s) from {d}', n=3, d='/x')\n"
    assert find_violations(source, "<ok>") == []


@pytest.mark.parametrize(
    "call",
    [
        'logger.info("plain literal {not_interp}")',  # literal braces, not an f-string
        'logger.info("value is %s", x)',  # %-style
        'logger.info("value is {}", x)',  # positional {} -style
        'logger.debug("")',  # empty literal
    ],
)
def test_non_fstring_messages_are_not_flagged(call: str) -> None:
    """Only f-string (JoinedStr) message arguments are violations."""
    source = f"from loguru import logger\n\n{call}\n"
    assert find_violations(source, "<ok>") == []


def test_all_loguru_methods_detected(tmp_path: Path) -> None:
    """Every loguru level method is covered, not just info/warning/error."""
    source = (
        "from loguru import logger\n"
        "logger.trace(f't {1}')\n"
        "logger.debug(f'd {1}')\n"
        "logger.info(f'i {1}')\n"
        "logger.success(f's {1}')\n"
        "logger.warning(f'w {1}')\n"
        "logger.error(f'e {1}')\n"
        "logger.critical(f'c {1}')\n"
        "logger.exception(f'ex {1}')\n"
    )
    methods = {v.method for v in find_violations(source, "<all>")}
    assert methods == {
        "trace",
        "debug",
        "info",
        "success",
        "warning",
        "error",
        "critical",
        "exception",
    }


def test_log_method_message_is_second_argument() -> None:
    """logger.log(level, msg) takes the level first, so only msg is checked."""
    source = (
        "from loguru import logger\n"
        "import logging\n"
        "logger.log(logging.WARNING, f'msg {1}')\n"  # f-string message -> violation
        "logger.log(logging.WARNING, 'plain')\n"  # literal -> ok
    )
    violations = find_violations(source, "<log>")
    assert len(violations) == 1
    assert violations[0].method == "log"


def test_logger_chain_calls_detected() -> None:
    """logger.opt(...)/logger.bind(...).<method>(f'...') is also flagged."""
    source = (
        "from loguru import logger\n"
        "logger.opt(exception=True).warning(f'chain {1}')\n"
        "logger.bind(k=1).error(f'bound {1}')\n"
    )
    methods = sorted(v.method for v in find_violations(source, "<chain>"))
    assert methods == ["error", "warning"]


def test_argparse_and_non_logger_receivers_not_flagged() -> None:
    """parser.error(f'...') and unrelated obj.info(f'...') are not logger calls."""
    source = (
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.error(f'bad {1}')\n"
        "obj.info(f'not logger {1}')\n"
    )
    assert find_violations(source, "<nope>") == []


def _allowlist_file(tmp_path: Path, entries: list[str]) -> Path:
    allow = tmp_path / "allow.txt"
    allow.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return allow


def test_main_blocks_non_allowlisted_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-allowlisted file with an f-string logger call fails the hook."""
    monkeypatch.chdir(tmp_path)
    bad = _write(tmp_path, "robot_sf/mod.py", "from loguru import logger\nlogger.info(f'x {1}')\n")
    allow = _allowlist_file(tmp_path, [])
    assert _HOOK.main([str(bad), "--allowlist", str(allow)]) == 1


def test_main_allows_grandfathered_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An allowlisted file is permitted (ratchet), so the hook exits 0."""
    monkeypatch.chdir(tmp_path)
    bad = _write(tmp_path, "robot_sf/mod.py", "from loguru import logger\nlogger.info(f'x {1}')\n")
    allow = _allowlist_file(tmp_path, ["robot_sf/mod.py"])
    assert _HOOK.main([str(bad), "--allowlist", str(allow)]) == 0


def test_main_clean_file_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean structured-style file passes regardless of allowlist."""
    monkeypatch.chdir(tmp_path)
    good = _write(
        tmp_path,
        "robot_sf/mod.py",
        "from loguru import logger\nlogger.info('ok {n}', n=1)\n",
    )
    allow = _allowlist_file(tmp_path, [])
    assert _HOOK.main([str(good), "--allowlist", str(allow)]) == 0


def test_repo_allowlist_is_superset_of_files_with_violations() -> None:
    """The shipped allowlist must cover every robot_sf file that still has violations.

    A regression guard that is not green is useless: if a file gains an
    f-string logger call it must either be migrated or listed here.
    """
    repo_root = Path(__file__).resolve().parents[2]
    allow_path = repo_root / "hooks" / _HOOK.ALLOWLIST_FILENAME
    allowlisted = {line.strip() for line in allow_path.read_text("utf-8").splitlines()}
    allowlisted = {x for x in allowlisted if x and not x.startswith("#")}
    uncovered: list[str] = []
    for py in sorted((repo_root / "robot_sf").rglob("*.py")):
        rel = py.relative_to(repo_root).as_posix()
        if rel in allowlisted:
            continue  # grandfathered; whether or not it currently has calls is fine
        src = py.read_text(encoding="utf-8")
        if find_violations(src, rel):
            uncovered.append(rel)
    assert not uncovered, (
        "Files with f-string logger calls are neither migrated nor allowlisted: "
        f"{uncovered}. Either migrate them or add them to {allow_path.as_posix()}."
    )


def test_hot_path_files_not_in_allowlist() -> None:
    """The migrated hot-path files must NOT be allowlisted (guard enforces them)."""
    repo_root = Path(__file__).resolve().parents[2]
    allow_path = repo_root / "hooks" / _HOOK.ALLOWLIST_FILENAME
    allowlisted = {line.strip() for line in allow_path.read_text("utf-8").splitlines()}
    allowlisted = {x for x in allowlisted if x and not x.startswith("#")}
    hot_paths = {
        "robot_sf/sim/simulator.py",
        "robot_sf/gym_env/base_env.py",
        "robot_sf/gym_env/pedestrian_env.py",
    }
    assert not (hot_paths & allowlisted), (
        f"migrated hot-path files must be absent from allowlist: {hot_paths & allowlisted}"
    )
