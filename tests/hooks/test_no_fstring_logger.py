"""Contract tests for the no-fstring-logger pre-commit hook (issue #6468).

The hook AST-scans ``logger.<method>(f"...")`` calls because ruff rule G004
cannot see ``loguru``. These tests lock the detector behavior (including the
allowlist ratchet) and prove it flags a deliberately injected f-string.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

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
    assert len(violations) == 1
    assert violations[0].lineno == 5
    assert violations[0].method == "info"
    assert violations[0].preview == "f'value is {x}'"
    assert violations[0].scope == "f"
    assert len(violations[0].fingerprint) == _HOOK.FINGERPRINT_LENGTH


@pytest.mark.parametrize(
    ("source", "expected_repr", "expected_fingerprint"),
    [
        # Version-stable canonical f-string AST serialization (issue #6566).
        # ``ast.dump`` diverges across interpreters: 3.13 omits empty fields such
        # as ``keywords=[]`` on Call nodes, while 3.11 still emits the removed
        # ``Constant.kind`` field. These pinned bytes are the canonical form
        # produced identically on Python 3.11, 3.12, and 3.13; reverting the
        # fingerprint to ``ast.dump`` would change them on 3.11/3.12.
        (
            'f"value is {x}"',
            "JoinedStr(values=[Constant(value='value is '), "
            "FormattedValue(value=Name(id='x', ctx=Load()), conversion=-1)])",
            "84844c275ea48afc",
        ),
        (
            'f"len is {len(items)}"',
            "JoinedStr(values=[Constant(value='len is '), "
            "FormattedValue(value=Call(func=Name(id='len', ctx=Load()), "
            "args=[Name(id='items', ctx=Load())]), conversion=-1)])",
            "74505df916c2050d",
        ),
        (
            'f"dump {json.dumps(x, indent=2)}"',
            "JoinedStr(values=[Constant(value='dump '), "
            "FormattedValue(value=Call(func=Attribute(value=Name(id='json', "
            "ctx=Load()), attr='dumps', ctx=Load()), "
            "args=[Name(id='x', ctx=Load())], "
            "keywords=[keyword(arg='indent', value=Constant(value=2))]), conversion=-1)])",
            "00556df024a1897a",
        ),
    ],
)
def test_fstring_fingerprint_is_version_stable(
    source: str, expected_repr: str, expected_fingerprint: str
) -> None:
    """The f-string fingerprint must not depend on the running interpreter.

    A ``Call`` with no keyword arguments is the exact case where ``ast.dump``
    diverges (3.13 omits ``keywords=[]``), so it is pinned explicitly. These
    canonical bytes are identical on Python 3.11, 3.12, and 3.13.
    """
    message = ast.parse(source, mode="eval").body
    assert isinstance(message, ast.JoinedStr)
    assert _HOOK._canonical_ast_repr(message) == expected_repr
    assert _HOOK._message_fingerprint(message) == expected_fingerprint


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


def test_supported_loguru_binding_shapes_detected() -> None:
    """Imported aliases, module access, assignments, and bound loggers are detected."""
    source = (
        "import loguru as lu\n"
        "from loguru import logger as audit_logger\n"
        "from robot_sf.common.logging import get_logger\n"
        "logger = lu.logger\n"
        "bound_logger = get_logger(__name__)\n"
        "lu.logger.info(f'module {1}')\n"
        "audit_logger.info(f'alias {1}')\n"
        "logger.info(f'assigned {1}')\n"
        "bound_logger.info(f'bound {1}')\n"
    )
    assert len(find_violations(source, "<bindings>")) == 4


def test_non_loguru_receivers_and_arbitrary_chains_not_flagged() -> None:
    """Only proven Loguru roots and supported chain methods are inspected."""
    source = (
        "import argparse\n"
        "import logging\n"
        "from loguru import logger\n"
        "parser = argparse.ArgumentParser()\n"
        "stdlib_logger = logging.getLogger(__name__)\n"
        "parser.error(f'bad {1}')\n"
        "obj.info(f'not logger {1}')\n"
        "self.logger.info(f'attribute {1}')\n"
        "stdlib_logger.info(f'stdlib {1}')\n"
        "logger.factory().info(f'arbitrary chain {1}')\n"
    )
    assert find_violations(source, "<nope>") == []


def _allowlist_file(tmp_path: Path, entries: list[str]) -> Path:
    allow = tmp_path / "allow.txt"
    allow.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return allow


def _baseline_lines(path: str, source: str) -> list[str]:
    """Return deterministic allowlist lines for violations in one source file."""
    counts = Counter(
        (item.scope, item.method, item.fingerprint) for item in find_violations(source, path)
    )
    return [
        "\t".join((path, scope, method, fingerprint, str(count)))
        for (scope, method, fingerprint), count in sorted(counts.items())
    ]


def test_main_blocks_non_allowlisted_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-allowlisted file with an f-string logger call fails the hook."""
    monkeypatch.chdir(tmp_path)
    bad = _write(tmp_path, "robot_sf/mod.py", "from loguru import logger\nlogger.info(f'x {1}')\n")
    allow = _allowlist_file(tmp_path, [])
    assert _HOOK.main([str(bad), "--allowlist", str(allow)]) == 1


def test_main_allows_grandfathered_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An allowlisted file is permitted (ratchet), so the hook exits 0."""
    monkeypatch.chdir(tmp_path)
    source = "from loguru import logger\nlogger.info(f'x {1}')\n"
    bad = _write(tmp_path, "robot_sf/mod.py", source)
    allow = _allowlist_file(tmp_path, _baseline_lines("robot_sf/mod.py", source))
    assert _HOOK.main([str(bad), "--allowlist", str(allow)]) == 0


def test_main_blocks_new_call_in_grandfathered_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-call baseline cannot hide a new violation in a legacy file."""
    monkeypatch.chdir(tmp_path)
    baseline = "from loguru import logger\nlogger.info(f'old {1}')\n"
    changed = baseline + "logger.warning(f'new {2}')\n"
    bad = _write(tmp_path, "robot_sf/mod.py", changed)
    allow = _allowlist_file(tmp_path, _baseline_lines("robot_sf/mod.py", baseline))
    assert _HOOK.main([str(bad), "--allowlist", str(allow)]) == 1


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
    """The shipped baseline must cover every exact grandfathered violation.

    New identities or multiplicities fail even inside files with historical calls.
    """
    repo_root = Path(__file__).resolve().parents[2]
    allow_path = repo_root / "hooks" / _HOOK.ALLOWLIST_FILENAME
    remaining = _HOOK._load_allowlist(allow_path)
    uncovered: list[tuple[str, Violation]] = []
    for py in sorted((repo_root / "robot_sf").rglob("*.py")):
        rel = py.relative_to(repo_root).as_posix()
        src = py.read_text(encoding="utf-8")
        for violation in find_violations(src, rel):
            key = _HOOK.AllowlistKey(
                rel,
                violation.scope,
                violation.method,
                violation.fingerprint,
            )
            if remaining[key] > 0:
                remaining[key] -= 1
            else:
                uncovered.append((rel, violation))
    assert not uncovered, (
        "F-string logger calls are absent from the exact ratchet baseline: "
        f"{uncovered}. Either migrate them or add them to {allow_path.as_posix()}."
    )


def test_hot_path_files_not_in_allowlist() -> None:
    """The migrated hot-path files must NOT be allowlisted (guard enforces them)."""
    repo_root = Path(__file__).resolve().parents[2]
    allow_path = repo_root / "hooks" / _HOOK.ALLOWLIST_FILENAME
    allowlisted = {key.path for key in _HOOK._load_allowlist(allow_path)}
    hot_paths = {
        "robot_sf/sim/simulator.py",
        "robot_sf/gym_env/base_env.py",
        "robot_sf/gym_env/pedestrian_env.py",
    }
    assert not (hot_paths & allowlisted), (
        f"migrated hot-path files must be absent from allowlist: {hot_paths & allowlisted}"
    )


def test_precommit_runs_full_scan_for_source_and_guard_changes() -> None:
    """The hook cannot be bypassed by pre-commit filename filtering."""
    repo_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((repo_root / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hook = next(
        hook
        for repo in config["repos"]
        if repo["repo"] == "local"
        for hook in repo["hooks"]
        if hook["id"] == "no-fstring-logger"
    )
    assert hook["pass_filenames"] is False
    pattern = re.compile(hook["files"])
    assert pattern.fullmatch("robot_sf/gym_env/base_env.py")
    assert pattern.fullmatch("hooks/no_fstring_logger.py")
    assert pattern.fullmatch("hooks/no_fstring_logger_allowlist.txt")
    assert not pattern.fullmatch("tests/hooks/test_no_fstring_logger.py")
