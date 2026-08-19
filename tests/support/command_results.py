"""Small parametrizable factories for structured subprocess-test results."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

CommandObserver = Callable[[str, list[str], Path, int], None]


def command_result_factory(
    result_type: Callable[..., Any],
    *,
    returncode: int | None = 0,
    failure_summary: str | None = None,
    stdout_tail: str = "ok",
    stderr_tail: str = "",
    on_call: CommandObserver | None = None,
) -> Callable[[str, list[str], Path, int], Any]:
    """Return a probe-compatible fake command runner.

    The factory deliberately only owns the homogeneous ``CommandResult``
    construction shared by probe tests. Scenario-specific routing and payload
    assertions stay in each test.
    """

    def fake_run(name: str, command: list[str], cwd: Path, timeout_seconds: int) -> Any:
        if on_call is not None:
            on_call(name, command, cwd, timeout_seconds)
        return result_type(
            name=name,
            command=command,
            returncode=returncode,
            failure_summary=failure_summary,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )

    return fake_run
