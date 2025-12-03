"""Execute all CI-enabled examples to guard against regressions.

The manifest (`examples/examples_manifest.yaml`) controls which scripts run as part of
this smoke harness. Examples flagged with ``ci_enabled: false`` are excluded; their
``ci_reason`` is surfaced via the manifest and documented in `examples/_archived/`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from robot_sf.examples.manifest_loader import ExampleManifest, ExampleScript, load_manifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST: ExampleManifest = load_manifest(validate_paths=True)
_CI_EXAMPLES: tuple[ExampleScript, ...] = tuple(_MANIFEST.iter_ci_enabled_examples())


def _id(example: ExampleScript) -> str:
    """TODO docstring. Document this function.

    Args:
        example: TODO docstring.

    Returns:
        TODO docstring.
    """
    return example.path.as_posix()


def _merge_pythonpath(root: Path, existing: str | None) -> str:
    """TODO docstring. Document this function.

    Args:
        root: TODO docstring.
        existing: TODO docstring.

    Returns:
        TODO docstring.
    """
    parts: list[str] = [str(root)]
    if existing:
        parts.extend(element for element in existing.split(os.pathsep) if element)
    seen: set[str] = set()
    ordered: list[str] = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            ordered.append(part)
    return os.pathsep.join(ordered)


def _tail(text: str | bytes | None, limit: int = 20) -> str:
    """TODO docstring. Document this function.

    Args:
        text: TODO docstring.
        limit: TODO docstring.

    Returns:
        TODO docstring.
    """
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    lines = [line.rstrip() for line in text.splitlines()]
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join(lines[-limit:])


@pytest.fixture(scope="module", name="repo_root_path")
def _repo_root_path() -> Path:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    return _REPO_ROOT


@pytest.fixture(scope="module", name="example_env")
def _example_env(repo_root_path: Path) -> dict[str, str]:
    """TODO docstring. Document this function.

    Args:
        repo_root_path: TODO docstring.

    Returns:
        TODO docstring.
    """
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("DISPLAY", "")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env["PYTHONPATH"] = _merge_pythonpath(repo_root_path, env.get("PYTHONPATH"))
    env.setdefault("ROBOT_SF_FAST_DEMO", "1")
    env.setdefault("ROBOT_SF_EXAMPLES_MAX_STEPS", "64")
    return env


@pytest.mark.parametrize("example", _CI_EXAMPLES, ids=_id)
def test_example_runs_without_error(
    example: ExampleScript,
    repo_root_path: Path,
    example_env: dict[str, str],
    perf_policy,
) -> None:
    """TODO docstring. Document this function.

    Args:
        example: TODO docstring.
        repo_root_path: TODO docstring.
        example_env: TODO docstring.
        perf_policy: TODO docstring.
    """
    script_path = _MANIFEST.resolve_example_path(example)
    assert script_path.is_file(), f"Example path missing: {script_path}"

    timeout_seconds = getattr(perf_policy, "hard_timeout_seconds", 120.0)
    command = [sys.executable, str(script_path)]

    start = time.perf_counter()
    try:
        completed = subprocess.run(  # noqa: PL subprocess-run-check
            command,
            cwd=repo_root_path,
            env=example_env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - handled as failure path
        stdout_tail = _tail(exc.stdout)
        stderr_tail = _tail(exc.stderr)
        pytest.fail(
            f"{example.path} exceeded {timeout_seconds:.1f}s timeout."
            f"\nstdout tail:\n{stdout_tail}\n---\nstderr tail:\n{stderr_tail}"
        )

    duration = time.perf_counter() - start
    if completed.returncode != 0:
        stdout_tail = _tail(completed.stdout)
        stderr_tail = _tail(completed.stderr)
        pytest.fail(
            f"{example.path} exited with code {completed.returncode} after {duration:.2f}s."
            f"\nstdout tail:\n{stdout_tail}\n---\nstderr tail:\n{stderr_tail}"
        )


@pytest.mark.skipif(len(_CI_EXAMPLES) == 0, reason="No CI-enabled examples declared in manifest")
def test_manifest_has_ci_enabled_entries() -> None:
    """TODO docstring. Document this function."""
    assert _CI_EXAMPLES, "Expected at least one CI-enabled example in manifest"
