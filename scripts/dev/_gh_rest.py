"""Small, fail-closed transport primitives for ``gh api`` REST helpers.

The scripts in this directory intentionally keep endpoint validation and
write-verification local to each caller.  This module owns only process
execution, JSON stdin handling, bounded timeouts, and the shared JSON/string
normalization that is safe to keep identical across callers.
"""

from __future__ import annotations

import json
import subprocess
from functools import partial
from typing import Any


def run_gh_api(
    path: str,
    payload: object | None = None,
    *,
    method: str | None = None,
    extra_args: list[str] | None = None,
    timeout: int = 30,
    timeout_context: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one ``gh api`` request without a shell and return its result.

    JSON payloads are always sent through stdin, never placed in argv.  Missing
    ``gh`` and timeouts become deterministic failed ``CompletedProcess`` values
    so endpoint helpers can preserve their own structured error contracts.
    """
    args = ["gh", "api"]
    if method is not None:
        args.extend(["--method", method])
    args.append(path)
    if extra_args:
        args.extend(extra_args)
    stdin_payload: str | None = None
    if payload is not None:
        args.extend(["--input", "-"])
        stdin_payload = json.dumps(payload)

    run_kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": timeout,
    }
    if stdin_payload is not None:
        run_kwargs["input"] = stdin_payload
    try:
        return subprocess.run(args, check=False, **run_kwargs)
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            args=args,
            returncode=127,
            stdout="",
            stderr="gh CLI not found on PATH; install GitHub CLI (https://cli.github.com/)",
        )
    except subprocess.TimeoutExpired:
        detail = f"gh api timed out after {timeout} seconds"
        if timeout_context:
            detail += f"; {timeout_context}"
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout="",
            stderr=detail,
        )


def run_gh_api_or_raise(
    path: str,
    *,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run a read request while preserving the audit tool's exception contract."""
    result = run_gh_api(path, timeout=timeout)
    if result.returncode == 127:
        raise RuntimeError("GitHub CLI 'gh' was not found")
    if result.returncode == 124:
        raise RuntimeError(f"GitHub REST read timed out after {timeout}s ({path})")
    return result


def parse_json(result: subprocess.CompletedProcess[str], *, what: str) -> tuple[Any, str]:
    """Parse a REST result using the shared bounded diagnostic format."""
    if result.returncode != 0:
        detail = result.stderr.strip() or f"gh api exited with code {result.returncode}"
        return None, f"{what} failed: {detail}"
    try:
        return json.loads(result.stdout), ""
    except json.JSONDecodeError as exc:
        snippet = result.stdout.strip()[:200]
        return None, f"{what} returned invalid JSON: {exc}; stdout snippet: {snippet!r}"


def as_str(raw: Any) -> str:
    """Coerce a JSON value to ``str`` while mapping explicit ``None`` to empty."""
    return "" if raw is None else str(raw)


gh_api_post = partial(run_gh_api, method="POST", timeout_context="label update was not verified")
gh_api_delete = partial(
    run_gh_api, method="DELETE", timeout_context="label update was not verified"
)
gh_api_get = partial(run_gh_api)
gh_api_label_get = partial(run_gh_api, timeout_context="could not read labels")
gh_api_comments_get = partial(run_gh_api, timeout_context="PR comments were not read")
gh_api_patch = partial(run_gh_api, method="PATCH", timeout_context="body update was not verified")
gh_api_metadata_get = partial(run_gh_api, timeout_context="PR metadata was not verified")
