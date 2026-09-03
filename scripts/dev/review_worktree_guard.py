#!/usr/bin/env python3
"""Protect review worktrees from remote writes and unsafe synthetic merges.

Review worktrees are deliberately read-only with respect to remote references.
``configure --mode review`` stores the mode in the linked worktree's private
Git config, installs the tracked pre-push hook, and routes push destinations to
a non-repository URL.  Worktree-local URL and protocol barriers catch direct
URLs and explicit push URLs as well; the integration probe reads remote state
through the common Git config so it can retain a read-only remote comparison.
This is a Git-level workflow guard, not an operating-system sandbox; a caller
who deliberately overrides the worktree's Git configuration can bypass these
local barriers.

``integrate`` is the canonical read-only merge probe.  It snapshots all refs
reported by ``git ls-remote --refs`` before and after a ``--no-commit --no-ff``
merge, always attempts ``git merge --abort``, and succeeds only when the
worktree and remote snapshot are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "review_worktree_guard.v1"
BACKUP_SCHEMA_VERSION = "review_worktree_guard_backup.v1"
WORKTREE_MODE_KEY = "robot-sf.worktree-mode"
BACKUP_KEY = "robot-sf.review-push-guard-backup"
BLOCKED_URL_KEY = "robot-sf.review-push-blocked-url"
REVIEW_MODE = "review"
IMPLEMENTATION_MODE = "implementation"
HOOK_RELATIVE_PATH = Path("scripts/dev/git_hooks/pre-push")
GUARD_RELATIVE_PATH = Path("scripts/dev/review_worktree_guard.py")
DEFAULT_TIMEOUT_SECONDS = 120
PROTOCOL_POLICY_KEYS = (
    "protocol.allow",
    "protocol.ext.allow",
    "protocol.file.allow",
    "protocol.git.allow",
    "protocol.http.allow",
    "protocol.https.allow",
    "protocol.ssh.allow",
)


class GuardError(ValueError):
    """A deterministic, user-actionable guard failure."""


def _run_git(
    worktree: Path,
    *args: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run Git without invoking a shell and preserve bounded diagnostics."""
    command = ["git", "-C", str(worktree), *args]
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            command,
            returncode=124,
            stdout="",
            stderr=f"command timed out after {timeout} seconds",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(command, returncode=127, stdout="", stderr=str(exc))


def _run_common_git(
    identity: dict[str, Path],
    *args: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run a common-config Git command without the review worktree config."""
    command = [
        "git",
        "-C",
        str(identity["path"]),
        "--git-dir",
        str(identity["common_git_dir"]),
        *args,
    ]
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            command,
            returncode=124,
            stdout="",
            stderr=f"command timed out after {timeout} seconds",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(command, returncode=127, stdout="", stderr=str(exc))


def _command_detail(result: subprocess.CompletedProcess[str]) -> str:
    """Return a bounded command diagnostic suitable for JSON output."""
    detail = (result.stderr or result.stdout).strip().replace("\x00", "")
    return detail[:2_000] or f"exit status {result.returncode}"


def _git(worktree: Path, *args: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    result = _run_git(worktree, *args, timeout=timeout)
    if result.returncode != 0:
        raise GuardError(f"git {' '.join(args)} failed: {_command_detail(result)}")
    return result.stdout.strip()


def _git_optional(worktree: Path, *args: str) -> str | None:
    """Read an optional Git value, failing on errors other than absence."""
    result = _run_git(worktree, *args)
    if result.returncode == 0:
        return result.stdout.strip() or None
    if result.returncode == 1:
        return None
    raise GuardError(f"git {' '.join(args)} failed: {_command_detail(result)}")


def _absolute_directory(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_symlink():
        raise GuardError("worktree path must not be a symlink")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_dir():
        raise GuardError(f"worktree path is not an existing directory: {value}")
    return resolved


def _identity(worktree: str | Path) -> dict[str, Path]:
    """Resolve and validate that a path is a registered linked worktree."""
    path = _absolute_directory(worktree)
    top_level = Path(_git(path, "rev-parse", "--show-toplevel")).resolve()
    if top_level != path:
        raise GuardError(f"git top-level {top_level} does not match worktree {path}")
    git_dir = Path(_git(path, "rev-parse", "--path-format=absolute", "--git-dir")).resolve()
    common_git_dir = Path(
        _git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
    ).resolve()
    if git_dir == common_git_dir:
        raise GuardError("review guard requires a linked worktree, not the main checkout")
    return {"path": path, "git_dir": git_dir, "common_git_dir": common_git_dir}


def _worktree_config_file(identity: dict[str, Path]) -> Path:
    config_path = Path(
        _git(
            identity["path"],
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "config.worktree",
        )
    )
    if not config_path.is_absolute():
        config_path = identity["git_dir"] / config_path
    # Some Git versions resolve the final component for ``--git-path``. Check
    # the canonical per-worktree location before accepting that resolved path,
    # otherwise a symlinked config could be followed and mutated.
    canonical_path = identity["git_dir"] / "config.worktree"
    if canonical_path.is_symlink() or config_path.is_symlink():
        raise GuardError("linked worktree config must not be a symlink")
    return config_path.resolve()


def _worktree_values(identity: dict[str, Path], key: str) -> list[str]:
    """Read only values physically stored in the linked worktree config."""
    config_path = _worktree_config_file(identity)
    if not config_path.exists():
        return []
    result = _run_git(identity["path"], "config", "--file", str(config_path), "--get-all", key)
    if result.returncode == 1:
        return []
    if result.returncode != 0:
        raise GuardError(f"git config read failed for {key}: {_command_detail(result)}")
    return result.stdout.splitlines()


def _worktree_set(identity: dict[str, Path], key: str, value: str) -> None:
    _git(identity["path"], "config", "--worktree", "--replace-all", key, value)


def _worktree_add(identity: dict[str, Path], key: str, value: str) -> None:
    _git(identity["path"], "config", "--worktree", "--add", key, value)


def _worktree_unset(identity: dict[str, Path], key: str) -> None:
    result = _run_git(identity["path"], "config", "--worktree", "--unset-all", key)
    if result.returncode not in (0, 1, 5):
        raise GuardError(f"git config unset failed for {key}: {_command_detail(result)}")


def _ensure_worktree_config(identity: dict[str, Path]) -> None:
    value = _git_optional(identity["path"], "config", "--get", "extensions.worktreeConfig")
    if value is None or value.lower() != "true":
        _git(identity["path"], "config", "extensions.worktreeConfig", "true")


def _configured_mode(identity: dict[str, Path]) -> str | None:
    return _git_optional(identity["path"], "config", "--get", WORKTREE_MODE_KEY)


def _remote_names(identity: dict[str, Path]) -> list[str]:
    output = _git(identity["path"], "remote")
    return [line for line in output.splitlines() if line]


def _remote_urls(identity: dict[str, Path], remote: str) -> list[str]:
    urls: list[str] = []
    for options in (("--all",), ("--all", "--push")):
        result = _run_git(identity["path"], "remote", "get-url", *options, remote)
        if result.returncode != 0:
            raise GuardError(
                f"could not resolve configured URL for remote {remote}: {_command_detail(result)}"
            )
        urls.extend(line for line in result.stdout.splitlines() if line)
    return list(dict.fromkeys(urls))


def _blocked_url(identity: dict[str, Path]) -> str:
    """Return a stable URL that cannot be a Git repository."""
    blocked_path = identity["git_dir"] / ".robot-sf-review-push-blocked"
    return blocked_path.resolve().as_uri()


def _blocked_receivepack(identity: dict[str, Path]) -> str:
    """Return a nonexistent per-worktree receive-pack command path."""
    return str(identity["git_dir"] / ".robot-sf-review-push-blocked-receive-pack")


def _url_rule_key(blocked_url: str) -> str:
    return f"url.{blocked_url}.pushInsteadOf"


def _url_catchall_key(blocked_url: str) -> str:
    return f"url.{blocked_url}.insteadOf"


def _capture_backup(identity: dict[str, Path], original_mode: str | None) -> dict[str, Any]:
    remotes = _remote_names(identity)
    return {
        "schema": BACKUP_SCHEMA_VERSION,
        "mode": original_mode,
        "core_hooks_path": _worktree_values(identity, "core.hooksPath"),
        "remote_pushurls": {
            remote: _worktree_values(identity, f"remote.{remote}.pushurl") for remote in remotes
        },
        "remote_receivepacks": {
            remote: _worktree_values(identity, f"remote.{remote}.receivepack") for remote in remotes
        },
        "remote_urls": {remote: _remote_urls(identity, remote) for remote in remotes},
        "protocol_allows": {key: _worktree_values(identity, key) for key in PROTOCOL_POLICY_KEYS},
    }


def _write_backup(identity: dict[str, Path], backup: dict[str, Any]) -> None:
    _worktree_set(
        identity,
        BACKUP_KEY,
        json.dumps(backup, sort_keys=True, separators=(",", ":")),
    )


def _validate_string_list_mapping(value: Any, label: str) -> None:
    """Validate a backup mapping whose values are string lists."""
    if not isinstance(value, dict):
        raise GuardError(f"review guard backup has malformed {label} values")
    for key, values in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(values, list)
            or not all(isinstance(value, str) for value in values)
        ):
            raise GuardError(f"review guard backup has malformed {label} values")


def _load_backup(identity: dict[str, Path]) -> dict[str, Any] | None:
    values = _worktree_values(identity, BACKUP_KEY)
    if not values:
        return None
    if len(values) != 1:
        raise GuardError("review guard backup has multiple values")
    try:
        backup = json.loads(values[0])
    except json.JSONDecodeError as exc:
        raise GuardError(f"review guard backup is not valid JSON: {exc}") from exc
    if not isinstance(backup, dict) or backup.get("schema") != BACKUP_SCHEMA_VERSION:
        raise GuardError("review guard backup schema is missing or unsupported")
    if backup.get("mode") not in (None, IMPLEMENTATION_MODE):
        raise GuardError("review guard backup contains an unsupported original mode")
    if not isinstance(backup.get("core_hooks_path"), list) or not all(
        isinstance(value, str) for value in backup["core_hooks_path"]
    ):
        raise GuardError("review guard backup has malformed core.hooksPath values")
    _validate_string_list_mapping(backup.get("remote_pushurls"), "remote.pushurl")
    _validate_string_list_mapping(backup.get("remote_receivepacks", {}), "remote.receivepack")
    _validate_string_list_mapping(backup.get("remote_urls"), "remote URL")
    _validate_string_list_mapping(backup.get("protocol_allows", {}), "protocol policy")
    return backup


def _check_hook_files(identity: dict[str, Path], hook_source_root: str | Path | None) -> Path:
    if hook_source_root is None:
        hook = identity["path"] / HOOK_RELATIVE_PATH
        guard = identity["path"] / GUARD_RELATIVE_PATH
    else:
        source_root = Path(hook_source_root)
        if source_root.is_symlink() or not source_root.is_dir():
            raise GuardError(f"hook source root must be a real directory: {hook_source_root}")
        source_root = source_root.resolve()
        hook = source_root / "git_hooks" / HOOK_RELATIVE_PATH.name
        guard = source_root / GUARD_RELATIVE_PATH.name
    for candidate, label in ((hook, "pre-push hook"), (guard, "guard script")):
        if candidate.is_symlink() or not candidate.is_file():
            raise GuardError(f"review mode requires a tracked {label}: {candidate}")
    if not os.access(hook, os.X_OK):
        raise GuardError(f"review mode requires an executable pre-push hook: {hook}")
    return hook


def _configure_review_mode(
    identity: dict[str, Path],
    *,
    original_mode: str | None,
    backup: dict[str, Any] | None,
    hook_source_root: str | Path | None,
) -> dict[str, Any]:
    """Install the hook and configured-remote barriers."""
    if original_mode == REVIEW_MODE and backup is None:
        raise GuardError("review mode is set but its restoration backup is missing")
    if original_mode != REVIEW_MODE and backup is not None:
        raise GuardError("stale review guard backup exists outside review mode")
    hook = _check_hook_files(identity, hook_source_root)
    if backup is None:
        backup = _capture_backup(identity, original_mode)
        _write_backup(identity, backup)
    _prepare_review_barriers(identity, original_mode, backup)
    _worktree_set(identity, "core.hooksPath", str(hook.parent))
    # Set the mode last so a partial setup remains fail-closed only after its
    # hook and configured-remote barriers are in place.
    _worktree_set(identity, WORKTREE_MODE_KEY, REVIEW_MODE)
    return _configure_result(identity, REVIEW_MODE, hook, len(_remote_names(identity)))


def _prepare_review_barriers(
    identity: dict[str, Path],
    original_mode: str | None,
    backup: dict[str, Any],
) -> str:
    """Apply inert push destinations and URL rewrites."""
    expected_url = _blocked_url(identity)
    blocked_values = _worktree_values(identity, BLOCKED_URL_KEY)
    if blocked_values and blocked_values != [expected_url]:
        raise GuardError("review guard has an unexpected blocked push URL")
    if original_mode == REVIEW_MODE and not blocked_values:
        raise GuardError("review mode is set but its blocked push URL is missing")
    _worktree_set(identity, BLOCKED_URL_KEY, expected_url)
    blocked_receivepack = _blocked_receivepack(identity)
    blocked_receivepack_path = Path(blocked_receivepack)
    if blocked_receivepack_path.exists() or blocked_receivepack_path.is_symlink():
        raise GuardError(
            f"review guard receive-pack barrier path already exists: {blocked_receivepack}"
        )
    rule_key = _url_rule_key(expected_url)
    _worktree_unset(identity, rule_key)
    catchall_key = _url_catchall_key(expected_url)
    _worktree_unset(identity, catchall_key)
    remotes = _remote_names(identity)
    configured_urls = {
        remote: (
            backup["remote_urls"].get(remote, [])
            if original_mode == REVIEW_MODE and remote in backup["remote_urls"]
            else _remote_urls(identity, remote)
        )
        for remote in remotes
    }
    for remote in remotes:
        # An empty higher-priority value resets inherited common-config
        # pushurl entries. Without it, Git pushes to every effective push URL
        # and may update the real destination before the inert one fails.
        pushurl_key = f"remote.{remote}.pushurl"
        _worktree_set(identity, pushurl_key, "")
        _worktree_add(identity, pushurl_key, expected_url)
        _worktree_set(identity, f"remote.{remote}.receivepack", blocked_receivepack)
        for url in configured_urls[remote]:
            _worktree_add(identity, rule_key, url)
    # Keep the URL barrier effective for remotes added after review mode is
    # configured, including explicit pushurl values. This all-URL rewrite is
    # intentionally broader than pushInsteadOf: Git ignores pushInsteadOf when
    # a remote has an explicit pushurl. Remote reads needed by ``integrate``
    # use the common Git config below, outside this worktree-local rule.
    _worktree_add(identity, catchall_key, "")
    # URL rewrite precedence is longest-prefix based, so a pre-existing common
    # config alias can otherwise beat the empty-prefix catch-all. Deny every
    # built-in transport in this worktree as the final barrier, including
    # remotes and aliases added after review mode is configured.
    for key in PROTOCOL_POLICY_KEYS:
        _worktree_set(identity, key, "never")
    return expected_url


def _configure_result(
    identity: dict[str, Path],
    mode: str,
    hook: Path | None,
    blocked_remote_count: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "command": "configure",
        "ok": True,
        "mode": mode,
        "worktree": str(identity["path"]),
        "hook_path": str(hook) if hook is not None else None,
        "blocked_remote_count": blocked_remote_count,
    }


def _restore_implementation_mode(
    identity: dict[str, Path], backup: dict[str, Any]
) -> dict[str, Any]:
    """Restore the worktree-local values captured before review mode."""
    blocked_values = _worktree_values(identity, BLOCKED_URL_KEY)
    if len(blocked_values) != 1:
        raise GuardError("review guard blocked push URL is missing or duplicated")
    _worktree_unset(identity, _url_rule_key(blocked_values[0]))
    _worktree_unset(identity, _url_catchall_key(blocked_values[0]))
    for remote in set(_remote_names(identity)) | set(backup["remote_pushurls"]):
        key = f"remote.{remote}.pushurl"
        _worktree_unset(identity, key)
        for value in backup["remote_pushurls"].get(remote, []):
            _worktree_add(identity, key, value)
        key = f"remote.{remote}.receivepack"
        _worktree_unset(identity, key)
        for value in backup.get("remote_receivepacks", {}).get(remote, []):
            _worktree_add(identity, key, value)
    for key in PROTOCOL_POLICY_KEYS:
        _worktree_unset(identity, key)
        for value in backup.get("protocol_allows", {}).get(key, []):
            _worktree_add(identity, key, value)
    _worktree_unset(identity, "core.hooksPath")
    for value in backup["core_hooks_path"]:
        _worktree_add(identity, "core.hooksPath", value)
    _worktree_unset(identity, BLOCKED_URL_KEY)
    _worktree_unset(identity, BACKUP_KEY)
    _worktree_unset(identity, WORKTREE_MODE_KEY)
    if backup["mode"] is not None:
        _worktree_set(identity, WORKTREE_MODE_KEY, backup["mode"])
    return _configure_result(identity, IMPLEMENTATION_MODE, None, 0)


def configure_worktree(
    worktree: str | Path,
    *,
    mode: str,
    hook_source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Configure or restore one linked worktree's review protection."""
    if mode not in (REVIEW_MODE, IMPLEMENTATION_MODE):
        raise GuardError(f"unsupported worktree mode: {mode}")
    identity = _identity(worktree)
    _ensure_worktree_config(identity)
    original_mode = _configured_mode(identity)
    if original_mode not in (None, REVIEW_MODE, IMPLEMENTATION_MODE):
        raise GuardError(f"unsupported configured worktree mode: {original_mode}")
    backup = _load_backup(identity)
    if mode == REVIEW_MODE:
        return _configure_review_mode(
            identity,
            original_mode=original_mode,
            backup=backup,
            hook_source_root=hook_source_root,
        )
    if original_mode == REVIEW_MODE:
        if backup is None:
            raise GuardError("cannot restore review mode without its backup")
        return _restore_implementation_mode(identity, backup)
    # Explicit implementation mode is useful when repairing a worktree's
    # local metadata; ordinary worktree creation does not need this mutation.
    _worktree_set(identity, WORKTREE_MODE_KEY, IMPLEMENTATION_MODE)
    return _configure_result(identity, IMPLEMENTATION_MODE, None, 0)


def pre_push_check(worktree: str | Path = ".") -> tuple[dict[str, Any], int]:
    """Reject a push from review mode; allow an ordinary implementation tree."""
    identity = _identity(worktree)
    mode = _configured_mode(identity)
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "command": "pre-push",
        "worktree": str(identity["path"]),
        "mode": mode,
        "ok": False,
        "blocked": False,
        "error": None,
    }
    if mode in (None, IMPLEMENTATION_MODE):
        payload.update(ok=True, blocked=False)
        return payload, 0
    if mode == REVIEW_MODE:
        payload.update(
            blocked=True,
            error="remote pushes are disabled in a read-only review worktree",
        )
        return payload, 1
    payload["error"] = f"unsupported configured worktree mode: {mode}"
    return payload, 1


def _status(identity: dict[str, Path]) -> str:
    return _git(identity["path"], "status", "--porcelain=v1", "--untracked-files=all")


def _merge_head(identity: dict[str, Path]) -> str | None:
    return _git_optional(identity["path"], "rev-parse", "-q", "--verify", "MERGE_HEAD")


def _remote_snapshot(identity: dict[str, Path], remote: str, *, timeout: int) -> str:
    result = _run_common_git(identity, "ls-remote", "--refs", remote, timeout=timeout)
    if result.returncode != 0:
        raise GuardError(f"git ls-remote failed for {remote}: {_command_detail(result)}")
    return result.stdout


def _snapshot_summary(snapshot: str) -> dict[str, int | str]:
    return {
        "sha256": hashlib.sha256(snapshot.encode("utf-8")).hexdigest(),
        "line_count": len(snapshot.splitlines()),
    }


def _integration_payload(source_ref: str, remote: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "command": "integrate",
        "ok": False,
        "worktree": None,
        "mode": None,
        "source_ref": source_ref,
        "source_commit": None,
        "remote": remote,
        "status_before": None,
        "status_after": None,
        "head_before": None,
        "head_after": None,
        "merge_returncode": None,
        "merge_stdout": None,
        "merge_stderr": None,
        "abort_attempted": False,
        "abort_returncode": None,
        "abort_stdout": None,
        "abort_stderr": None,
        "orig_head_before": None,
        "orig_head_after": None,
        "orig_head_restore_returncode": None,
        "merge_head_after": None,
        "remote_refs_before": None,
        "remote_refs_after": None,
        "remote_refs_unchanged": None,
        "error": None,
    }


def _run_merge_probe(
    identity: dict[str, Path],
    source_ref: str,
    timeout: int,
    payload: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    result = _run_git(
        identity["path"],
        "merge",
        "--no-commit",
        "--no-ff",
        "--no-edit",
        source_ref,
        timeout=timeout,
    )
    payload["merge_returncode"] = result.returncode
    payload["merge_stdout"] = result.stdout[-2_000:] or None
    payload["merge_stderr"] = result.stderr[-2_000:] or None
    return result


def _abort_merge_probe(
    identity: dict[str, Path],
    timeout: int,
    payload: dict[str, Any],
) -> None:
    payload["abort_attempted"] = True
    result = _run_git(identity["path"], "merge", "--abort", timeout=timeout)
    payload["abort_returncode"] = result.returncode
    payload["abort_stdout"] = result.stdout[-2_000:] or None
    payload["abort_stderr"] = result.stderr[-2_000:] or None


def _collect_post_probe_state(
    identity: dict[str, Path],
    remote_before: str,
    timeout: int,
    payload: dict[str, Any],
    errors: list[str],
) -> None:
    try:
        payload["head_after"] = _git(identity["path"], "rev-parse", "HEAD")
        payload["status_after"] = _status(identity)
        payload["merge_head_after"] = _merge_head(identity)
    except GuardError as exc:
        errors.append(f"post-merge local-state check failed: {exc}")
    try:
        remote_after = _remote_snapshot(identity, payload["remote"], timeout=timeout)
        payload["remote_refs_after"] = _snapshot_summary(remote_after)
        payload["remote_refs_unchanged"] = remote_after == remote_before
    except GuardError as exc:
        errors.append(f"post-merge remote-state check failed: {exc}")


def _restore_orig_head(
    identity: dict[str, Path],
    payload: dict[str, Any],
    timeout: int,
    errors: list[str],
) -> None:
    """Restore the worktree's pre-probe ORIG_HEAD pseudo-ref."""
    desired = payload["orig_head_before"]
    command = ("update-ref", "ORIG_HEAD", desired) if desired else ("update-ref", "-d", "ORIG_HEAD")
    result = _run_git(identity["path"], *command, timeout=timeout)
    payload["orig_head_restore_returncode"] = result.returncode
    if result.returncode != 0:
        errors.append(f"ORIG_HEAD restoration failed: {_command_detail(result)}")
    try:
        payload["orig_head_after"] = _git_optional(
            identity["path"], "rev-parse", "-q", "--verify", "ORIG_HEAD"
        )
    except GuardError as exc:
        errors.append(f"ORIG_HEAD post-restore check failed: {exc}")
        return
    if payload["orig_head_after"] != desired:
        errors.append("ORIG_HEAD changed during the integration probe")


def _validate_probe(
    payload: dict[str, Any],
    merge_result: subprocess.CompletedProcess[str],
) -> list[str]:
    errors: list[str] = []
    if merge_result.returncode != 0:
        errors.append(f"synthetic merge failed: {_command_detail(merge_result)}")
    merge_aborted = payload["abort_returncode"] == 0
    abort_noop = (
        payload["abort_returncode"] not in (None, 0)
        and payload["merge_head_after"] is None
        and payload["head_after"] == payload["head_before"]
        and payload["status_after"] == ""
    )
    if not merge_aborted and not abort_noop:
        errors.append("merge abort did not leave a clean, non-merge worktree")
    if payload["merge_head_after"] is not None:
        errors.append("MERGE_HEAD remains after the integration probe")
    if payload["head_after"] != payload["head_before"]:
        errors.append("HEAD changed during the no-commit integration probe")
    if payload["status_after"] != "":
        errors.append("worktree is dirty after the integration probe")
    if payload["remote_refs_unchanged"] is not True:
        errors.append("remote refs changed or could not be compared")
    return errors


def integrate_worktree(
    worktree: str | Path,
    *,
    source_ref: str,
    remote: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], int]:
    """Probe a merge and prove that both local and remote state were restored."""
    payload = _integration_payload(source_ref, remote)
    identity: dict[str, Path] | None = None
    remote_before: str | None = None
    merge_result: subprocess.CompletedProcess[str] | None = None
    errors: list[str] = []

    try:
        identity = _identity(worktree)
        payload["worktree"] = str(identity["path"])
        payload["mode"] = _configured_mode(identity)
        if payload["mode"] != REVIEW_MODE:
            raise GuardError("synthetic integration requires a review-mode worktree")
        status_before = _status(identity)
        payload["status_before"] = status_before
        if status_before:
            raise GuardError("synthetic integration requires a clean worktree")
        merge_head_before = _merge_head(identity)
        if merge_head_before is not None:
            raise GuardError("synthetic integration refuses an existing merge state")
        payload["head_before"] = _git(identity["path"], "rev-parse", "HEAD")
        payload["orig_head_before"] = _git_optional(
            identity["path"], "rev-parse", "-q", "--verify", "ORIG_HEAD"
        )
        payload["source_commit"] = _git(
            identity["path"], "rev-parse", "--verify", f"{source_ref}^{{commit}}"
        )
        remote_before = _remote_snapshot(identity, remote, timeout=timeout)
        payload["remote_refs_before"] = _snapshot_summary(remote_before)
        merge_result = _run_merge_probe(identity, source_ref, timeout, payload)
    except GuardError as exc:
        errors.append(str(exc))
    finally:
        if merge_result is not None and identity is not None:
            _abort_merge_probe(identity, timeout, payload)
            _restore_orig_head(identity, payload, timeout, errors)

        if identity is not None and remote_before is not None:
            _collect_post_probe_state(identity, remote_before, timeout, payload, errors)

    if merge_result is not None:
        errors.extend(_validate_probe(payload, merge_result))

    if errors:
        payload["error"] = "; ".join(dict.fromkeys(errors))[:4_000]
        return payload, 1
    if merge_result is None:
        payload["error"] = "integration probe did not run"
        return payload, 1
    payload["ok"] = True
    return payload, 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    configure = subparsers.add_parser("configure", help="configure one linked worktree mode")
    configure.add_argument("--worktree", required=True)
    configure.add_argument("--mode", choices=(REVIEW_MODE, IMPLEMENTATION_MODE), required=True)
    configure.add_argument(
        "--hook-source-root",
        help="use this scripts/dev directory when the target base lacks the guard files",
    )

    pre_push = subparsers.add_parser("pre-push", help="run the review worktree push guard")
    pre_push.add_argument("--worktree", default=".")
    pre_push.add_argument("hook_arguments", nargs="*", help=argparse.SUPPRESS)

    integrate = subparsers.add_parser("integrate", help="run an aborting synthetic merge probe")
    integrate.add_argument("--worktree", default=".")
    integrate.add_argument("--source-ref", default="origin/main")
    integrate.add_argument("--remote", default="origin")
    integrate.add_argument("--timeout", type=_positive_int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the review-worktree guard CLI and emit one JSON result."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "configure":
            payload = configure_worktree(
                args.worktree,
                mode=args.mode,
                hook_source_root=args.hook_source_root,
            )
            return_code = 0
        elif args.command == "pre-push":
            payload, return_code = pre_push_check(args.worktree)
        else:
            payload, return_code = integrate_worktree(
                args.worktree,
                source_ref=args.source_ref,
                remote=args.remote,
                timeout=args.timeout,
            )
    except (GuardError, OSError) as exc:
        payload = {
            "schema": SCHEMA_VERSION,
            "command": args.command,
            "ok": False,
            "error": str(exc),
        }
        return_code = 1
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return return_code


if __name__ == "__main__":
    sys.exit(main())
