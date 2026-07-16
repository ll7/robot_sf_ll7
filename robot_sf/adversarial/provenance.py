"""Reproducibility provenance helpers for adversarial campaigns.

Issue #5303 (and the issue-3079 evidence-grade promotion plan) require every
adversarial job directory to pin execution context (hostname, CPU model,
thread environment, commit) and to emit a receipt manifest that records the
configs/seeds/digests archived for a run. This module centralises that
provenance so the transfer-matrix archival stage and future adversarial jobs
share one fail-closed, reproducible contract.

Capability-not-evidence boundary: these helpers only record *what ran*; they
make no benchmark or paper-facing claim. The recorded digests describe the
local repository state, not the scientific result.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_EXECUTION_CONTEXT_SCHEMA = "adversarial_execution_context.v1"
_RECEIPT_SCHEMA = "adversarial_receipt_manifest.v1"

# Thread-environment variables that must be pinned per the evidence-grade plan.
_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _git_commit_sha(repo_root: Path) -> str | None:
    """Return the current git HEAD sha, or None when not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip() or None


def _cpu_model() -> str:
    """Best-effort CPU model string across Linux/macOS."""
    system = platform.system()
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return result.stdout.strip() or platform.processor()
        except (subprocess.SubprocessError, OSError):
            pass
    return platform.processor() or "unknown"


@dataclass(frozen=True)
class ExecutionContext:
    """Pinned execution environment for one adversarial job directory."""

    schema_version: str = _EXECUTION_CONTEXT_SCHEMA
    hostname: str = ""
    cpu_model: str = ""
    python_version: str = ""
    platform: str = ""
    commit_sha: str | None = None
    thread_env: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""
        return {
            "schema_version": self.schema_version,
            "hostname": self.hostname,
            "cpu_model": self.cpu_model,
            "python_version": self.python_version,
            "platform": self.platform,
            "commit_sha": self.commit_sha,
            "thread_env": dict(self.thread_env),
        }


def gather_execution_context(*, repo_root: Path | str | None = None) -> ExecutionContext:
    """Collect the host execution context for reproducible provenance.

    Parameters
    ----------
    repo_root : Path | str | None
        Repository root used to resolve the git commit. Defaults to the current
        working directory.

    Returns
    -------
    ExecutionContext
        Pinned hostname, CPU model, python/platform, commit sha, and thread env.
    """
    root = Path(repo_root) if repo_root else Path.cwd()
    thread_env = {
        var: (os.environ.get(var) if os.environ.get(var) is not None else "unset")
        for var in _THREAD_ENV_VARS
    }
    return ExecutionContext(
        hostname=platform.node(),
        cpu_model=_cpu_model(),
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
        commit_sha=_git_commit_sha(root),
        thread_env=thread_env,
    )


def write_execution_context(
    out_dir: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> Path:
    """Write ``execution_context.txt`` (JSON) into ``out_dir`` and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = gather_execution_context(repo_root=repo_root)
    path = out_dir / "execution_context.txt"
    path.write_text(json.dumps(context.to_json(), indent=2) + "\n", encoding="utf-8")
    return path


def sha256_of_file(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file, streaming to bound memory."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReceiptItem:
    """One archived artifact row in the receipt manifest."""

    artifact: str
    path: str
    digest: str | None = None
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""
        return {
            "artifact": self.artifact,
            "path": self.path,
            "digest": self.digest,
            "note": self.note,
        }


@dataclass(frozen=True)
class ReceiptManifest:
    """Manifest of archived artifacts for one adversarial run."""

    schema_version: str = _RECEIPT_SCHEMA
    run_id: str = ""
    execution_context_path: str = ""
    items: tuple[ReceiptItem, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "execution_context_path": self.execution_context_path,
            "items": [item.to_json() for item in self.items],
        }


def write_receipt_manifest(
    out_dir: Path | str,
    *,
    run_id: str,
    items: list[ReceiptItem],
    execution_context_path: str,
) -> Path:
    """Write ``receipt_manifest.json`` into ``out_dir`` and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = ReceiptManifest(
        run_id=run_id,
        execution_context_path=execution_context_path,
        items=tuple(items),
    )
    path = out_dir / "receipt_manifest.json"
    path.write_text(json.dumps(manifest.to_json(), indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "ExecutionContext",
    "ReceiptItem",
    "ReceiptManifest",
    "gather_execution_context",
    "sha256_of_file",
    "write_execution_context",
    "write_receipt_manifest",
]
