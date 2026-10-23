"""Reconstruct a pinned acquisition-only overlay without replacing historical planner code.

The audited patch contains only receipt plumbing. Every original and installed byte is bound
in harness.json; the sweep reconstructs the same overlay from committed objects before allowing
these acquisition paths to differ from the runtime source.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import cast

SCRIPT = "scripts/validation/run_empty_world_sweep.py"
PATCH = "scripts/validation/receipt_acquisition.patch"
RUNTIME_PATHS = (
    "robot_sf/benchmark/types.py",
    "robot_sf/benchmark/map_runner/map_runner.py",
    "robot_sf/benchmark/map_runner/map_runner_batch_runner.py",
    "robot_sf/benchmark/camera_ready/_resume_plan.py",
)
MODULES = (
    "robot_sf/benchmark/receipt_stages.py",
    "robot_sf/benchmark/receipt_contract.py",
    "scripts/validation/receipt_canary.py",
    "scripts/validation/receipt_harness.py",
    "configs/benchmarks/releases/behaviour_gate_0_1_0.json",
)


def _blob(repo: Path, source: str, relative: str, *, optional: bool = False) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{source}:{relative}"], cwd=repo, capture_output=True, check=False
    )
    if result.returncode:
        if optional:
            return None
        raise ValueError("pinned acquisition bundle is missing a required committed file")
    return result.stdout


def _digest(value: bytes | None) -> str | None:
    return hashlib.sha256(value).hexdigest() if value is not None else None


def _patched_files(repo: Path, runtime: str, harness: str, patch: bytes) -> dict[str, bytes]:
    originals = {name: cast("bytes", _blob(repo, runtime, name)) for name in RUNTIME_PATHS}
    desired = {name: cast("bytes", _blob(repo, harness, name)) for name in RUNTIME_PATHS}
    if originals == desired:
        return originals
    # Limit patch input and output to the audited acquisition files. Never substitute an
    # entire newer map runner: its policy/control implementation belongs to the runtime source.
    with tempfile.TemporaryDirectory(prefix="receipt-acquisition-") as temporary:
        directory = Path(temporary)
        for relative, content in originals.items():
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        headers = [line for line in patch.decode().splitlines() if line.startswith("+++ ")]
        if not headers or any(line[6:] not in RUNTIME_PATHS for line in headers):
            raise ValueError("acquisition patch escapes its bounded runtime paths")
        checked = subprocess.run(
            ["git", "apply", "--check", "-"],
            input=patch,
            cwd=directory,
            capture_output=True,
            check=False,
        )
        if checked.returncode:
            raise ValueError("historical source is incompatible with the pinned acquisition patch")
        subprocess.run(
            ["git", "apply", "-"], input=patch, cwd=directory, check=True, capture_output=True
        )
        return {relative: (directory / relative).read_bytes() for relative in RUNTIME_PATHS}


def bundle_contents(repo: Path, runtime: str, harness: str) -> tuple[dict[str, bytes], dict] | None:
    """Build the exact overlay from committed objects, preserving unrelated historical code.

    Returns:
        Installed bytes and provenance additions, or None for a historical single-file harness.
    """
    patch = _blob(repo, harness, PATCH, optional=True)
    if patch is None:
        return None
    files = {name: cast("bytes", _blob(repo, harness, name)) for name in (SCRIPT, *MODULES)}
    files.update(_patched_files(repo, runtime, harness, patch))
    manifest = {
        "acquisition_patch_sha256": _digest(patch),
        "bundle_files": {
            name: {
                "sha256": _digest(content),
                "original_sha256": _digest(_blob(repo, runtime, name, optional=True)),
            }
            for name, content in sorted(files.items())
        },
    }
    return files, manifest


def install_bundle(checkout: Path, files: dict[str, bytes], manifest: dict) -> None:
    """Install only reviewed old-or-new files; partial installation can converge on retry."""
    for relative, content in files.items():
        target = checkout / relative
        if target.is_symlink() or any(
            parent.is_symlink() for parent in target.parents if parent != checkout.parent
        ):
            raise ValueError("acquisition bundle source path contains a symlink")
        if target.exists() and _digest(target.read_bytes()) not in {
            manifest["bundle_files"][relative]["original_sha256"],
            _digest(content),
        }:
            raise ValueError("refusing to overwrite uncommitted acquisition bundle edits")
    for relative, content in files.items():
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".receipt-install-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)


def verify_bundle(repo: Path, output_dir: Path, runtime: str) -> tuple[str, ...]:
    """Reconstruct installed files independently of the manifest's asserted digests.

    Returns:
        Only the exact verified acquisition paths allowed to differ from runtime HEAD.
    """
    record = json.loads((output_dir / "harness.json").read_bytes())
    if record["runtime_source_sha"] != runtime:
        raise ValueError("acquisition bundle runtime source differs")
    result = bundle_contents(repo, runtime, record["harness_source_sha"])
    if result is None:
        raise ValueError("incremental acquisition bundle is missing")
    files, expected = result
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("acquisition bundle provenance differs from committed overlay")
    for relative, content in files.items():
        target = repo / relative
        if target.is_symlink() or not target.is_file() or target.read_bytes() != content:
            raise ValueError("installed acquisition bundle differs from committed overlay")
    return tuple(files)
