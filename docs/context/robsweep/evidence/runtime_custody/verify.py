"""Verify recovered acquisition identity using archived trees and a public base."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CUSTODY = Path(__file__).resolve().parent


def object_id(kind: str, data: bytes) -> str:
    """Compute Git's object identifier, including its type and byte-length header."""
    return hashlib.sha1(f"{kind} {len(data)}\0".encode() + data).hexdigest()


def archived_objects(custody: Path, identity: dict) -> list[tuple[str, bytes]]:
    """Check artifact checksums and Git hashes before restoring archived metadata."""
    result = []
    for item in identity["objects"]:
        data = (custody / Path(item["artifact_path"]).name).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Archived object SHA256 mismatch")
        if object_id(item["kind"], data) != item["git_object_id"]:
            raise ValueError("Archived Git object identity mismatch")
        result.append((item["kind"], data))
    return result


def verify_sources(recovered: dict) -> int:
    """Bind exact acquisition source bytes to the recovered commit's tracked blobs."""
    source = json.loads(
        (ROOT / "docs/context/robsweep/evidence/acquisition_source.json").read_text()
    )
    for item in source["source_files"]:
        path = "/".join(item.get("acquisition_path_parts", [])) or item["path"]
        data = (ROOT / item["artifact_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Archived acquisition-source SHA256 mismatch")
        if object_id("blob", data) != recovered[path][2].decode():
            raise ValueError("Archived acquisition source differs from recovered commit")
    return len(source["source_files"])


def verify(public_repository: Path, custody: Path = CUSTODY) -> dict:
    """Reconstruct metadata without requiring the unpublished diagnostic blobs."""
    identity = json.loads((custody / "identity.json").read_text())
    base = identity["public_base_commit"]
    commit = identity["recovered_episode_commit"]
    public_objects = subprocess.check_output(
        ["git", "rev-parse", "--git-path", "objects"], cwd=public_repository, text=True
    ).strip()
    public_objects = (public_repository / public_objects).resolve()
    with tempfile.TemporaryDirectory() as directory:
        env = {
            **os.environ,
            "GIT_OBJECT_DIRECTORY": directory,
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(public_objects),
            "GIT_NO_LAZY_FETCH": "1",
        }

        def git(*args: str, data: bytes | None = None) -> bytes:
            return subprocess.check_output(
                ["git", *args], cwd=public_repository, env=env, input=data
            )

        for kind, data in archived_objects(custody, identity):
            # Partial output blobs are unavailable; their identities remain in the trees.
            # Their existence is unnecessary for metadata comparison after hash verification.
            git("hash-object", "--literally", "-t", kind, "-w", "--stdin", data=data)

        def leaves(ref: str) -> dict:
            result = {}
            for entry in git("ls-tree", "-r", "-z", ref).split(b"\0"):
                if entry:
                    metadata, path = entry.split(b"\t", 1)
                    result[path.decode()] = metadata.split()
            return result

        old, recovered = leaves(base), leaves(commit)
        if any(recovered.get(path) != value for path, value in old.items()):
            raise ValueError("Recovered commit changes a tracked public-base path")
        added = sorted(recovered.keys() - old.keys())
        if added != ["/".join(parts) for parts in identity["added_git_path_parts"]] or not all(
            path.startswith(("configs/benchmarks/robsweep/", "docs/context/robsweep/"))
            or path == "scripts/validation/examiner_robustness_20261012.py"
            for path in added
        ):
            raise ValueError("Recovered additions differ from the diagnostic scope")
        if (
            git("diff", "--raw", "--no-abbrev", "--no-renames", base, commit)
            != (custody / "changes.raw.txt").read_bytes()
        ):
            raise ValueError("Archived public-base diff disagrees with reconstructed trees")
        runtime_diff = git(
            "diff", "--binary", base, commit, "--", *identity["unchanged_root_scopes"]
        )
        if runtime_diff or runtime_diff != (custody / "runtime.diff.patch").read_bytes():
            raise ValueError("Runtime diff is not empty")
        return {
            "preserved_base_paths": len(old),
            "diagnostic_additions": len(added),
            "verified_git_objects": len(identity["objects"]),
            "verified_source_files": verify_sources(recovered),
            "runtime_diff_bytes": len(runtime_diff),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-repository", type=Path, default=ROOT)
    parser.add_argument("--custody-directory", type=Path, default=CUSTODY)
    args = parser.parse_args()
    print(json.dumps(verify(args.public_repository.resolve(), args.custody_directory.resolve())))
