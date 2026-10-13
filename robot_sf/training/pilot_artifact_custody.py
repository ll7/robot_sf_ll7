"""Hub-first custody for opt-in training pilots with deferred quota-blocked uploads."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import wandb


def sha(path: str | Path) -> str:
    """Return a streaming file SHA-256 digest."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    """Atomically write a JSON metadata file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def portable(value: str) -> str:
    """Replace private home roots with a reconstructable alias.

    Returns:
        Text with portable home-root references."""
    return value.replace(str(Path.home()), "$HOME")


def _snapshot(work: Path, hub: Path, tag: str) -> tuple[Path, list[dict[str, Any]]]:
    """Create a new immutable hub snapshot and verify every copied file.

    Returns:
        The snapshot directory and its verified file inventory."""
    snapshot = hub / tag
    snapshot.mkdir(parents=True, exist_ok=False)
    files = []
    for p in sorted(work.rglob("*")):
        rel = p.relative_to(work)
        if not p.is_file() or p.is_symlink() or "wandb" in rel.parts:
            continue
        if p.name.endswith(".tmp") or (
            p.name == "episodes.jsonl" and p.with_suffix(".jsonl.gz").exists()
        ):
            continue
        q = snapshot / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in {".json", ".jsonl", ".yaml", ".yml", ".log", ".txt", ".py", ".md"}:
            text = portable(p.read_text())
            # Live reset receipts may end with an in-progress write. The snapshot
            # contains only committed whole lines; final admission reads all lines.
            if "training_seed_receipts" in rel.parts and text and not text.endswith("\n"):
                text = text.rsplit("\n", 1)[0] + "\n" if "\n" in text else ""
            q.write_text(text)
        else:
            shutil.copy2(p, q)
        files.append({"artifact": str(rel), "sha256": sha(q), "bytes": q.stat().st_size})
    dump(
        snapshot / "preservation-manifest.json",
        {
            "schema": "controlled-seedpol-custody.v1",
            "files": files,
            "exclusions": "runtime wandb files, symlinks, temporary writes; raw final evaluation is losslessly compressed",
            "path_alias": "$HOME replaces home-root paths in text; binary checkpoint bytes unchanged",
        },
    )
    for item in files:
        if sha(snapshot / item["artifact"]) != item["sha256"]:
            raise ValueError("hub hash readback failed")
    return snapshot, files


def _quota_allows(
    work: Path, hub: Path, snapshot: Path, tag: str, files: list[dict[str, Any]]
) -> bool:
    """Queue known storage refusals without weakening the upload preflight.

    Returns:
        Whether a W&B upload passes the unchanged quota policy."""
    report = work / f"quota-{tag}.json"
    with report.open("w") as output:
        preflight = subprocess.run(
            [
                sys.executable,
                os.environ["SEEDPOL_QUOTA_CHECK"],
                "--json",
                "--planned-upload-gb",
                str(max(0.01, sum(x["bytes"] for x in files) / 1e9)),
                "--min-headroom-gb",
                "10",
            ],
            stdout=output,
            check=False,
        )
    if preflight.returncode:
        quota = json.loads(report.read_text())
        if quota.get("status") != "insufficient" or "upload_breaches_min_headroom" not in quota.get(
            "reason_codes", []
        ):
            raise RuntimeError("artifact preflight failed for an unclassified reason")
        receipt = {
            "state": "DEFERRED_QUOTA",
            "hub_hash_readback": True,
            "wandb_hash_readback": False,
            "manifest_artifact": tag + "/preservation-manifest.json",
            "manifest_sha256": sha(snapshot / "preservation-manifest.json"),
            "reason": "upload_breaches_min_headroom",
            "retry": "after quota preflight succeeds",
        }
        dump(hub / f"preservation-{tag}.json", receipt)
        return False
    return True


def preserve(work: Path, hub: Path, run: Any, tag: str, artifact_name: str) -> dict[str, Any]:
    """Snapshot and verify hub bytes before an optional W&B upload.

    Returns:
        A committed dual-custody receipt, or an explicit quota-deferred receipt.
    """
    snapshot, files = _snapshot(work, hub, tag)
    if not _quota_allows(work, hub, snapshot, tag, files):
        return json.loads((hub / f"preservation-{tag}.json").read_text())
    art = wandb.Artifact(artifact_name + "-" + tag, type="controlled-development-replication")
    art.add_dir(str(snapshot))
    art = run.log_artifact(art)
    art.wait()
    remote = wandb.Api().artifact("ll7/robot_sf/" + art.name)
    assert remote.state == "COMMITTED"
    readback = work / "wandb" / "custody-readback" / tag
    for item in files + [
        {
            "artifact": "preservation-manifest.json",
            "sha256": sha(snapshot / "preservation-manifest.json"),
        }
    ]:
        assert sha(snapshot / item["artifact"]) == item["sha256"], "hub hash readback failed"
        got = remote.get_entry(item["artifact"]).download(root=str(readback))
        assert sha(got) == item["sha256"], "W&B hash readback failed"
    receipt = {
        "artifact": "ll7/robot_sf/" + art.name,
        "state": remote.state,
        "hub": str(snapshot.relative_to(hub.parent.parent)),
        "manifest_artifact": tag + "/preservation-manifest.json",
        "manifest_sha256": sha(snapshot / "preservation-manifest.json"),
        "hub_hash_readback": True,
        "wandb_hash_readback": True,
        "file_count": len(files),
    }
    dump(hub / f"preservation-{tag}.json", receipt)
    return receipt
