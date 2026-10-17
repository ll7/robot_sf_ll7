"""Quota refusal preserves checkpoints and does not abort the training owner."""

import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from robot_sf.training import pilot_artifact_custody as current


def custody_under_test():
    """Allow the same behavioral regression to run against the original pilot."""
    path = os.environ.get("PILOT_CUSTODY_UNDER_TEST")
    if not path:
        return current
    spec = importlib.util.spec_from_file_location("old_pilot_custody", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quota_refusal_keeps_hub_checkpoint_and_returns_pending(tmp_path, monkeypatch):
    """Known quota refusal queues upload while the caller continues training."""
    module = custody_under_test()
    work, hub = tmp_path / "work", tmp_path / "hub"
    work.mkdir()
    payload = b"immutable-checkpoint-bytes"
    (work / "checkpoint.zip").write_bytes(payload)
    monkeypatch.setenv("SEEDPOL_QUOTA_CHECK", "quota-program")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace())

    def refused(command, *, stdout, check):
        stdout.write(
            json.dumps({"status": "insufficient", "reason_codes": ["upload_breaches_min_headroom"]})
        )
        stdout.flush()
        if check:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(module.subprocess, "run", refused)
    receipt = module.preserve(work, hub, None, "checkpoint", "pilot")
    assert receipt["state"] == "DEFERRED_QUOTA"
    assert receipt["hub_hash_readback"] and not receipt["wandb_hash_readback"]
    assert (hub / "checkpoint/checkpoint.zip").read_bytes() == payload
    with pytest.raises(FileExistsError):
        module.preserve(work, hub, None, "checkpoint", "pilot")


def test_unclassified_preflight_error_is_not_reported_as_preservation(tmp_path, monkeypatch):
    """Unknown failures remain visible rather than being labeled quota deferrals."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("SEEDPOL_QUOTA_CHECK", "quota-program")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace())

    def refused(command, *, stdout, check):
        stdout.write('{"status": "unknown", "reason_codes": ["unauthorised"]}')
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(current.subprocess, "run", refused)
    with pytest.raises(RuntimeError, match="unclassified"):
        current.preserve(work, tmp_path / "hub", None, "snapshot", "pilot")


@pytest.mark.parametrize(
    "failure,message",
    [
        ("state", "not committed"),
        ("hub", "hub hash readback failed"),
        ("wandb", "W&B hash readback failed"),
    ],
)
def test_optimized_python_still_refuses_unverified_custody(tmp_path, monkeypatch, failure, message):
    """Artifact admission must remain fail-closed when Python strips assertions."""
    from pathlib import Path

    namespace = {"__name__": "optimized_pilot_custody"}
    source = Path(current.__file__).read_text()
    exec(compile(source, current.__file__, "exec", optimize=1), namespace)
    work, hub = tmp_path / "work", tmp_path / "hub"
    work.mkdir()
    (work / "checkpoint.zip").write_bytes(b"checkpoint")
    downloaded = tmp_path / "downloaded"

    def download(*, root):
        downloaded.write_bytes(b"wrong" if failure == "wandb" else b"checkpoint")
        return downloaded

    remote = SimpleNamespace(
        state="PENDING" if failure == "state" else "COMMITTED",
        get_entry=lambda name: SimpleNamespace(download=download),
    )
    artifact = SimpleNamespace(name="pilot:v0", add_dir=lambda path: None, wait=lambda: None)

    def log_artifact(art):
        if failure == "hub":
            (hub / "snapshot/checkpoint.zip").write_bytes(b"corrupted")
        return art

    namespace["wandb"] = SimpleNamespace(
        Artifact=lambda *args, **kwargs: artifact,
        Api=lambda: SimpleNamespace(artifact=lambda name: remote),
    )
    namespace["_quota_allows"] = lambda *args: True
    with pytest.raises((ValueError, RuntimeError), match=message):
        namespace["preserve"](
            work, hub, SimpleNamespace(log_artifact=log_artifact), "snapshot", "pilot"
        )
    assert not (hub / "preservation-snapshot.json").exists()
