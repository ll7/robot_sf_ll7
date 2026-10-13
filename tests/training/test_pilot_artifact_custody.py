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
