"""Exercise the actual pilot shell on trainer failure and termination, without submission."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "SLURM/ftplan/released_ppo_pilot.slurm"


@pytest.mark.parametrize(
    "mode,expected_exit", [("failed", 23), ("terminated", 143), ("success", 0)]
)
def test_pilot_preserves_partial_outputs_and_original_exit(tmp_path, mode, expected_exit):
    """Stubs replace external training/W&B, while the production custody Python runs."""
    root = tmp_path / "checkout"
    wrapper = root / "scripts/dev/run_worktree_shared_venv.sh"
    wrapper.parent.mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git = fake_bin / "git"
    git.write_text(
        '#!/bin/sh\nif [ "$1" = rev-parse ]; then printf \'%s\\n\' "$FTPLAN_SOURCE_HEAD"; fi\n'
    )
    git.chmod(0o755)
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "wandb.py").write_text("""import os
from pathlib import Path
class Artifact:
    def __init__(self, *args, **kwargs): pass
    def add_dir(self, path): pass
class Run:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def log_artifact(self, artifact): return self
    def wait(self): Path(os.environ['TEST_WANDB_COMMIT']).write_text('committed')
def init(**kwargs): return Run()
""")
    wrapper.write_text("""#!/usr/bin/env python3
import os, signal, subprocess, sys, time
from pathlib import Path
root = Path(os.environ['ROBOT_SF_ARTIFACT_ROOT'])
if '-m' in sys.argv:
    Path(os.environ['TEST_TRAINER_READY']).touch()
    if os.environ['TEST_MODE'] == 'terminated':
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        while True: time.sleep(0.05)
    sys.exit(23 if os.environ['TEST_MODE'] == 'failed' else 0)
code = sys.stdin.read()
if 'import wandb' in code:
    sys.exit(subprocess.run([sys.executable, '-c', code]).returncode)
(root / 'partial.zip').write_bytes(b'partial-checkpoint')
""")
    wrapper.chmod(0o755)
    hub = tmp_path / "hub"
    ready = tmp_path / "trainer-ready"
    committed = tmp_path / "wandb-committed"
    env = os.environ.copy()
    env.update(
        {
            "PATH": str(fake_bin) + os.pathsep + env["PATH"],
            "PYTHONPATH": str(modules),
            "FTPLAN_REPO_ROOT": str(root),
            "FTPLAN_SOURCE_HEAD": "a" * 40,
            "FTPLAN_REPLICATION_MANIFEST": "unused",
            "FTPLAN_SELECTION_RECEIPT": "unused",
            "FTPLAN_ACTION_CONTRACT_RECEIPT": "unused",
            "FTPLAN_HUB_ROOT": str(hub),
            "FTPLAN_SEGMENT": "1",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_JOB_ID": "test",
            "TEST_MODE": mode,
            "TEST_TRAINER_READY": str(ready),
            "TEST_WANDB_COMMIT": str(committed),
        }
    )
    proc = subprocess.Popen(
        ["bash", str(SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        if mode == "terminated":
            deadline = time.monotonic() + 10
            while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert ready.exists(), "stub trainer did not reach its signal wait"
            os.killpg(proc.pid, signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=15)
        assert proc.returncode == expected_exit, (stdout, stderr)
        dest = hub / "ppo-ftplan-20261012/init100001/segment1"
        assert (dest / "partial.zip").exists(), "exit path did not preserve partial checkpoints"
        assert (dest / "partial.zip").read_bytes() == b"partial-checkpoint"
        status = json.loads((dest / "segment_status.json").read_text())
        assert status["original_exit_code"] == expected_exit
        assert status["status"] == ("complete" if mode == "success" else "incomplete")
        assert committed.exists(), "exit path did not commit W&B custody"
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
