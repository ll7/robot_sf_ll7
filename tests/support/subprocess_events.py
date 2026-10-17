"""Capture child completion with a generous hang guard and group cleanup."""

import subprocess

from tests.support.process_teardown import terminate_process_group


def capture_completion(command, *, cwd=None, env=None, hang_guard_seconds=300):
    """Wait for completion; a hang fails after reaping the entire child group."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=hang_guard_seconds)
    except subprocess.TimeoutExpired:
        terminate_process_group(process)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
