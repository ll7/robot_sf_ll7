"""Exercise the real shell wrapper in a clean temporary Git repository."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tests.support.environment_guards import configure_git_identity

ROOT = Path(__file__).resolve().parents[2]


def wrapper_repository(tmp_path):
    """Copy owned entrypoints and replace only dependency/test execution boundaries."""
    repo = tmp_path / "repo"
    scripts = repo / "scripts/dev"
    scripts.mkdir(parents=True)
    for name in (
        "run_tests_parallel.sh",
        "common_setup.sh",
        "affected_test_selection.py",
        "train_suite_receipt.py",
    ):
        source = ROOT / "scripts/dev" / name
        if source.exists():
            target = scripts / name
            shutil.copyfile(source, target)
            target.chmod(0o755)
    support = repo / "tests/support"
    support.mkdir(parents=True)
    (support / "optional_test_allowlist.txt").write_text("")
    (repo / "tests/test_one.py").write_text("def test_one(): assert True\n")
    (repo / "fast-pysf/tests").mkdir(parents=True)
    (repo / "fast-pysf/tests/test_two.py").write_text("def test_two(): assert True\n")
    (repo / ".gitignore").write_text("/.venv/\n/fake-bin/\n/captured.jsonl\n/output/\n")
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    fake_bin = repo / "fake-bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(f"""#!{sys.executable}
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with Path('captured.jsonl').open('a') as stream:
    stream.write(json.dumps({{'args': args, 'addopts': os.environ.get('PYTEST_ADDOPTS'), 'shards': os.environ.get('PYTEST_SHARD_COUNT')}}) + '\\n')
if args[:2] == ['run', 'python']:
    name = Path(args[2]).name
    if name == 'resolve_pytest_workers.py': print('2')
    elif name == 'affected_test_selection.py' and os.environ.get('SELECTION_MODE'):
        print(os.environ['SELECTION_MODE'])
    elif name in {{'affected_test_selection.py', 'train_suite_receipt.py'}}:
        sys.exit(subprocess.run([sys.executable, *args[2:]], check=False).returncode)
    elif name == 'diagnose_xdist_crash.py': pass
    else: sys.exit(99)
elif args[:2] == ['run', 'pytest']:
    if os.environ.get('MOVE_HEAD'):
        Path('tests/test_one.py').write_text('def test_one(): assert 1 == 1\\n')
        subprocess.run(['git', 'add', 'tests/test_one.py'], check=True)
        subprocess.run(['git', 'commit', '-qm', 'moved head'], check=True)
    if os.environ.get('DIRTY_AFTER'):
        Path('untracked_after.py').write_text('pass\\n')
    if os.environ.get('REAL_PYTEST'):
        sys.exit(subprocess.run([sys.executable, '-m', 'pytest', *args[2:]], check=False).returncode)
    print('2 passed, 1 skipped in 0.01s')
    sys.exit(int(os.environ.get('FAKE_PYTEST_EXIT', '0')))
else: sys.exit(99)
""")
    uv.chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    configure_git_identity(repo)
    subprocess.run(
        ["git", "add", ".gitignore", "scripts/dev", "tests", "fast-pysf/tests"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    env = {
        **os.environ,
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        "PYTEST_NUM_WORKERS": "2",
        "PYTEST_FAST_FAIL": "0",
        "PYTEST_ORDER_MODE": "none",
        "PYTEST_SHARD_COUNT": "2",
        "PYTEST_SHARD_INDEX": "1",
        "PYTEST_ADDOPTS": "",
        "ROBOT_SF_SHARD_INCLUDE_SLOW": "0",
        "ROBOT_SF_PYTEST_COVERAGE": "0",
        "PYTEST_DEBUG_TEMPROOT": "fixture",
        "CI": "false",
    }
    for key in (
        "ROBOT_SF_AFFECTED_BASE_REF",
        "ROBOT_SF_AFFECTED_SELECTION_FILE",
        "PR_READY_SERIAL_FALLBACK",
    ):
        env.pop(key, None)
    return repo, env


def run_wrapper(repo, env, *args):
    """Run the copied shell entrypoint and preserve its real status and arguments."""
    return subprocess.run(
        ["bash", "scripts/dev/run_tests_parallel.sh", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def captured_calls(repo):
    """Read fixture-only calls; no ambient environment or secrets are recorded."""
    path = repo / "captured.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
