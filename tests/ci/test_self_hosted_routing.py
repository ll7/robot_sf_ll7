"""Evaluate the CI runner expression against trusted and untrusted events."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github/workflows/ci.yml"
SETUP_SCRIPT = ROOT / "scripts/ci/self_hosted/setup.sh"
ROUTED_JOBS = (
    "fast-feedback",
    "smoke-artifacts",
    "wheel-smoke-install",
    "examples-smoke",
    "notebooks-smoke",
)


def _evaluate_node(node: ast.AST, context: dict[str, Any]) -> Any:
    """Evaluate only the expression features used in the runs-on contract."""
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, context)
    if isinstance(node, ast.Name):
        assert node.id in {"github", "vars"}
        return context[node.id]
    if isinstance(node, ast.Attribute):
        parent = _evaluate_node(node.value, context)
        return parent.get(node.attr) if isinstance(parent, dict) else None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Compare):
        assert len(node.ops) == len(node.comparators) == 1
        assert isinstance(node.ops[0], ast.Eq)
        return _evaluate_node(node.left, context) == _evaluate_node(node.comparators[0], context)
    if isinstance(node, ast.BoolOp):
        return _evaluate_bool_op(node, context)
    raise AssertionError(f"Unsupported runs-on expression node: {ast.dump(node)}")


def _evaluate_bool_op(node: ast.BoolOp, context: dict[str, Any]) -> Any:
    """Preserve GitHub's short-circuit value selection for and/or."""
    assert isinstance(node.op, (ast.And, ast.Or))
    result: Any = isinstance(node.op, ast.And)
    for value in node.values:
        result = _evaluate_node(value, context)
        if (isinstance(node.op, ast.And) and not result) or (
            isinstance(node.op, ast.Or) and result
        ):
            return result
    return result


def _resolve_runs_on(expression: str, github: dict[str, Any], *, enabled: bool = True) -> str:
    assert expression.startswith("${{") and expression.endswith("}}")
    inner = expression[3:-2].strip().replace("&&", "and").replace("||", "or")
    resolved = _evaluate_node(
        ast.parse(inner, mode="eval"),
        {
            "github": github,
            "vars": {"ROBOT_SF_SELF_HOSTED_CI_ENABLED": "true" if enabled else ""},
        },
    )
    assert isinstance(resolved, str)
    return resolved


def _github_context(
    event: str,
    *,
    actor: str = "ll7",
    triggering_actor: str = "ll7",
    repository: str = "ll7/robot_sf_ll7",
    head_repository: str = "ll7/robot_sf_ll7",
) -> dict[str, Any]:
    return {
        "event_name": event,
        "actor": actor,
        "triggering_actor": triggering_actor,
        "repository": repository,
        "event": {"pull_request": {"head": {"repo": {"full_name": head_repository}}}},
    }


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (_github_context("push"), "robot-sf-ci-ephemeral"),
        (_github_context("pull_request"), "robot-sf-ci-ephemeral"),
        (_github_context("pull_request", head_repository="outsider/robot_sf_ll7"), "ubuntu-latest"),
        (_github_context("pull_request", actor="dependabot[bot]"), "ubuntu-latest"),
        (_github_context("pull_request", triggering_actor="dependabot[bot]"), "ubuntu-latest"),
        (_github_context("pull_request", triggering_actor="maintainer"), "ubuntu-latest"),
        (_github_context("pull_request_target"), "ubuntu-latest"),
        (_github_context("workflow_run"), "ubuntu-latest"),
        (_github_context("issue_comment"), "ubuntu-latest"),
        (_github_context("pull_request_review_comment"), "ubuntu-latest"),
        (_github_context("workflow_dispatch"), "ubuntu-latest"),
        (_github_context("merge_group"), "ubuntu-latest"),
        (_github_context("push", repository="outsider/robot_sf_ll7"), "ubuntu-latest"),
    ],
)
def test_routed_jobs_resolve_to_expected_runner(context: dict[str, Any], expected: str) -> None:
    """Only author-started push and same-repository PR jobs reach the private label."""
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    expressions = {jobs[name]["runs-on"] for name in ROUTED_JOBS}
    assert len(expressions) == 1
    for name in ROUTED_JOBS:
        job = jobs[name]
        assert _resolve_runs_on(job["runs-on"], context) == expected, name
        assert job["permissions"] == {"contents": "read"}


def test_unset_rollout_switch_keeps_trusted_jobs_hosted() -> None:
    """A PR can complete its own hosted CI before runner installation."""
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    for name in ROUTED_JOBS:
        assert (
            _resolve_runs_on(jobs[name]["runs-on"], _github_context("pull_request"), enabled=False)
            == "ubuntu-latest"
        )


def test_routed_jobs_do_not_persist_checkout_credentials() -> None:
    """No routed checkout may leave its job token in the disk-backed work volume."""
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    routed = {
        name: job
        for name, job in jobs.items()
        if "robot-sf-ci-ephemeral" in str(job.get("runs-on", ""))
    }
    assert set(ROUTED_JOBS) <= routed.keys()
    for name, job in routed.items():
        checkouts = [
            step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@")
        ]
        assert checkouts, name
        for step in checkouts:
            assert (step.get("with") or {}).get("persist-credentials") is False, (
                name,
                step.get("name"),
            )


def test_routed_jobs_skip_package_install_only_on_self_hosted() -> None:
    """The image owns system packages; hosted fallback keeps its existing setup."""
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    expected = "${{ runner.environment == 'github-hosted' && 'true' || 'false' }}"
    for name in ROUTED_JOBS:
        setup = next(
            step
            for step in jobs[name]["steps"]
            if step.get("uses") == "./.github/actions/setup-ci-python"
        )
        assert setup["with"]["install-system-packages"] == expected


def test_container_setup_keeps_ephemeral_and_no_host_mounts() -> None:
    """Keep the setup's core isolation controls visible in the reviewed script."""
    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    for required in (
        "--ephemeral",
        "--read-only",
        "--user 1001:1001",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--tmpfs /home/runner:",
        "read -r cpus memory workers < <(slot_limits)",
        '--cpus "$cpus" --memory "$memory" --memory-swap "$memory"',
        "--jq .token |",
        "--rm --detach --interactive",
        "flock -x",
        '--network "$network"',
        "probe_network",
        "check_docker_disk",
        "ACTIONS_RUNNER_HOOK_JOB_STARTED=/usr/local/libexec/robot-sf-job-started.sh",
        "--ephemeral --disableupdate --replace",
    ):
        assert required in script
    assert "--volume" not in script
    assert re.findall(r"--mount(?:\s+|=)([^\s\\]+)", script) == [
        "type=volume,dst=/home/runner/_work"
    ]
    assert re.search(r"(?<!\S)-v(?:\s|=)", script) is None
    assert "src=" not in script
    assert "source=" not in script
    assert "/var/run/docker.sock" not in script
    # actions/setup-python's Linux executable embeds /opt/hostedtoolcache in
    # RUNPATH; keep it resolvable when a test launches Python with a clean env.
    assert "ln -s /home/runner/_tool /opt/hostedtoolcache" in script
    for environment in (
        "RUNNER_TEMP=/home/runner/_work/_temp",
        "RUNNER_TOOL_CACHE=/home/runner/_tool",
        "UV_CACHE_DIR=/home/runner/_work/_uv_cache",
        "TMPDIR=/home/runner/_work/_tmp",
        "PIP_CACHE_DIR=/home/runner/_work/_pip_cache",
        'PYTEST_NUM_WORKERS="$workers"',
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
    ):
        assert environment in script


@pytest.mark.parametrize(
    ("hostname", "expected"),
    (
        ("imech156-u", "8 16g 4"),
        ("imech036", "4 8g 2"),
        ("imech039", "4 8g 2"),
        ("auxme-imech036", "4 8g 2"),
        ("auxme-imech039", "4 8g 2"),
    ),
)
def test_slot_limits_keep_each_host_bounded(hostname: str, expected: str) -> None:
    """Execute the read-only limits command without depending on this host."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            'hostname() { printf "%s\\n" "$TEST_HOST"; }; source "$1" limits',
            "slot-limits-test",
            str(SETUP_SCRIPT),
        ],
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "TEST_HOST": hostname},
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == expected


def test_runner_temporary_paths_resolve_to_intended_mounts() -> None:
    """Credential scratch stays in memory; build scratch has volume capacity."""
    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    assert "--work _work" in script
    docker_run = script.split("if ! docker run", 1)[1].split('"$image" "$name"', 1)[0]
    mounts = [
        (spec.split(":", 1)[0], "tmpfs") for spec in re.findall(r"--tmpfs\s+([^\s\\]+)", docker_run)
    ]
    mounts += [
        (destination, "volume")
        for destination in re.findall(r"--mount\s+type=volume,dst=([^\s\\]+)", docker_run)
    ]
    environment = dict(re.findall(r"--env\s+([A-Z_]+)=([^\s\\]+)", docker_run))

    def filesystem(path: str) -> str:
        candidates = [
            (len(destination), kind)
            for destination, kind in mounts
            if path == destination or path.startswith(f"{destination}/")
        ]
        assert candidates, path
        return max(candidates)[1]

    assert environment["RUNNER_TEMP"] == "/home/runner/_work/_temp"
    for key in ("RUNNER_TEMP", "RUNNER_TOOL_CACHE"):
        assert filesystem(environment[key]) == "tmpfs", key
    assert filesystem("/home/runner/_work") == "volume"
    for key in ("TMPDIR", "PIP_CACHE_DIR", "UV_CACHE_DIR"):
        assert filesystem(environment[key]) == "volume", key


def test_container_image_preloads_routed_job_tools() -> None:
    """Pin native builds, video output, and release hydration to image packages."""
    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    apt_packages = script.split("apt-get install -y --no-install-recommends", 1)[1].split(
        "&& rm -rf /var/lib/apt/lists/*", 1
    )[0]
    assert {"build-essential", "cmake", "ffmpeg", "gh", "git-lfs"} <= set(apt_packages.split())
    assert "https://nodejs.org/dist/v22.23.2/node-v22.23.2-linux-x64.tar.gz" in script
    assert "b294a556e639d64338823920e5866c21c02741742d2e1529ee1a225c1ec9252a" in script
    assert "sha256sum -c -" in script
    for tool in ("node", "npm", "npx"):
        assert f"ln -s /opt/node/bin/{tool} /usr/local/bin/{tool}" in script


def test_job_started_hook_path_has_runner_accepted_extension() -> None:
    """The runner rejects hook paths without .sh/.ps1/.js and fails every job."""
    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    env_lines = [line for line in script.splitlines() if "ACTIONS_RUNNER_HOOK_JOB_STARTED=" in line]
    assert env_lines
    for line in env_lines:
        hook_path = line.split("ACTIONS_RUNNER_HOOK_JOB_STARTED=", 1)[1].strip()
        assert hook_path.endswith((".sh", ".ps1", ".js")), hook_path
        assert f"job_started_hook.sh {hook_path}" in script
