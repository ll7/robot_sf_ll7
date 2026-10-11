"""Evaluate the CI runner expression against trusted and untrusted events."""

from __future__ import annotations

import ast
import copy
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
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name) and node.func.id in {"cancelled", "always"}
        assert not node.args and not node.keywords
        return True if node.func.id == "always" else context["cancelled"]
    if isinstance(node, ast.UnaryOp):
        assert isinstance(node.op, ast.Not)
        return not _evaluate_node(node.operand, context)
    if isinstance(node, ast.Name):
        assert node.id in {"github", "vars", "needs", "steps"}
        return context[node.id]
    if isinstance(node, ast.Attribute):
        parent = _evaluate_node(node.value, context)
        return parent.get(node.attr) if isinstance(parent, dict) else None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Compare):
        assert len(node.ops) == len(node.comparators) == 1
        assert isinstance(node.ops[0], (ast.Eq, ast.NotEq))
        equal = _evaluate_node(node.left, context) == _evaluate_node(node.comparators[0], context)
        return equal if isinstance(node.ops[0], ast.Eq) else not equal
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


def _resolve_runs_on(
    expression: str,
    github: dict[str, Any],
    *,
    enabled: bool = True,
    available: bool = True,
    provenance: str = "true",
    admission_result: str = "success",
) -> str:
    assert expression.startswith("${{") and expression.endswith("}}")
    inner = expression[3:-2].strip().replace("&&", "and").replace("||", "or")
    inner = inner.replace("needs.runner-availability", "needs.runner_availability")
    inner = inner.replace("self-hosted-admission", "self_hosted_admission")
    resolved = _evaluate_node(
        ast.parse(inner, mode="eval"),
        {
            "github": {**github, "run_attempt": int(github["run_attempt"])},
            "vars": {"ROBOT_SF_SELF_HOSTED_CI_ENABLED": "true" if enabled else ""},
            "needs": {
                "self_hosted_admission": {
                    "result": admission_result,
                    "outputs": {"self_hosted": provenance},
                },
                "runner_availability": {
                    "outputs": {
                        name.replace("-", "_"): "true" if available else "" for name in ROUTED_JOBS
                    }
                },
            },
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
    author: str | None = "ll7",
) -> dict[str, Any]:
    return {
        "event_name": event,
        "actor": actor,
        "triggering_actor": triggering_actor,
        "repository": repository,
        "run_attempt": "1",
        "event": {
            "pull_request": {
                "head": {"repo": {"full_name": head_repository}},
                "user": {"login": author} if author is not None else {},
            }
        },
    }


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (_github_context("push"), "robot-sf-ci-ephemeral"),
        (_github_context("pull_request"), "robot-sf-ci-ephemeral"),
        pytest.param(
            _github_context("pull_request", author="dependabot[bot]"),
            "ubuntu-latest",
            id="bot-author-maintainer-trigger",
        ),
        pytest.param(
            _github_context("pull_request", author="outsider"),
            "ubuntu-latest",
            id="other-author-maintainer-trigger",
        ),
        pytest.param(
            _github_context("pull_request", author=None),
            "ubuntu-latest",
            id="missing-author",
        ),
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
    assert len(expressions) == len(ROUTED_JOBS)
    for name in ROUTED_JOBS:
        job = jobs[name]
        assert _resolve_runs_on(job["runs-on"], context) == expected, name
        expected_permissions = {"contents": "read"}
        if name == "fast-feedback":
            # The run-scoped snapshot API needs read access even on failed-job retries.
            expected_permissions["actions"] = "read"
        assert job["permissions"] == expected_permissions


def test_unset_rollout_switch_keeps_trusted_jobs_hosted() -> None:
    """A PR can complete its own hosted CI before runner installation."""
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    for name in ROUTED_JOBS:
        assert (
            _resolve_runs_on(jobs[name]["runs-on"], _github_context("pull_request"), enabled=False)
            == "ubuntu-latest"
        )


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_missing_idle_capacity_keeps_trusted_jobs_hosted(event: str) -> None:
    """A successful trust check alone must not strand work on a missing runner."""
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    for name in ROUTED_JOBS:
        assert (
            _resolve_runs_on(jobs[name]["runs-on"], _github_context(event), available=False)
            == "ubuntu-latest"
        ), name


def test_retry_is_hosted_even_with_stale_successful_capacity_output() -> None:
    """A watchdog or manual retry cannot repeat the self-hosted queue failure."""
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    context = {**_github_context("push"), "run_attempt": "2"}
    for name in ROUTED_JOBS:
        assert _resolve_runs_on(jobs[name]["runs-on"], context) == "ubuntu-latest", name


@pytest.mark.parametrize(
    ("provenance", "available", "attempt", "expected"),
    [
        ("true", True, "1", "robot-sf-ci-ephemeral"),
        ("true", False, "1", "ubuntu-latest"),
        ("false", True, "1", "ubuntu-latest"),
        ("", True, "1", "ubuntu-latest"),
        ("unknown", True, "1", "ubuntu-latest"),
        ("true", True, "2", "ubuntu-latest"),
    ],
)
def test_provenance_and_availability_are_both_required(
    provenance: str, available: bool, attempt: str, expected: str
) -> None:
    """A healthy runner cannot override denied or missing provenance admission."""
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    context = {**_github_context("push"), "run_attempt": attempt}
    for name in ROUTED_JOBS:
        assert (
            _resolve_runs_on(
                jobs[name]["runs-on"], context, provenance=provenance, available=available
            )
            == expected
        ), name


def test_inventory_token_and_recovery_authority_stay_in_hosted_trusted_jobs() -> None:
    """Never execute PR source with the inventory or cancellation credentials."""
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    router = jobs["runner-availability"]
    assert router["runs-on"] == "ubuntu-latest"
    assert router["permissions"] == {"contents": "read"}
    assert router["continue-on-error"] is True
    assert "self-hosted-admission" in router["needs"]
    probe = next(step for step in router["steps"] if step.get("id") == "route")
    assert "needs.self-hosted-admission.outputs.self_hosted == 'true'" in probe["if"]
    assert "needs.self-hosted-admission.result == 'success'" in probe["if"]
    assert "github.event.pull_request.base.ref == 'main'" in probe["if"]
    assert "github.ref == 'refs/heads/main'" in probe["if"]
    _assert_router_app_token_boundary(yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8")))
    checkout = router["steps"][0]
    assert "pull_request.base.sha" in checkout["with"]["ref"]
    assert "pull_request.head.sha" not in checkout["with"]["ref"]
    assert checkout["with"]["persist-credentials"] is False
    for name in ROUTED_JOBS:
        assert "secrets." not in str(jobs[name])
        assert "always()" not in jobs[name]["if"]
        assert "!cancelled()" in jobs[name]["if"]
        assert "needs.dispatch-ownership.result == 'success'" in jobs[name]["if"]
    watchdog = yaml.safe_load(
        (ROOT / ".github/workflows/ci-runner-watchdog.yml").read_text(encoding="utf-8")
    )
    recovery = watchdog["jobs"]["recover"]
    assert recovery["runs-on"] == "ubuntu-latest"
    assert watchdog["permissions"] == {"contents": "read", "actions": "write"}
    assert recovery["steps"][0]["with"]["ref"] == "${{ github.sha }}"
    assert "download-artifact" not in str(recovery)


def test_failed_provenance_job_cannot_use_stale_positive_outputs() -> None:
    """Failed admission never grants access even if an earlier step wrote true."""
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    for name in ROUTED_JOBS:
        assert (
            _resolve_runs_on(
                jobs[name]["runs-on"], _github_context("push"), admission_result="failure"
            )
            == "ubuntu-latest"
        ), name


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


APP_TOKEN_ACTION = "actions/create-github-app-token@fee1f7d63c2ff003460e3d139729b119787bc349"
APP_TOKEN_INPUT = "${{ steps.router-token.outputs.token }}"


def _token_steps(workflow: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    steps = workflow["jobs"]["runner-availability"]["steps"]
    mint = [step for step in steps if step.get("id") == "router-token"]
    assert len(mint) == 1, "exactly one inventory App token mint is required"
    return mint[0], next(step for step in steps if step.get("id") == "route")


def _token_condition(
    step: dict[str, Any],
    github: dict[str, Any],
    *,
    enabled: str = "true",
    provenance: str = "true",
    admission: str = "success",
    mint_outcome: str = "success",
    token_present: bool = True,
) -> bool:
    expression = step["if"].strip().removeprefix("${{").removesuffix("}}")
    expression = expression.replace("&&", "and").replace("||", "or")
    expression = expression.replace("self-hosted-admission", "self_hosted_admission")
    expression = expression.replace("router-token", "router_token").replace("== false", "== False")
    return bool(
        _evaluate_node(
            ast.parse(expression.strip(), mode="eval"),
            {
                "github": github,
                "vars": {"ROBOT_SF_SELF_HOSTED_CI_ENABLED": enabled},
                "needs": {
                    "self_hosted_admission": {
                        "result": admission,
                        "outputs": {"self_hosted": provenance},
                    }
                },
                "steps": {
                    "router_token": {
                        "outcome": mint_outcome,
                        "outputs": {"token": "present" if token_present else ""},
                    }
                },
            },
        )
    )


def _mint_context(**changes: Any) -> dict[str, Any]:
    return {
        "event_name": "pull_request",
        "repository": "ll7/robot_sf_ll7",
        "actor": "ll7",
        "triggering_actor": "ll7",
        "run_attempt": 1,
        "ref": "refs/pull/123/merge",
        "event": {
            "pull_request": {
                "base": {"ref": "main"},
                "head": {"repo": {"full_name": "ll7/robot_sf_ll7", "fork": False}},
                "user": {"login": "ll7"},
            }
        },
        **changes,
    }


def _assert_router_app_token_boundary(workflow: dict[str, Any]) -> None:
    """Check real workflow credential custody, scope, failure policy and event admission."""
    jobs = workflow["jobs"]
    router = jobs["runner-availability"]
    mint, probe = _token_steps(workflow)
    mint_locations = [
        name
        for name, job in jobs.items()
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("actions/create-github-app-token@")
    ]
    assert mint_locations == ["runner-availability"], "mint must stay in availability only"
    assert mint["uses"] == APP_TOKEN_ACTION, "App action must use the reviewed full release SHA"
    assert mint["with"] == {
        "app-id": "${{ secrets.ROBOT_SF_ROUTER_APP_ID }}",
        "private-key": "${{ secrets.ROBOT_SF_ROUTER_APP_KEY }}",
        "owner": "ll7",
        "repositories": "robot_sf_ll7",
        "permission-administration": "read",
    }, "App inputs must request only administration read for the single repository with revocation"
    assert router["runs-on"] == "ubuntu-latest", "credential job must stay hosted"
    assert router["permissions"] == {"contents": "read"}, "job token stays read-only"
    for label, subject in [("job", router), ("mint", mint), ("probe", probe)]:
        assert subject.get("continue-on-error") is True, f"{label} failure must not fail CI"
    assert [step.get("id") for step in router["steps"]] == [None, "router-token", "route"], (
        "no extra step may execute with the credential"
    )
    checkout = router["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["ref"] == (
        "${{ github.event_name == 'pull_request' && github.event.pull_request.base.sha || github.sha }}"
    ), "credential consumer must use trusted base source"
    assert checkout["with"]["persist-credentials"] is False
    assert probe["with"]["github-token"] == APP_TOKEN_INPUT, "probe must use the App token"
    assert probe["with"]["retries"] == 0, "inventory failures must not retry"
    _assert_token_references(workflow)
    for options in [
        {"mint_outcome": "failure"},
        {"mint_outcome": "skipped"},
        {"token_present": False},
    ]:
        assert not _token_condition(probe, _mint_context(), **options), (
            "failed/skipped/empty mint must skip inventory and leave hosted outputs empty"
        )
    for options in [
        {"enabled": ""},
        {"enabled": "false"},
        {"provenance": "false"},
        {"provenance": ""},
        {"admission": "failure"},
        {"admission": "skipped"},
    ]:
        for step in (mint, probe):
            assert not _token_condition(step, _mint_context(), **options), (
                "mint and probe require rollout and successful commit provenance"
            )
    for case, context, expected in _mint_event_cases():
        for step in (mint, probe):
            assert _token_condition(step, context) is expected, f"mint/probe event policy: {case}"


def _assert_token_references(workflow: dict[str, Any]) -> None:
    """Refuse token or App-secret references outside their exact action input locations."""
    allowed = {
        ("jobs", "runner-availability", "steps", 1, "with", "app-id"),
        ("jobs", "runner-availability", "steps", 1, "with", "private-key"),
        ("jobs", "runner-availability", "steps", 2, "if"),
        ("jobs", "runner-availability", "steps", 2, "with", "github-token"),
    }

    def inspect(value: Any, path: tuple[Any, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                inspect(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, (*path, index))
        elif isinstance(value, str) and (
            "ROBOT_SF_ROUTER_APP_" in value or "steps.router-token" in value
        ):
            assert path in allowed, (
                "App credential must not escape mint inputs and probe input/guard"
            )

    inspect(workflow)


def _mint_event_cases() -> list[tuple[str, dict[str, Any], bool]]:
    cases = [("trusted PR", _mint_context(), True)]
    cases.append(("main push", _mint_context(event_name="push", ref="refs/heads/main"), True))
    for field, value in [
        ("repository", "outsider/robot_sf_ll7"),
        ("actor", "dependabot[bot]"),
        ("actor", "other"),
        ("triggering_actor", "github-actions[bot]"),
        ("triggering_actor", "other"),
        ("run_attempt", 2),
    ]:
        cases.append((f"{field}={value}", _mint_context(**{field: value}), False))
    for event in ["pull_request_target", "workflow_run", "workflow_dispatch", "merge_group"]:
        cases.append((event, _mint_context(event_name=event), False))
    cases.append(
        ("feature push", _mint_context(event_name="push", ref="refs/heads/feature"), False)
    )
    for case, section, value in [
        ("fork repository", "head", {"repo": {"full_name": "outsider/repo", "fork": True}}),
        ("fork flag", "head", {"repo": {"full_name": "ll7/robot_sf_ll7", "fork": True}}),
        ("missing fork flag", "head", {"repo": {"full_name": "ll7/robot_sf_ll7"}}),
        ("other PR author", "user", {"login": "other"}),
        ("non-main base", "base", {"ref": "feature"}),
    ]:
        context = _mint_context()
        context["event"]["pull_request"][section] = value
        cases.append((case, context, False))
    return cases


def test_router_app_token_boundary() -> None:
    """Inventory authentication is scoped and usable only by the trusted hosted probe."""
    _assert_router_app_token_boundary(yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("extra-job", "availability only"),
        ("write", "only administration read"),
        ("extra-permission", "only administration read"),
        ("all-repositories", "single repository"),
        ("disable-revoke", "revocation"),
        ("tag", "full release SHA"),
        ("job-output", "must not escape"),
        ("env", "must not escape"),
        ("script", "must not escape"),
        ("pr-head", "trusted base source"),
        ("mint-failure", "mint failure must not fail CI"),
        ("probe-failure", "probe failure must not fail CI"),
        ("job-failure", "job failure must not fail CI"),
        ("failed-mint-probe", "failed/skipped/empty mint"),
        ("bot-mint", "event policy: actor=dependabot"),
        ("fork-mint", "event policy: fork flag"),
        ("provenance", "successful commit provenance"),
    ],
)
def test_app_token_check_rejects_unsafe_workflow_mutations(mutation: str, reason: str) -> None:
    """The check must detect credential widening/leakage, lost fallback and unsafe minting."""
    workflow = copy.deepcopy(yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8")))
    router = workflow["jobs"]["runner-availability"]
    mint, probe = _token_steps(workflow)
    if mutation == "extra-job":
        workflow["jobs"]["extra"] = {"steps": [copy.deepcopy(mint)]}
    elif mutation in {"write", "extra-permission", "all-repositories", "disable-revoke"}:
        key, value = {
            "write": ("permission-administration", "write"),
            "extra-permission": ("permission-contents", "read"),
            "all-repositories": ("repositories", ""),
            "disable-revoke": ("skip-token-revoke", True),
        }[mutation]
        mint["with"][key] = value
    elif mutation == "tag":
        mint["uses"] = "actions/create-github-app-token@v2"
    elif mutation == "job-output":
        router["outputs"]["token"] = APP_TOKEN_INPUT
    elif mutation == "env":
        probe["env"]["GH_TOKEN"] = APP_TOKEN_INPUT
    elif mutation == "script":
        probe["with"]["script"] += f"\nconsole.log('{APP_TOKEN_INPUT}');"
    elif mutation == "pr-head":
        router["steps"][0]["with"]["ref"] = "${{ github.event.pull_request.head.sha }}"
    elif mutation in {"mint-failure", "probe-failure", "job-failure"}:
        {"mint-failure": mint, "probe-failure": probe, "job-failure": router}[mutation][
            "continue-on-error"
        ] = False
    elif mutation == "failed-mint-probe":
        probe["if"] = mint["if"]
    else:
        guard = {
            "bot-mint": "github.actor == 'll7' &&",
            "fork-mint": "github.event.pull_request.head.repo.fork == false &&",
            "provenance": "needs.self-hosted-admission.result == 'success' &&",
        }[mutation]
        assert guard in mint["if"], "mutation must change the intended guard"
        mint["if"] = mint["if"].replace(guard, "")
    with pytest.raises(AssertionError, match=re.escape(reason)):
        _assert_router_app_token_boundary(workflow)
