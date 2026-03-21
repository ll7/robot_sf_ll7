"""Probe a compatible SocialForce runtime for Social-Navigation-PyEnvs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ISSUE_NUMBER = 653
DEFAULT_TIMEOUT_SECONDS = 120
EXPECTED_POLICY_FILE = Path("crowd_nav/policy_no_train/socialforce.py")
DEFAULT_BACKEND_SPEC = "socialforce==0.2.3"


@dataclass(frozen=True)
class CommandResult:
    """Result of one external probe command."""

    name: str
    command: list[str]
    returncode: int
    failure_summary: str | None
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ProbeReport:
    """Structured runtime-compatibility report."""

    issue: int
    repo_root: str
    repo_remote_url: str | None
    backend_spec: str
    verdict: str
    failure_stage: str | None
    failure_summary: str | None
    source_contract: dict[str, Any]
    commands: list[CommandResult]


def _extract_remote_url(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "config", "--get", "remote.origin.url"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _run_command(name: str, command: list[str], cwd: Path, timeout_seconds: int) -> CommandResult:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            name=name,
            command=command,
            returncode=124,
            failure_summary=f"timed out after {timeout_seconds}s",
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )

    failure_summary = (
        None if result.returncode == 0 else _detect_failure_summary(result.stdout, result.stderr)
    )
    return CommandResult(
        name=name,
        command=command,
        returncode=result.returncode,
        failure_summary=failure_summary,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _detect_failure_summary(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}"
    lowered = text.lower()
    if "modulenotfounderror" in lowered and "socialforce" in lowered:
        return "missing external socialforce runtime"
    if "unexpected keyword argument 'initial_speed'" in lowered:
        return "current socialforce package API mismatches upstream simulator constructor"
    if "attributeerror" in lowered and "step" in lowered:
        return "compatibility shim did not provide CrowdNav-style simulator step()"
    return "runtime compatibility probe failed"


def _compat_probe_script(repo_root: Path) -> str:
    repo = str(repo_root.resolve())
    return f"""
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import socialforce as backend_socialforce

repo_root = Path({repo!r})
sys.path.insert(0, str(repo_root))
original_socialforce = sys.modules.get('socialforce')

compat = types.ModuleType('socialforce')
compat.__dict__['__doc__'] = 'CrowdNav-style compatibility shim over socialforce==0.2.3'
compat.__dict__['__version__'] = getattr(backend_socialforce, '__version__', 'unknown')

class CompatSimulator:
    def __init__(self, state, *, delta_t=0.4, initial_speed=1.0, v0=10, sigma=0.3):
        self._sim = backend_socialforce.Simulator(delta_t=delta_t)
        self._state = torch.as_tensor(state, dtype=torch.float32)
        self.state = np.asarray(state, dtype=float)
        self.initial_speed = initial_speed
        self.v0 = v0
        self.sigma = sigma
    def step(self):
        out = self._sim.forward(self._state)
        self._state = out.detach().clone()
        self.state = out.detach().cpu().numpy()
        return self.state

compat.Simulator = CompatSimulator
sys.modules['socialforce'] = compat
try:
    action_mod = importlib.import_module('crowd_nav.utils.action')
    state_mod = importlib.import_module('crowd_nav.utils.state')
    policy_mod = importlib.import_module('crowd_nav.policy_no_train.socialforce')
    policy = policy_mod.SocialForce()
    policy.time_step = 0.1
    self_state = state_mod.FullState(0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.0, 1.0, 0.0)
    humans = [state_mod.ObservableState(1.0, 0.5, 0.0, 0.0, 0.3)]
    action = policy.predict(state_mod.JointState(self_state, humans))
    if not isinstance(action, action_mod.ActionXY):
        raise TypeError(f'unexpected action type: {{type(action).__name__}}')
    print(json.dumps({{
        'backend_version': getattr(backend_socialforce, '__version__', 'unknown'),
        'shim_accepts_initial_speed': True,
        'upstream_policy': 'crowd_nav.policy_no_train.socialforce.SocialForce',
        'action_xy': [float(action.vx), float(action.vy)],
        'sim_state_shape': list(policy_mod.socialforce.Simulator(np.array([[0,0,0,0,1,0]], dtype=float), delta_t=0.1, initial_speed=1.0, v0=10, sigma=0.3).step().shape),
    }}))
finally:
    if original_socialforce is None:
        sys.modules.pop('socialforce', None)
    else:
        sys.modules['socialforce'] = original_socialforce
"""


def _backend_signature_script() -> str:
    return """
import inspect
import json
import socialforce
from socialforce import Simulator
print(json.dumps({
    'backend_version': getattr(socialforce, '__version__', 'unknown'),
    'simulator_signature': str(inspect.signature(Simulator.__init__)),
}))
"""


def _extract_source_contract(repo_root: Path) -> dict[str, Any]:
    policy_path = repo_root / EXPECTED_POLICY_FILE
    text = policy_path.read_text(encoding="utf-8")
    expected_kwargs = [
        name
        for name in ("initial_speed", "v0", "sigma")
        if re.search(rf"\b{name}\s*=\s*self\.{name}\b", text)
    ]
    return {
        "upstream_policy": "crowd_nav.policy_no_train.socialforce.SocialForce",
        "upstream_policy_file": str(policy_path),
        "upstream_action_contract": "ActionXY",
        "upstream_state_contract": "JointState(FullState, list[ObservableState])",
        "external_simulator_constructor_expects": expected_kwargs,
        "runtime_strategy": "compatibility shim around external socialforce package",
        "notes": (
            "This issue validates only source-harness runtime compatibility for the upstream "
            "SocialForce policy. Benchmark-facing wrapper retry is separate."
        ),
    }


def run_probe(
    repo_root: Path,
    *,
    backend_spec: str = DEFAULT_BACKEND_SPEC,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> ProbeReport:
    """Run the runtime-compatibility probe against a checked-out upstream repo."""
    repo_root = repo_root.resolve()
    policy_file = repo_root / EXPECTED_POLICY_FILE
    if not policy_file.exists():
        raise FileNotFoundError(f"Required upstream policy file missing: {policy_file}")

    commands: list[CommandResult] = []
    backend_cmd = [
        "uv",
        "run",
        "--with",
        backend_spec,
        "python",
        "-c",
        _backend_signature_script(),
    ]
    backend_result = _run_command(
        "backend_signature", backend_cmd, cwd=repo_root, timeout_seconds=timeout_seconds
    )
    commands.append(backend_result)
    if backend_result.returncode != 0:
        return ProbeReport(
            issue=ISSUE_NUMBER,
            repo_root=str(repo_root),
            repo_remote_url=_extract_remote_url(repo_root),
            backend_spec=backend_spec,
            verdict="blocked by dependency/runtime mismatch",
            failure_stage=backend_result.name,
            failure_summary=backend_result.failure_summary,
            source_contract=_extract_source_contract(repo_root),
            commands=commands,
        )

    compat_cmd = [
        "uv",
        "run",
        "--with",
        backend_spec,
        "python",
        "-c",
        _compat_probe_script(repo_root),
    ]
    compat_result = _run_command(
        "compat_predict_minimal", compat_cmd, cwd=repo_root, timeout_seconds=timeout_seconds
    )
    commands.append(compat_result)

    verdict = "compatible runtime reproduced"
    failure_stage = None
    failure_summary = None
    if compat_result.returncode != 0:
        verdict = "blocked by dependency/runtime mismatch"
        failure_stage = compat_result.name
        failure_summary = compat_result.failure_summary

    contract = _extract_source_contract(repo_root)
    for result in commands:
        if result.returncode == 0 and result.stdout.strip().startswith("{"):
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                continue
            if result.name == "backend_signature":
                contract["backend_version"] = payload.get("backend_version")
                contract["backend_simulator_signature"] = payload.get("simulator_signature")
            elif result.name == "compat_predict_minimal":
                contract["compat_probe_action_xy"] = payload.get("action_xy")
                contract["compat_probe_sim_state_shape"] = payload.get("sim_state_shape")
                contract["shim_accepts_initial_speed"] = payload.get("shim_accepts_initial_speed")

    return ProbeReport(
        issue=ISSUE_NUMBER,
        repo_root=str(repo_root),
        repo_remote_url=_extract_remote_url(repo_root),
        backend_spec=backend_spec,
        verdict=verdict,
        failure_stage=failure_stage,
        failure_summary=failure_summary,
        source_contract=contract,
        commands=commands,
    )


def _render_markdown(report: ProbeReport) -> str:
    lines = [
        f"# Issue {report.issue} Social-Navigation-PyEnvs SocialForce Runtime Probe",
        "",
        f"Verdict: `{report.verdict}`",
        "",
        "## Summary",
        "",
    ]
    if report.verdict == "compatible runtime reproduced":
        lines.extend(
            [
                "The upstream `crowd_nav.policy_no_train.socialforce.SocialForce` policy can run ",
                "against the current external `socialforce` package when the package is wrapped ",
                "with a narrow CrowdNav-style compatibility shim. The upstream policy logic stays ",
                "unchanged; only the external simulator API surface is adapted.",
            ]
        )
    else:
        lines.extend(
            [
                "The upstream SocialForce policy could not be reproduced with the tested external ",
                "runtime. The blocker remains in the dependency/API layer, so benchmark-facing retry ",
                "work is not justified yet.",
            ]
        )
    if report.failure_summary:
        lines.extend(["", f"Failure summary: `{report.failure_summary}`"])
    lines.extend(
        [
            "",
            "## Source contract",
            "",
            f"- upstream policy: `{report.source_contract.get('upstream_policy')}`",
            f"- external constructor kwargs expected by upstream: `{report.source_contract.get('external_simulator_constructor_expects')}`",
            f"- backend package: `{report.backend_spec}`",
            f"- backend simulator signature: `{report.source_contract.get('backend_simulator_signature', 'unknown')}`",
            f"- compatibility runtime strategy: `{report.source_contract.get('runtime_strategy')}`",
        ]
    )
    if report.verdict == "compatible runtime reproduced":
        lines.extend(
            [
                f"- compat probe action: `{report.source_contract.get('compat_probe_action_xy')}`",
                f"- compat simulator state shape after `step()`: `{report.source_contract.get('compat_probe_sim_state_shape')}`",
            ]
        )
    lines.extend(["", "## Commands", ""])
    for command in report.commands:
        lines.append(f"- `{command.name}`: returncode `{command.returncode}`")
        if command.failure_summary:
            lines.append(f"  - failure summary: `{command.failure_summary}`")
    return "\n".join(lines) + "\n"


def _write_outputs(report: ProbeReport, *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    output_md.write_text(_render_markdown(report), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the SocialForce runtime probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("output/repos/Social-Navigation-PyEnvs"),
        help="Path to the Social-Navigation-PyEnvs checkout.",
    )
    parser.add_argument(
        "--backend-spec",
        default=DEFAULT_BACKEND_SPEC,
        help="Extra dependency spec passed to `uv run --with ...`.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(
            os.environ.get("SOCIAL_NAV_SOCIALFORCE_PROBE_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "output/benchmarks/external/social_navigation_pyenvs_socialforce_runtime_probe/report.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "output/benchmarks/external/social_navigation_pyenvs_socialforce_runtime_probe/report.md"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the probe and write report artifacts."""
    args = parse_args(argv)
    report = run_probe(
        args.repo_root,
        backend_spec=args.backend_spec,
        timeout_seconds=args.timeout_seconds,
    )
    _write_outputs(report, output_json=args.output_json, output_md=args.output_md)
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.verdict == "compatible runtime reproduced" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
