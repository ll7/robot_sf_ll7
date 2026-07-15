#!/usr/bin/env python3
"""Run or adjudicate the issue #4981 VecEnv throughput acceptance workload.

The four-mode comparator remains a diagnostic measurement tool. This wrapper derives its command
from a tracked profile, requires the single-environment fallback/equivalence tests, and reports
``met``, ``not_met``, or ``blocked`` without promoting incomplete evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.validation.run_vecenv_worker_mode_throughput import (  # noqa: E402
    _git_commit,
    _load_yaml_raw,
    _provenance_path,
    _resolve_scenario_path,
    _sha256_file,
)

PROFILE_SCHEMA = "issue_4981_vecenv_throughput_acceptance_profile.v1"
REPORT_SCHEMA = "issue_4981_vecenv_throughput_acceptance.v1"
SOURCE_SCHEMA = "vecenv_throughput_comparator.v2"
SOURCE_CLAIM_BOUNDARY = "diagnostic_only_not_benchmark_evidence"
DEFAULT_PROFILE = _REPO_ROOT / "configs/training/lidar/issue_4981_vecenv_throughput_acceptance.yaml"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "output/issue_4981_vecenv_throughput_acceptance"
_COMPARATOR = _REPO_ROOT / "scripts/validation/run_vecenv_worker_mode_throughput.py"
_STANDARD_CONFIG = Path("configs/training/lidar/lidar_ppo_mlp_smoke_issue_1662.yaml")
_SUPPORTED_MODES = ("dummy", "subproc", "threaded", "threaded_lidar_batch")
_SPEEDUP_STATISTIC = "median_transitions_per_second_ratio"
_CLAIM_BOUNDARY = (
    "Host-specific CPU implementation-performance acceptance for issue #4981 only; this does not "
    "establish training quality, navigation-benchmark performance, cross-host speedup, GPU speedup, "
    "or any paper/dissertation claim."
)


@dataclasses.dataclass(frozen=True)
class AcceptanceProfile:
    """Validated parameters for one issue #4981 acceptance decision."""

    path: Path
    config_path: Path
    num_envs: int
    modes: tuple[str, ...]
    repetitions: int
    base_seed: int
    warmup_steps: int
    measure_steps: int
    candidate_modes: tuple[str, ...]
    minimum_speedup_exclusive: float
    required_pytest_nodes: tuple[str, ...]
    claim_boundary: str


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def _strings(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a non-empty sequence of strings")
    resolved = tuple(item.strip() for item in value if isinstance(item, str))
    if len(resolved) != len(value) or not resolved or any(not item for item in resolved):
        raise ValueError(f"{field} must be a non-empty sequence of strings")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field} must not contain duplicates")
    return resolved


def _integer(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_threshold(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) != 3.0
    ):
        raise ValueError("acceptance.minimum_speedup_exclusive must be exactly 3.0")
    return float(value)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load profile {path}: {exc}") from exc
    return _mapping(payload, "profile")


def _repository_file(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not (_REPO_ROOT / path).is_file():
        raise ValueError(f"{field} must name an existing repository file")
    return path


def _required_test_nodes(root: Mapping[str, Any]) -> tuple[str, ...]:
    nodes = _strings(root.get("required_pytest_nodes"), "required_pytest_nodes")
    for node in nodes:
        test_path = Path(node.split("::", maxsplit=1)[0])
        if (
            "::" not in node
            or test_path.is_absolute()
            or not str(test_path).startswith("tests/")
            or not (_REPO_ROOT / test_path).is_file()
        ):
            raise ValueError(f"invalid required pytest node: {node!r}")
    return nodes


def load_profile(path: Path) -> AcceptanceProfile:
    """Load the tracked profile and fail closed on unsupported acceptance semantics."""
    profile_path = path.resolve()
    root = _load_yaml_mapping(profile_path)
    workload = _mapping(root.get("standard_workload"), "standard_workload")
    acceptance = _mapping(root.get("acceptance"), "acceptance")
    modes = _strings(workload.get("modes"), "standard_workload.modes")
    candidates = _strings(acceptance.get("candidate_modes"), "acceptance.candidate_modes")
    threshold = _strict_threshold(acceptance.get("minimum_speedup_exclusive"))
    claim_boundary = str(root.get("claim_boundary", "")).strip()
    config_path = _repository_file(workload.get("config_path"), "standard_workload.config_path")
    num_envs = _integer(workload.get("num_envs"), "standard_workload.num_envs", minimum=2)
    repetitions = _integer(workload.get("repetitions"), "standard_workload.repetitions", minimum=5)
    warmup_steps = _integer(
        workload.get("warmup_steps"), "standard_workload.warmup_steps", minimum=1000
    )
    measure_steps = _integer(
        workload.get("measure_steps"), "standard_workload.measure_steps", minimum=10000
    )

    if root.get("schema") != PROFILE_SCHEMA or root.get("issue") != 4981:
        raise ValueError(f"profile must use {PROFILE_SCHEMA!r} for issue 4981")
    if modes != _SUPPORTED_MODES:
        raise ValueError(f"standard_workload.modes must be {_SUPPORTED_MODES!r}")
    if config_path != _STANDARD_CONFIG or num_envs != 4:
        raise ValueError("standard workload must use the reviewed issue #1662 config with 4 envs")
    if any(mode not in {"threaded", "threaded_lidar_batch"} for mode in candidates):
        raise ValueError("candidate modes must be in-process threaded modes")
    if acceptance.get("source_schema") != SOURCE_SCHEMA:
        raise ValueError(f"acceptance.source_schema must be {SOURCE_SCHEMA!r}")
    if acceptance.get("source_claim_boundary") != SOURCE_CLAIM_BOUNDARY:
        raise ValueError("acceptance.source_claim_boundary must preserve the diagnostic boundary")
    expected_acceptance = {
        "baseline_mode": "dummy",
        "baseline_num_envs": 1,
        "speedup_statistic": _SPEEDUP_STATISTIC,
    }
    for field, expected in expected_acceptance.items():
        if acceptance.get(field) != expected:
            raise ValueError(f"acceptance.{field} must be {expected!r}")
    if claim_boundary != _CLAIM_BOUNDARY:
        raise ValueError("claim_boundary must match the host-specific non-transfer boundary")

    return AcceptanceProfile(
        path=profile_path,
        config_path=config_path,
        num_envs=num_envs,
        modes=modes,
        repetitions=repetitions,
        base_seed=_integer(workload.get("base_seed"), "standard_workload.base_seed", minimum=0),
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        candidate_modes=candidates,
        minimum_speedup_exclusive=threshold,
        required_pytest_nodes=_required_test_nodes(root),
        claim_boundary=claim_boundary,
    )


def expected_source_provenance(profile: AcceptanceProfile) -> dict[str, str]:
    """Return the config/scenario paths and hashes required from source evidence."""
    config_path = (_REPO_ROOT / profile.config_path).resolve()
    scenario_path = _resolve_scenario_path(_load_yaml_raw(config_path), config_path)
    if not scenario_path.is_file():
        raise ValueError(f"standard workload scenario does not exist: {scenario_path}")
    return {
        "config_path": _provenance_path(config_path),
        "config_sha256": _sha256_file(config_path),
        "scenario_path": _provenance_path(scenario_path),
        "scenario_sha256": _sha256_file(scenario_path),
    }


def build_comparator_command(
    profile: AcceptanceProfile,
    evidence_path: Path,
    *,
    python_executable: str = sys.executable,
) -> list[str]:
    """Derive the comparator command entirely from the tracked profile."""
    return [
        python_executable,
        str(_COMPARATOR),
        "--config",
        str((_REPO_ROOT / profile.config_path).resolve()),
        "--num-envs",
        str(profile.num_envs),
        "--repetitions",
        str(profile.repetitions),
        "--base-seed",
        str(profile.base_seed),
        "--warmup-steps",
        str(profile.warmup_steps),
        "--measure-steps",
        str(profile.measure_steps),
        "--modes",
        *profile.modes,
        "--output",
        str(evidence_path.resolve()),
    ]


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


class _EvidenceValidator:
    """Validate comparator JSON against the frozen profile while collecting every blocker."""

    def __init__(
        self,
        profile: AcceptanceProfile,
        evidence: Mapping[str, Any],
        current_commit: str,
        preflight: Mapping[str, Any],
    ) -> None:
        self.profile = profile
        self.evidence = evidence
        self.current_commit = current_commit
        self.preflight = preflight
        self.blockers: list[str] = []

    def _expect(self, field: str, expected: object) -> None:
        actual = self.evidence.get(field)
        if actual != expected:
            self.blockers.append(f"{field} must be {expected!r}, got {actual!r}")

    def _summary(self, record: object, *, label: str, mode: str) -> float | None:
        if not isinstance(record, Mapping):
            self.blockers.append(f"{label} must be a mapping")
            return None
        for field, expected in {"mode": mode, "status": "ok", "error": None}.items():
            if record.get(field) != expected:
                self.blockers.append(f"{label}.{field} must be {expected!r}")
        aggregate = _number(record.get("transitions_per_second"))
        if aggregate is None or aggregate <= 0:
            self.blockers.append(f"{label}.transitions_per_second must be positive and finite")
            aggregate = None

        samples = record.get("repetition_results")
        if not isinstance(samples, list) or len(samples) != self.profile.repetitions:
            self.blockers.append(
                f"{label}.repetition_results must contain exactly {self.profile.repetitions} rows"
            )
            return aggregate
        throughputs = self._repetition_throughputs(samples, label=label, mode=mode)
        if aggregate is not None and len(throughputs) == self.profile.repetitions:
            expected_median = round(statistics.median(throughputs), 2)
            if not math.isclose(aggregate, expected_median, rel_tol=0.0, abs_tol=0.011):
                self.blockers.append(
                    f"{label}.transitions_per_second must equal median repetition throughput "
                    f"{expected_median}, got {aggregate}"
                )
        return aggregate

    def _repetition_throughputs(
        self,
        samples: list[object],
        *,
        label: str,
        mode: str,
    ) -> list[float]:
        """Validate retained repetitions and return their positive throughputs."""
        throughputs: list[float] = []
        for index, sample in enumerate(samples):
            sample_label = f"{label}.repetition_results[{index}]"
            if not isinstance(sample, Mapping):
                self.blockers.append(f"{sample_label} must be a mapping")
                continue
            expected_fields = {
                "mode": mode,
                "repetition": index,
                "status": "ok",
                "error": None,
            }
            for field, expected in expected_fields.items():
                if sample.get(field) != expected:
                    self.blockers.append(f"{sample_label}.{field} must be {expected!r}")
            throughput = _number(sample.get("transitions_per_second"))
            if throughput is None or throughput <= 0:
                self.blockers.append(
                    f"{sample_label}.transitions_per_second must be positive and finite"
                )
            else:
                throughputs.append(throughput)
        return throughputs

    def _validate_source_contract(self) -> None:
        expected_fields: dict[str, object] = {
            "schema": SOURCE_SCHEMA,
            **expected_source_provenance(self.profile),
            "num_envs": self.profile.num_envs,
            "repetitions": self.profile.repetitions,
            "base_seed": self.profile.base_seed,
            "warmup_steps": self.profile.warmup_steps,
            "measure_steps": self.profile.measure_steps,
            "modes": list(self.profile.modes),
            "baseline_mode": "dummy",
            "baseline_num_envs": 1,
            "claim_boundary": SOURCE_CLAIM_BOUNDARY,
            "scenario_selection": {"strategy": "first", "index": 0},
        }
        for field, expected in expected_fields.items():
            self._expect(field, expected)
        # Only baseline and candidate-mode failures block acceptance.
        # Comparison-only modes (subproc, 4-env dummy) may fail without blocking,
        # provided their mode is identifiable. Unidentified or missing-mode failures
        # are treated as critical since they may mask regressions.
        failures = self.evidence.get("failures") or []
        critical_failures = [
            f for f in failures
            if f.get("scope") == "baseline"
            or f.get("mode") in self.profile.candidate_modes
        ]
        unknown_failures = [
            f for f in failures
            if f not in critical_failures and (
                f.get("mode") is None or f.get("mode") not in self.profile.modes
            )
        ]
        if critical_failures:
            self.blockers.append(
                f"baseline or candidate-mode failures block acceptance: {critical_failures}"
            )
        if unknown_failures:
            self.blockers.append(
                f"unidentified or unrecognized-mode failures block acceptance: {unknown_failures}"
            )
        # Source status may be "failed" when only non-critical comparison modes
        # failed with known mode identities. It must still block when status
        # is "failed" with no documented failures (inconsistent state).
        if self.evidence.get("status") != "ok":
            if not failures:
                self.blockers.append(
                    f"source status must be 'ok', got {self.evidence.get('status')!r}"
                )
            elif critical_failures or unknown_failures:
                self.blockers.append(
                    f"source status must be 'ok', got {self.evidence.get('status')!r}"
                )
        if self.evidence.get("commit") != self.current_commit:
            self.blockers.append(
                f"source commit must match current HEAD {self.current_commit!r}, "
                f"got {self.evidence.get('commit')!r}"
            )
        if not re.fullmatch(r"[0-9a-f]{40}", self.current_commit):
            self.blockers.append(
                f"current HEAD must be a full Git commit, got {self.current_commit!r}"
            )
        for field in ("host", "python", "platform"):
            value = self.evidence.get(field)
            if not isinstance(value, str) or not value.strip():
                self.blockers.append(f"{field} must be a non-empty provenance string")
        if self.preflight.get("status") != "passed":
            self.blockers.append(
                f"required fallback/equivalence preflight failed: {self.preflight!r}"
            )

    def validate(self) -> tuple[float | None, list[dict[str, Any]]]:
        """Return validated baseline throughput and in-process candidate speedups."""
        self._validate_source_contract()
        baseline = self._summary(self.evidence.get("baseline"), label="baseline", mode="dummy")
        results = self.evidence.get("results")
        if not isinstance(results, list) or any(not isinstance(row, Mapping) for row in results):
            self.blockers.append("results must be a list of mappings")
            return baseline, []
        modes = [row.get("mode") for row in results]
        if modes != list(self.profile.modes):
            self.blockers.append(
                f"result mode order must be {list(self.profile.modes)!r}, got {modes!r}"
            )

        candidate_speedups = []
        for mode, row in zip(self.profile.modes, results, strict=False):
            # Only validate comparison-only modes if they succeeded; skip
            # detailed checks for non-critical modes that are known to have
            # failed (e.g. subproc connection resets on some hosts).
            if mode not in self.profile.candidate_modes and row.get("status") != "ok":
                continue
            throughput = self._summary(row, label=f"results[{mode}]", mode=mode)
            if baseline is None or throughput is None:
                continue
            speedup = round(throughput / baseline, 3)
            reported = _number(row.get("speedup_vs_baseline"))
            if reported is None or not math.isclose(reported, speedup, rel_tol=0.0, abs_tol=0.0011):
                self.blockers.append(
                    f"results[{mode}].speedup_vs_baseline must equal {speedup}, got {reported!r}"
                )
            if mode in self.profile.candidate_modes:
                candidate_speedups.append(
                    {
                        "mode": mode,
                        "transitions_per_second": throughput,
                        "speedup_vs_baseline": speedup,
                        "threshold_met": speedup > self.profile.minimum_speedup_exclusive,
                    }
                )
        if baseline is not None:
            reported = _number(
                self.evidence.get("baseline", {}).get("speedup_vs_baseline")
                if isinstance(self.evidence.get("baseline"), Mapping)
                else None
            )
            if reported != 1.0:
                self.blockers.append("baseline.speedup_vs_baseline must equal 1.0")
        if not candidate_speedups:
            self.blockers.append("no valid candidate-mode speedups were available")
        return baseline, candidate_speedups


def adjudicate(
    profile: AcceptanceProfile,
    evidence: Mapping[str, Any],
    *,
    evidence_path: Path,
    current_commit: str,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one comparator artifact and classify the issue throughput criterion."""
    validator = _EvidenceValidator(profile, evidence, current_commit, preflight)
    baseline, speedups = validator.validate()
    best = max(speedups, key=lambda row: row["speedup_vs_baseline"], default=None)
    acceptance_met = (
        None
        if validator.blockers
        else bool(best and best["speedup_vs_baseline"] > profile.minimum_speedup_exclusive)
    )
    status = "blocked" if validator.blockers else "met" if acceptance_met else "not_met"
    return {
        "schema": REPORT_SCHEMA,
        "issue": 4981,
        "status": status,
        "acceptance_met": acceptance_met,
        "evidence_status": (
            "blocked_invalid_or_incomplete_evidence"
            if validator.blockers
            else "host_specific_implementation_performance_acceptance"
        ),
        "claim_boundary": profile.claim_boundary,
        "profile_path": _provenance_path(profile.path),
        "profile_sha256": _sha256_file(profile.path),
        "source_evidence_path": _provenance_path(evidence_path),
        "source_evidence_sha256": _sha256_file(evidence_path) if evidence_path.is_file() else None,
        "source_commit": evidence.get("commit"),
        "source_host": evidence.get("host"),
        "source_platform": evidence.get("platform"),
        "preflight": dict(preflight),
        "decision_rule": {
            "statistic": _SPEEDUP_STATISTIC,
            "baseline_mode": "dummy",
            "baseline_num_envs": 1,
            "candidate_modes": list(profile.candidate_modes),
            "minimum_speedup_exclusive": profile.minimum_speedup_exclusive,
            "required_repetitions": profile.repetitions,
            "warmup_steps_per_repetition": profile.warmup_steps,
            "measure_steps_per_repetition": profile.measure_steps,
        },
        "baseline_transitions_per_second": baseline,
        "mode_speedups": speedups,
        "best_mode": best["mode"] if best else None,
        "best_speedup_vs_baseline": best["speedup_vs_baseline"] if best else None,
        "blockers": validator.blockers,
        "residual_risks": [
            "The decision is specific to the recorded CPU host and current commit.",
            "It does not measure policy learning quality or navigation benchmark outcomes.",
        ],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _clean_tracked_tree() -> tuple[bool, str]:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return False, f"could not inspect tracked worktree state: {exc}"
    return not output, output


def _run_required_tests(profile: AcceptanceProfile, output_dir: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", *profile.required_pytest_nodes]
    completed = subprocess.run(command, cwd=_REPO_ROOT, text=True, capture_output=True, check=False)
    log_path = output_dir / "fallback_preflight.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "command": command,
        "exit_code": completed.returncode,
        "pytest_nodes": list(profile.required_pytest_nodes),
        "log_path": _provenance_path(log_path),
    }


def _blocked_report(
    profile: AcceptanceProfile,
    blocker: str,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "issue": 4981,
        "status": "blocked",
        "acceptance_met": None,
        "evidence_status": "blocked_before_measurement",
        "claim_boundary": profile.claim_boundary,
        "profile_path": _provenance_path(profile.path),
        "profile_sha256": _sha256_file(profile.path),
        "preflight": dict(preflight),
        "blockers": [blocker],
    }


def _read_evidence(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"comparator evidence does not exist: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"could not read comparator evidence: {exc}"
    if not isinstance(payload, Mapping):
        return None, "comparator evidence root must be a JSON object"
    return payload, None


def _execute(
    args: argparse.Namespace,
    profile: AcceptanceProfile,
    output_dir: Path,
    command: list[str],
) -> tuple[dict[str, Any], int]:
    """Run required proof, optionally measure, and return a decision plus exit code."""
    clean, detail = _clean_tracked_tree()
    if not clean:
        report = _blocked_report(
            profile,
            "tracked worktree must be clean before acceptance measurement/adjudication",
            {"status": "failed", "tracked_tree": detail},
        )
        return report, 3

    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = _run_required_tests(profile, output_dir)
    if preflight["status"] != "passed":
        report = _blocked_report(
            profile,
            "required single-environment fallback/equivalence tests failed",
            preflight,
        )
        return report, 3

    evidence_path = args.evidence.resolve() if args.evidence else output_dir / "comparison.json"
    comparator_exit_code = None
    if args.run:
        completed = subprocess.run(
            command,
            cwd=_REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        comparator_exit_code = completed.returncode
        log_path = output_dir / "comparator.log"
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        preflight.update(
            {
                "comparator_exit_code": comparator_exit_code,
                "comparator_log_path": _provenance_path(log_path),
            }
        )

    evidence, blocker = _read_evidence(evidence_path)
    if evidence is None:
        return _blocked_report(profile, blocker or "could not load evidence", preflight), 3
    report = adjudicate(
        profile,
        evidence,
        evidence_path=evidence_path,
        current_commit=_git_commit(_REPO_ROOT),
        preflight=preflight,
    )
    if comparator_exit_code not in {None, 0} and report["blockers"]:
        # Record the comparator exit code only when the adjudicator already
        # has blockers. If the adjudicator is clean (e.g. only non-critical
        # comparison modes failed), the adjudicator's own verdict is authoritative
        # and we must not override it with the coarse comparator exit.
        report["blockers"].append(f"comparator exited with code {comparator_exit_code}")
    return report, {"met": 0, "not_met": 2}.get(str(report["status"]), 3)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the explicit preview/run/existing-evidence interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true", help="Write the exact command only.")
    action.add_argument("--run", action="store_true", help="Run and adjudicate the profile.")
    action.add_argument("--evidence", type=Path, help="Adjudicate existing current-HEAD JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested phase and return zero only for preview-ready or met outcomes."""
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    decision_path = output_dir / "decision.json"
    try:
        profile = load_profile(DEFAULT_PROFILE)
        expected_source_provenance(profile)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    command = build_comparator_command(profile, output_dir / "comparison.json")

    if args.dry_run:
        report = {
            "schema": REPORT_SCHEMA,
            "issue": 4981,
            "status": "dry_run_ready",
            "acceptance_met": None,
            "evidence_status": "preflight_only_not_performance_evidence",
            "claim_boundary": (
                "Preflight only; no throughput measurement ran and no issue, benchmark, or "
                "paper/dissertation claim is supported."
            ),
            "profile_path": _provenance_path(profile.path),
            "profile_sha256": _sha256_file(profile.path),
            "comparator_command": command,
            "required_pytest_nodes": list(profile.required_pytest_nodes),
            "blockers": [],
        }
        exit_code = 0
    else:
        report, exit_code = _execute(args, profile, output_dir, command)

    _write_json(decision_path, report)
    print(
        f"status={report['status']} best_mode={report.get('best_mode')} "
        f"best_speedup={report.get('best_speedup_vs_baseline')} decision={decision_path}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
