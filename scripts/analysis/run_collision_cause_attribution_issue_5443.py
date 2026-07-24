"""Run the label-blind #5443 collision-cause analyser and frozen scorer.

This CPU-only runner is controlled-fixture injected-fault validation. It makes
no real-trace, campaign, legal, or moral root-cause claim. The analyser consumes
only low-level trace events and counterfactual repairs; the manifest answer key
is loaded separately by the scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.benchmark.collision_cause_analyser import (
    DetectedFault,
    analyse_suite,
    analyser_config,
)
from robot_sf.benchmark.collision_cause_attribution import (
    REPORT_STATUS_SCORED,
    VERDICT_PASS,
    build_validation_report,
)
from robot_sf.benchmark.last_avoidable_fixtures import (
    ObservableTraceEvent,
    build_collision_cause_fixtures,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFEST = (
    _REPO_ROOT / "tests/benchmark/fixtures/collision_cause_attribution_manifest_5443.json"
)
_RUN_SCHEMA = _REPO_ROOT / "robot_sf/benchmark/schemas/collision_cause_analyser_run.v1.json"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=None,
        help="Optional path for the schema-validated analyser payload.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Frozen ground-truth manifest used only by the scorer.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score analyser verdicts against the frozen manifest.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path for the scored validation report.",
    )
    return parser.parse_args(argv)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _git_output(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _event_mapping(event: ObservableTraceEvent) -> dict[str, object]:
    return {
        "step": event.step,
        "channel": event.channel,
        "field": event.field,
        "expected": event.expected,
        "observed": event.observed,
    }


def _detected_fault_mapping(detected_fault: DetectedFault) -> dict[str, object]:
    return {
        "predicted_cause": detected_fault.predicted_cause,
        "activation_step": detected_fault.activation_step,
        "gates_applied_command": detected_fault.fault.gates_applied_command,
        "events": [_event_mapping(event) for event in detected_fault.fault.events],
    }


def validate_verdicts_payload(payload: dict[str, Any]) -> None:
    """Validate a generated run payload against its committed JSON Schema."""
    schema = json.loads(_RUN_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)


def build_verdicts_payload(
    *,
    manifest_path: Path = _DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Build and schema-validate verdicts, evidence, and exact provenance."""
    manifest_path = manifest_path.resolve()
    config = analyser_config()
    result = analyse_suite(build_collision_cause_fixtures())
    evidence = [
        {
            "fixture_id": item.fixture_id,
            "replay_verdict": item.replay.verdict,
            "replay_t_uca": item.replay.t_uca,
            "replay_t_inevitable": item.replay.t_inevitable,
            "replay_t_contact": item.replay.config.t_contact,
            "replay_deterministic": item.replay.determinism.deterministic,
            "decisive_faults": [_detected_fault_mapping(fault) for fault in item.decisive_faults],
            "present_faults": [_detected_fault_mapping(fault) for fault in item.present_faults],
            "metric_quirk_onset": item.metric_quirk_onset,
            "avoidable_pred": item.avoidable_pred,
        }
        for item in result.evidence
    ]
    payload: dict[str, Any] = {
        "schema_version": "collision_cause_analyser_run.v1",
        "analyser": "rule_based_kinematic_replay",
        "issue": 5443,
        "claim_boundary": (
            "controlled-fixture injected-fault validation only; not a real-trace "
            "root-cause claim; assigns no legal or moral fault"
        ),
        "provenance": {
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_tree_dirty": bool(_git_output("status", "--porcelain", "--untracked-files=no")),
            "manifest_path": manifest_path.relative_to(_REPO_ROOT).as_posix(),
            "manifest_sha256": _sha256_path(manifest_path),
            "analyser_config": config,
            "analyser_config_sha256": _canonical_sha256(config),
            "payload_schema_sha256": _sha256_path(_RUN_SCHEMA),
        },
        "verdicts": result.verdict_mappings(),
        "evidence": evidence,
    }
    validate_verdicts_payload(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run the analyser and optionally score its verdicts."""
    args = _parse_args(argv)
    payload = build_verdicts_payload(manifest_path=args.manifest)

    if args.verdicts is not None:
        args.verdicts.parent.mkdir(parents=True, exist_ok=True)
        args.verdicts.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if not args.score:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    fixtures = json.loads(args.manifest.read_text(encoding="utf-8"))["fixtures"]
    report = build_validation_report(fixtures, payload["verdicts"])
    text = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if report.status == REPORT_STATUS_SCORED and report.report is not None:
        return 0 if report.report.verdict == VERDICT_PASS else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
