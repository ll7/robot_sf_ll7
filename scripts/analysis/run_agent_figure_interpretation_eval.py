#!/usr/bin/env python3
"""Replay frozen agent-figure interpretation evaluation fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.agent_figure_interpretation_eval import (
    AgentFigureEvalError,
    canonical_json,
    evaluate_manifest,
    list_fixture_mutations,
    replay_all_fixture_mutations,
    replay_fixture_mutation,
    validate_candidate_envelope,
)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "agent_figure_interpretation_eval"
    / "v1"
    / "manifest.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Digest-pinned fixture manifest to replay.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print indented JSON instead of canonical compact JSON.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_inventory",
        help="List verified source fixtures, mutations, and expected detectors.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        help="Provider-free candidate envelope JSON for one-pair or replay-all mode.",
    )
    parser.add_argument(
        "--replay-all",
        action="store_true",
        help="Replay a JSON array of candidate envelopes against every verified pair.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate one candidate envelope without replaying it.",
    )
    parser.add_argument("--fixture-id", help="Require this fixture ID in one-pair mode.")
    parser.add_argument("--mutation-id", help="Require this mutation ID in one-pair mode.")
    return parser


def _load_candidate_payload(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentFigureEvalError(f"{path}: unreadable candidate JSON: {exc}") from exc


def _run_list(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.candidate or args.replay_all or args.validate or args.fixture_id or args.mutation_id:
        raise AgentFigureEvalError("--list cannot be combined with replay arguments")
    return list_fixture_mutations(args.manifest), 0


def _run_all(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if not args.candidate:
        raise AgentFigureEvalError("--replay-all requires --candidate")
    if args.fixture_id or args.mutation_id:
        raise AgentFigureEvalError("--replay-all cannot use --fixture-id or --mutation-id")
    result = replay_all_fixture_mutations(args.manifest, _load_candidate_payload(args.candidate))
    return result, 0 if result["detector_status"] == "pass" else 1


def _run_validate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if not args.candidate:
        raise AgentFigureEvalError("--validate requires --candidate")
    if args.replay_all or args.fixture_id or args.mutation_id:
        raise AgentFigureEvalError("--validate cannot use replay selectors")
    payload = _load_candidate_payload(args.candidate)
    if not isinstance(payload, dict):
        raise AgentFigureEvalError("--validate candidate must be a JSON object")
    validate_candidate_envelope(payload)
    return {
        "schema_version": "agent_figure_interpretation_replay.v1",
        "status": "evaluation_artifacts_only",
        "claim_boundary": "fixture replay only; no external model calls, no benchmark claims, and no generated evidence promotion",
        "mode": "validate",
        "detector_status": "not_run",
        "verdict": "valid",
    }, 0


def _run_one(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if (args.fixture_id is None) != (args.mutation_id is None):
        raise AgentFigureEvalError("--fixture-id and --mutation-id must be supplied together")
    result = replay_fixture_mutation(
        args.manifest,
        _load_candidate_payload(args.candidate),
        fixture_id=args.fixture_id,
        mutation_id=args.mutation_id,
    )
    return result, 0 if result["detector_status"] == "pass" else 1


def _run_requested_mode(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.list_inventory:
        return _run_list(args)
    if args.replay_all:
        return _run_all(args)
    if args.validate:
        return _run_validate(args)
    if args.candidate:
        return _run_one(args)
    if args.fixture_id or args.mutation_id:
        raise AgentFigureEvalError("fixture/mutation selectors require --candidate")
    return evaluate_manifest(args.manifest), 0


def main(argv: list[str] | None = None) -> int:
    """Run fixture-only evaluation or the provider-free replay harness."""

    args = _parser().parse_args(argv)
    try:
        result, exit_code = _run_requested_mode(args)
    except AgentFigureEvalError as exc:
        print(f"agent figure interpretation eval failed closed: {exc}", file=sys.stderr)
        return 2
    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(canonical_json(result))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
