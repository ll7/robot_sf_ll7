#!/usr/bin/env python3
"""Manage the versioned adversarial counterexample corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.adversarial.counterexample_corpus import (
    CorpusError,
    admit_case_record,
    append_planner_evaluation,
    export_regression_slice,
    import_issue9645_packet,
    import_issue9656_candidates,
    load_corpus,
    new_corpus,
    promote_historical_candidate,
    recompute_planner_status,
    save_corpus,
)


def main(argv: list[str] | None = None) -> int:
    """Run one corpus management command and return its process status."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return _run_init(args)
        corpus = _load_command_corpus(args)
        handler = _COMMAND_HANDLERS.get(args.command)
        if handler is None:
            parser.error(f"unsupported command {args.command}")
        return handler(args, corpus)
    except (CorpusError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    return 2


def _run_init(args: argparse.Namespace) -> int:
    corpus_path = Path(args.corpus)
    created = not corpus_path.exists()
    corpus = new_corpus() if created else load_corpus(corpus_path)
    if created:
        save_corpus(corpus_path, corpus)
    _write_json({"created": created, "schema_version": corpus["schema_version"]}, args.output)
    return 0


def _load_command_corpus(args: argparse.Namespace) -> dict[str, Any]:
    create = args.command in {
        "import-9645",
        "import-9656-candidates",
        "admit-case",
        "promote-candidate",
    }
    return load_corpus(args.corpus, create=create)


def _save_receipt(args: argparse.Namespace, corpus: dict[str, Any], receipt: dict[str, Any]) -> int:
    save_corpus(args.corpus, corpus)
    _write_json(receipt, args.output)
    return 2 if receipt.get("decision") == "rejected" else 0


def _run_import_9645(args: argparse.Namespace, corpus: dict[str, Any]) -> int:
    corpus, receipt = import_issue9645_packet(args.payload, corpus, corpus_root=args.corpus_root)
    return _save_receipt(args, corpus, receipt)


def _run_import_9656(args: argparse.Namespace, corpus: dict[str, Any]) -> int:
    corpus, receipt = import_issue9656_candidates(
        args.summary,
        args.materialized_root,
        args.evidence_root,
        args.campaign_root,
        corpus,
        corpus_root=args.corpus_root,
    )
    return _save_receipt(args, corpus, receipt)


def _run_admit_case(args: argparse.Namespace, corpus: dict[str, Any]) -> int:
    case_record = _read_json_object(args.case)
    corpus, receipt = admit_case_record(
        case_record,
        corpus,
        corpus_root=args.corpus_root,
        artifact_root=args.artifact_root,
        source_kind=args.source_kind,
        source_id=args.source_id,
    )
    return _save_receipt(args, corpus, receipt)


def _run_promote_candidate(args: argparse.Namespace, corpus: dict[str, Any]) -> int:
    case_record = _read_json_object(args.case)
    corpus, receipt = promote_historical_candidate(
        args.candidate_id,
        case_record,
        corpus,
        corpus_root=args.corpus_root,
    )
    return _save_receipt(args, corpus, receipt)


def _run_record_evaluation(args: argparse.Namespace, corpus: dict[str, Any]) -> int:
    evaluation = _read_json_object(args.observation)
    corpus_root = Path(args.corpus).resolve().parent
    append_planner_evaluation(corpus, evaluation, corpus_root=corpus_root)
    save_corpus(args.corpus, corpus)
    _write_json(
        {"recorded": True, "evaluation_id": corpus["planner_evaluations"][-1]["evaluation_id"]},
        args.output,
    )
    return 0


def _run_status(args: argparse.Namespace, corpus: dict[str, Any]) -> int:
    report = recompute_planner_status(
        corpus,
        planner_id=args.planner_id,
        planner_config_identity=args.planner_config_identity,
        corpus_root=Path(args.corpus).resolve().parent,
    )
    _write_json(report, args.output)
    return 0


def _run_export_slice(args: argparse.Namespace, corpus: dict[str, Any]) -> int:
    manifest = export_regression_slice(
        corpus,
        corpus_root=args.corpus_root,
        output_dir=args.output_dir,
    )
    _write_json(manifest, args.output)
    return 0


_COMMAND_HANDLERS = {
    "import-9645": _run_import_9645,
    "import-9656-candidates": _run_import_9656,
    "admit-case": _run_admit_case,
    "promote-candidate": _run_promote_candidate,
    "record-evaluation": _run_record_evaluation,
    "status": _run_status,
    "export-slice": _run_export_slice,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="create or validate an empty corpus")
    init_parser.add_argument("--corpus", required=True)
    init_parser.add_argument("--output")

    import_parser = subparsers.add_parser(
        "import-9645", help="admit the replay-verified #1501 case from the #9645 packet"
    )
    import_parser.add_argument("--payload", required=True)
    import_parser.add_argument("--corpus", required=True)
    import_parser.add_argument("--corpus-root", required=True)
    import_parser.add_argument("--output")

    historical_parser = subparsers.add_parser(
        "import-9656-candidates",
        help="append historical #9656 mined rows as pending replay candidates",
    )
    historical_parser.add_argument("--summary", required=True)
    historical_parser.add_argument("--materialized-root", required=True)
    historical_parser.add_argument("--evidence-root", required=True)
    historical_parser.add_argument("--campaign-root", required=True)
    historical_parser.add_argument("--corpus", required=True)
    historical_parser.add_argument("--corpus-root", required=True)
    historical_parser.add_argument("--output")

    admission_parser = subparsers.add_parser(
        "admit-case",
        help="admit a complete replay-verified case from corpus-staged artifacts",
    )
    admission_parser.add_argument("--case", required=True)
    admission_parser.add_argument("--artifact-root", required=True)
    admission_parser.add_argument("--source-kind", default="adversarial_case_record")
    admission_parser.add_argument("--source-id", required=True)
    admission_parser.add_argument("--corpus", required=True)
    admission_parser.add_argument("--corpus-root", required=True)
    admission_parser.add_argument("--output")

    promotion_parser = subparsers.add_parser(
        "promote-candidate",
        help="promote a pending #9656 candidate with exact replay and admission evidence",
    )
    promotion_parser.add_argument("--candidate-id", required=True)
    promotion_parser.add_argument("--case", required=True)
    promotion_parser.add_argument("--corpus", required=True)
    promotion_parser.add_argument("--corpus-root", required=True)
    promotion_parser.add_argument("--output")

    evaluation_parser = subparsers.add_parser(
        "record-evaluation", help="append one complete or explicitly incomplete planner result"
    )
    evaluation_parser.add_argument("--observation", required=True)
    evaluation_parser.add_argument("--corpus", required=True)
    evaluation_parser.add_argument("--output")

    status_parser = subparsers.add_parser(
        "status", help="recompute solved/unsolved status for one planner configuration"
    )
    status_parser.add_argument("--corpus", required=True)
    status_parser.add_argument("--planner-id", required=True)
    status_parser.add_argument("--planner-config-identity", required=True)
    status_parser.add_argument("--output")

    slice_parser = subparsers.add_parser(
        "export-slice", help="export admitted cases as a replay matrix and inputs"
    )
    slice_parser.add_argument("--corpus", required=True)
    slice_parser.add_argument("--corpus-root", required=True)
    slice_parser.add_argument("--output-dir", required=True)
    slice_parser.add_argument("--output")
    return parser


def _read_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CorpusError("input JSON must be an object")
    return value


def _write_json(value: Any, output_path: str | None) -> None:
    rendered = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    raise SystemExit(main())
