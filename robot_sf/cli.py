"""Top-level ``robot-sf`` command line interface.

Thin, user-facing entry point for everyday Robot SF workflows. The
``doctor`` subcommand wraps the existing runtime diagnostics in
:mod:`robot_sf.benchmark.doctor` behind one obvious command, adding
friendly, remedy-bearing output for beginners. The ``models`` and
``datasets`` subcommands wrap the existing model registry and external-data
provenance/checksum machinery (issue #5797) behind list/verify/download and
list/verify/prepare commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf import cli_datasets, cli_models
from robot_sf.benchmark.doctor import collect_doctor_report, doctor_exit_code

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with its subcommands.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="robot-sf",
        description="Robot SF top-level command line interface.",
    )
    sub = parser.add_subparsers(dest="cmd")
    doc = sub.add_parser(
        "doctor",
        help="Environment/readiness check with friendly remedies (uv run robot-sf doctor)",
    )
    doc.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )
    doc.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("output"),
        help="Artifact root to probe for temporary write access (default: output)",
    )
    doc.add_argument(
        "--skip-env-smoke",
        action="store_true",
        default=False,
        help="Skip the minimal reset/step environment smoke check",
    )
    _add_models_subparser(sub)
    _add_datasets_subparser(sub)
    return parser


def _handle_doctor(args: argparse.Namespace) -> int:
    """Run the doctor check and print the report.

    Returns:
        int: Doctor command exit code.
    """
    report = collect_doctor_report(
        artifact_root=args.artifact_root,
        run_env_smoke=not args.skip_env_smoke,
    )
    if args.format == "json":
        import json  # noqa: PLC0415

        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        from robot_sf.benchmark.doctor import _format_human  # noqa: PLC0415

        sys.stdout.write(_format_human(report))
    return doctor_exit_code(report)


def _add_models_subparser(sub: argparse._SubParsersAction) -> None:
    """Register the ``robot-sf models`` subcommand tree."""
    models = sub.add_parser(
        "models",
        help="List, download, and verify registered model artifacts (uv run robot-sf models ...)",
    )
    models_sub = models.add_subparsers(dest="models_cmd", required=True)

    verify = models_sub.add_parser("verify", help="Verify each model artifact's pinned checksum.")
    verify.add_argument(
        "model_id",
        nargs="*",
        help="Optional model ids to verify (default: all registered models).",
    )
    verify.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )
    verify.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to model/registry.yaml (default: the bundled registry).",
    )

    lst = models_sub.add_parser(
        "list",
        help="List registered models and local artifact status.",
    )
    lst.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )
    lst.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to model/registry.yaml (default: the bundled registry).",
    )

    download = models_sub.add_parser(
        "download",
        help="Download a model artifact through the registry download path.",
    )
    download.add_argument("model_id", help="Registry model id to download.")
    download.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to model/registry.yaml (default: the bundled registry).",
    )
    download.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for downloaded artifacts (default: output/model_cache).",
    )
    download.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )


def _add_datasets_subparser(sub: argparse._SubParsersAction) -> None:
    """Register the ``robot-sf datasets`` subcommand tree."""
    datasets = sub.add_parser(
        "datasets",
        help="List, prepare, and verify external datasets (uv run robot-sf datasets ...)",
    )
    datasets_sub = datasets.add_subparsers(dest="datasets_cmd", required=True)

    dlist = datasets_sub.add_parser("list", help="List registered datasets and local status.")
    dlist.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )

    dverify = datasets_sub.add_parser(
        "verify", help="Verify each dataset's local layout and pinned checksum."
    )
    dverify.add_argument(
        "asset_id",
        nargs="*",
        help="Optional dataset asset ids to verify (default: all registered datasets).",
    )
    dverify.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )

    dprepare = datasets_sub.add_parser(
        "prepare",
        help="Print acquisition instructions and verify local layout WITHOUT downloading.",
    )
    dprepare.add_argument("asset_id", help="Dataset asset id to prepare.")
    dprepare.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Override the local source path to verify (default: the resolved asset path).",
    )
    dprepare.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )


def _handle_models(args: argparse.Namespace) -> int:
    """Dispatch the ``robot-sf models`` subcommand.

    Returns:
        int: Process-style exit code (0 success, 2 verify failure, 1 download error).
    """
    cmd = args.models_cmd
    if cmd == "list":
        rows = cli_models.list_models(registry_path=args.registry)
        if args.format == "json":
            sys.stdout.write(json.dumps(rows, indent=2) + "\n")
        else:
            _format_models_list(rows)
        return 0
    if cmd == "verify":
        report = cli_models.verify_models(
            registry_path=args.registry, model_ids=args.model_id or None
        )
        if args.format == "json":
            sys.stdout.write(json.dumps(report, indent=2) + "\n")
        else:
            _format_models_verify(report)
        # Non-zero when at least one pinned checksum failed to pass.
        return 0 if report["ok"] else 2
    if cmd == "download":
        try:
            report = cli_models.download_model(
                args.model_id,
                registry_path=args.registry,
                cache_dir=args.cache_dir,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a clean CLI error
            sys.stderr.write(f"error: could not download model '{args.model_id}': {exc}\n")
            return 1
        if args.format == "json":
            sys.stdout.write(json.dumps(report, indent=2) + "\n")
        else:
            sys.stdout.write(f"Resolved model '{report['model_id']}' to: {report['path']}\n")
        return 0
    parser = _build_parser()  # pragma: no cover - defensive
    parser.error(f"unknown models command: {cmd}")
    return 2


def _format_models_list(rows: list[dict]) -> None:
    """Print a friendly table for ``models list``."""
    if not rows:
        sys.stdout.write("No models registered.\n")
        return
    sys.stdout.write(f"{len(rows)} model(s) registered:\n\n")
    for row in rows:
        present = "present" if row["present_locally"] else "absent"
        tags = ", ".join(row["tags"]) if row["tags"] else "-"
        boundary = row["claim_boundary"] or "-"
        sys.stdout.write(f"- {row['model_id']}  [{present}]\n")
        sys.stdout.write(f"    name: {row['display_name']}\n")
        sys.stdout.write(f"    tags: {tags}\n")
        sys.stdout.write(f"    claim boundary: {boundary}\n")
        if row["local_path"]:
            sys.stdout.write(f"    local path: {row['local_path']}\n")
        if row["local_only"]:
            sys.stdout.write("    local-only: unavailable unless staged on this machine\n")
        sys.stdout.write("\n")


def _format_models_verify(report: dict) -> None:
    """Print a friendly per-artifact checksum report for ``models verify``."""
    sys.stdout.write(
        f"Model artifact verification: {report['passed']}/{report['pinned_checksums']} "
        f"pinned checksums passed ({report['checked']} checked).\n\n"
    )
    for result in report["results"]:
        sys.stdout.write(f"- {result['model_id']}: {result['status']}\n")
        if result.get("display_name"):
            sys.stdout.write(f"    name: {result['display_name']}\n")
        if result["pinned"]:
            sys.stdout.write(f"    expected sha256: {result['expected_sha256']}\n")
            if result.get("observed_sha256"):
                sys.stdout.write(f"    observed sha256: {result['observed_sha256']}\n")
        sys.stdout.write("\n")
    if report["ok"]:
        sys.stdout.write("All pinned checksums verified.\n")
    else:
        sys.stdout.write(
            "One or more pinned checksums failed or are missing. Run "
            "`uv run robot-sf models download <id>` to restore the artifact.\n"
        )


def _handle_datasets(args: argparse.Namespace) -> int:
    """Dispatch the ``robot-sf datasets`` subcommand.

    Returns:
        int: Process-style exit code (0 success, 2 verify/layout failure).
    """
    cmd = args.datasets_cmd
    if cmd == "list":
        rows = cli_datasets.list_datasets()
        if args.format == "json":
            sys.stdout.write(json.dumps(rows, indent=2) + "\n")
        else:
            _format_datasets_list(rows)
        return 0
    if cmd == "verify":
        report = cli_datasets.verify_datasets(asset_ids=args.asset_id or None)
        if args.format == "json":
            sys.stdout.write(json.dumps(report, indent=2) + "\n")
        else:
            _format_datasets_verify(report)
        return 0 if report["ok"] else 2
    if cmd == "prepare":
        payload = cli_datasets.prepare_dataset(args.asset_id, source_path=args.source)
        if args.format == "json":
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        else:
            _format_datasets_prepare(payload)
        return 0 if payload["local_layout"]["ok"] else 2
    parser = _build_parser()  # pragma: no cover - defensive
    parser.error(f"unknown datasets command: {cmd}")
    return 2


def _format_datasets_list(rows: list[dict]) -> None:
    """Print a friendly table for ``datasets list``."""
    if not rows:
        sys.stdout.write("No datasets registered.\n")
        return
    sys.stdout.write(f"{len(rows)} dataset(s) registered:\n\n")
    for row in rows:
        download = "auto-download" if row["auto_download_allowed"] else "manual/license-gated"
        sys.stdout.write(f"- {row['asset_id']}  [{row['status']}]\n")
        sys.stdout.write(f"    title: {row['title']}\n")
        sys.stdout.write(f"    source: {row['source_url']}\n")
        sys.stdout.write(f"    license: {row['license_note']}\n")
        sys.stdout.write(f"    acquisition: {download}\n")
        sys.stdout.write("\n")


def _format_datasets_verify(report: dict) -> None:
    """Print a friendly per-artifact layout+checksum report for ``datasets verify``."""
    sys.stdout.write(
        f"Dataset verification: {report['passed']}/{report['pinned_checksums']} "
        f"pinned checksums passed ({report['checked']} checked).\n\n"
    )
    for result in report["results"]:
        layout_ok = "yes" if result["layout_ok"] else "no"
        sys.stdout.write(
            f"- {result['asset_id']}: layout={result['layout_status']} ({layout_ok}); "
            f"checksum={result['checksum_status']}\n"
        )
        if result["pinned_checksum"]:
            sys.stdout.write(f"    expected tree sha256: {result['expected_tree_sha256']}\n")
            if result.get("observed_tree_sha256"):
                sys.stdout.write(f"    observed tree sha256: {result['observed_tree_sha256']}\n")
        if result.get("missing_required_paths"):
            sys.stdout.write(
                f"    missing required paths: {', '.join(result['missing_required_paths'])}\n"
            )
        sys.stdout.write("\n")
    if report["ok"]:
        sys.stdout.write("All pinned dataset checksums verified.\n")
    else:
        sys.stdout.write(
            "One or more datasets are missing, incomplete, or failed checksum. "
            "Run `uv run robot-sf datasets prepare <id>` for acquisition instructions.\n"
        )


def _format_datasets_prepare(payload: dict) -> None:
    """Print acquisition instructions and a local-layout verdict for ``datasets prepare``."""
    layout = payload["local_layout"]
    sys.stdout.write(f"Dataset: {payload['asset_id']} — {payload['title']}\n\n")
    download = "auto-download" if payload["auto_download_allowed"] else "manual/license-gated"
    sys.stdout.write("Acquisition\n")
    sys.stdout.write(f"  source: {payload['source_url']}\n")
    if payload.get("license_url"):
        sys.stdout.write(f"  license url: {payload['license_url']}\n")
    sys.stdout.write(f"  license: {payload['license_note']}\n")
    sys.stdout.write(f"  acquisition: {download}\n")
    sys.stdout.write(f"  instructions: {payload['access_note']}\n")
    if payload.get("acquisition_doc"):
        sys.stdout.write(f"  full guide: {payload['acquisition_doc']}\n")
    sys.stdout.write("\nRequired local layout\n")
    for required in payload["required_paths"]:
        sys.stdout.write(
            f"  - {required['pattern']} ({required['kind']}): {required['description']}\n"
        )
    sys.stdout.write("\nLocal verification (no download)\n")
    sys.stdout.write(f"  status: {layout['status']} (ok={layout['ok']})\n")
    sys.stdout.write(f"  checked path: {layout['source_path']}\n")
    if layout.get("matched_required_paths"):
        sys.stdout.write(f"  matched: {', '.join(layout['matched_required_paths'])}\n")
    if layout.get("missing_required_paths"):
        sys.stdout.write(f"  missing: {', '.join(layout['missing_required_paths'])}\n")
    if layout.get("action"):
        sys.stdout.write(f"  action: {layout['action']}\n")


_HANDLERS = {
    "doctor": _handle_doctor,
    "models": _handle_models,
    "datasets": _handle_datasets,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Top-level ``robot-sf`` entry point.

    Returns:
        int: Process-style exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 1
    handler = _HANDLERS.get(args.cmd)
    if handler is None:  # pragma: no cover - defensive
        parser.error(f"unknown command: {args.cmd}")
        return 2
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - entrypoint
    raise SystemExit(main())
