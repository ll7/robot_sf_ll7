"""Top-level ``robot-sf`` command line interface.

Thin, user-facing entry point for everyday Robot SF workflows. The
``doctor`` subcommand wraps the existing runtime diagnostics in
:mod:`robot_sf.benchmark.doctor` behind one obvious command, adding
friendly, remedy-bearing output for beginners. The ``models`` and
``datasets`` subcommands wrap the existing model registry and external-data
provenance/checksum machinery (issue #5797) behind list/verify/download and
list/verify/prepare commands. The ``demo`` subcommand exposes the one-command
visual demo from the adoption/UX epic. The ``gallery`` subcommand builds a
self-contained static scenario/planner gallery from existing rendering tools.
More everyday workflows can be added here without creating another top-level
entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf import cli_datasets, cli_models
from robot_sf.benchmark.doctor import collect_doctor_report, doctor_exit_code
from robot_sf.examples_cli import examples_cli_main

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_GALLERY_MATRIX = "configs/baselines/example_matrix.yaml"
DEFAULT_GALLERY_OUT_DIR = "output/gallery"


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``robot-sf`` argument parser.

    Returns:
        argparse.ArgumentParser: Parser with the registered subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="robot-sf",
        description="Robot SF top-level command line interface.",
    )
    subparsers = parser.add_subparsers(dest="cmd")

    doctor = subparsers.add_parser(
        "doctor",
        help="Environment/readiness check with friendly remedies.",
    )
    doctor.add_argument(
        "--format",
        choices=("friendly", "json"),
        default="friendly",
        help="Output format (default: friendly).",
    )
    doctor.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("output"),
        help="Artifact root to probe for temporary write access (default: output).",
    )
    doctor.add_argument(
        "--skip-env-smoke",
        action="store_true",
        help="Skip the minimal reset/step environment smoke check.",
    )
    doctor.add_argument(
        "--skip-quickstart-smoke",
        action="store_true",
        default=False,
        help="Skip executing the manifest-declared quickstart examples",
    )
    _add_models_subparser(subparsers)
    _add_datasets_subparser(subparsers)
    # The ``examples`` subcommand owns its own sub-subcommand parser
    # (``list``/``run``); it is registered here only so the top-level parser
    # recognises the token. Remaining args are forwarded by the handler.
    subparsers.add_parser(
        "examples",
        add_help=False,
        help="List and run examples from examples_manifest.yaml (issue #5794)",
    )

    demo = subparsers.add_parser(
        "demo",
        help="Run the one-command visual demo (tiny deterministic episode + viewer).",
    )
    demo.add_argument("--output-root", type=Path, default=None)
    demo.add_argument("--scenario", type=Path, default=None)
    demo.add_argument("--seed", type=int, default=None)
    demo.add_argument("--verbose", action="store_true")

    gallery = subparsers.add_parser(
        "gallery",
        help="Build a static scenario/planner gallery (issue #5796)",
    )
    gallery_sub = gallery.add_subparsers(dest="gallery_cmd", required=True)
    g_build = gallery_sub.add_parser(
        "build",
        help=(
            "Render per-scenario thumbnails (reusing existing tooling) and emit a "
            "self-contained static HTML gallery plus a JSON manifest. "
            "Discoverability artifact only — not benchmark evidence."
        ),
    )
    g_build.add_argument(
        "--matrix",
        default=DEFAULT_GALLERY_MATRIX,
        help=f"Path to scenario matrix YAML (default: {DEFAULT_GALLERY_MATRIX})",
    )
    g_build.add_argument(
        "--out-dir",
        default=DEFAULT_GALLERY_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_GALLERY_OUT_DIR})",
    )
    g_build.add_argument("--base-seed", type=int, default=0, help="Base seed for thumbnails")
    g_build.add_argument(
        "--horizon",
        type=int,
        default=100,
        help="Horizon (steps) used for the expected-runtime estimate",
    )
    g_build.add_argument(
        "--no-thumbnails",
        action="store_true",
        default=False,
        help="Skip thumbnail rendering (cards show a placeholder)",
    )
    g_build.add_argument(
        "--link-thumbnails",
        action="store_true",
        default=False,
        help="Reference thumbnails by relative path instead of embedding as data URIs",
    )
    g_build.add_argument(
        "--sample-rollout-root",
        default=None,
        help=(
            "Optional directory searched for per-scenario sample rollouts "
            "(<id>.mp4/.webm/.jsonl/.html)"
        ),
    )
    g_build.add_argument(
        "--title",
        default=None,
        help="Optional page title (defaults to a name derived from the matrix)",
    )
    g_build.set_defaults(gallery_cmd="build")
    return parser


def _handle_doctor(args: argparse.Namespace) -> int:
    """Run the doctor check and print the report.

    Returns:
        int: Process-style doctor exit code.
    """
    report = collect_doctor_report(
        artifact_root=args.artifact_root,
        run_env_smoke=not args.skip_env_smoke,
        run_quickstart_smoke=not args.skip_quickstart_smoke,
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


def _handle_gallery_build(args: argparse.Namespace) -> int:
    """Handle ``gallery build`` and emit a self-contained inspection artifact.

    Returns:
        Exit code (0 for success, 2 for invalid input or build failure).
    """
    from robot_sf.benchmark.runner import load_scenario_matrix  # noqa: PLC0415
    from robot_sf.gallery.builder import build_gallery  # noqa: PLC0415

    try:
        scenarios = load_scenario_matrix(args.matrix)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"error: failed to load matrix {args.matrix}: {exc}\n")
        return 2

    try:
        result = build_gallery(
            scenarios,
            matrix_path=str(args.matrix),
            out_dir=Path(args.out_dir),
            base_seed=int(args.base_seed),
            horizon_steps=int(args.horizon),
            render_thumbnails=not bool(args.no_thumbnails),
            embed_thumbnails=not bool(args.link_thumbnails),
            sample_rollout_root=args.sample_rollout_root,
            title=args.title,
        )
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    payload = {
        "html_path": str(result.html_path),
        "manifest_path": str(result.manifest_path),
        "scenario_count": len(result.cards),
        "matrix_path": result.matrix_path,
        "schema_version": result.schema_version,
    }
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


def _handle_gallery(args: argparse.Namespace) -> int:
    """Dispatch the ``gallery`` command group.

    Returns:
        Exit code from the selected gallery subcommand.
    """
    if args.gallery_cmd != "build":
        return 2
    return _handle_gallery_build(args)


_HANDLERS = {
    "doctor": _handle_doctor,
    "models": _handle_models,
    "datasets": _handle_datasets,
    "gallery": _handle_gallery,
}


def _handle_examples(extra_args: Sequence[str]) -> int:
    """Forward to the examples discovery CLI.

    Args:
        extra_args: The arguments following the ``examples`` token.

    Returns:
        int: Process-style exit code from the examples CLI.
    """
    return examples_cli_main(list(extra_args))


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to the requested ``robot-sf`` subcommand.

    Returns:
        int: Process exit status code.
    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    # The ``examples`` subcommand owns its own sub-parser (``list``/``run``), so
    # forward everything after the token verbatim and avoid letting the
    # top-level parser consume example-specific options.
    if args_list and args_list[0] == "examples":
        return _handle_examples(args_list[1:])

    parser = _build_parser()
    args = parser.parse_args(args_list)

    if args.cmd == "demo":
        from scripts.demo.quickstart_demo import main as demo_main  # noqa: PLC0415

        demo_argv = []
        if args.output_root is not None:
            demo_argv += ["--output-root", str(args.output_root)]
        if args.scenario is not None:
            demo_argv += ["--scenario", str(args.scenario)]
        if args.seed is not None:
            demo_argv += ["--seed", str(args.seed)]
        if args.verbose:
            demo_argv.append("--verbose")
        return demo_main(demo_argv)

    if args.cmd is None:
        parser.print_help()
        return 1
    handler = _HANDLERS.get(args.cmd)
    if handler is None:  # pragma: no cover - defensive
        parser.error(f"unknown command: {args.cmd}")
        return 2
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
