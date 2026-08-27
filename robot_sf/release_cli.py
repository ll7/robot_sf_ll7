"""Maintainer CLI for benchmark-release diagnostics and direct Zenodo publication."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark import zenodo_publisher
from robot_sf.benchmark.post_execution_release_doctor import (
    EXPECTED_CAMPAIGN_ID,
    EXPECTED_VALIDATOR_SHA,
    collect_post_execution_release_doctor_report,
)
from robot_sf.benchmark.release_doctor import collect_release_doctor_report
from robot_sf.benchmark.release_protocol import load_release_manifest, validate_release_manifest

if TYPE_CHECKING:
    import argparse


def build_subparser(subparsers: Any) -> None:
    """Register the ``robot-sf release`` command tree."""
    release = subparsers.add_parser("release", help="Benchmark-data release operations.")
    modes = release.add_subparsers(dest="release_cmd", required=True)
    zenodo = modes.add_parser("zenodo", help="Direct Zenodo benchmark-dataset publisher.")
    zenodo_modes = zenodo.add_subparsers(dest="zenodo_mode", required=True)
    for mode in ("reserve", "recover", "upload", "publish", "verify"):
        parser = zenodo_modes.add_parser(mode)
        parser.add_argument("--token-file", type=Path, required=True)
        parser.add_argument("--state", type=Path, required=True)
        parser.add_argument("--api-base", default=zenodo_publisher.ZENODO_API_BASE)
        parser.add_argument(
            "--manifest",
            type=Path,
            required=mode == "recover",
            help="Validated benchmark release manifest to bind Zenodo operations to.",
        )
        if mode in {"reserve", "recover", "publish", "verify"}:
            parser.add_argument("--metadata", type=Path, required=True)
        if mode == "recover":
            parser.add_argument("--deposition-id", type=int, required=True)
        if mode == "upload":
            parser.add_argument("files", nargs="+", type=Path)
    doctor = modes.add_parser("doctor", help="Fail-closed full benchmark-release diagnostics.")
    doctor.add_argument("--repo", type=Path, default=Path.cwd())
    doctor.add_argument(
        "--manifest",
        type=Path,
        help="Validated release manifest (required for the pre-execution doctor path).",
    )
    doctor.add_argument("--expected-release-sha", required=True)
    doctor.add_argument("--expected-base-sha", required=True)
    doctor.add_argument("--tag", required=True)
    doctor.add_argument(
        "--expected-campaign-id",
        help="Require the admitted private packet and queue row to use this fixed campaign ID.",
    )
    doctor.add_argument("--checkpoint-receipt", type=Path)
    doctor.add_argument(
        "--checkpoint-path-map",
        action="append",
        default=[],
        metavar="RECEIPT_PATH=LOCAL_PATH",
        help=(
            "Explicitly remap an exact receipt resolved_path to a local checkpoint; repeatable. "
            "The local path must be a regular file beneath --repo."
        ),
    )
    doctor.add_argument("--private-launch-packet", type=Path)
    doctor.add_argument("--private-queue", type=Path)
    doctor.add_argument(
        "--private-ops-repository",
        type=Path,
        help=(
            "Trusted private-ops Git checkout used only for object-addressed ledger reads at "
            "the packet-pinned commit."
        ),
    )
    doctor.add_argument(
        "--post-execution",
        action="store_true",
        help=(
            "Validate preserved derived evidence after a terminal campaign; this mode does not "
            "require the historical queue row to remain dispatchable."
        ),
    )
    doctor.add_argument("--derived-revalidation-receipt", type=Path)
    doctor.add_argument("--publication-bundle", type=Path)
    doctor.add_argument("--publication-archive", type=Path)
    doctor.add_argument("--publication-preflight", type=Path)
    doctor.add_argument("--private-jobs", type=Path)
    doctor.add_argument("--private-evaluation-receipt", type=Path)
    doctor.add_argument("--expected-job-id", default="14890")
    doctor.add_argument("--expected-validator-sha")
    doctor.add_argument("--dissertation", type=Path)
    doctor.add_argument("--token-file", type=Path)
    doctor.add_argument("--expected-cells", type=int, default=20160)
    doctor.add_argument("--minimum-free-gib", type=float, default=100.0)
    doctor.add_argument("--require-zenodo-webhook-disabled", action="store_true")
    doctor.add_argument(
        "--publication-mode",
        choices=("pre-publication", "final"),
        default="pre-publication",
        help="Use final for fail-closed publication admission.",
    )
    doctor.add_argument(
        "--final",
        action="store_true",
        help="Alias for --publication-mode final.",
    )


def _print(payload: dict[str, Any]) -> None:
    """Print a stable JSON object without credentials."""
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_release_binding(args: argparse.Namespace) -> tuple[Any, dict[str, Any]] | None:
    """Load and validate an optional benchmark manifest for Zenodo operations.

    Returns:
        The parsed manifest and credential-free publisher binding, or ``None``
        for the generic publisher route.
    """
    manifest_path = getattr(args, "manifest", None)
    if manifest_path is None:
        return None
    manifest = load_release_manifest(Path(manifest_path).resolve())
    validation = validate_release_manifest(manifest)
    if validation["status"] != "valid":
        problems = "; ".join(str(problem) for problem in validation["problems"])
        raise zenodo_publisher.ZenodoPublisherError(f"release manifest is invalid: {problems}")
    binding = zenodo_publisher.build_release_binding(manifest)
    metadata_path = getattr(args, "metadata", None)
    if metadata_path is not None and Path(metadata_path).resolve() != binding["metadata_path"]:
        raise zenodo_publisher.ZenodoPublisherError(
            "Zenodo metadata path does not match the release manifest"
        )
    return manifest, binding


def _repo_relative_path(path: Path | None, repo: Path) -> Path | None:
    """Resolve a doctor argument against the explicit ``--repo`` root when relative.

    The doctor may run from a separate tooling worktree while ``--repo`` names the
    exact release checkout; a relative manifest/config path must then resolve
    against ``--repo``, not the process working directory, so every manifest,
    config, and scenario lookup reads the intended checkout (issue #7794).

    Returns:
        The absolute resolved path, or ``None`` when the input path is ``None``.
    """
    if path is None:
        return None
    if path.is_absolute():
        return path
    return (repo / path).resolve()


def _handle_post_execution_doctor(args: argparse.Namespace, repo_root: Path) -> int:
    """Run the read-only preserved-evidence doctor path.

    Returns:
        Process-style exit code.
    """
    report = collect_post_execution_release_doctor_report(
        repo=repo_root,
        manifest_path=_repo_relative_path(args.manifest, repo_root),
        derived_revalidation_receipt=_repo_relative_path(
            args.derived_revalidation_receipt, repo_root
        ),
        publication_bundle=_repo_relative_path(args.publication_bundle, repo_root),
        publication_archive=_repo_relative_path(args.publication_archive, repo_root),
        publication_preflight=_repo_relative_path(args.publication_preflight, repo_root),
        private_queue=_repo_relative_path(args.private_queue, repo_root),
        private_jobs=_repo_relative_path(args.private_jobs, repo_root),
        private_launch_packet=_repo_relative_path(args.private_launch_packet, repo_root),
        private_evaluation_receipt=_repo_relative_path(args.private_evaluation_receipt, repo_root),
        dissertation=_repo_relative_path(args.dissertation, repo_root),
        token_file=_repo_relative_path(args.token_file, repo_root),
        minimum_free_gib=args.minimum_free_gib,
        require_zenodo_webhook_disabled=(
            args.require_zenodo_webhook_disabled or args.final or args.publication_mode == "final"
        ),
        expected_source_sha=args.expected_release_sha,
        expected_base_sha=args.expected_base_sha,
        tag=args.tag,
        expected_campaign_id=args.expected_campaign_id or EXPECTED_CAMPAIGN_ID,
        expected_job_id=args.expected_job_id,
        expected_validator_sha=args.expected_validator_sha or EXPECTED_VALIDATOR_SHA,
    )
    _print(report)
    return 0 if report["status"] == "pass" else 2


def handle(args: argparse.Namespace) -> int:  # noqa: C901
    """Dispatch release operations and return a process exit code.

    Returns:
        Zero for success and two for a blocked or failed publication operation.
    """
    if args.release_cmd == "doctor":
        repo_root = args.repo.resolve()
        if getattr(args, "post_execution", False):
            return _handle_post_execution_doctor(args, repo_root)
        if args.manifest is None:
            _print(
                {"status": "blocked", "reason": "--manifest is required without --post-execution"}
            )
            return 2
        report = collect_release_doctor_report(
            repo=repo_root,
            manifest_path=_repo_relative_path(args.manifest, repo_root),
            expected_release_sha=args.expected_release_sha,
            expected_base_sha=args.expected_base_sha,
            tag=args.tag,
            expected_campaign_id=getattr(args, "expected_campaign_id", None),
            checkpoint_receipt=_repo_relative_path(args.checkpoint_receipt, repo_root),
            checkpoint_path_map=getattr(args, "checkpoint_path_map", None),
            private_launch_packet=_repo_relative_path(args.private_launch_packet, repo_root),
            private_queue=_repo_relative_path(getattr(args, "private_queue", None), repo_root),
            private_ops_repository=(
                args.private_ops_repository.resolve()
                if getattr(args, "private_ops_repository", None) is not None
                else None
            ),
            dissertation=_repo_relative_path(args.dissertation, repo_root),
            token_file=_repo_relative_path(args.token_file, repo_root),
            expected_cells=args.expected_cells,
            minimum_free_gib=args.minimum_free_gib,
            require_zenodo_webhook_disabled=args.require_zenodo_webhook_disabled,
            publication_mode=(
                "final"
                if getattr(args, "final", False)
                else getattr(args, "publication_mode", "pre-publication")
            ),
        )
        _print(report)
        return 0 if report["status"] == "pass" else 2
    try:
        release_context = _load_release_binding(args)
        release_manifest = release_context[0] if release_context is not None else None
        release_binding = release_context[1] if release_context is not None else None
        session = zenodo_publisher.build_session(args.token_file)
        if args.zenodo_mode in {"reserve", "recover"}:
            metadata_kwargs = (
                {
                    "expected_source_tag": release_manifest.release_tag,
                    "expected_metadata_sha256": release_manifest.metadata_sha256,
                }
                if release_manifest is not None
                else {}
            )
            metadata = zenodo_publisher.load_dataset_metadata(args.metadata, **metadata_kwargs)
            operation_kwargs = (
                {"release_binding": release_binding} if release_binding is not None else {}
            )
            if args.zenodo_mode == "recover":
                if release_binding is None:
                    raise zenodo_publisher.ZenodoPublisherError(
                        "Zenodo recovery requires a validated release manifest"
                    )
                state = zenodo_publisher.recover(
                    session,
                    args.deposition_id,
                    metadata,
                    api_base=args.api_base,
                    release_binding=release_binding,
                )
            else:
                state = zenodo_publisher.reserve(
                    session, metadata, api_base=args.api_base, **operation_kwargs
                )
            zenodo_publisher.write_state(args.state, state)
            _print(state)
            return 0
        state = zenodo_publisher.load_state(args.state)
        if args.zenodo_mode == "upload":
            operation_kwargs = (
                {"release_binding": release_binding} if release_binding is not None else {}
            )
            state = zenodo_publisher.upload(
                session, state, args.files, api_base=args.api_base, **operation_kwargs
            )
            zenodo_publisher.write_state(args.state, state)
            _print(state)
            return 0
        if args.zenodo_mode == "publish":
            metadata_kwargs = (
                {
                    "expected_source_tag": release_manifest.release_tag,
                    "expected_metadata_sha256": release_manifest.metadata_sha256,
                }
                if release_manifest is not None
                else {}
            )
            metadata = zenodo_publisher.load_dataset_metadata(args.metadata, **metadata_kwargs)
            operation_kwargs = (
                {"release_binding": release_binding} if release_binding is not None else {}
            )
            state = zenodo_publisher.publish(
                session,
                state,
                metadata,
                api_base=args.api_base,
                **operation_kwargs,
            )
            zenodo_publisher.write_state(args.state, state)
            _print(state)
            return 0
        metadata_kwargs = (
            {
                "expected_source_tag": release_manifest.release_tag,
                "expected_metadata_sha256": release_manifest.metadata_sha256,
            }
            if release_manifest is not None
            else {}
        )
        metadata = zenodo_publisher.load_dataset_metadata(args.metadata, **metadata_kwargs)
        operation_kwargs = (
            {"release_binding": release_binding} if release_binding is not None else {}
        )
        report = zenodo_publisher.verify(
            session, state, metadata, api_base=args.api_base, **operation_kwargs
        )
        if report.get("status") == "pass":
            zenodo_publisher.write_state(args.state, state)
        _print(report)
        return 0 if report["status"] == "pass" else 2
    except (FileNotFoundError, ValueError, zenodo_publisher.ZenodoPublisherError) as exc:
        _print({"status": "blocked", "reason": str(exc)})
        return 2


__all__ = ["build_subparser", "handle"]
