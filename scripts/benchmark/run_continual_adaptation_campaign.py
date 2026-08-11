#!/usr/bin/env python3
"""Wire continual-adaptation campaign evidence for the promotion gate (issue #6657).

Builds nominal/shift/forgetting result references and a versioned evidence
bundle naming the validator-derived adapted-policy identifier, then validates
the promotion gate via ``check_continual_adaptation_run``.

This script is metadata-only: it does not launch training, run evaluations,
write checkpoints, mutate the safety wrapper, or promote a policy. Only a
positively identified native record is accepted; every other status fails closed.

Example::

    source .venv/bin/activate
    uv run python scripts/benchmark/run_continual_adaptation_campaign.py \\
        --manifest configs/benchmark/continual_adaptation_promotion_fixture.yaml \\
        --validate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from robot_sf.benchmark.continual_adaptation_campaign import (
    ContinualAdaptationCampaignError,
    ContinualAdaptationEvidenceBundle,
    build_continual_adaptation_evidence,
    prepare_promotion_manifest,
    validate_promotion_readiness,
    verify_local_result_references,
    write_evidence_bundle,
    write_promotion_manifest,
)
from robot_sf.research.continual_adaptation_protocol import (
    ContinualAdaptationProtocolError,
    check_continual_adaptation_run,
    load_continual_adaptation_run,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a continual_adaptation_run.v1 YAML manifest.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the manifest against the promotion gate and exit.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT,
        help="Root used to resolve repository-relative result URIs.",
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        help="Exact path to write the deterministic evidence bundle YAML.",
    )
    parser.add_argument(
        "--promotion-manifest-out",
        type=Path,
        help="Path to write the promotion-ready manifest YAML.",
    )
    parser.add_argument(
        "--execution-mode",
        default="native",
        help="Execution mode label (only native is accepted; every other status fails closed).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def _result_uri(manifest: dict, name: str, fallback: str) -> str:
    """Return an existing result URI or a deterministic issue-scoped fallback."""
    results = manifest.get("results")
    if isinstance(results, dict):
        ref = results.get(name)
        if isinstance(ref, dict):
            uri = ref.get("uri")
            if isinstance(uri, str) and uri.strip():
                return uri
    return fallback


def _read_repo_relative_artifact(root: Path, uri: str) -> bytes:
    """Read one safe repository-relative artifact path."""
    relative = Path(uri)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContinualAdaptationCampaignError(
            f"artifact URI must be a safe repository-relative path: {uri}"
        )
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ContinualAdaptationCampaignError(f"artifact URI escapes root: {uri}") from exc
    if not path.is_file():
        raise ContinualAdaptationCampaignError(f"artifact does not exist: {uri}", source=path)
    return path.read_bytes()


def main() -> int:
    """Run the continual-adaptation campaign evidence builder."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()

    try:
        return _run(args)
    except (ContinualAdaptationCampaignError, ContinualAdaptationProtocolError) as exc:
        LOGGER.log(
            logging.ERROR,
            "campaign validation failed: %s",
            exc,
            extra={"failure": str(exc)},
        )
        return 1


def _run(args: argparse.Namespace) -> int:
    """Execute the fail-closed campaign command for parsed arguments."""
    manifest = load_continual_adaptation_run(args.manifest)
    artifact_root = args.artifact_root.resolve()
    if args.validate:
        return _validate_existing_manifest(manifest, args, artifact_root)
    return _run_generation(manifest, args, artifact_root)


def _validate_existing_manifest(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    artifact_root: Path,
) -> int:
    """Validate one existing promotion manifest and its exact local artifacts."""
    report = check_continual_adaptation_run(manifest, source=args.manifest)
    print(json.dumps(report.to_dict(), indent=2))
    if report.protocol_status != "valid":
        LOGGER.error(
            "protocol validation failed: status=%s blockers=%s",
            report.protocol_status,
            report.blockers,
            extra={
                "protocol_status": report.protocol_status,
                "blockers": list(report.blockers),
            },
        )
        return 1
    if not report.promotion_ready:
        LOGGER.error(
            "promotion gate not satisfied: blockers=%s",
            report.blockers,
            extra={
                "protocol_status": report.protocol_status,
                "blockers": list(report.blockers),
            },
        )
        return 1
    verified = verify_local_result_references(
        manifest,
        artifact_root,
        source=args.manifest,
    )
    LOGGER.info(
        "protocol gate passed with %d exact local artifact checksums",
        len(verified),
        extra={
            "protocol_status": report.protocol_status,
            "verified_artifact_count": len(verified),
        },
    )
    return 0


def _run_generation(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    artifact_root: Path,
) -> int:
    """Build validated metadata and write only explicitly requested outputs."""
    evidence_dir = f"docs/context/evidence/issue_{manifest['issue']}_continual_adaptation_campaign"
    nominal_uri = _result_uri(manifest, "nominal_result", f"{evidence_dir}/nominal_result.json")
    shift_uri = _result_uri(manifest, "shift_result", f"{evidence_dir}/shift_result.json")
    forgetting_uri = _result_uri(
        manifest,
        "forgetting_result",
        f"{evidence_dir}/forgetting_result.json",
    )
    evidence_bundle_uri = _result_uri(
        manifest,
        "evidence_bundle",
        f"{evidence_dir}/evidence_bundle.yaml",
    )
    evidence = build_continual_adaptation_evidence(
        manifest,
        nominal_uri=nominal_uri,
        shift_uri=shift_uri,
        forgetting_uri=forgetting_uri,
        evidence_bundle_uri=evidence_bundle_uri,
        evidence_bundle_identifier=f"continual_adaptation_evidence_{manifest['issue']}_v1",
        execution_mode=args.execution_mode,
        nominal_content=_read_repo_relative_artifact(artifact_root, nominal_uri),
        shift_content=_read_repo_relative_artifact(artifact_root, shift_uri),
        forgetting_content=_read_repo_relative_artifact(artifact_root, forgetting_uri),
        source=args.manifest,
    )

    requested_evidence_out = _resolve_requested_evidence_output(
        args.evidence_out,
        artifact_root=artifact_root,
        evidence_bundle_uri=evidence_bundle_uri,
    )
    promoted = _prepare_requested_promotion(manifest, evidence, args, requested_evidence_out)
    promoted_for_input_validation = promoted or prepare_promotion_manifest(manifest, evidence)
    verify_local_result_references(
        promoted_for_input_validation,
        artifact_root,
        include_evidence_bundle=False,
        source=args.manifest,
    )
    _preflight_output_collisions(
        (requested_evidence_out, args.promotion_manifest_out),
        overwrite=args.overwrite,
    )

    if requested_evidence_out is not None:
        path = write_evidence_bundle(
            evidence,
            requested_evidence_out,
            artifact_root=artifact_root,
            overwrite=args.overwrite,
        )
        LOGGER.info("evidence bundle written: %s", path, extra={"output_path": str(path)})

    if promoted is not None and args.promotion_manifest_out is not None:
        verify_local_result_references(promoted, artifact_root, source=args.promotion_manifest_out)
        path = write_promotion_manifest(
            promoted, args.promotion_manifest_out, overwrite=args.overwrite
        )
        LOGGER.info("promotion manifest written: %s", path, extra={"output_path": str(path)})
        LOGGER.info(
            "protocol fixture gate and exact artifact checksums satisfied",
            extra={"protocol_status": "valid", "blockers": []},
        )

    print(json.dumps(evidence.to_dict(), indent=2))
    return 0


def _resolve_requested_evidence_output(
    requested: Path | None,
    *,
    artifact_root: Path,
    evidence_bundle_uri: str,
) -> Path | None:
    """Resolve and bind ``--evidence-out`` to the manifest-declared URI."""
    if requested is None:
        return None
    expected_out = (artifact_root / evidence_bundle_uri).resolve()
    requested_out = (requested if requested.is_absolute() else artifact_root / requested).resolve()
    if requested_out != expected_out:
        raise ContinualAdaptationCampaignError(
            f"--evidence-out must match the manifest URI: expected {expected_out}, "
            f"got {requested_out}"
        )
    return requested_out


def _prepare_requested_promotion(
    manifest: dict[str, Any],
    evidence: ContinualAdaptationEvidenceBundle,
    args: argparse.Namespace,
    requested_evidence_out: Path | None,
) -> dict[str, Any] | None:
    """Prepare and validate promotion metadata before any artifact write."""
    if args.promotion_manifest_out is None:
        return None
    if requested_evidence_out is None:
        raise ContinualAdaptationCampaignError(
            "--promotion-manifest-out requires --evidence-out so its checksum target exists"
        )
    promoted = prepare_promotion_manifest(manifest, evidence)
    validation = validate_promotion_readiness(promoted, source=args.promotion_manifest_out)
    if not validation.is_promotion_ready:
        raise ContinualAdaptationCampaignError(
            f"promotion gate not satisfied: {validation.blockers}",
            source=args.promotion_manifest_out,
        )
    return promoted


def _preflight_output_collisions(
    outputs: tuple[Path | None, ...],
    *,
    overwrite: bool,
) -> None:
    """Reject known multi-output collisions before the first write."""
    resolved_outputs = [path.resolve() for path in outputs if path is not None]
    if len(resolved_outputs) != len(set(resolved_outputs)):
        raise ContinualAdaptationCampaignError(
            "evidence and promotion-manifest output paths must be distinct"
        )
    if overwrite:
        return
    existing_outputs = [path for path in resolved_outputs if path.exists()]
    if existing_outputs:
        raise ContinualAdaptationCampaignError(
            f"refusing to overwrite existing campaign outputs: {existing_outputs}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
