"""SREV-04 review-package component: build relocatable verified packages.

This module owns the SREV-04 leaf surface only: a ``run(request)`` adapter plus a
standalone CLI that consumes the SREV-01 shared contracts, relocates bundle
payloads into a verified package, and reports verification. Sources are
copied, never modified; remote, absolute, traversal, and symlink-escaping
references are refused; video payloads are never staged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_request_from_dict,
    review_bundle_from_dict,
)

COMPONENT_ID = "srev04-review-package"
COMPONENT_VERSION = "1.0.0"

REQUIRED_CAPABILITIES = ("review-bundle",)
OPTIONAL_CAPABILITIES: tuple[str, ...] = ()

OUTPUT_MANIFEST_FILENAME = "manifest.json"
OUTPUT_REPORT_FILENAME = "verification-report.json"
OUTPUT_PACKAGE_DIRNAME = "package"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

_VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})

_DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=("review-package.v1", "missing-capability-report.v1"),
    required_capabilities=REQUIRED_CAPABILITIES,
    optional_capabilities=OPTIONAL_CAPABILITIES,
)


@dataclass
class _RelocatedEntry:
    """Verified payload staged for the package."""

    artifact_id: str
    source_uri: str
    relocated_path: str
    sha256: str
    size_bytes: int


def descriptor() -> dict[str, Any]:
    """Return this component's self-contained capability descriptor."""
    return asdict(_DESCRIPTOR)


def _sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Atomically write strict-JSON.

    Returns:
        Hex digest of the written bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return _sha256_bytes(text.encode("utf-8"))


def _check_version_compatible(config: dict[str, Any]) -> str | None:
    """Return a failure reason when the request demands a newer component.

    Returns:
        Failure reason string, or None when the component version satisfies it.
    """
    minimum = config.get("min_component_version")
    if minimum is None:
        return None
    try:
        wanted = int(str(minimum).split(".", maxsplit=1)[0])
        ours = int(COMPONENT_VERSION.split(".", maxsplit=1)[0])
    except ValueError:
        return f"incompatible_component_version: malformed min_component_version: {minimum!r}"
    if wanted > ours:
        return f"incompatible_component_version: request needs v{wanted}, component is v{ours}"
    return None


def _reject_not_applicable(request: ComponentRequest) -> ComponentResult | None:
    """Reject requests this component cannot serve.

    Returns:
        An unavailable/failed result, or None when the request applies.
    """
    if request.component_id != COMPONENT_ID:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_UNAVAILABLE,
            reason=f"unsupported_component: {request.component_id}",
        )
    supported = set(_DESCRIPTOR.required_capabilities) | set(_DESCRIPTOR.optional_capabilities)
    missing = [name for name in request.required_capabilities if name not in supported]
    if missing:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_UNAVAILABLE,
            reason=f"missing_required_capabilities: {', '.join(sorted(missing))}",
        )
    version_error = _check_version_compatible(request.config)
    if version_error is not None:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=version_error,
        )
    return None


def _resolve_local(uri: str, root: Path) -> Path | None:
    """Resolve a local relative URI, rejecting remote/absolute/traversal refs.

    Returns:
        Resolved path, or None with the refusal recorded by the caller.
    """
    if "://" in uri:
        return None
    candidate = Path(uri)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return root / candidate


def _is_safe_source(path: Path) -> str | None:
    """Return a refusal code when a source path is unsafe to copy.

    Returns:
        Refusal code, or None when the path is a plain readable file.
    """
    try:
        if path.is_symlink():
            return "symlink_refused"
        if not path.is_file():
            return "source_unreadable"
    except OSError:
        return "source_unreadable"
    return None


def _relocate_reference(
    reference: dict[str, Any],
    root: Path,
    package_dir: Path,
    operations: list[str],
    diagnostics: list[str],
    dry_run: bool,
) -> _RelocatedEntry | None:
    """Verify and stage one bundle reference.

    Returns:
        Relocated entry, or None when the reference is refused or unavailable.
    """
    artifact_id = str(reference.get("artifact_id", "unknown"))
    uri = str(reference.get("uri", ""))
    suffix = Path(uri).suffix.lower()
    if suffix in _VIDEO_SUFFIXES:
        diagnostics.append(f"{artifact_id}: video_not_staged")
        return None
    target = _resolve_local(uri, root)
    if target is None:
        diagnostics.append(f"{artifact_id}: reference_not_local")
        return None
    refusal = _is_safe_source(target)
    if refusal is not None:
        diagnostics.append(f"{artifact_id}: {refusal}")
        return None
    try:
        raw = target.read_bytes()
    except OSError:
        diagnostics.append(f"{artifact_id}: source_unreadable")
        return None
    observed = _sha256_bytes(raw)
    declared = str(reference.get("sha256", ""))
    if declared and declared != observed:
        diagnostics.append(f"{artifact_id}: tampered_payload")
        return None
    try:
        relative = Path(uri)
        destination = package_dir / relative
        destination.relative_to(package_dir)
    except ValueError:
        diagnostics.append(f"{artifact_id}: reference_not_local")
        return None
    operations.append(f"copy {uri} -> {destination.relative_to(package_dir.parent)}")
    if dry_run:
        return _RelocatedEntry(
            artifact_id=artifact_id,
            source_uri=uri,
            relocated_path=str(destination.relative_to(package_dir.parent)),
            sha256=observed,
            size_bytes=len(raw),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "rb") as source, open(destination, "wb") as staged:
        shutil.copyfileobj(source, staged)
    return _RelocatedEntry(
        artifact_id=artifact_id,
        source_uri=uri,
        relocated_path=str(destination.relative_to(package_dir.parent)),
        sha256=observed,
        size_bytes=len(raw),
    )


class _BundleWork:
    """Mutable per-run bundle processing state."""

    def __init__(self, package_dir: Path, dry_run: bool) -> None:
        """Record the staging target and dry-run mode."""
        self.package_dir = package_dir
        self.dry_run = dry_run
        self.diagnostics: list[str] = []
        self.operations: list[str] = []
        self.relocated: list[_RelocatedEntry] = []

    def add_bundle_source(self, *, artifact_id: str, uri: str, root: Path) -> None:
        """Verify and stage every reference of one bundle source."""
        bundle_path = _resolve_local(uri, root)
        if bundle_path is None:
            self.diagnostics.append(f"{artifact_id}: reference_not_local")
            return
        try:
            bundle_payload = json.loads(bundle_path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            self.diagnostics.append(f"{artifact_id}: bundle_unreadable")
            return
        try:
            bundle = review_bundle_from_dict(bundle_payload, source=uri)
        except ReviewContractsValidationError as error:
            self.diagnostics.append(f"{artifact_id}: bundle_invalid:" + error.errors[0][:80])
            return
        for episode in bundle.episodes:
            for reference in episode["references"]:
                entry = _relocate_reference(
                    reference,
                    root,
                    self.package_dir,
                    self.operations,
                    self.diagnostics,
                    self.dry_run,
                )
                if entry is not None:
                    self.relocated.append(entry)


def _prepare_directories(
    request: ComponentRequest, root: Path
) -> tuple[Path, Path, bool] | ComponentResult:
    """Validate output/package paths and the dry-run flag.

    Returns:
        Either an error result or the (output_dir, package_dir, dry_run) tuple.
    """
    output_dir = root / request.output_directory
    if output_dir.exists():
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=f"output_collision: already exists: {request.output_directory}",
        )
    dry_run = request.config.get("dry_run") is True
    package_name = request.config.get("package_dirname", OUTPUT_PACKAGE_DIRNAME)
    if not isinstance(package_name, str) or not package_name or ".." in Path(package_name).parts:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason="invalid_config: package_dirname must be a plain relative name",
        )
    return output_dir, output_dir / package_name, dry_run


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Relocate bundle payloads into a verified package.

    Args:
        request: Validated component request referencing review bundles.
        base: Base directory source URIs and the output directory resolve under.

    Returns:
        Component result: ``complete`` only when every reference verified and
        staged, ``partial`` on refused/unavailable entries, ``unavailable`` when
        the component or a required capability does not apply, ``failed`` on
        validation, version, collision, or internal errors.
    """
    root = base if base is not None else Path.cwd()
    rejected = _reject_not_applicable(request)
    if rejected is not None:
        return rejected
    prepared = _prepare_directories(request, root)
    if isinstance(prepared, ComponentResult):
        return prepared
    output_dir, package_dir, dry_run = prepared
    work = _BundleWork(package_dir, dry_run)
    diagnostics = work.diagnostics
    try:
        bundle_refs = [ref for ref in request.sources if ref.format == "review-bundle"]
        for ref in request.sources:
            if ref.format != "review-bundle":
                diagnostics.append(f"{ref.artifact_id}: unknown_source_format:{ref.format}")
        if not bundle_refs:
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="no_review_bundle_source",
            )
        for ref in bundle_refs:
            work.add_bundle_source(artifact_id=ref.artifact_id, uri=ref.uri, root=root)
        relocated = work.relocated
        operations = work.operations
        manifest = {
            "schema_version": "review-package.v1",
            "request_id": request.request_id,
            "dry_run": dry_run,
            "operations": sorted(set(operations)),
            "entries": [
                {
                    "artifact_id": entry.artifact_id,
                    "source_uri": entry.source_uri,
                    "relocated_path": entry.relocated_path,
                    "sha256": entry.sha256,
                    "size_bytes": entry.size_bytes,
                }
                for entry in sorted(relocated, key=lambda item: item.artifact_id)
            ],
        }
        verification = {
            "schema_version": "package-verification-report.v1",
            "verified_entries": len(relocated),
            "diagnostics": sorted(set(diagnostics)),
            "videos_staged": [],
        }
        if not dry_run:
            _write_json(output_dir / "manifest.json", manifest)
            _write_json(output_dir / OUTPUT_REPORT_FILENAME, verification)
        partial = bool(diagnostics)
        status = STATUS_PARTIAL if partial else STATUS_COMPLETE
        reason = "" if status == STATUS_COMPLETE else "; ".join(sorted(set(diagnostics))[:8])
        artifacts: tuple[dict[str, Any], ...] = ()
        if status == STATUS_COMPLETE and not dry_run:
            manifest_text = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
            artifacts = (
                {
                    "artifact_id": "manifest.json",
                    "uri": str(Path(request.output_directory) / "manifest.json"),
                    "sha256": _sha256_bytes(manifest_text.encode()),
                },
            )
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=status,
            artifacts=artifacts,
            diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
            provenance={
                "output_directory": request.output_directory,
                "relocated_entries": len(relocated),
                "operations": sorted(set(operations)),
                "dry_run": dry_run,
            },
            reason=reason,
        )
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason="; ".join(error.errors),
        )


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the review-package component."""
    parser = argparse.ArgumentParser(description="Package verified review artifacts.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base dir for resolution.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-package component.

    Returns:
        Process exit code (0 only when the result status is complete).
    """
    args = _build_parser().parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewContractsValidationError([f"cannot read request: {error}"]) from error
    if args.config is not None:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewContractsValidationError([f"cannot read config: {error}"]) from error
        if isinstance(config, dict):
            payload = {**payload, "config": {**payload.get("config", {}), **config}}
    payload = {**payload, "output_directory": args.output}
    request = component_request_from_dict(payload, source=args.input)
    result = run(request, base=Path(args.base) if args.base is not None else None)
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == STATUS_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
