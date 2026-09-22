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
    source_provenance: dict[str, Any]


class _PackageOperationError(RuntimeError):
    """Identify a filesystem operation that prevents a complete package."""

    def __init__(self, code: str) -> None:
        """Record the stable operation failure code."""
        self.code = code
        super().__init__(code)


def descriptor() -> dict[str, Any]:
    """Return this component's self-contained capability descriptor."""
    return asdict(_DESCRIPTOR)


def _sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    The lexical path is returned so a symlink at the leaf remains visible to
    ``_is_safe_source``. Its real path is checked first so existing parent
    symlinks cannot redirect reads outside ``root``.

    Returns:
        Resolved path, or None with the refusal recorded by the caller.
    """
    if "://" in uri:
        return None
    try:
        candidate = Path(uri)
        if candidate.is_absolute() or ".." in candidate.parts:
            return None
        resolved_root = root.resolve(strict=False)
        lexical = resolved_root / candidate
        lexical.resolve(strict=False).relative_to(resolved_root)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return lexical


def _realpath_within(path: Path, root: Path) -> bool:
    """Return whether a path's real path remains below a real root."""
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _declared_source_provenance(reference: dict[str, Any]) -> dict[str, Any]:
    """Copy the source identity fields declared by a validated bundle reference.

    Returns:
        A standalone mapping of the declared source identity and integrity fields.
    """
    return {
        "artifact_id": str(reference["artifact_id"]),
        "uri": str(reference["uri"]),
        "format": str(reference["format"]),
        "schema": str(reference["schema"]),
        "sha256": str(reference["sha256"]),
        "source_commit": str(reference["source_commit"]),
        "config_identity": str(reference.get("config_identity", "")),
        "units": str(reference["units"]),
        "coordinate_frame": str(reference["coordinate_frame"]),
    }


def _is_video_format(format_name: str) -> bool:
    """Return whether a declared reference format identifies video content."""
    normalized = format_name.strip().lower()
    return normalized == "video" or normalized.startswith(("video-", "video_", "video/", "video."))


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


def _stage_verified_file(target: Path, destination: Path, expected_sha256: str) -> None:
    """Copy one source file and verify the published payload digest."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "rb") as source, open(destination, "wb") as staged:
            shutil.copyfileobj(source, staged)
        if _sha256_file(destination) != expected_sha256:
            raise _PackageOperationError("staging_failed")
    except (OSError, TypeError, ValueError) as error:
        raise _PackageOperationError("staging_failed") from error


def _relocate_reference(
    reference: dict[str, Any],
    root: Path,
    package_dir: Path,
    operations: list[str],
    diagnostics: list[str],
    dry_run: bool,
    source_provenance: dict[str, Any],
) -> _RelocatedEntry | None:
    """Verify and stage one bundle reference.

    Returns:
        Relocated entry, or None when the reference is refused or unavailable.
    """
    artifact_id = str(reference.get("artifact_id", "unknown"))
    uri = str(reference.get("uri", ""))
    if _is_video_format(str(reference.get("format", ""))):
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
        if not _realpath_within(destination, package_dir):
            raise ValueError("destination resolves outside package")
        relocated_path = str(destination.relative_to(package_dir.parent))
    except (TypeError, ValueError):
        diagnostics.append(f"{artifact_id}: reference_not_local")
        return None
    operations.append(f"copy {uri} -> {relocated_path}")
    if dry_run:
        return _RelocatedEntry(
            artifact_id=artifact_id,
            source_uri=uri,
            relocated_path=relocated_path,
            sha256=observed,
            size_bytes=len(raw),
            source_provenance=source_provenance,
        )
    _stage_verified_file(target, destination, observed)
    return _RelocatedEntry(
        artifact_id=artifact_id,
        source_uri=uri,
        relocated_path=relocated_path,
        sha256=observed,
        size_bytes=len(raw),
        source_provenance=source_provenance,
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
        self.source_provenance: list[dict[str, Any]] = []
        self.videos_not_staged: list[dict[str, Any]] = []

    def add_bundle_source(self, *, artifact_id: str, uri: str, root: Path) -> None:
        """Verify and stage every reference of one bundle source."""
        bundle_path = _resolve_local(uri, root)
        if bundle_path is None:
            self.diagnostics.append(f"{artifact_id}: reference_not_local")
            return
        refusal = _is_safe_source(bundle_path)
        if refusal is not None:
            self.diagnostics.append(f"{artifact_id}: {refusal}")
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
                source_provenance = _declared_source_provenance(reference)
                self.source_provenance.append(source_provenance)
                if _is_video_format(source_provenance["format"]):
                    self.videos_not_staged.append(source_provenance)
                entry = _relocate_reference(
                    reference,
                    root,
                    self.package_dir,
                    self.operations,
                    self.diagnostics,
                    self.dry_run,
                    source_provenance,
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
    try:
        output_relative = Path(request.output_directory)
        if (
            not request.output_directory
            or output_relative.is_absolute()
            or ".." in output_relative.parts
        ):
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="invalid_output_path: output directory must be a plain relative path",
            )
        resolved_root = root.resolve(strict=False)
        output_dir = resolved_root / output_relative
        if not _realpath_within(output_dir, resolved_root):
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="unsafe_output_path: output directory must resolve within base",
            )
        if output_dir.exists():
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason=f"output_collision: already exists: {request.output_directory}",
            )
        dry_run = request.config.get("dry_run") is True
        package_name = request.config.get("package_dirname", OUTPUT_PACKAGE_DIRNAME)
        if not isinstance(package_name, str) or not package_name:
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="invalid_config: package_dirname must be a plain relative name",
            )
        package_relative = Path(package_name)
        if package_relative.is_absolute() or ".." in package_relative.parts:
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="invalid_config: package_dirname must be a plain relative name",
            )
        package_dir = output_dir / package_relative
        if not _realpath_within(package_dir, output_dir):
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="invalid_config: package_dirname must resolve within output",
            )
        return output_dir, package_dir, dry_run
    except (OSError, RuntimeError, TypeError, ValueError):
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason="invalid_output_path: output directory must resolve within base",
        )


def _source_provenance_sort_key(source: dict[str, Any]) -> tuple[str, str]:
    """Return the deterministic ordering key for one declared source."""
    return str(source["artifact_id"]), str(source["uri"])


def _sorted_source_provenance(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copied source provenance records in deterministic order."""
    return [dict(source) for source in sorted(sources, key=_source_provenance_sort_key)]


def _result_provenance(request: ComponentRequest, work: _BundleWork) -> dict[str, Any]:
    """Build result provenance without weakening the diagnostic-only boundary.

    Returns:
        Result provenance with declared source identities and no scientific claims.
    """
    source_provenance = _sorted_source_provenance(work.source_provenance)
    return {
        "output_directory": request.output_directory,
        "relocated_entries": len(work.relocated),
        "operations": sorted(set(work.operations)),
        "dry_run": work.dry_run,
        "source_provenance": source_provenance,
        "source_commits": sorted({str(source["source_commit"]) for source in source_provenance}),
    }


def _manifest_entry(entry: _RelocatedEntry) -> dict[str, Any]:
    """Project one relocated entry with its declared source provenance.

    Returns:
        Manifest fields for the staged payload and its declared source identity.
    """
    source = entry.source_provenance
    return {
        "artifact_id": entry.artifact_id,
        "source_uri": entry.source_uri,
        "format": source["format"],
        "schema": source["schema"],
        "source_commit": source["source_commit"],
        "config_identity": source["config_identity"],
        "units": source["units"],
        "coordinate_frame": source["coordinate_frame"],
        "relocated_path": entry.relocated_path,
        "sha256": entry.sha256,
        "size_bytes": entry.size_bytes,
    }


def _published_artifact(
    path: Path,
    output_dir: Path,
    artifact_id: str,
    uri: str,
    expected_sha256: str,
) -> dict[str, str]:
    """Digest one complete output only after validating its published path.

    Returns:
        Artifact record containing the validated URI and observed digest.
    """
    if not _realpath_within(path, output_dir):
        raise _PackageOperationError("publication_failed")
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("published output is not a regular file")
        observed = _sha256_file(path)
    except (OSError, TypeError, ValueError) as error:
        raise _PackageOperationError("publication_failed") from error
    if observed != expected_sha256:
        raise _PackageOperationError("publication_failed")
    return {"artifact_id": artifact_id, "uri": uri, "sha256": observed}


def _complete_artifacts(
    output_dir: Path,
    output_directory: str,
    relocated: list[_RelocatedEntry],
    manifest_digest: str,
    report_digest: str,
) -> tuple[dict[str, str], ...]:
    """Expose and digest every file promised by a complete package result.

    Returns:
        Ordered artifact records for the manifest, report, and payload files.
    """
    output_root = Path(output_directory)
    artifacts = [
        _published_artifact(
            output_dir / OUTPUT_MANIFEST_FILENAME,
            output_dir,
            OUTPUT_MANIFEST_FILENAME,
            str(output_root / OUTPUT_MANIFEST_FILENAME),
            manifest_digest,
        ),
        _published_artifact(
            output_dir / OUTPUT_REPORT_FILENAME,
            output_dir,
            OUTPUT_REPORT_FILENAME,
            str(output_root / OUTPUT_REPORT_FILENAME),
            report_digest,
        ),
    ]
    package_entries: dict[str, _RelocatedEntry] = {}
    for entry in relocated:
        previous = package_entries.get(entry.relocated_path)
        if previous is not None and previous.sha256 != entry.sha256:
            raise _PackageOperationError("publication_failed")
        package_entries[entry.relocated_path] = entry
    for relocated_path, entry in sorted(package_entries.items()):
        artifacts.append(
            _published_artifact(
                output_dir / relocated_path,
                output_dir,
                relocated_path,
                str(output_root / relocated_path),
                entry.sha256,
            )
        )
    return tuple(artifacts)


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:  # noqa: C901
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
            "source_provenance": _sorted_source_provenance(work.source_provenance),
            "videos_not_staged": _sorted_source_provenance(work.videos_not_staged),
            "entries": [
                _manifest_entry(entry)
                for entry in sorted(relocated, key=lambda item: item.artifact_id)
            ],
        }
        verification = {
            "schema_version": "package-verification-report.v1",
            "verified_entries": len(relocated),
            "diagnostics": sorted(set(diagnostics)),
            "videos_staged": [],
            "videos_not_staged": _sorted_source_provenance(work.videos_not_staged),
        }
        manifest_digest: str | None = None
        report_digest: str | None = None
        if not dry_run:
            try:
                manifest_digest = _write_json(output_dir / OUTPUT_MANIFEST_FILENAME, manifest)
                report_digest = _write_json(output_dir / OUTPUT_REPORT_FILENAME, verification)
            except (OSError, TypeError, ValueError) as error:
                raise _PackageOperationError("publication_failed") from error
        partial = bool(diagnostics)
        status = STATUS_PARTIAL if partial else STATUS_COMPLETE
        reason = "" if status == STATUS_COMPLETE else "; ".join(sorted(set(diagnostics))[:8])
        artifacts: tuple[dict[str, Any], ...] = ()
        if status == STATUS_COMPLETE and not dry_run:
            if manifest_digest is None or report_digest is None:
                raise _PackageOperationError("publication_failed")
            artifacts = _complete_artifacts(
                output_dir,
                request.output_directory,
                relocated,
                manifest_digest,
                report_digest,
            )
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=status,
            artifacts=artifacts,
            diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
            provenance=_result_provenance(request, work),
            reason=reason,
        )
    except _PackageOperationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            diagnostics=({"code": error.code},),
            provenance=_result_provenance(request, work),
            reason=error.code,
        )
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason="; ".join(error.errors),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            diagnostics=({"code": "package_failed"},),
            provenance=_result_provenance(request, work),
            reason=f"package_failed: {type(error).__name__}",
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
