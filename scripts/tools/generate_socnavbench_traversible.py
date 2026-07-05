#!/usr/bin/env python3
"""Reproducibly build a SocNavBench custom-map traversible (``traversibles/<MAP>/data.pkl``).

Context (issue #4291, closing the last gap from issue #1498)
------------------------------------------------------------
The official SocNavBench asset package ships S3DIS traversibles (areas 1-6) but
*not* traversibles for the custom maps (ETH/Hotel/Univ/Zara/DoubleHotel).
SocNavBench generates those from the curated per-map mesh the first time a map is
loaded with meshes enabled: :meth:`SBPDRenderer.get_config` builds the traversible
from the mesh and pickles it to ``traversibles/<MAP>/data.pkl``. This wrapper makes
that generation an explicit, reproducible, provenance-recorded step so issue #1134
(ETH map conversion) can unblock once the derived ``data.pkl`` exists.

Design constraints
------------------
- **Derived data stays out of git.** The generated ``data.pkl`` is written *into the
  external data root* (the same tree that stages the mesh), never into the repository.
- **CI-safe by construction.** Input validation (``--dry-run`` and skip-if-absent) uses
  only the standard library plus the repository's external-data path resolver, so it runs
  in the main environment. The heavy SocNavBench mesh pipeline (``dotmap``, ``mp_env``,
  swiftshader, staged meshes) is imported lazily and only reached on an explicit build.
- **Fail closed.** When the required per-map mesh is not staged, the tool prints an
  actionable message and exits ``2`` instead of silently producing nothing.

Exit codes
----------
``0`` build succeeded, output already present (idempotent skip), or ``--dry-run`` inputs valid.
``2`` blocked: the required mesh is not staged (or another required input is missing).

Typical usage
-------------
Validate inputs without building (CI/local, no SocNavBench deps required)::

    uv run python scripts/tools/generate_socnavbench_traversible.py --map ETH --dry-run

Run the actual generation (maintainer step, SocNavBench environment with staged mesh)::

    uv run python scripts/tools/generate_socnavbench_traversible.py --map ETH

The build prints the SHA-256 of the produced ``data.pkl`` plus registry-style output
tree checksum metadata so the external-data registry pin can be updated after the
maintainer re-seeds the internal store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.tools.manage_external_data import (
    EXTERNAL_DATA_ROOT_ENV,
    resolve_asset_local_path_by_id,
)

# The external-data asset whose staged tree owns the SocNavBench custom-map meshes and
# traversibles. Reusing this id keeps this tool consistent with the registry and with
# ``validate_socnav_map_batch.py`` instead of hard-coding a parallel path convention.
SOCNAV_ASSET_ID = "socnavbench-s3dis-eth"

# S3DIS/SBPD dataset sub-tree inside the SocNavBench root; both meshes and generated
# traversibles live under here, matching SocNavBench's ``central_params`` layout and the
# ``socnavbench_import_batches.yaml`` manifest.
DATASET_SUBPATH = Path("sd3dis/stanford_building_parser_dataset")

# Blocked exit code, matching ``validate_socnav_map_batch.py`` so callers can treat a
# missing-source-asset condition uniformly across the SocNavBench staging tooling.
EXIT_BLOCKED = 2

STATUS_READY = "ready"
STATUS_ALREADY_PRESENT = "already_present"
STATUS_BLOCKED_MISSING_MESH = "blocked_missing_mesh"


class TraversibleGenerationError(RuntimeError):
    """Raised when a traversible build cannot proceed or does not produce its output."""


@dataclass(frozen=True)
class TraversiblePaths:
    """Resolved filesystem paths for one map's mesh input and traversible output."""

    map_name: str
    socnav_root: Path
    mesh_dir: Path
    traversible_dir: Path
    output_pkl: Path


def resolve_paths(map_name: str, *, root: Path | None = None) -> TraversiblePaths:
    """Resolve the mesh input and traversible output paths for one SocNavBench map.

    The socnav root honors the shared external-data root (``ROBOT_SF_EXTERNAL_DATA_ROOT``)
    via the registry, falling back to the in-repo ``third_party/socnavbench`` location.
    ``map_name`` is used verbatim as SocNavBench's ``building_name`` (e.g. ``ETH``).
    """
    name = map_name.strip()
    if not name or "/" in name or name in {".", ".."}:
        raise ValueError(f"Invalid map name: {map_name!r}")
    socnav_root = resolve_asset_local_path_by_id(SOCNAV_ASSET_ID, root=root)
    dataset_root = socnav_root / DATASET_SUBPATH
    mesh_dir = dataset_root / "mesh" / name
    traversible_dir = dataset_root / "traversibles" / name
    return TraversiblePaths(
        map_name=name,
        socnav_root=socnav_root,
        mesh_dir=mesh_dir,
        traversible_dir=traversible_dir,
        output_pkl=traversible_dir / "data.pkl",
    )


def _mesh_is_staged(mesh_dir: Path) -> bool:
    """Return whether the mesh directory exists and holds at least one file.

    An empty directory is treated as *not staged* (a placeholder shell), mirroring the
    fail-closed convention in ``prepare_socnav_assets.py`` so a hollow directory can never
    be mistaken for a usable mesh.
    """
    return mesh_dir.is_dir() and any(child.is_file() for child in mesh_dir.rglob("*"))


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_tree_checksum(paths: TraversiblePaths) -> dict[str, Any]:
    """Return registry-style checksum metadata for the generated traversible output tree."""
    if not paths.output_pkl.is_file():
        return {
            "output_tree_sha256": None,
            "output_tree_file_count": 0,
            "output_tree_total_size_bytes": 0,
        }
    file_sha = sha256_file(paths.output_pkl)
    size = paths.output_pkl.stat().st_size
    rel = paths.output_pkl.relative_to(paths.traversible_dir).as_posix()
    digest = hashlib.sha256()
    digest.update(rel.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(size).encode("ascii"))
    digest.update(b"\0")
    digest.update(file_sha.encode("ascii"))
    digest.update(b"\0")
    return {
        "output_tree_sha256": digest.hexdigest(),
        "output_tree_file_count": 1,
        "output_tree_total_size_bytes": size,
    }


def registry_pin_report(paths: TraversiblePaths, report: dict[str, Any]) -> dict[str, Any]:
    """Return maintainer-review metadata for pinning the generated traversible tree."""
    tree_sha256 = report.get("output_tree_sha256")
    if not isinstance(tree_sha256, str) or not tree_sha256:
        raise TraversibleGenerationError(
            "Cannot write registry pin report before traversible data.pkl exists."
        )
    return {
        "asset_id": SOCNAV_ASSET_ID,
        "map_name": paths.map_name,
        "status": "ready_for_trusted_pin_review",
        "expected_tree_sha256": tree_sha256,
        "expected_tree_sha256_status_after_pin": "pinned",
        "output_tree_file_count": report["output_tree_file_count"],
        "output_tree_total_size_bytes": report["output_tree_total_size_bytes"],
        "output_sha256": report["output_sha256"],
        "output_pkl": str(paths.output_pkl),
        "registry_owner": "scripts/tools/manage_external_data.py",
        "manual_review_required": True,
        "next_action": (
            "After maintainer verifies the trusted generated artifact and reseeds durable "
            "storage, copy expected_tree_sha256 into the socnavbench-s3dis-eth registry entry."
        ),
        "forbidden_actions": {
            "commit_generated_data": False,
            "treat_as_benchmark_evidence": False,
        },
    }


def preflight(map_name: str, *, root: Path | None = None) -> dict[str, Any]:
    """Inspect staged inputs and existing output for one map without building anything.

    The returned report classifies the map into one of:

    - ``blocked_missing_mesh``: the per-map mesh is not staged; generation cannot proceed.
    - ``already_present``: the traversible ``data.pkl`` already exists (idempotent skip).
    - ``ready``: the mesh is staged and no output exists yet, so a build would run.

    This function performs no SocNavBench imports and never writes to disk.
    """
    paths = resolve_paths(map_name, root=root)
    mesh_present = _mesh_is_staged(paths.mesh_dir)
    output_exists = paths.output_pkl.is_file()

    if not mesh_present:
        status = STATUS_BLOCKED_MISSING_MESH
        next_action = (
            f"Stage the curated SocNavBench '{paths.map_name}' mesh at "
            f"{paths.mesh_dir} (see docs/socnav_assets_setup.md), then re-run. "
            f"Set {EXTERNAL_DATA_ROOT_ENV} if the mesh lives in a shared data root."
        )
    elif output_exists:
        status = STATUS_ALREADY_PRESENT
        next_action = f"Traversible already present at {paths.output_pkl}. Pass --force to rebuild."
    else:
        status = STATUS_READY
        next_action = (
            "Inputs are staged. Run without --dry-run (in the SocNavBench environment) "
            "to build the traversible."
        )

    report: dict[str, Any] = {
        "map_name": paths.map_name,
        "asset_id": SOCNAV_ASSET_ID,
        "external_data_root_env": EXTERNAL_DATA_ROOT_ENV,
        "socnav_root": str(paths.socnav_root),
        "mesh_dir": str(paths.mesh_dir),
        "mesh_present": mesh_present,
        "output_pkl": str(paths.output_pkl),
        "output_exists": output_exists,
        "status": status,
        "blocked": status == STATUS_BLOCKED_MISSING_MESH,
        "next_action": next_action,
    }
    if output_exists:
        # Surface the current pin so callers can compare before/after a --force rebuild.
        report["output_sha256"] = sha256_file(paths.output_pkl)
        report.update(output_tree_checksum(paths))
    return report


def _build_socnav_traversible(paths: TraversiblePaths) -> None:  # pragma: no cover
    """Invoke SocNavBench's own renderer to build and pickle the traversible.

    Lazily imports the vendored SocNavBench package (which pulls heavy, environment-only
    dependencies such as ``dotmap``, ``mp_env`` and swiftshader), so this function is never
    imported during ``--dry-run`` or skip-if-absent preflight. It is marked ``no cover``
    because it can only run in the SocNavBench environment with a staged mesh — the same
    reason CI cannot exercise it — and is validated by the maintainer post-merge.

    Mechanism: :meth:`SBPDRenderer.get_config` reads ``self.building.env.resolution`` and
    ``self.building.traversible`` when ``load_meshes`` is set and the pickle is absent, then
    writes ``traversibles/<building_name>/data.pkl``. We drive it with the *occupancy-grid*
    camera modality so the mesh-derived traversible is built without instantiating the
    RGB/depth GL renderer.

    The renderer params object is reconstructed here as a flat ``DotMap`` carrying exactly
    the attributes ``SBPDRenderer.__init__``/``get_config`` read. The vendored tree's
    single renderer-params factory is commented out (``create_base_params`` does not exist),
    so this explicit reconstruction is the defensible invocation; the maintainer confirms it
    on the first real run.
    """
    socnav_pkg = Path(__file__).resolve().parents[2] / "third_party" / "socnavbench"
    if not socnav_pkg.is_dir():
        raise TraversibleGenerationError(f"Vendored SocNavBench package not found at {socnav_pkg}.")
    # SocNavBench modules import each other by bare top-level names (``sbpd``, ``params``,
    # ``mp_env`` ...), so its root must be importable directly.
    sys.path.insert(0, str(socnav_pkg))
    try:
        from dotmap import DotMap  # type: ignore
        from params import central_params as central  # type: ignore
        from sbpd.sbpd_renderer import SBPDRenderer  # type: ignore
    except ImportError as exc:
        raise TraversibleGenerationError(
            "Failed to import SocNavBench. Generation must run in the SocNavBench "
            "environment with its dependencies installed (dotmap, mp_env, swiftshader). "
            f"Underlying error: {exc}"
        ) from exc

    building_params = central.create_building_params()
    camera_params = central.create_camera_params()
    # The occupancy-grid branch requires a square top view; force width == height so the
    # mesh-only path is taken instead of the RGB/depth GL renderer.
    camera_params.modalities = ["occupancy_grid"]
    camera_params.height = camera_params.width

    params = DotMap()
    params.dataset_name = building_params.dataset_name
    params.building_name = paths.map_name  # override to the requested custom map
    params.robot_params = central.create_robot_params()
    params.flip = False
    params.sbpd_data_dir = central.get_sbpd_data_dir()
    params.traversible_dir = central.get_traversible_dir()
    params.camera_params = camera_params
    # Force a fresh mesh-based build: load the mesh, and do NOT load a pre-existing pickle
    # so get_config() regenerates the traversible and pickles it to data.pkl.
    params.load_meshes = True
    params.load_traversible_from_pickle_file = False

    renderer = SBPDRenderer.get_renderer(params)
    renderer.get_config()


def build_traversible(
    map_name: str, *, root: Path | None = None, force: bool = False
) -> dict[str, Any]:
    """Build traversible one map, failing closed on missing inputs.

    Returns report dict including SHA-256 produced ``data.pkl`` and registry-style output
    tree checksum metadata so external-data registry pin can be updated. Raises
    :class:`TraversibleGenerationError` if mesh is not staged or the build does not
    produce expected output.
    """
    paths = resolve_paths(map_name, root=root)
    if not _mesh_is_staged(paths.mesh_dir):
        raise TraversibleGenerationError(
            f"Cannot generate the '{paths.map_name}' traversible: mesh not staged at "
            f"{paths.mesh_dir}. Stage it (docs/socnav_assets_setup.md) or set "
            f"{EXTERNAL_DATA_ROOT_ENV}, then re-run."
        )

    if paths.output_pkl.is_file() and not force:
        return {
            "map_name": paths.map_name,
            "output_pkl": str(paths.output_pkl),
            "status": STATUS_ALREADY_PRESENT,
            "built": False,
            "output_sha256": sha256_file(paths.output_pkl),
            **output_tree_checksum(paths),
        }

    if force and paths.output_pkl.is_file():
        # Remove the stale pickle so SocNavBench regenerates it rather than reloading it.
        paths.output_pkl.unlink()

    paths.traversible_dir.mkdir(parents=True, exist_ok=True)
    _build_socnav_traversible(paths)

    if not paths.output_pkl.is_file():
        raise TraversibleGenerationError(
            f"SocNavBench did not produce the expected traversible at {paths.output_pkl}."
        )
    return {
        "map_name": paths.map_name,
        "output_pkl": str(paths.output_pkl),
        "status": "generated",
        "built": True,
        "output_sha256": sha256_file(paths.output_pkl),
        **output_tree_checksum(paths),
    }


def _emit(report: dict[str, Any], report_json: Path | None) -> None:
    """Write the report JSON to an optional path and print it to stdout."""
    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _write_pin_report(report: dict[str, Any], args: argparse.Namespace) -> None:
    """Write optional registry-pin sidecar for already-present/generated output."""
    if args.pin_report_json is None:
        return
    paths = resolve_paths(args.map, root=args.root)
    pin_report = registry_pin_report(paths, report)
    args.pin_report_json.parent.mkdir(parents=True, exist_ok=True)
    args.pin_report_json.write_text(json.dumps(pin_report, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        default="ETH",
        help="SocNavBench custom-map / building name to generate (default: ETH).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Override the SocNavBench root directory. Defaults to the registry-resolved "
            f"path (honors {EXTERNAL_DATA_ROOT_ENV})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate staged inputs and report the planned action without building.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the traversible data.pkl already exists.",
    )
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument(
        "--pin-report-json",
        type=Path,
        default=None,
        help=(
            "Write a maintainer-review JSON sidecar with expected_tree_sha256 fields for "
            "registry pinning. Requires an existing or newly generated data.pkl."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run preflight or generation for one map and return a process exit code."""
    args = parse_args(argv)
    report = preflight(args.map, root=args.root)

    if args.dry_run:
        report["dry_run"] = True
        if args.pin_report_json is not None:
            try:
                _write_pin_report(report, args)
            except TraversibleGenerationError as exc:
                report.update({"status": "blocked_missing_output", "error": str(exc)})
                _emit(report, args.report_json)
                return EXIT_BLOCKED
        _emit(report, args.report_json)
        return EXIT_BLOCKED if report["blocked"] else 0

    if report["blocked"]:
        _emit(report, args.report_json)
        return EXIT_BLOCKED

    try:
        build_report = build_traversible(args.map, root=args.root, force=args.force)
    except TraversibleGenerationError as exc:
        error_report = dict(report)
        error_report.update({"status": "error", "error": str(exc)})
        _emit(error_report, args.report_json)
        return EXIT_BLOCKED

    # Merge the resolved preflight context with the build outcome for a complete record.
    merged = {**{k: report[k] for k in ("map_name", "socnav_root", "mesh_dir")}, **build_report}
    try:
        _write_pin_report(merged, args)
    except TraversibleGenerationError as exc:
        merged.update({"status": "blocked_missing_output", "error": str(exc)})
        _emit(merged, args.report_json)
        return EXIT_BLOCKED
    _emit(merged, args.report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
