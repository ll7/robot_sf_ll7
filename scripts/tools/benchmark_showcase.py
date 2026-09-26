#!/usr/bin/env python3
"""Build a no-simulation technical showcase from persisted campaign results.

Examples:
  uv run python scripts/tools/benchmark_showcase.py \
    --bundle-url https://github.com/ll7/robot_sf_ll7/releases/download/0.0.2/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz \
    --sha256 64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90 \
    --out-dir output/benchmark_showcase/release-0.0.2

The existing camera-ready analyzer remains the owner of campaign diagnostics. The existing episode
figure bridge is called only for rows with usable recorded ``replay_steps``; its scenario-based
resimulation fallback is deliberately skipped because that would launch a simulation and would not
replay an arbitrary recorded planner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from robot_sf.benchmark.episode_replay_figure import EpisodeRow, build_replay_from_episode_row
from robot_sf.benchmark.full_classic.replay import validate_replay_episode

SCHEMA_VERSION = "benchmark-showcase.v1"
DEFAULT_TOP_K = 3
DEFAULT_RENDER_LIMIT = 5
MAX_ARCHIVE_MEMBERS = 50_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 5 * 1024**3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYZER_SCRIPT = REPO_ROOT / "scripts" / "tools" / "analyze_camera_ready_campaign.py"
FIGURE_SCRIPT = REPO_ROOT / "scripts" / "replay_episode_figure.py"

# Keep components named and separate. In particular, outcome events are not derived from the
# legacy ``metrics.success`` or ``metrics.collisions`` aliases.
METRIC_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("clearing_distance_min", "minimum_clearance_m", "m"),
    ("near_misses", "near_misses", "count"),
    ("force_exceed_events", "force_exceed_events", "count"),
    ("comfort_exposure", "comfort_exposure", "reported units"),
    ("time_to_goal_norm", "time_to_goal_norm", "normalized time"),
    ("path_efficiency", "path_efficiency", "reported units"),
    ("snqi", "snqi", "reported units"),
    ("success", "success_metric", "reported units"),
    ("collisions", "collisions_metric", "reported units"),
    ("total_collision_count", "total_collision_count", "count"),
)

CASE_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "name": "collision_event",
        "description": "Canonical collision event",
        "event": "collision_event",
    },
    {"name": "timeout_event", "description": "Canonical timeout event", "event": "timeout_event"},
    {
        "name": "route_not_complete",
        "description": "Route completion is false",
        "event": "route_not_complete",
    },
    {
        "name": "minimum_clearance",
        "description": "Lowest recorded minimum-clearance values",
        "metric": "clearing_distance_min",
        "direction": "ascending",
    },
    {
        "name": "near_miss_extreme",
        "description": "Highest recorded near-miss counts",
        "metric": "near_misses",
        "direction": "descending",
    },
    {
        "name": "force_exceed_extreme",
        "description": "Highest recorded force-exceed event counts",
        "metric": "force_exceed_events",
        "direction": "descending",
    },
    {
        "name": "comfort_exposure_extreme",
        "description": "Highest recorded comfort-exposure values",
        "metric": "comfort_exposure",
        "direction": "descending",
    },
    {
        "name": "slow_normalized_time",
        "description": "Highest recorded normalized time-to-goal values",
        "metric": "time_to_goal_norm",
        "direction": "descending",
    },
    {
        "name": "low_path_efficiency",
        "description": "Lowest recorded path-efficiency values",
        "metric": "path_efficiency",
        "direction": "ascending",
    },
)


class ShowcaseError(RuntimeError):
    """A user-correctable issue in source, provenance, or command arguments."""


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest without loading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    """Load one JSON object, failing closed on malformed or wrong-shaped input."""
    if not path.is_file():
        if required:
            raise ShowcaseError(f"required JSON file is missing: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShowcaseError(f"could not read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ShowcaseError(f"JSON root must be an object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV artifact as rows while preserving its displayed values."""
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ShowcaseError(f"could not read CSV {path}: {exc}") from exc


def _extract_member(
    bundle: tarfile.TarFile, member: tarfile.TarInfo, root: Path, remaining_bytes: int
) -> int:
    """Validate and extract one archive member below ``root``; return its byte size."""
    member_path = PurePosixPath(member.name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ShowcaseError(f"unsafe archive member path: {member.name!r}")
    if not (member.isdir() or member.isfile()):
        raise ShowcaseError(f"unsupported archive member type: {member.name!r}")
    size = max(member.size, 0)
    if size > remaining_bytes:
        raise ShowcaseError("bundle exceeds the configured uncompressed-size limit")
    target = (root / Path(*member_path.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ShowcaseError(f"archive member escapes extraction root: {member.name!r}") from exc
    if member.isdir():
        target.mkdir(parents=True, exist_ok=True)
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    stream = bundle.extractfile(member)
    if stream is None:
        raise ShowcaseError(f"archive file has no data stream: {member.name!r}")
    with stream, target.open("xb") as output:
        shutil.copyfileobj(stream, output)
    return size


def _safe_extract_archive(archive: Path, destination: Path) -> None:
    """Extract regular files/directories only and reject traversal or link entries."""
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ShowcaseError(f"refusing to extract over nonempty directory: {destination}")
    root = destination.resolve()
    seen: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            members = bundle.getmembers()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ShowcaseError(f"bundle has too many archive members: {len(members)}")
            total_size = 0
            for member in members:
                if member.name in seen:
                    raise ShowcaseError(f"duplicate archive member path: {member.name!r}")
                seen.add(member.name)
                total_size += _extract_member(
                    bundle, member, root, MAX_ARCHIVE_UNCOMPRESSED_BYTES - total_size
                )
    except (OSError, tarfile.TarError) as exc:
        raise ShowcaseError(f"could not safely extract {archive}: {exc}") from exc


def _verify_embedded_checksums(bundle_root: Path) -> dict[str, Any]:
    """Verify release payload checksums when the archive supplies a checksum manifest."""
    manifest = bundle_root / "checksums.sha256"
    payload_root = bundle_root / "payload"
    if not manifest.is_file():
        return {"status": "not_provided", "checked_files": 0, "failures": []}
    failures: list[str] = []
    checked = 0
    checked_paths: set[str] = set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return {
            "status": "failed",
            "checked_files": 0,
            "failures": [f"could not read checksum manifest: {exc}"],
        }
    for line_number, raw_line in enumerate(lines, 1):
        _relative, failure, file_checked = _verify_checksum_record(
            payload_root, raw_line, line_number, checked_paths
        )
        if failure:
            failures.append(failure)
        checked += int(file_checked)
    for missing_from_manifest in _unlisted_payload_files(payload_root, checked_paths):
        failures.append(f"{missing_from_manifest}: file is not covered by checksum manifest")
    return {
        "status": "passed" if checked and not failures else "failed",
        "checked_files": checked,
        "failures": failures,
    }


def _verify_checksum_record(
    payload_root: Path, raw_line: str, line_number: int, checked_paths: set[str]
) -> tuple[str | None, str | None, bool]:
    """Validate and hash one sha256sum-format record."""
    line = raw_line.strip()
    if not line:
        return None, None, False
    parts = line.split(maxsplit=1)
    if len(parts) != 2 or not SHA256_RE.fullmatch(parts[0]):
        return None, f"line {line_number}: malformed checksum record", False
    relative = parts[1].lstrip(" *")
    rel_path = PurePosixPath(relative)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        return None, f"line {line_number}: unsafe checksum path", False
    normalized_path = rel_path.as_posix()
    if normalized_path in checked_paths:
        return None, f"line {line_number}: duplicate checksum path {normalized_path}", False
    checked_paths.add(normalized_path)
    candidate = (payload_root / Path(*rel_path.parts)).resolve()
    try:
        candidate.relative_to(payload_root.resolve())
    except ValueError:
        return normalized_path, f"line {line_number}: checksum path escapes payload", False
    if not candidate.is_file():
        return normalized_path, f"{relative}: file is missing", False
    if sha256_file(candidate) != parts[0]:
        return normalized_path, f"{relative}: SHA-256 mismatch", True
    return normalized_path, None, True


def _unlisted_payload_files(payload_root: Path, checked_paths: set[str]) -> list[str]:
    """Find payload files that are omitted from the supplied checksum manifest."""
    if not payload_root.is_dir():
        return []
    actual_paths = {
        path.relative_to(payload_root).as_posix()
        for path in payload_root.rglob("*")
        if path.is_file()
    }
    return sorted(actual_paths - checked_paths)


def _download_bundle(url: str, expected_sha256: str, cache_dir: Path) -> Path:
    """Download or reuse a release archive only after validating its required digest."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ShowcaseError("bundle URL must use HTTP or HTTPS")
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"bundle-{expected_sha256}.tar.gz"
    if destination.exists():
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise ShowcaseError(
                f"cached bundle digest mismatch at {destination}: expected {expected_sha256}, got {actual}"
            )
        return destination

    request = urllib.request.Request(url, headers={"User-Agent": "robot-sf-benchmark-showcase/1"})
    temp_name: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f"bundle-{expected_sha256}-",
                suffix=".partial",
                dir=cache_dir,
                delete=False,
            ) as output:
                temp_name = output.name
                shutil.copyfileobj(response, output)
        temporary = Path(temp_name)
        actual = sha256_file(temporary)
        if actual != expected_sha256:
            raise ShowcaseError(
                f"downloaded bundle digest mismatch: expected {expected_sha256}, got {actual}; retained {temporary}"
            )
        os.replace(temporary, destination)
        return destination
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise ShowcaseError(f"could not download bundle {url}: {exc}") from exc


def _locate_campaign_root(search_root: Path) -> Path:
    """Find the unique campaign root containing canonical summary and run paths."""
    candidates = [
        p.parent.parent
        for p in search_root.rglob("campaign_summary.json")
        if p.parent.name == "reports" and (p.parent.parent / "runs").is_dir()
    ]
    unique = sorted({p.resolve() for p in candidates})
    if len(unique) != 1:
        raise ShowcaseError(
            f"expected one camera-ready campaign under {search_root}, found {len(unique)}"
        )
    return unique[0]


def _find_bundle_root(campaign_root: Path) -> Path | None:
    """Find an enclosing bundle root whose checksum manifest covers ``payload``."""
    current = campaign_root.resolve()
    for parent in (current, *current.parents):
        if (parent / "checksums.sha256").is_file() and (parent / "payload").is_dir():
            try:
                current.relative_to((parent / "payload").resolve())
            except ValueError:
                continue
            return parent
    return None


def _resolve_episode_path(campaign_root: Path, run_entry: dict[str, Any]) -> Path:
    """Resolve a persisted run path inside the supplied campaign root only."""
    raw_path = run_entry.get("episodes_path")
    if isinstance(raw_path, str) and raw_path.strip():
        raw = Path(raw_path)
        direct = raw if raw.is_absolute() else campaign_root / raw
        if direct.is_file():
            resolved = direct.resolve()
            try:
                resolved.relative_to(campaign_root.resolve())
            except ValueError as exc:
                raise ShowcaseError(f"episode path escapes campaign root: {raw_path}") from exc
            return resolved
        parts = PurePosixPath(raw_path).parts
        if "runs" in parts:
            index = len(parts) - 1 - tuple(reversed(parts)).index("runs")
            suffix = parts[index:]
            candidate = campaign_root.joinpath(*suffix)
            if candidate.is_file():
                resolved = candidate.resolve()
                try:
                    resolved.relative_to(campaign_root.resolve())
                except ValueError as exc:
                    raise ShowcaseError(f"episode path escapes campaign root: {raw_path}") from exc
                return resolved
    planner = run_entry.get("planner") or {}
    key = planner.get("key") if isinstance(planner, dict) else None
    kinematics = planner.get("kinematics") if isinstance(planner, dict) else None
    if isinstance(key, str) and key and isinstance(kinematics, str) and kinematics:
        candidate = campaign_root / "runs" / f"{key}__{kinematics}" / "episodes.jsonl"
        if candidate.is_file():
            return candidate.resolve()
    return campaign_root / "__missing_episode_file__"


def _run_identity(run_entry: dict[str, Any]) -> tuple[str | None, str | None]:
    """Read planner and kinematics identity without trusting optional nested shapes."""
    planner = run_entry.get("planner") if isinstance(run_entry.get("planner"), dict) else {}
    summary = run_entry.get("summary") if isinstance(run_entry.get("summary"), dict) else {}
    key = planner.get("key")
    kind = planner.get("kinematics") or summary.get("kinematics")
    return (
        key if isinstance(key, str) and key else None,
        kind if isinstance(kind, str) and kind else None,
    )


def _finite_float(value: Any) -> float | None:
    """Return a finite numeric value, excluding booleans and nonnumeric strings."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _strict_bool(value: Any) -> bool | None:
    """Read canonical boolean values without treating missing or truthy strings as false."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    return None


def _scenario_inventory(  # noqa: C901 - preserves fail-closed inventory and seed provenance checks
    campaign_root: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """Read the frozen preflight scenario inventory and preserve its source strength."""
    preview_path = campaign_root / "preflight" / "preview_scenarios.json"
    preview = _read_json(preview_path, required=False)
    scenarios = preview.get("scenarios") if isinstance(preview, dict) else None
    if isinstance(scenarios, list) and preview.get("truncated") is not True:
        seed_policy = (
            manifest.get("seed_policy") if isinstance(manifest.get("seed_policy"), dict) else {}
        )
        global_seeds = _normalized_seeds(seed_policy.get("resolved_seeds"))
        normalized: list[dict[str, Any]] = []
        for item in scenarios:
            if not isinstance(item, dict):
                continue
            scenario_id = item.get("scenario_id") or item.get("id") or item.get("name")
            if not isinstance(scenario_id, str) or not scenario_id:
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            family = (
                metadata.get("archetype")
                or metadata.get("scenario_family")
                or item.get("scenario_family")
            )
            seeds = _normalized_seeds(item.get("seeds")) or global_seeds
            normalized.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_family": str(family or "unknown"),
                    "seeds": seeds,
                }
            )
        if normalized:
            declared_count = preview.get("scenario_count")
            if isinstance(declared_count, int) and declared_count != len(normalized):
                raise ShowcaseError(
                    f"preflight inventory declares {declared_count} scenarios but exposes {len(normalized)}"
                )
            if len({item["scenario_id"] for item in normalized}) != len(normalized):
                raise ShowcaseError("preflight scenario inventory contains duplicate scenario IDs")
            if any(not item["seeds"] for item in normalized):
                raise ShowcaseError(
                    "preflight scenario inventory is missing seeds for one or more scenarios"
                )
            return sorted(
                normalized, key=lambda item: item["scenario_id"]
            ), "preflight_preview_scenarios"

    rows = _read_csv(campaign_root / "reports" / "scenario_breakdown.csv")
    mapping: dict[str, str] = {}
    for row in rows:
        scenario_id = str(row.get("scenario_id") or "").strip()
        family = str(row.get("scenario_family") or "unknown").strip()
        if not scenario_id:
            continue
        previous = mapping.setdefault(scenario_id, family)
        if previous != family:
            raise ShowcaseError(f"scenario family conflicts across breakdown rows: {scenario_id}")
    if mapping:
        seed_policy = manifest.get("seed_policy")
        seeds = seed_policy.get("resolved_seeds") if isinstance(seed_policy, dict) else []
        normalized_seeds = _normalized_seeds(seeds)
        if not normalized_seeds:
            raise ShowcaseError("scenario breakdown is available but manifest seeds are missing")
        return [
            {
                "scenario_id": scenario_id,
                "scenario_family": mapping[scenario_id],
                "seeds": normalized_seeds,
            }
            for scenario_id in sorted(mapping)
        ], "scenario_breakdown_fallback"
    return [], "unavailable"


def _normalized_seeds(values: Any) -> list[int]:
    """Normalize a seed list without accepting booleans or non-integral values."""
    if not isinstance(values, list):
        return []
    seeds = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            seeds.add(int(numeric))
    return sorted(seeds)


def _expected_runs(
    manifest: dict[str, Any], run_entries: list[Any]
) -> tuple[list[tuple[str, str]], str]:
    """Enumerate expected planner/kinematics arms from the frozen manifest."""
    planners = manifest.get("planners")
    kinematics_values = manifest.get("kinematics_matrix")
    if (
        isinstance(planners, list)
        and isinstance(kinematics_values, list)
        and planners
        and kinematics_values
    ):
        keys = sorted(
            {
                str(planner.get("key"))
                for planner in planners
                if isinstance(planner, dict)
                and planner.get("enabled", True) is not False
                and isinstance(planner.get("key"), str)
                and planner.get("key")
            }
        )
        kinematics = sorted(
            {str(value) for value in kinematics_values if isinstance(value, str) and value}
        )
        if keys and kinematics:
            return [(key, kind) for key in keys for kind in kinematics], "campaign_manifest"
    inferred: set[tuple[str, str]] = set()
    for entry in run_entries:
        if not isinstance(entry, dict):
            continue
        key, kind = _run_identity(entry)
        if key and kind:
            inferred.add((key, kind))
    return sorted(inferred), "campaign_summary_only"


def _metric_value(row: dict[str, Any], key: str) -> float | None:
    """Read one named canonical episode metric without substituting another metric."""
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    value = metrics.get(key)
    if key == "success" and isinstance(value, bool):
        return float(value)
    return _finite_float(value)


def _read_episode_rows(  # noqa: C901 - keeps malformed source rows and exact line provenance together
    campaign_root: Path,
    expected_runs: list[tuple[str, str]],
    run_entries: list[Any],
    family_by_scenario: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Path]]:
    """Read JSONL rows, keeping malformed lines and source references visible."""
    run_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in run_entries:
        if not isinstance(entry, dict):
            continue
        key, kind = _run_identity(entry)
        if key and kind:
            identity = (key, kind)
            if identity in run_by_key:
                raise ShowcaseError(f"campaign summary has duplicate run identity: {identity}")
            run_by_key[identity] = entry

    rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    episode_files: dict[str, Path] = {}
    for planner_key, kinematics in expected_runs:
        run_entry = run_by_key.get((planner_key, kinematics), {})
        file_path = _resolve_episode_path(campaign_root, run_entry)
        run_id = f"{planner_key}__{kinematics}"
        if not file_path.is_file():
            episode_files[run_id] = file_path
            continue
        resolved = file_path.resolve()
        try:
            resolved.relative_to(campaign_root.resolve())
        except ValueError as exc:
            raise ShowcaseError(f"episode file escapes campaign root: {resolved}") from exc
        episode_files[run_id] = resolved
        file_sha256 = sha256_file(resolved)
        run_status_details = _status_details(run_entry, resolved)
        run_status_details = _status_details(run_entry, resolved)
        with resolved.open("rb") as handle:
            for line_number, raw_bytes in enumerate(handle, 1):
                if not raw_bytes.strip():
                    continue
                try:
                    raw = json.loads(raw_bytes)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    malformed.append(
                        {
                            "planner_key": planner_key,
                            "kinematics": kinematics,
                            "path": resolved.relative_to(campaign_root.resolve()).as_posix(),
                            "line_number": line_number,
                            "error": str(exc),
                            "source_line_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                        }
                    )
                    continue
                if not isinstance(raw, dict):
                    malformed.append(
                        {
                            "planner_key": planner_key,
                            "kinematics": kinematics,
                            "path": resolved.relative_to(campaign_root.resolve()).as_posix(),
                            "line_number": line_number,
                            "error": "episode JSONL row is not an object",
                            "source_line_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                        }
                    )
                    continue
                outcome = raw.get("outcome") if isinstance(raw.get("outcome"), dict) else {}
                events = {
                    "route_complete": _strict_bool(
                        outcome.get("route_complete", outcome.get("success"))
                    ),
                    "collision_event": _strict_bool(
                        outcome.get("collision_event", outcome.get("collision"))
                    ),
                    "timeout_event": _strict_bool(
                        outcome.get("timeout_event", outcome.get("timeout"))
                    ),
                }
                scenario_id = str(raw.get("scenario_id") or "unknown")
                seed_value = raw.get("seed")
                normalized_seed = _normalized_seeds([seed_value])
                seed = normalized_seed[0] if normalized_seed else None
                episode_id = str(raw.get("episode_id") or "")
                case_key = "\0".join(
                    (planner_key, kinematics, scenario_id, str(seed), episode_id, str(line_number))
                )
                case_id = "case-" + hashlib.sha256(case_key.encode("utf-8")).hexdigest()[:16]
                metric_values = {
                    output_key: _metric_value(raw, source_key)
                    for source_key, output_key, _unit in METRIC_FIELDS
                }
                rows.append(
                    {
                        "case_id": case_id,
                        "planner_key": planner_key,
                        "kinematics": kinematics,
                        "scenario_id": scenario_id,
                        "scenario_family": family_by_scenario.get(scenario_id, "unknown"),
                        "seed": seed,
                        "episode_id": episode_id or None,
                        "outcome": events,
                        "termination_reason": str(raw.get("termination_reason") or "unknown"),
                        "episode_status": str(raw.get("status") or "unknown"),
                        "metrics": metric_values,
                        "benchmark_eligible": run_status_details["benchmark_eligible"],
                        "benchmark_exclusion_reasons": run_status_details[
                            "benchmark_exclusion_reasons"
                        ],
                        "replay_steps_available": bool(
                            isinstance(raw.get("replay_steps"), list) and raw.get("replay_steps")
                        ),
                        "raw": raw,
                        "source": {
                            "episode_file": resolved.relative_to(
                                campaign_root.resolve()
                            ).as_posix(),
                            "episode_file_sha256": file_sha256,
                            "line_number": line_number,
                            "record_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                        },
                    }
                )
    return rows, malformed, episode_files


def _status_details(run_entry: dict[str, Any] | None, episode_file: Path) -> dict[str, Any]:
    """Summarize operational run state separately from episode outcome status."""
    entry = run_entry if isinstance(run_entry, dict) else {}
    summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
    preflight = summary.get("preflight") if isinstance(summary.get("preflight"), dict) else {}
    availability = (
        summary.get("benchmark_availability")
        if isinstance(summary.get("benchmark_availability"), dict)
        else {}
    )
    run_status = str(entry.get("status") or summary.get("status") or "missing_run")
    preflight_status = str(preflight.get("status") or "unknown")
    availability_status = str(
        availability.get("availability_status") or availability.get("status") or "unknown"
    )
    normalized = {run_status.lower(), preflight_status.lower(), availability_status.lower()}
    failed_jobs = summary.get("failed_jobs")
    failed_job_count = (
        int(failed_jobs) if isinstance(failed_jobs, int | str) and str(failed_jobs).isdigit() else 0
    )
    fallback_or_degraded = bool(normalized & {"fallback", "degraded"})
    unavailable = bool(normalized & {"unavailable", "not_available", "missing", "blocked"})
    failed = bool(normalized & {"failed", "failure", "error"}) or failed_job_count > 0
    successful_run_states = {"ok", "success", "passed", "complete", "completed"}
    successful_availability_states = successful_run_states | {"available"}
    exclusion_reasons = []
    if run_status.lower() not in successful_run_states:
        exclusion_reasons.append(f"run_status={run_status}")
    if preflight_status.lower() not in successful_run_states:
        exclusion_reasons.append(f"preflight_status={preflight_status}")
    if availability_status.lower() not in successful_availability_states:
        exclusion_reasons.append(f"availability_status={availability_status}")
    if fallback_or_degraded:
        exclusion_reasons.append("fallback_or_degraded")
    if unavailable:
        exclusion_reasons.append("unavailable")
    if failed:
        exclusion_reasons.append("failed")
    if not episode_file.is_file():
        exclusion_reasons.append("episode_file_missing")
    return {
        "run_status": run_status,
        "preflight_status": preflight_status,
        "availability_status": availability_status,
        "episode_file_present": episode_file.is_file(),
        "reported_written": summary.get("written"),
        "reported_total_jobs": summary.get("total_jobs"),
        "failed_jobs": failed_jobs,
        "fallback_or_degraded": fallback_or_degraded,
        "unavailable": unavailable,
        "failed": failed,
        "benchmark_eligible": not exclusion_reasons,
        "benchmark_exclusion_reasons": exclusion_reasons,
    }


def _build_accounting(
    scenarios: list[dict[str, Any]],
    scenario_source: str,
    expected_runs: list[tuple[str, str]],
    run_source: str,
    run_entries: list[Any],
    rows: list[dict[str, Any]],
    malformed: list[dict[str, Any]],
    episode_files: dict[str, Path],
) -> dict[str, Any]:
    """Compute expected-versus-present planner/scenario/seed identity coverage."""
    expected: set[tuple[str, str, str, int]] = set()
    for planner, kind in expected_runs:
        for scenario in scenarios:
            for seed in scenario["seeds"]:
                expected.add((planner, kind, scenario["scenario_id"], seed))
    observed_all: list[tuple[str, str, str, int | None]] = [
        (row["planner_key"], row["kinematics"], row["scenario_id"], row["seed"]) for row in rows
    ]
    observed_valid = {identity for identity in observed_all if identity[3] is not None}
    duplicates = sorted(
        (key for key, count in Counter(observed_all).items() if count > 1),
        key=lambda item: tuple(str(value) for value in item),
    )
    missing = sorted(expected - observed_valid, key=lambda item: item)
    unexpected = sorted(observed_valid - expected, key=lambda item: item)

    run_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in run_entries:
        if not isinstance(entry, dict):
            continue
        key, kind = _run_identity(entry)
        if key and kind:
            run_by_key[(key, kind)] = entry
    row_counts = Counter((row["planner_key"], row["kinematics"]) for row in rows)
    episode_status_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        episode_status_counts[(row["planner_key"], row["kinematics"])][row["episode_status"]] += 1
    expected_counts = Counter((planner, kind) for planner, kind, _scenario, _seed in expected)
    per_run = []
    for planner, kind in expected_runs:
        run_id = f"{planner}__{kind}"
        file_path = episode_files.get(run_id, Path("__missing_episode_file__"))
        details = _status_details(run_by_key.get((planner, kind)), file_path)
        details.update(
            {
                "planner_key": planner,
                "kinematics": kind,
                "expected_episodes": expected_counts[(planner, kind)],
                "present_episode_rows": row_counts[(planner, kind)],
                "present_unique_identities": len(
                    {
                        identity
                        for identity in observed_valid
                        if identity[0] == planner and identity[1] == kind
                    }
                ),
                "episode_outcome_status_counts": dict(
                    sorted(episode_status_counts[(planner, kind)].items())
                ),
                "benchmark_eligible_episode_rows": sum(
                    1
                    for row in rows
                    if row["planner_key"] == planner
                    and row["kinematics"] == kind
                    and row["benchmark_eligible"]
                ),
                "excluded_episode_rows": sum(
                    1
                    for row in rows
                    if row["planner_key"] == planner
                    and row["kinematics"] == kind
                    and not row["benchmark_eligible"]
                ),
                "missing_episodes": sum(
                    1 for identity in missing if identity[:2] == (planner, kind)
                ),
            }
        )
        per_run.append(details)

    return {
        "scenario_inventory_source": scenario_source,
        "expected_run_source": run_source,
        "denominator_status": (
            "complete_preflight_inventory"
            if scenario_source == "preflight_preview_scenarios"
            else "observed_scenario_breakdown_only"
        ),
        "expected_run_count": len(expected_runs),
        "expected_identity_count": len(expected),
        "present_episode_rows": len(rows),
        "present_unique_identities": len(observed_valid),
        "missing_identity_count": len(missing),
        "missing_identities": [list(identity) for identity in missing],
        "unexpected_identity_count": len(unexpected),
        "unexpected_identities": [list(identity) for identity in unexpected],
        "duplicate_identity_count": len(duplicates),
        "duplicate_identities": [list(identity) for identity in duplicates],
        "malformed_line_count": len(malformed),
        "malformed_lines": malformed,
        "run_statuses": per_run,
    }


def _metric_summary(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    """Summarize one named metric with its own nonmissing denominator."""
    values = [row["metrics"].get(metric_name) for row in rows]
    present = [float(value) for value in values if value is not None]
    return {
        "mean": (sum(present) / len(present)) if present else None,
        "present_count": len(present),
        "missing_count": len(values) - len(present),
    }


def _build_matrix(
    scenarios: list[dict[str, Any]],
    expected_runs: list[tuple[str, str]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build explicit planner/family outcome summaries and separate named metric means."""
    scenarios_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        scenarios_by_family[scenario["scenario_family"]].append(scenario)
    rows_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_key[(row["planner_key"], row["kinematics"], row["scenario_family"])].append(row)

    result = []
    family_names = sorted(set(scenarios_by_family) | {row["scenario_family"] for row in rows})
    for planner, kind in expected_runs:
        for family in family_names:
            family_scenarios = scenarios_by_family.get(family, [])
            expected_count = sum(len(scenario["seeds"]) for scenario in family_scenarios)
            group_rows = sorted(
                rows_by_key.get((planner, kind, family), []),
                key=lambda row: (row["scenario_id"], row["seed"] or -1, row["episode_id"] or ""),
            )
            eligible_rows = [row for row in group_rows if row["benchmark_eligible"]]
            events: dict[str, Any] = {}
            for event in ("route_complete", "collision_event", "timeout_event"):
                known = [
                    row["outcome"][event]
                    for row in eligible_rows
                    if row["outcome"][event] is not None
                ]
                events[event] = {
                    "count": sum(1 for value in known if value),
                    "denominator": len(known),
                    "rate": (sum(1 for value in known if value) / len(known)) if known else None,
                    "unknown_count": len(eligible_rows) - len(known),
                }
            metrics = {
                output_name: _metric_summary(eligible_rows, output_name)
                for _source_name, output_name, _unit in METRIC_FIELDS
            }
            result.append(
                {
                    "planner_key": planner,
                    "kinematics": kind,
                    "scenario_family": family,
                    "scenario_count": len(family_scenarios),
                    "expected_episodes": expected_count,
                    "present_episodes": len(group_rows),
                    "benchmark_eligible_episodes": len(eligible_rows),
                    "excluded_episodes": len(group_rows) - len(eligible_rows),
                    "missing_episodes": max(0, expected_count - len(group_rows)),
                    "outcomes": events,
                    "metric_summaries": metrics,
                }
            )
    return result


def _stable_identity(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    """Return deterministic tie-break identity for a case row."""
    return (
        str(row["planner_key"]),
        str(row["scenario_family"]),
        str(row["scenario_id"]),
        int(row["seed"] if row["seed"] is not None else -1),
        str(row["episode_id"] or ""),
    )


def _select_diverse(  # noqa: C901 - explicit deterministic ordering is easier to audit in one place
    candidates: list[dict[str, Any]], group: dict[str, Any], top_k: int
) -> list[dict[str, Any]]:
    """Select deterministic top cases while round-robining planners, then families."""
    metric = group.get("metric")
    direction = group.get("direction")

    def ordering(row: dict[str, Any]) -> tuple[Any, ...]:
        value = (
            row["metrics"].get(
                next((output for source, output, _unit in METRIC_FIELDS if source == metric), "")
            )
            if metric
            else None
        )
        severity = (
            value
            if value is not None
            else (float("inf") if direction == "ascending" else float("-inf"))
        )
        if direction == "ascending":
            return (severity, *_stable_identity(row))
        if direction == "descending":
            return (-severity, *_stable_identity(row))
        return _stable_identity(row)

    buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        buckets[row["planner_key"]][row["scenario_family"]].append(row)
    for family_buckets in buckets.values():
        for bucket_rows in family_buckets.values():
            bucket_rows.sort(key=ordering)
    planner_keys = sorted(buckets)
    family_positions = dict.fromkeys(planner_keys, 0)
    selected: list[dict[str, Any]] = []
    while len(selected) < top_k:
        advanced = False
        for planner_key in planner_keys:
            families = sorted(buckets[planner_key])
            if not families:
                continue
            position = family_positions[planner_key] % len(families)
            for offset in range(len(families)):
                family_position = (position + offset) % len(families)
                family = families[family_position]
                if not buckets[planner_key][family]:
                    continue
                selected.append(buckets[planner_key][family].pop(0))
                family_positions[planner_key] = (family_position + 1) % len(families)
                advanced = True
                break
            if len(selected) == top_k:
                break
        if not advanced:
            break
    return selected


def _build_case_groups(
    rows: list[dict[str, Any]], top_k: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Group rows by explicit events/features and select representatives without a scalar score."""
    groups = []
    case_index: dict[str, dict[str, Any]] = {}
    for spec in CASE_GROUPS:
        if "event" in spec:
            event = spec["event"]
            if event == "route_not_complete":
                candidates = [row for row in rows if row["outcome"]["route_complete"] is False]
            else:
                candidates = [row for row in rows if row["outcome"][event] is True]
        else:
            source_metric = spec["metric"]
            output_metric = next(
                output for source, output, _unit in METRIC_FIELDS if source == source_metric
            )
            candidates = [row for row in rows if row["metrics"].get(output_metric) is not None]
        selected = _select_diverse(candidates, spec, top_k)
        case_ids = []
        for row in selected:
            case_id = row["case_id"]
            case_ids.append(case_id)
            case_record = case_index.setdefault(
                case_id,
                {
                    "case_id": case_id,
                    "planner_key": row["planner_key"],
                    "kinematics": row["kinematics"],
                    "scenario_id": row["scenario_id"],
                    "scenario_family": row["scenario_family"],
                    "seed": row["seed"],
                    "episode_id": row["episode_id"],
                    "termination_reason": row["termination_reason"],
                    "episode_status": row["episode_status"],
                    "outcome": row["outcome"],
                    "metrics": row["metrics"],
                    "benchmark_eligible": row["benchmark_eligible"],
                    "benchmark_exclusion_reasons": row["benchmark_exclusion_reasons"],
                    "source": row["source"],
                    "replay": {
                        "status": "not_attempted",
                        "artifact_status": "not_generated",
                        "reason": "rendering has not been attempted",
                        "renderer_called": False,
                        "artifacts": [],
                    },
                    "_raw": row["raw"],
                },
            )
            case_record.setdefault("selected_groups", []).append(spec["name"])
        groups.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "feature": spec.get("event", spec.get("metric")),
                "direction": spec.get("direction", "event membership"),
                "candidate_count": len(candidates),
                "selected_case_ids": case_ids,
            }
        )
    for record in case_index.values():
        record.pop("_raw", None)
        record["selected_groups"] = sorted(set(record["selected_groups"]))
    return groups, case_index


def _render_direct_replay(
    case: dict[str, Any],
    row: dict[str, Any],
    campaign_root: Path,
    output_dir: Path,
    render_limit_reached: bool,
) -> dict[str, Any]:
    """Render recorded replay steps only; never invoke the renderer's simulation fallback."""
    raw = row["raw"]
    replay = raw.get("replay_steps")
    if not isinstance(replay, list) or not replay:
        return {
            "status": "unavailable",
            "artifact_status": "not_generated",
            "reason": "source row has no recorded replay_steps; renderer was skipped to avoid resimulation",
            "renderer_called": False,
            "artifacts": [],
        }
    validation_error = _recorded_replay_validation_error(raw)
    if validation_error:
        return {
            "status": "unavailable",
            "artifact_status": "not_generated",
            "reason": validation_error,
            "renderer_called": False,
            "artifacts": [],
        }
    if render_limit_reached:
        return {
            "status": "not_attempted",
            "artifact_status": "not_generated",
            "reason": "configured render limit reached",
            "renderer_called": False,
            "artifacts": [],
        }

    episode_file, source_record_error = _verified_episode_source(campaign_root, case, row)
    if source_record_error:
        return {
            "status": "mismatch",
            "artifact_status": "not_generated",
            "reason": source_record_error,
            "renderer_called": False,
            "artifacts": [],
        }
    assert episode_file is not None

    case_output = output_dir / "renders" / case["case_id"]
    command = [
        sys.executable,
        str(FIGURE_SCRIPT),
        "--episodes",
        str(episode_file),
        "--episode-id",
        str(case["episode_id"]),
        "--outputs",
        "still,filmstrip,trajectory",
        "--out-dir",
        str(case_output),
        "--campaign-root",
        str(campaign_root),
    ]
    invocation_failure = _invoke_renderer(command)
    if invocation_failure:
        return invocation_failure
    sidecar_path = case_output / "replay_provenance.json"
    payload, sidecar_error = _read_renderer_sidecar(sidecar_path)
    if payload is None:
        return {
            "status": "unavailable",
            "artifact_status": "not_generated",
            "reason": sidecar_error,
            "renderer_called": True,
            "command": command,
            "artifacts": [],
        }
    if (
        payload.get("episode_id") != case["episode_id"]
        or payload.get("scenario_id") != case["scenario_id"]
        or payload.get("seed") != case["seed"]
        or payload.get("source_episodes_jsonl_sha256") != case["source"]["episode_file_sha256"]
        or payload.get("resimulated") is not False
    ):
        return {
            "status": "mismatch",
            "artifact_status": "not_generated",
            "reason": "renderer provenance does not match the selected persisted row",
            "renderer_called": True,
            "command": command,
            "provenance_sidecar": str(sidecar_path.relative_to(output_dir)),
            "artifacts": [],
        }
    determinism = str(payload.get("determinism_check_status") or "unknown")
    artifact_paths, artifact_failures = _read_renderer_artifacts(payload, output_dir)
    if artifact_failures:
        return {
            "status": "unavailable",
            "artifact_status": "mismatch",
            "reason": "; ".join(artifact_failures),
            "renderer_called": True,
            "command": command,
            "provenance_sidecar": str(sidecar_path.relative_to(output_dir)),
            "artifacts": artifact_paths,
        }
    status = "verified" if determinism == "pass" else "unavailable"
    return {
        "status": status,
        "artifact_status": "generated",
        "reason": None
        if status == "verified"
        else f"renderer determinism status was {determinism}",
        "renderer_called": True,
        "determinism_check_status": determinism,
        "command": command,
        "provenance_sidecar": str(sidecar_path.relative_to(output_dir)),
        "provenance_sidecar_sha256": sha256_file(sidecar_path),
        "artifacts": artifact_paths,
    }


def _read_episode_source_binding(
    episode_file: Path, line_number: int, episode_id: str
) -> tuple[dict[str, Any] | None, str | None, int]:
    """Return the exact source row, its line digest, and renderer-ID match count."""
    selected_raw: dict[str, Any] | None = None
    selected_digest: str | None = None
    matching_episode_ids = 0
    with episode_file.open("rb") as handle:
        for current_line, raw_bytes in enumerate(handle, 1):
            if not raw_bytes.strip():
                continue
            try:
                parsed = json.loads(raw_bytes)
            except (UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(parsed, dict):
                continue
            matching_episode_ids += parsed.get("episode_id") == episode_id
            if current_line == line_number:
                selected_raw = parsed
                selected_digest = hashlib.sha256(raw_bytes).hexdigest()
    return selected_raw, selected_digest, matching_episode_ids


def _verified_episode_source(
    campaign_root: Path, case: dict[str, Any], row: dict[str, Any]
) -> tuple[Path | None, str | None]:
    """Resolve the source file and verify its selected row before renderer invocation."""
    episode_file, path_error = _resolve_episode_file(campaign_root, case)
    if path_error:
        return None, path_error
    assert episode_file is not None
    record_error = _selected_episode_record_error(episode_file, case, row)
    if record_error:
        return None, record_error
    return episode_file, None


def _resolve_episode_file(
    campaign_root: Path, case: dict[str, Any]
) -> tuple[Path | None, str | None]:
    """Resolve the selected episode path without allowing it to escape the campaign root."""
    source = case.get("source")
    episode_file_value = source.get("episode_file") if isinstance(source, dict) else None
    if not isinstance(episode_file_value, str) or not episode_file_value:
        return None, "selected source row has incomplete episode-file provenance"
    campaign_root_resolved = campaign_root.resolve()
    episode_file = (campaign_root_resolved / episode_file_value).resolve()
    try:
        episode_file.relative_to(campaign_root_resolved)
    except ValueError:
        return None, "selected episode file escapes the campaign root"
    return episode_file, None


def _selected_episode_record_error(
    episode_file: Path, case: dict[str, Any], row: dict[str, Any]
) -> str | None:
    """Fail closed unless the renderer's episode ID resolves to this exact source record."""
    source = case.get("source")
    selected_raw = row.get("raw")
    if not isinstance(source, dict) or not isinstance(selected_raw, dict):
        return "selected source row has incomplete provenance"
    expected_file_sha256 = source.get("episode_file_sha256")
    line_number = source.get("line_number")
    record_sha256 = source.get("record_sha256")
    episode_id = case.get("episode_id")
    if (
        not isinstance(expected_file_sha256, str)
        or not SHA256_RE.fullmatch(expected_file_sha256)
        or type(line_number) is not int
        or line_number < 1
        or not isinstance(record_sha256, str)
        or not SHA256_RE.fullmatch(record_sha256)
        or not isinstance(episode_id, str)
        or not episode_id
    ):
        return "selected source row has incomplete provenance or identity"

    try:
        actual_file_sha256 = sha256_file(episode_file)
        persisted_raw, persisted_digest, matching_episode_ids = _read_episode_source_binding(
            episode_file, line_number, episode_id
        )
    except OSError as exc:
        return f"could not verify selected source file: {exc}"
    if actual_file_sha256 != expected_file_sha256:
        return "source episodes file changed since the selected row was read"
    if persisted_raw is None or persisted_digest != record_sha256:
        return "selected source row record digest changed or its line is unavailable"
    try:
        persisted_canonical = json.dumps(
            persisted_raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        selected_canonical = json.dumps(
            selected_raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    except (TypeError, ValueError) as exc:
        return f"selected source row payload could not be compared: {exc}"
    if persisted_canonical != selected_canonical:
        return "selected source row payload differs from the replay input"
    if (
        persisted_raw.get("episode_id") != episode_id
        or persisted_raw.get("scenario_id") != case.get("scenario_id")
        or persisted_raw.get("seed") != case.get("seed")
    ):
        return "selected source row identity does not match the replay case"
    if matching_episode_ids != 1:
        return (
            f"duplicate or unresolved episode ID {episode_id!r} appears "
            f"{matching_episode_ids} times; renderer cannot bind it to the selected row"
        )
    return None


def _recorded_replay_validation_error(raw: dict[str, Any]) -> str | None:
    """Return a reason unless recorded replay steps form a renderable episode."""
    try:
        replay_episode = build_replay_from_episode_row(EpisodeRow.from_dict(raw))
    except (KeyError, TypeError, ValueError) as exc:
        return str(exc)
    if replay_episode is None or not validate_replay_episode(replay_episode, min_length=2):
        return "recorded replay_steps are not renderable"
    return None


def _invoke_renderer(command: list[str]) -> dict[str, Any] | None:
    """Invoke the existing episode renderer and normalize process failures."""
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    except OSError as exc:
        message = f"could not invoke existing renderer: {exc}"
        status = "unavailable"
    else:
        if result.returncode == 0:
            return None
        message = (result.stderr or result.stdout).strip()[-1000:]
        if not message:
            message = f"renderer exited with {result.returncode}"
        status = "mismatch" if "Determinism check failed" in message else "unavailable"
    return {
        "status": status,
        "artifact_status": "not_generated",
        "reason": message,
        "renderer_called": True,
        "command": command,
        "artifacts": [],
    }


def _read_renderer_sidecar(sidecar_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load renderer provenance and return a readable diagnostic when unavailable."""
    try:
        payload = _read_json(sidecar_path, required=False)
    except ShowcaseError as exc:
        return None, str(exc)
    if payload is None:
        return None, "renderer exited successfully but did not emit replay provenance"
    return payload, None


def _read_renderer_artifacts(
    payload: dict[str, Any], output_dir: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate renderer artifact paths, types, and digests from its provenance sidecar."""
    artifact_paths = []
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raw_artifacts = []
    artifact_failures = []
    for artifact in raw_artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            artifact_failures.append("renderer provenance contains an invalid artifact record")
            continue
        path = Path(artifact["path"])
        try:
            relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
        except ValueError:
            artifact_failures.append(
                "renderer wrote an artifact outside the showcase output directory"
            )
            continue
        if not path.is_file():
            artifact_failures.append(f"renderer artifact is missing: {relative}")
            continue
        expected_digest = artifact.get("sha256")
        actual_digest = sha256_file(path)
        if not isinstance(expected_digest, str) or actual_digest != expected_digest:
            artifact_failures.append(f"renderer artifact checksum mismatch: {relative}")
            continue
        artifact_paths.append(
            {
                "type": artifact.get("type"),
                "path": relative,
                "sha256": actual_digest,
            }
        )
    required_types = {"still", "filmstrip", "trajectory"}
    observed_types = {artifact["type"] for artifact in artifact_paths}
    if len(artifact_paths) != len(required_types) or observed_types != required_types:
        artifact_failures.append("renderer did not produce every requested visualization type")
    return artifact_paths, artifact_failures


def _attach_replays(
    case_index: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    campaign_root: Path,
    output_dir: Path,
    render_limit: int,
) -> None:
    """Attempt bounded direct rendering for selected cases with recorded replay steps."""
    rows_by_case = {row["case_id"]: row for row in rows}
    attempted = 0
    for case_id in sorted(case_index):
        case = case_index[case_id]
        row = rows_by_case[case_id]
        has_steps = isinstance(row["raw"].get("replay_steps"), list) and bool(
            row["raw"].get("replay_steps")
        )
        limit_reached = has_steps and attempted >= render_limit
        if has_steps and not limit_reached:
            attempted += 1
        case["replay"] = _render_direct_replay(
            case, row, campaign_root, output_dir, render_limit_reached=limit_reached
        )


def _tool_provenance() -> dict[str, Any]:
    """Capture source revision and hashes of the showcase and reused analyzer/renderer."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "git_revision": revision,
        "files": {
            str(path.relative_to(REPO_ROOT).as_posix()): sha256_file(path)
            for path in (Path(__file__).resolve(), ANALYZER_SCRIPT, FIGURE_SCRIPT)
            if path.is_file()
        },
    }


def _source_file_hashes(campaign_root: Path, episode_files: dict[str, Path]) -> dict[str, str]:
    """Hash compact source manifests, summaries, preview inventory, and run JSONL files."""
    paths = [
        campaign_root / "campaign_manifest.json",
        campaign_root / "run_meta.json",
        campaign_root / "preflight" / "preview_scenarios.json",
        campaign_root / "reports" / "campaign_summary.json",
        campaign_root / "reports" / "scenario_breakdown.csv",
        campaign_root / "reports" / "scenario_family_breakdown.csv",
    ]
    paths.extend(path for path in episode_files.values() if path.is_file())
    result = {}
    for path in sorted(set(paths)):
        if path.is_file():
            try:
                relative = path.resolve().relative_to(campaign_root.resolve()).as_posix()
            except ValueError as exc:
                raise ShowcaseError(f"provenance file escapes campaign root: {path}") from exc
            result[relative] = sha256_file(path)
    return result


def _run_existing_analyzer(
    campaign_root: Path, output_dir: Path
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Run the existing camera-ready analyzer and retain its diagnostic outcome."""
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_json = analysis_dir / "campaign_analysis.json"
    output_md = analysis_dir / "campaign_analysis.md"
    command = [
        sys.executable,
        str(ANALYZER_SCRIPT),
        "--campaign-root",
        str(campaign_root),
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]
    try:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        return None, {"status": "failed", "reason": str(exc), "command": command}
    if completed.returncode != 0:
        reason = (completed.stderr or completed.stdout).strip()[-1500:]
        return None, {
            "status": "failed",
            "reason": reason or f"analyzer exited with {completed.returncode}",
            "command": command,
        }
    payload = _read_json(output_json, required=False)
    if payload is None:
        return None, {
            "status": "failed",
            "reason": "analyzer reported success but emitted no JSON",
            "command": command,
        }
    return payload, {
        "status": "passed",
        "command": command,
        "summary_path": "analysis/campaign_analysis.json",
        "report_path": "analysis/campaign_analysis.md",
        "summary_sha256": sha256_file(output_json),
        "report_sha256": sha256_file(output_md) if output_md.is_file() else None,
    }


def _summary_payload(
    campaign_root: Path,
    output_dir: Path,
    *,
    bundle_url: str | None,
    bundle_sha256: str | None,
    checksum_status: dict[str, Any],
    top_k: int,
    render_limit: int,
) -> dict[str, Any]:
    """Build machine-readable showcase output from persisted campaign artifacts."""
    manifest = _read_json(campaign_root / "campaign_manifest.json")
    run_meta = _read_json(campaign_root / "run_meta.json", required=False) or {}
    campaign_summary = _read_json(campaign_root / "reports" / "campaign_summary.json")
    campaign = (
        campaign_summary.get("campaign")
        if isinstance(campaign_summary.get("campaign"), dict)
        else {}
    )
    run_entries = (
        campaign_summary.get("runs") if isinstance(campaign_summary.get("runs"), list) else []
    )
    expected_runs, run_source = _expected_runs(manifest, run_entries)
    scenarios, scenario_source = _scenario_inventory(campaign_root, manifest)
    if not expected_runs:
        raise ShowcaseError("no expected planner/kinematics runs could be reconstructed")
    if not scenarios:
        raise ShowcaseError(
            "no complete scenario inventory is available for expected-row accounting"
        )
    family_by_scenario = {item["scenario_id"]: item["scenario_family"] for item in scenarios}
    rows, malformed, episode_files = _read_episode_rows(
        campaign_root, expected_runs, run_entries, family_by_scenario
    )
    analysis, analyzer_status = _run_existing_analyzer(campaign_root, output_dir)
    groups, case_index = _build_case_groups(rows, top_k)
    _attach_replays(case_index, rows, campaign_root, output_dir, render_limit)

    accounting = _build_accounting(
        scenarios,
        scenario_source,
        expected_runs,
        run_source,
        run_entries,
        rows,
        malformed,
        episode_files,
    )
    matrix = _build_matrix(scenarios, expected_runs, rows)
    analyzer_findings = list(analysis.get("findings", [])) if isinstance(analysis, dict) else []
    analyzer_integrity = (
        analysis.get("campaign_integrity")
        if isinstance(analysis, dict) and isinstance(analysis.get("campaign_integrity"), dict)
        else None
    )
    bundle_root = _find_bundle_root(campaign_root)
    if bundle_root is not None:
        checksum_status = _verify_embedded_checksums(bundle_root)
        if checksum_status["status"] == "failed":
            raise ShowcaseError(
                "embedded bundle checksums failed: " + "; ".join(checksum_status["failures"])
            )
    source_commit = manifest.get("git_hash")
    if not source_commit and isinstance(manifest.get("git"), dict):
        source_commit = manifest["git"].get("commit")
    source = {
        "campaign_id": campaign.get("campaign_id") or manifest.get("campaign_id") or "unknown",
        "campaign_source_revision": source_commit,
        "scenario_matrix": manifest.get("scenario_matrix"),
        "scenario_matrix_hash": manifest.get("scenario_matrix_hash"),
        "config_hash": manifest.get("config_hash"),
        "seed_policy": manifest.get("seed_policy"),
        "scenario_inventory_source": scenario_source,
        "expected_run_source": run_source,
        "bundle_url": bundle_url,
        "bundle_sha256": bundle_sha256,
        "embedded_checksums": checksum_status,
        "campaign_declared_benchmark_success": campaign.get("benchmark_success"),
        "campaign_integrity_status": (analyzer_integrity or {}).get("status"),
        "campaign_integrity_claim_boundary": (analyzer_integrity or {}).get("claim_boundary"),
        "source_files_sha256": _source_file_hashes(campaign_root, episode_files),
        "run_meta_campaign_id": run_meta.get("campaign_id"),
    }
    replay_counts = Counter(case["replay"]["status"] for case in case_index.values())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "analysis_and_visualization_only",
        "claim_boundary": (
            "This report describes only the persisted rows and metrics in the named bundle. "
            "It establishes no real-world safety claim or universal planner ranking."
        ),
        "source": source,
        "tool_provenance": _tool_provenance(),
        "accounting": accounting,
        "planner_family_matrix": matrix,
        "collision_event_metric_consistency": _collision_event_metric_consistency(rows),
        "critical_case_groups": groups,
        "cases": [case_index[key] for key in sorted(case_index)],
        "replay_status_counts": dict(sorted(replay_counts.items())),
        "camera_ready_analyzer": {
            **analyzer_status,
            "finding_count": len(analyzer_findings),
            "findings": analyzer_findings,
            "credibility_status": (
                analysis.get("credibility_scorecard", {}).get("status")
                if isinstance(analysis, dict)
                and isinstance(analysis.get("credibility_scorecard"), dict)
                else None
            ),
        },
        "output_paths": {"summary_json": "showcase_summary.json", "report_markdown": "report.md"},
        "selection": {
            "top_k_per_group": top_k,
            "render_limit_unique_cases": render_limit,
            "ordering": "named feature/event, then planner and scenario-family round-robin, then stable case identity",
            "opaque_composite_score_used": False,
        },
    }
    return summary


def _fmt(value: Any) -> str:
    """Format numbers for compact human-readable tables."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _render_markdown(summary: dict[str, Any]) -> str:
    """Render a repository-standard report with source, coverage, and replay boundaries."""
    source = summary["source"]
    accounting = summary["accounting"]
    collision_consistency = summary["collision_event_metric_consistency"]
    lines = [
        f"# Benchmark showcase: `{source['campaign_id']}`",
        "",
        "## Scope and provenance",
        "",
        "This is a report of persisted simulator results. It does not rank planners with a combined score or make a real-world safety claim.",
        "",
        f"- Campaign source revision: `{source.get('campaign_source_revision') or 'unavailable'}`",
        f"- Scenario matrix: `{source.get('scenario_matrix') or 'unavailable'}` (`{source.get('scenario_matrix_hash') or 'hash unavailable'}`)",
        f"- Source bundle SHA-256: `{source.get('bundle_sha256') or 'not supplied (campaign-root input)'}`",
        f"- Embedded payload checksums: `{source['embedded_checksums']['status']}` ({source['embedded_checksums']['checked_files']} files)",
        f"- Campaign summary declared `benchmark_success`: `{source.get('campaign_declared_benchmark_success')}`",
        f"- Camera-ready analyzer campaign-integrity status: `{source.get('campaign_integrity_status') or 'not reported'}`",
        f"- Analyzer campaign-integrity claim boundary: `{source.get('campaign_integrity_claim_boundary') or 'not reported'}`",
        f"- Expected episode identities: {accounting['expected_identity_count']}; present rows: {accounting['present_episode_rows']}; missing identities: {accounting['missing_identity_count']}; duplicates: {accounting['duplicate_identity_count']}; malformed lines: {accounting['malformed_line_count']}",
        "",
        _denominator_claim_text(accounting),
        "",
        "## Collision-event and metric consistency",
        "",
        _collision_event_metric_consistency_text(collision_consistency),
        "",
        "## Planner-by-family outcomes",
        "",
        "Canonical outcome flags and named metrics are reported in separate columns. Only rows from operationally eligible runs contribute to these summaries. Fallback, degraded, unavailable, failed, and unknown run outputs remain visible in row accounting and selected cases, but are excluded from planner outcome and metric aggregates.",
        "",
        "| Planner | Kinematics | Family | Expected | Present | Eligible | Excluded | Missing | Route complete | Collision event | Timeout event | Minimum clearance (m) | Near misses | Force events | Comfort exposure | Normalized time | Path efficiency | SNQI | Success metric | Collision metric | Total collision count |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_render_matrix_rows(summary["planner_family_matrix"]))
    lines.extend(_render_run_rows(accounting["run_statuses"]))
    lines.extend(_render_case_groups(summary))
    lines.extend(_render_diagnostics(summary))
    return "\n".join(lines)


def _collision_event_metric_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count persisted canonical collision events alongside their named count metrics."""
    collision_events = [row for row in rows if row["outcome"].get("collision_event") is True]
    collision_terminations = [
        row for row in collision_events if row["termination_reason"] == "collision"
    ]
    both_metrics = [
        (row["metrics"].get("collisions_metric"), row["metrics"].get("total_collision_count"))
        for row in collision_terminations
    ]
    both_available = [
        values for values in both_metrics if values[0] is not None and values[1] is not None
    ]
    both_nonpositive = [values for values in both_available if values[0] <= 0 and values[1] <= 0]
    return {
        "scope": "all_persisted_episode_rows",
        "metric_fields": ["metrics.collisions", "metrics.total_collision_count"],
        "canonical_collision_event_count": len(collision_events),
        "collision_event_and_collision_termination_count": len(collision_terminations),
        "both_count_metrics_available_count": len(both_available),
        "both_count_metrics_nonpositive_count": len(both_nonpositive),
        "missing_at_least_one_count_metric_count": len(collision_terminations)
        - len(both_available),
    }


def _collision_event_metric_consistency_text(diagnostics: dict[str, Any]) -> str:
    """Explain canonical collision events and count metrics without reconciling either."""
    event_count = diagnostics["canonical_collision_event_count"]
    collision_termination_count = diagnostics["collision_event_and_collision_termination_count"]
    nonpositive_count = diagnostics["both_count_metrics_nonpositive_count"]
    missing_count = diagnostics["missing_at_least_one_count_metric_count"]
    text = (
        f"Across all persisted episode rows, {event_count} have canonical "
        "`outcome.collision_event=true`; "
        f"{collision_termination_count} of those also have `termination_reason=collision`. "
    )
    if nonpositive_count:
        text += (
            f"**Data-quality warning:** {nonpositive_count} of the collision-terminated rows "
            "have both `metrics.collisions` and `metrics.total_collision_count` at or below "
            "zero. The matrix keeps the canonical event and source metrics separate; it does "
            "not infer or correct one from the other."
        )
    else:
        text += (
            f"{nonpositive_count} collision-terminated rows have both named count metrics at "
            "or below zero. The matrix keeps canonical events and source metrics separate."
        )
    if missing_count:
        text += f" {missing_count} collision-terminated rows lack one or both count metrics."
    return text


def _denominator_claim_text(accounting: dict[str, Any]) -> str:
    """State the strength of the expected-row denominator in plain language."""
    if accounting.get("denominator_status") == "complete_preflight_inventory":
        return (
            "The expected denominator comes from the campaign run manifest and complete preflight "
            "scenario inventory. Missing rows and operational run states remain visible below."
        )
    return (
        "The scenario inventory is inferred from the recorded scenario breakdown, which may omit "
        "scenarios with no persisted outcomes. Expected and missing identity counts are therefore "
        "partial diagnostics, not a complete campaign denominator."
    )


def _render_matrix_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Render planner/family outcomes with canonical events and metrics separated."""
    lines = []
    for row in rows:
        outcomes = row["outcomes"]
        metric = row["metric_summaries"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["planner_key"],
                    row["kinematics"],
                    row["scenario_family"],
                    str(row["expected_episodes"]),
                    str(row["present_episodes"]),
                    str(row["benchmark_eligible_episodes"]),
                    str(row["excluded_episodes"]),
                    str(row["missing_episodes"]),
                    _fmt(outcomes["route_complete"]["rate"]),
                    _fmt(outcomes["collision_event"]["rate"]),
                    _fmt(outcomes["timeout_event"]["rate"]),
                    _fmt(metric["minimum_clearance_m"]["mean"]),
                    _fmt(metric["near_misses"]["mean"]),
                    _fmt(metric["force_exceed_events"]["mean"]),
                    _fmt(metric["comfort_exposure"]["mean"]),
                    _fmt(metric["time_to_goal_norm"]["mean"]),
                    _fmt(metric["path_efficiency"]["mean"]),
                    _fmt(metric["snqi"]["mean"]),
                    _fmt(metric["success_metric"]["mean"]),
                    _fmt(metric["collisions_metric"]["mean"]),
                    _fmt(metric["total_collision_count"]["mean"]),
                ]
            )
            + " |"
        )

    return lines


def _render_run_rows(runs: list[dict[str, Any]]) -> list[str]:
    """Render per-run operational status and row accounting."""
    lines = [
        "",
        "## Run accounting",
        "",
        "| Planner | Kinematics | Run status | Preflight | Availability | Expected | Present | Eligible | Excluded | Missing | Benchmark eligible | Fallback/degraded | Unavailable | Failed |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for run in runs:
        lines.append(
            f"| {run['planner_key']} | {run['kinematics']} | {run['run_status']} | {run['preflight_status']} | {run['availability_status']} | {run['expected_episodes']} | {run['present_episode_rows']} | {run['benchmark_eligible_episode_rows']} | {run['excluded_episode_rows']} | {run['missing_episodes']} | {run['benchmark_eligible']} | {run['fallback_or_degraded']} | {run['unavailable']} | {run['failed']} |"
        )
    return lines


def _render_case_groups(summary: dict[str, Any]) -> list[str]:
    """Render selected cases with replay status and exact JSONL provenance."""
    lines = [
        "",
        "## Representative critical cases",
        "",
        "Groups use named outcome events or individual recorded metrics. Candidates include diagnostic-only rows, which are labeled as excluded; selection is round-robin across planners and scenario families with stable case-ID tie breaks.",
        "",
    ]
    cases = {case["case_id"]: case for case in summary["cases"]}
    for group in summary["critical_case_groups"]:
        lines.extend(
            [
                f"### {group['name']}",
                "",
                f"{group['description']}. Candidates: {group['candidate_count']}; selected: {len(group['selected_case_ids'])}.",
                "",
            ]
        )
        if not group["selected_case_ids"]:
            lines.extend(["No case in this group has the required recorded event/metric.", ""])
            continue
        lines.extend(
            [
                "| Case | Planner / family | Scenario / seed | Outcome | Evidence | Selected groups | Replay | Source row |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for case_id in group["selected_case_ids"]:
            case = cases[case_id]
            outcome = case["outcome"]
            outcome_text = (
                ", ".join(f"{key}={value}" for key, value in outcome.items() if value is not None)
                or "outcome unknown"
            )
            replay = case["replay"]
            replay_text = replay["status"]
            if replay.get("reason"):
                replay_text += f": {replay['reason']}"
            source = case["source"]
            source_text = f"`{source['episode_file']}#L{source['line_number']}` (row SHA `{source['record_sha256'][:12]}…`)"
            evidence_text = (
                "benchmark eligible"
                if case["benchmark_eligible"]
                else "excluded: " + ", ".join(case["benchmark_exclusion_reasons"])
            )
            lines.append(
                f"| `{case_id}` | {case['planner_key']} / {case['scenario_family']} | {case['scenario_id']} / {case['seed']} | {outcome_text} | {evidence_text} | {', '.join(case['selected_groups'])} | {replay_text} | {source_text} |"
            )
        lines.append("")
    return lines


def _render_diagnostics(summary: dict[str, Any]) -> list[str]:
    """Render analyzer findings and replay/provenance limitations."""
    lines = []
    analyzer = summary["camera_ready_analyzer"]
    lines.extend(["## Existing analyzer diagnostics", ""])
    if analyzer["status"] != "passed":
        lines.append(
            f"The existing camera-ready analyzer did not complete: {analyzer.get('reason') or 'no detail provided'}."
        )
    elif analyzer["findings"]:
        lines.append(
            f"The existing analyzer reported {analyzer['finding_count']} finding(s); these are preserved without reinterpretation:"
        )
        lines.extend(f"- {finding}" for finding in analyzer["findings"])
    else:
        lines.append("The existing camera-ready analyzer reported no findings.")
    lines.extend(
        [
            "",
            f"Analyzer machine report: `{analyzer.get('summary_path', 'unavailable')}`. The analyzer's credibility status is `{analyzer.get('credibility_status') or 'unavailable'}` and is diagnostic, not a planner ranking.",
            "",
            "## Replay limitation",
            "",
        ]
    )
    if summary["replay_status_counts"].get("unavailable", 0):
        lines.append(
            "Some selected rows do not include recorded `replay_steps`. The figure renderer was skipped for those rows because its fallback re-simulates with `simple_policy`; that would both launch a simulation and fail to reproduce arbitrary recorded planners. Their source JSONL row remains linked above."
        )
    if summary["replay_status_counts"].get("verified", 0):
        lines.append(
            "A `verified` replay means the existing renderer consumed recorded replay steps and its endpoint determinism check passed; it does not promote the replay as new benchmark evidence."
        )
    lines.extend(
        [
            "",
            "## Provenance files",
            "",
            "Source file digests are in `showcase_summary.json`. The downloaded archive is reused only after its SHA-256 matches the requested digest; any embedded `checksums.sha256` manifest is verified before analysis.",
            "",
        ]
    )
    return lines


def run_showcase(
    *,
    campaign_root: Path | None,
    out_dir: Path,
    bundle_path: Path | None = None,
    bundle_url: str | None = None,
    bundle_sha256: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    render_limit: int = DEFAULT_RENDER_LIMIT,
) -> dict[str, Any]:
    """Run one persisted-result showcase and write deterministic JSON/Markdown outputs."""
    _validate_showcase_inputs(
        campaign_root, bundle_path, bundle_url, bundle_sha256, top_k, render_limit
    )
    output_dir = out_dir.resolve()
    if campaign_root is not None:
        source_root = campaign_root.resolve()
        if output_dir == source_root or output_dir.is_relative_to(source_root):
            raise ShowcaseError("--out-dir must be outside the supplied campaign root")
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_root, source_url, expected_sha, checksum_status = _resolve_showcase_source(
        campaign_root,
        bundle_path,
        bundle_url,
        bundle_sha256,
        output_dir,
    )
    campaign_root = campaign_root.resolve()
    if not campaign_root.is_dir():
        raise ShowcaseError(f"campaign root is not a directory: {campaign_root}")
    summary = _summary_payload(
        campaign_root,
        output_dir,
        bundle_url=source_url,
        bundle_sha256=expected_sha,
        checksum_status=checksum_status,
        top_k=top_k,
        render_limit=render_limit,
    )
    summary_path = output_dir / "showcase_summary.json"
    report_path = output_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_render_markdown(summary) + "\n", encoding="utf-8")
    return summary


def _validate_showcase_inputs(
    campaign_root: Path | None,
    bundle_path: Path | None,
    bundle_url: str | None,
    bundle_sha256: str | None,
    top_k: int,
    render_limit: int,
) -> None:
    """Validate CLI/programmatic input combinations before creating outputs."""
    if top_k < 1 or top_k > 100:
        raise ShowcaseError("--top-k must be between 1 and 100")
    if render_limit < 0 or render_limit > 100:
        raise ShowcaseError("--render-limit must be between 0 and 100")
    if (campaign_root is None) == (bundle_path is None and bundle_url is None):
        raise ShowcaseError("provide exactly one of campaign_root, bundle_path, or bundle_url")
    if bundle_path is not None and bundle_url is not None:
        raise ShowcaseError("provide either a local bundle or a bundle URL, not both")
    if bundle_path is not None or bundle_url is not None:
        if bundle_sha256 is None or not SHA256_RE.fullmatch(bundle_sha256):
            raise ShowcaseError("a 64-character --sha256 is required for bundle inputs")


def _resolve_showcase_source(
    campaign_root: Path | None,
    bundle_path: Path | None,
    bundle_url: str | None,
    bundle_sha256: str | None,
    output_dir: Path,
) -> tuple[Path, str | None, str | None, dict[str, Any]]:
    """Resolve and checksum a persisted campaign or a pinned bundle input."""
    checksum_status: dict[str, Any] = {
        "status": "not_applicable",
        "checked_files": 0,
        "failures": [],
    }
    resolved_bundle_path: Path | None = None
    source_url = bundle_url
    expected_sha = bundle_sha256
    if bundle_url is not None:
        resolved_bundle_path = _download_bundle(
            bundle_url, str(bundle_sha256), output_dir / "input" / "cache"
        )
    elif bundle_path is not None:
        resolved_bundle_path = bundle_path.resolve()
        if not resolved_bundle_path.is_file():
            raise ShowcaseError(f"bundle file not found: {resolved_bundle_path}")
        actual_sha = sha256_file(resolved_bundle_path)
        if actual_sha != bundle_sha256:
            raise ShowcaseError(
                f"bundle SHA-256 mismatch: expected {bundle_sha256}, got {actual_sha}"
            )

    if resolved_bundle_path is not None:
        extraction = output_dir / "input" / f"extracted-{bundle_sha256}"
        if not extraction.exists():
            _safe_extract_archive(resolved_bundle_path, extraction)
        campaign_root = _locate_campaign_root(extraction)
        bundle_root = _find_bundle_root(campaign_root)
        if bundle_root is not None:
            checksum_status = _verify_embedded_checksums(bundle_root)
            if checksum_status["status"] != "passed":
                raise ShowcaseError("embedded bundle checksum validation failed")
    if campaign_root is None:
        raise ShowcaseError("no campaign root was resolved from the supplied input")
    return campaign_root, source_url, expected_sha, checksum_status


def _build_parser() -> argparse.ArgumentParser:
    """Build the no-simulation showcase CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--campaign-root", type=Path, help="Existing camera-ready campaign root")
    inputs.add_argument("--bundle", type=Path, help="Local release archive; requires --sha256")
    inputs.add_argument("--bundle-url", help="HTTP(S) release archive URL; requires --sha256")
    parser.add_argument("--sha256", help="Required SHA-256 for --bundle/--bundle-url")
    parser.add_argument("--out-dir", type=Path, required=True, help="Ignored output directory")
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Representative cases per named group"
    )
    parser.add_argument(
        "--render-limit",
        type=int,
        default=DEFAULT_RENDER_LIMIT,
        help="Maximum unique direct replays to render",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    try:
        summary = run_showcase(
            campaign_root=args.campaign_root,
            bundle_path=args.bundle,
            bundle_url=args.bundle_url,
            bundle_sha256=args.sha256,
            out_dir=args.out_dir,
            top_k=args.top_k,
            render_limit=args.render_limit,
        )
    except ShowcaseError as exc:
        print(
            json.dumps({"schema_version": SCHEMA_VERSION, "status": "error", "error": str(exc)}),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "campaign_id": summary["source"]["campaign_id"],
                "expected_identities": summary["accounting"]["expected_identity_count"],
                "present_episode_rows": summary["accounting"]["present_episode_rows"],
                "missing_identity_count": summary["accounting"]["missing_identity_count"],
                "replay_status_counts": summary["replay_status_counts"],
                "summary": str(args.out_dir.resolve() / "showcase_summary.json"),
                "report": str(args.out_dir.resolve() / "report.md"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
