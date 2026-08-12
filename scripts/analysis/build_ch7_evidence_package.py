"""Build the release-level Chapter 7 evidence package.

This builder consumes only digest-verified, already-produced artifacts.  It does
not run RobotSF, copy raw traces, populate the trusted source registry, or
admit a dissertation claim.  Cell-level release statistics can therefore be
rendered while trace-level dossiers remain explicitly unavailable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jsonschema import Draft202012Validator

PACKAGE_SCHEMA = "ch7-evidence-package.v1"
EXPECTED_SOURCE_SHA256SUMS = "011c644bac469a1ce6255ddb8731c53c84bd310887759174f4c734b54d6bb543"
EXPECTED_RELEASE_ARCHIVE_SHA256 = "3cfefaaa39aab6cae541cece9573848a7e0afc5e1d9e4c9a7bbf48df2330b1a7"
DEFAULT_CONFIG = Path("configs/analysis/ch7_evidence_package.v1.yaml")
REQUIRED_SCENARIOS = (
    "classic_realworld_double_bottleneck_high",
    "francis2023_blind_corner",
    "francis2023_narrow_doorway",
)
HYBRID_ARMS = (
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
)
COLOR_SAFE = {
    "route_complete": "#009E73",
    "collision_event": "#D55E00",
    "timeout": "#0072B2",
}


class Ch7EvidencePackageError(ValueError):
    """Raised when an input or package contract is not satisfied."""


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Ch7EvidencePackageError(f"invalid JSON input: {path}") from exc
    if not isinstance(payload, Mapping):
        raise Ch7EvidencePackageError(f"JSON input must be an object: {path}")
    return dict(payload)


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise Ch7EvidencePackageError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _assert_not_nested(output: Path, inputs: Iterable[Path]) -> None:
    output_resolved = output.resolve()
    for source in inputs:
        source_resolved = source.resolve()
        if output_resolved == source_resolved or source_resolved in output_resolved.parents:
            raise Ch7EvidencePackageError("output must not be the source or a child of an input")


def _parse_sha256sums(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        if "  " not in raw:
            raise Ch7EvidencePackageError(f"malformed SHA256SUMS line {line_number}: {path}")
        digest, relative = raw.split("  ", 1)
        digest = _require_sha256(digest, f"SHA256SUMS line {line_number}")
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise Ch7EvidencePackageError(f"unsafe SHA256SUMS path: {relative}")
        entries.append((digest, relative_path.as_posix()))
    if not entries:
        raise Ch7EvidencePackageError(f"empty SHA256SUMS: {path}")
    return entries


def verify_source_package(source_package: Path, expected_sha256sums: str) -> dict[str, Any]:
    """Verify every source-package member and return compact source metadata."""

    source_package = source_package.resolve()
    sums_path = source_package / "SHA256SUMS"
    if not source_package.is_dir() or not sums_path.is_file():
        raise Ch7EvidencePackageError("approved source package must contain SHA256SUMS")
    sums_digest = _sha256_file(sums_path)
    if sums_digest != expected_sha256sums:
        raise Ch7EvidencePackageError("approved source SHA256SUMS digest mismatch")
    entries = _parse_sha256sums(sums_path)
    for expected, relative in entries:
        member = source_package / relative
        if not member.is_file() or _sha256_file(member) != expected:
            raise Ch7EvidencePackageError(f"approved source member hash mismatch: {relative}")
    package_complete = _read_json(source_package / "package_complete.json")
    package_manifest = _read_json(source_package / "package_manifest.json")
    if (
        package_complete.get("visualization_only") is not True
        or package_manifest.get("visualization_only") is not True
    ):
        raise Ch7EvidencePackageError("source package must remain visualization-only")
    counts = {
        "requested": package_manifest.get("n_requested"),
        "admitted": package_manifest.get("n_admitted"),
        "excluded": package_manifest.get("n_excluded"),
    }
    if counts != {"requested": 90, "admitted": 88, "excluded": 2}:
        raise Ch7EvidencePackageError(f"unexpected source mapping counts: {counts}")
    mapping = _read_json(source_package / "mapping_receipt.json")
    rows = mapping.get("rows")
    if mapping.get("n_rows") != 90 or not isinstance(rows, list) or len(rows) != 90:
        raise Ch7EvidencePackageError("source mapping receipt is not the complete 90-row ledger")
    return {
        "sha256sums_sha256": sums_digest,
        "package_manifest_sha256": _sha256_file(source_package / "package_manifest.json"),
        "package_complete_sha256": _sha256_file(source_package / "package_complete.json"),
        "mapping_receipt_sha256": _sha256_file(source_package / "mapping_receipt.json"),
        "counts": counts,
        "member_count": len(entries),
        "mapping": mapping,
    }


def _safe_extract(tar: tarfile.TarFile, target: Path) -> None:
    members = tar.getmembers()
    for member in members:
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise Ch7EvidencePackageError(f"unsafe release archive member: {member.name}")
    tar.extractall(target, filter="data")


def _find_release_payload(root: Path) -> Path:
    candidates = sorted(
        path for path in root.rglob("campaign_manifest.json") if path.parent.name == "payload"
    )
    if len(candidates) != 1:
        raise Ch7EvidencePackageError(
            "release archive must contain exactly one payload/campaign_manifest.json"
        )
    return candidates[0].parent


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    except (OSError, csv.Error) as exc:
        raise Ch7EvidencePackageError(f"invalid release CSV: {path}") from exc


def _bool(value: Any) -> bool:
    return value is True or (isinstance(value, (int, str)) and value in {1, "1", "true"})


def _terminal_label(record: Mapping[str, Any]) -> str:
    outcome = record.get("outcome")
    if not isinstance(outcome, Mapping):
        outcome = {}
    if _bool(outcome.get("route_complete")):
        return "route_complete"
    if _bool(outcome.get("collision_event")):
        return "collision_event"
    termination = str(record.get("termination_reason") or "").lower()
    if termination in {"terminated", "timeout", "max_steps", "truncated", "horizon"}:
        return "timeout"
    return "unavailable"


def _episode_terminal_counts(
    payload: Path, scenarios: Sequence[str]
) -> dict[tuple[str, str], dict[str, int]]:
    wanted = set(scenarios)
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for episodes_path in sorted((payload / "runs").glob("*/episodes.jsonl")):
        planner_key = episodes_path.parent.name.rsplit("__", 1)[0]
        try:
            with episodes_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, Mapping) or record.get("scenario_id") not in wanted:
                        continue
                    key = (str(record["scenario_id"]), planner_key)
                    counts[key][_terminal_label(record)] += 1
        except (OSError, json.JSONDecodeError) as exc:
            raise Ch7EvidencePackageError(f"invalid release episode rows: {episodes_path}") from exc
    return {key: dict(value) for key, value in counts.items()}


def verify_release_archive(
    archive: Path, expected_digest: str
) -> tuple[Path, tempfile.TemporaryDirectory[str], dict[str, Any]]:
    """Extract and verify the release archive into a disposable directory."""

    archive = archive.resolve()
    if not archive.is_file() or _sha256_file(archive) != expected_digest:
        raise Ch7EvidencePackageError("release archive SHA-256 mismatch")
    temporary = tempfile.TemporaryDirectory(prefix="ch7-release-")
    extraction_root = Path(temporary.name)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract(tar, extraction_root)
        payload = _find_release_payload(extraction_root)
        release_manifest = _read_json(payload / "release/release_manifest.resolved.json")
        if release_manifest.get("release_tag") != "0.0.3":
            raise Ch7EvidencePackageError("unexpected release tag")
        required = (
            "campaign_manifest.json",
            "reports/scenario_breakdown.csv",
            "reports/matrix_summary.json",
        )
        if any(not (payload / item).is_file() for item in required):
            raise Ch7EvidencePackageError("release archive is missing required report inputs")
        metadata = {
            "archive_sha256": expected_digest,
            "campaign_manifest_sha256": _sha256_file(payload / "campaign_manifest.json"),
            "release_manifest_sha256": _sha256_file(
                payload / "release/release_manifest.resolved.json"
            ),
            "scenario_breakdown_sha256": _sha256_file(payload / "reports/scenario_breakdown.csv"),
            "matrix_summary_sha256": _sha256_file(payload / "reports/matrix_summary.json"),
            "release_tag": release_manifest.get("release_tag"),
            "release_id": release_manifest.get("release_id"),
        }
        return payload, temporary, metadata
    except Exception:
        temporary.cleanup()
        raise


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Ch7EvidencePackageError(f"invalid package config: {path}") from exc
    if not isinstance(payload, Mapping):
        raise Ch7EvidencePackageError("package config must be a mapping")
    return dict(payload)


def _validate_config(config: Mapping[str, Any]) -> None:
    """Validate the fixed-input and claim-boundary contract."""

    if not config:
        return
    if config.get("schema_version") is not None and config.get("schema_version") != PACKAGE_SCHEMA:
        raise Ch7EvidencePackageError("unsupported Chapter 7 package config schema")
    if config.get("issue") is not None and config.get("issue") != 6792:
        raise Ch7EvidencePackageError("Chapter 7 package config issue mismatch")
    if (
        config.get("approved_source_sha256sums") is not None
        and config.get("approved_source_sha256sums") != EXPECTED_SOURCE_SHA256SUMS
    ):
        raise Ch7EvidencePackageError("package config source digest does not match approved input")
    if (
        config.get("release_archive_sha256") is not None
        and config.get("release_archive_sha256") != EXPECTED_RELEASE_ARCHIVE_SHA256
    ):
        raise Ch7EvidencePackageError("package config release digest does not match approved input")
    counts = config.get("required_counts")
    if counts is not None and counts != {"requested": 90, "admitted": 88, "excluded": 2}:
        raise Ch7EvidencePackageError("package config mapping counts are not 90/88/2")
    policy = config.get("trace_policy")
    if isinstance(policy, Mapping) and (
        policy.get("dtw") not in {None, "forbidden"}
        or policy.get("counterfactual_branching") not in {None, "forbidden"}
    ):
        raise Ch7EvidencePackageError("package config relaxes a forbidden trace comparison")


def _arm_context(payload: Path) -> dict[str, dict[str, Any]]:
    summary = _read_json(payload / "reports/matrix_summary.json")
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise Ch7EvidencePackageError("matrix_summary.json has no planner arms")
    context: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("planner_key"):
            raise Ch7EvidencePackageError("matrix summary contains malformed arm row")
        planner = str(row["planner_key"])
        context[planner] = {
            "arm_id": planner,
            "planner_key": planner,
            "algo": row.get("algo"),
            "planner_group": row.get("planner_group"),
            "kinematics": row.get("kinematics"),
            "configuration_identity": row.get("config_hash"),
            "campaign_id": row.get("campaign_id"),
            "git_commit": row.get("git_commit"),
            "horizon": row.get("horizon"),
            "resolved_seeds": row.get("resolved_seeds"),
        }
    return context


def _cell_rows(payload: Path, arm_context: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = _read_csv(payload / "reports/scenario_breakdown.csv")
    if not rows:
        raise Ch7EvidencePackageError("scenario_breakdown.csv has no rows")
    result: list[dict[str, Any]] = []
    for row in rows:
        planner = row.get("planner_key", "")
        if planner not in arm_context:
            raise Ch7EvidencePackageError(f"scenario row references unknown planner arm: {planner}")
        try:
            cell = {
                "scenario_id": row["scenario_id"],
                "scenario_family": row.get("scenario_family", ""),
                "planner_key": planner,
                "arm_id": planner,
                "configuration_identity": arm_context[planner].get("configuration_identity"),
                "kinematics": arm_context[planner].get("kinematics"),
                "episodes": int(row["episodes"]),
                "success_fraction": float(row["success_mean"]),
                "collision_fraction": float(row["collisions_mean"]),
                "ped_collision_fraction": float(row["ped_collision_count_mean"]),
                "obstacle_collision_fraction": float(row["obstacle_collision_count_mean"]),
                "total_collision_fraction": float(row["total_collision_count_mean"]),
                "near_misses_mean": float(row["near_misses_mean"]),
                "time_to_goal_norm_mean": float(row["time_to_goal_norm_mean"]),
                "path_efficiency_mean": float(row["path_efficiency_mean"]),
                "snqi_mean": float(str(row["snqi_mean"]).lstrip("'")),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise Ch7EvidencePackageError(f"malformed scenario row: {row}") from exc
        cell["source_row_sha256"] = _sha256_bytes(_canonical_bytes(row))
        result.append(cell)
    result.sort(
        key=lambda row: (row["scenario_id"], row["planner_key"], str(row["configuration_identity"]))
    )
    return result


def _selected_rows(
    cells: Sequence[Mapping[str, Any]], terminal_counts: Mapping[tuple[str, str], Mapping[str, int]]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for cell in cells:
        scenario = cell["scenario_id"]
        planner = cell["planner_key"]
        if scenario not in REQUIRED_SCENARIOS:
            continue
        if scenario != "francis2023_narrow_doorway" and planner not in {"ppo", *HYBRID_ARMS}:
            continue
        row = dict(cell)
        row["terminal_counts"] = dict(terminal_counts.get((scenario, planner), {}))
        selected.append(row)
    return selected


def _mapping_projection(mapping: Mapping[str, Any]) -> dict[str, Any]:
    rows = mapping.get("rows")
    if not isinstance(rows, list):
        raise Ch7EvidencePackageError("mapping ledger rows are missing")
    return {
        "schema_version": mapping.get("schema_version"),
        "n_rows": mapping.get("n_rows"),
        "provenance": mapping.get("provenance"),
        "rows": rows,
        "counts": {
            "requested": len(rows),
            "admitted": sum(
                row.get("admission_status") == "admitted"
                for row in rows
                if isinstance(row, Mapping)
            ),
            "excluded": sum(
                row.get("admission_status") == "excluded"
                for row in rows
                if isinstance(row, Mapping)
            ),
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _sidecar(
    *,
    figure_id: str,
    cells: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    release: Mapping[str, Any],
    claim: str,
    limitations: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": "ch7-publication-sidecar.v1",
        "figure_id": figure_id,
        "status": "preview_pending_domain_approval",
        "evidence_grain": "release_cell",
        "release_cell_count": len(cells),
        "sample_count_per_cell": sorted({cell["episodes"] for cell in cells}),
        "arm_identities": sorted({str(cell["arm_id"]) for cell in cells}),
        "source_hashes": {
            "approved_trace_package_sha256sums": source["sha256sums_sha256"],
            "release_archive_sha256": release["archive_sha256"],
            "scenario_breakdown_sha256": release["scenario_breakdown_sha256"],
            "matrix_summary_sha256": release["matrix_summary_sha256"],
        },
        "observed_result": claim,
        "caption_ready_text": claim,
        "limitations": list(limitations),
        "trace_level": "unavailable",
        "causal_language_allowed": False,
        "dtw_used": False,
    }


def _render_figure(
    path_pdf: Path, path_svg: Path, selected: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["svg.hashsalt"] = "ch7-evidence-package.v1"
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:  # pragma: no cover - exercised by lean environments
        raise Ch7EvidencePackageError(
            "Matplotlib is required to render the Chapter 7 preview"
        ) from exc

    def find(scenario: str, planner: str) -> Mapping[str, Any]:
        for row in selected:
            if row["scenario_id"] == scenario and row["planner_key"] == planner:
                return row
        raise Ch7EvidencePackageError(f"missing publication cell: {scenario}/{planner}")

    scenarios = ["classic_realworld_double_bottleneck_high", "francis2023_blind_corner"]
    bars = ["ppo", *HYBRID_ARMS]
    labels = ["PPO", "Hybrid static", "Hybrid continuous"]
    figure = plt.figure(figsize=(7.2, 6.8), dpi=150)
    grid = figure.add_gridspec(2, 1, height_ratios=(1, 1.55), hspace=0.52)
    top = figure.add_subplot(grid[0, 0])
    x = list(range(len(scenarios)))
    width = 0.22
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for index, (planner, label, color) in enumerate(zip(bars, labels, colors, strict=True)):
        values = [find(scenario, planner)["success_fraction"] for scenario in scenarios]
        top.bar(
            [value + (index - 1) * width for value in x],
            values,
            width=width,
            label=label,
            color=color,
        )
    top.set_xticks(x, ["Double bottleneck", "Blind corner"])
    top.set_ylim(0, 1.08)
    top.set_ylabel("Success fraction")
    top.set_title(
        "Observed cross-cell inversion (release cells, n=30 each)", loc="left", fontsize=10
    )
    top.grid(axis="y", color="#dddddd", linewidth=0.6)
    top.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))

    bottom = figure.add_subplot(grid[1, 0])
    doorway = [row for row in selected if row["scenario_id"] == "francis2023_narrow_doorway"]
    doorway.sort(key=lambda row: row["planner_key"])
    short = {
        "scenario_adaptive_hybrid_orca_v1": "hybrid ORCA v1",
        "scenario_adaptive_hybrid_orca_v2_collision_guard": "hybrid ORCA v2",
        "hybrid_rule_v3_fast_progress_static_escape": "hybrid static",
        "hybrid_rule_v3_fast_progress_static_escape_continuous": "hybrid continuous",
        "prediction_planner": "prediction",
        "predictive_mppi": "predictive MPPI",
        "socnav_sampling": "socnav sampling",
    }
    y = list(range(len(doorway)))
    left = [0] * len(doorway)
    for label in ("route_complete", "collision_event", "timeout"):
        values = [int(row.get("terminal_counts", {}).get(label, 0)) for row in doorway]
        bottom.barh(y, values, left=left, color=COLOR_SAFE[label], label=label.replace("_", " "))
        left = [a + b for a, b in zip(left, values, strict=True)]
    bottom.set_yticks(
        y, [short.get(row["planner_key"], row["planner_key"]) for row in doorway], fontsize=7
    )
    bottom.set_xlim(0, 30)
    bottom.set_xlabel("Episodes (n=30)")
    bottom.set_title("Narrow-doorway terminal signatures (release cell)", loc="left", fontsize=10)
    bottom.grid(axis="x", color="#dddddd", linewidth=0.6)
    bottom.legend(
        frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22)
    )
    figure.subplots_adjust(left=0.24, right=0.98, top=0.96, bottom=0.18)
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(
        path_pdf,
        metadata={
            "Title": "Chapter 7 release-cell evidence",
            "Author": "RobotSF",
            "CreationDate": None,
            "ModDate": None,
        },
    ) as pdf:
        pdf.savefig(figure)
    figure.savefig(path_svg, format="svg", metadata={"Date": None})
    plt.close(figure)
    return {
        "status": "passed",
        "pdf_bytes": path_pdf.stat().st_size,
        "svg_bytes": path_svg.stat().st_size,
        "final_width_mm": 170,
        "color_safe_palette": True,
        "greyscale_review_required": True,
        "manual_rendered_page_inspection_required": True,
    }


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        rows.append(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="ascii")


def _build_once(
    *,
    output: Path,
    source_package: Path,
    release_archive: Path,
    issue6814_compact: Path,
    portfolio_config: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise Ch7EvidencePackageError(f"refusing to overwrite package output: {output}")
    _assert_not_nested(
        output, (source_package, release_archive, issue6814_compact, portfolio_config)
    )
    source = verify_source_package(source_package, EXPECTED_SOURCE_SHA256SUMS)
    compact_sha = _sha256_file(issue6814_compact / "compact_packet.json")
    compact_sums = _sha256_file(issue6814_compact / "SHA256SUMS")
    if (
        _sha256_file(issue6814_compact / "compact_packet.json")
        != _parse_sha256sums(issue6814_compact / "SHA256SUMS")[0][0]
    ):
        raise Ch7EvidencePackageError("#6814 compact packet checksum mismatch")
    compact = _read_json(issue6814_compact / "compact_packet.json")
    if (
        compact.get("schema_version") != "issue_6814_compact_packet.v1"
        or compact.get("disposition") != "unsupported"
    ):
        raise Ch7EvidencePackageError("#6814 compact input is not the expected unsupported packet")
    payload, release_temp, release = verify_release_archive(
        release_archive, EXPECTED_RELEASE_ARCHIVE_SHA256
    )
    try:
        arm_context = _arm_context(payload)
        cells = _cell_rows(payload, arm_context)
        terminal = _episode_terminal_counts(payload, REQUIRED_SCENARIOS)
        selected = _selected_rows(cells, terminal)
        if not selected:
            raise Ch7EvidencePackageError("no Chapter 7 release cells selected")
        staging = output.parent / f".{output.name}.staging"
        if staging.exists():
            raise Ch7EvidencePackageError(f"refusing to reuse staging directory: {staging}")
        staging.mkdir(parents=True)
        try:
            _write_json(staging / "mapping_ledger.json", _mapping_projection(source["mapping"]))
            _write_csv(staging / "audit/campaign_atlas.csv", cells, tuple(cells[0].keys()))
            _write_json(
                staging / "audit/summary.json",
                {
                    "schema_version": "ch7-audit-atlas.v1",
                    "cell_count": len(cells),
                    "planner_arm_count": len(arm_context),
                    "scenario_count": len({cell["scenario_id"] for cell in cells}),
                    "release_rows_source": "reports/scenario_breakdown.csv",
                    "claim_boundary": "release-cell descriptive context; not trace-level or causal evidence",
                },
            )
            _write_json(staging / "audit/arm_context.json", dict(sorted(arm_context.items())))
            publication_columns = tuple(selected[0].keys())
            _write_csv(staging / "publication/reduced_atlas.csv", selected, publication_columns)
            _write_json(
                staging / "publication/reduced_atlas.json",
                {
                    "schema_version": "ch7-reduced-publication-atlas.v1",
                    "cells": selected,
                    "roles": ["feasibility_criticism", "cross_cell_inversion"],
                    "claim_boundary": "release-cell descriptive figure source only",
                },
            )
            overlay = {
                "schema_version": "ch7-materialization-overlay.v1",
                "source_portfolio": {
                    "path": portfolio_config.name,
                    "sha256": _sha256_file(portfolio_config),
                },
                "roles": {
                    "planner_upset": {
                        "status": "unavailable",
                        "reason": "#6814 incompatible seed-118 starts and no shared-start receipt",
                    },
                    "seed_sensitivity": {
                        "status": "unavailable",
                        "reason": "#6814 shared_prefix=false and unequal starts",
                    },
                    "feasibility_criticism": {
                        "status": "available",
                        "grain": "release_cell_geometry",
                    },
                    "cross_cell_inversion": {"status": "available", "grain": "release_cell"},
                },
                "frozen_portfolio_unchanged": True,
            }
            _write_json(staging / "publication/materialization_overlay.json", overlay)
            double_cells = [row for row in selected if row["scenario_id"] in REQUIRED_SCENARIOS[:2]]
            doorway_cells = [row for row in selected if row["scenario_id"] == REQUIRED_SCENARIOS[2]]
            claim_double = (
                "Across the observed release cells, the selected hybrid arms complete the double-bottleneck cell "
                "at 30/30 episodes while PPO completes 0/30; in the blind-corner cell PPO completes 22/30 while "
                "the selected hybrid arms complete 0/30. This is a cross-cell descriptive inversion, not a causal mechanism or universal ranking."
            )
            claim_doorway = (
                "The narrow-doorway release cell has a terminal-signature mixture that is arm-specific: PPO and most "
                "arms terminate in collision_event, ORCA and social_force terminate in timeout, and the continuous "
                "hybrid arm contains both collision_event and timeout outcomes."
            )
            _write_json(
                staging / "publication/cross_cell_inversion.sidecar.json",
                _sidecar(
                    figure_id="cross_cell_inversion",
                    cells=double_cells,
                    source=source,
                    release=release,
                    claim=claim_double,
                    limitations=(
                        "release-cell grain only",
                        "hybrid arms remain separate",
                        "no trace or mechanism evidence",
                        "no causal or universal ranking claim",
                    ),
                ),
            )
            _write_json(
                staging / "publication/narrow_doorway_terminal_signature.sidecar.json",
                _sidecar(
                    figure_id="narrow_doorway_terminal_signature",
                    cells=doorway_cells,
                    source=source,
                    release=release,
                    claim=claim_doorway,
                    limitations=(
                        "release-cell grain only",
                        "terminal categories are observed outcomes",
                        "no trace or mechanism evidence",
                        "no causal interpretation",
                    ),
                ),
            )
            qa = _render_figure(
                staging / "publication/chapter7_release_cells.pdf",
                staging / "publication/chapter7_release_cells.svg",
                selected,
            )
            qa["source_status"] = "verified_but_domain_approval_pending"
            _write_json(
                staging / "qa/figure_qa.json",
                {
                    "schema_version": "ch7-figure-qa.v1",
                    "figures": [qa],
                    "status": "passed",
                    "claim_boundary": "rendering QA only; not evidence admission",
                },
            )
            unavailable_common = {
                "status": "unavailable",
                "grain": "trace",
                "source_gate": "blocked_visualization_only",
                "issue6814_compact_sha256": compact_sha,
                "issue6814_compact_sha256sums_sha256": compact_sums,
                "reasons": [
                    "missing_run_configuration",
                    "missing_provenance",
                    "initial_state_incompatible",
                    "shared_prefix_false",
                    "trace_publication_gate_blocked",
                ],
                "claim_boundary": "No trajectory, control-sequence, causal-divergence, or matched-start claim is available.",
            }
            _write_json(
                staging / "unavailable/doorway_ppo_seed113_114.json",
                {
                    **unavailable_common,
                    "case_id": "ch7-role-seed-sensitivity--classic-doorway-medium--ppo--seeds-113-114",
                },
            )
            _write_json(
                staging / "unavailable/double_bottleneck_goal_ppo_seed118.json",
                {
                    **unavailable_common,
                    "case_id": "ch7-role-planner-upset--classic-realworld-double-bottleneck-high--goal-vs-ppo--seed-118",
                },
            )
            _write_json(
                staging / "unavailable/trace_viewer.json",
                {
                    "status": "unavailable",
                    "reason": "trace_gate_blocked",
                    "source": "issue6814 compact unsupported",
                },
            )
            _write_json(
                staging / "unavailable/publication_comparison.json",
                {
                    "status": "unavailable",
                    "reason": "trace_gate_blocked",
                    "source": "issue6814 compact unsupported",
                },
            )
            _write_json(
                staging / "review/source_verification.json",
                {
                    "status": "verified",
                    "source": source,
                    "release": release,
                    "issue6814": {
                        "compact_packet_sha256": compact_sha,
                        "compact_sha256sums_sha256": compact_sums,
                    },
                },
            )
            _write_json(
                staging / "manifest.json",
                {
                    "schema_version": PACKAGE_SCHEMA,
                    "issue": 6792,
                    "status": "blocked_pending_domain_approval",
                    "admission_status": "not_admitted",
                    "source_integrity_gate": "blocked_pending_domain_approval",
                    "source": {
                        "approved_package_sha256sums": source["sha256sums_sha256"],
                        "release_archive_sha256": release["archive_sha256"],
                        "issue6814_compact_packet_sha256": compact_sha,
                    },
                    "inputs": {
                        "portfolio_config": {
                            "name": portfolio_config.name,
                            "sha256": _sha256_file(portfolio_config),
                        },
                        "source_package_member_count": source["member_count"],
                    },
                    "counts": source["counts"],
                    "atlas": {
                        "audit_cells": len(cells),
                        "publication_cells": len(selected),
                        "planner_arms": len(arm_context),
                    },
                    "roles": overlay["roles"],
                    "claim_boundary": "Release-cell descriptive evidence package. No trace-level or causal claim is admitted.",
                    "raw_traces_included": False,
                    "release_archive_included": False,
                    "deterministic_serialization": "strict-json-sort-keys-utf8-newline.v1",
                },
            )
            _write_checksums(staging)
            schema_path = (
                Path(__file__).parents[2]
                / "robot_sf/benchmark/schemas/ch7-evidence-package.v1.json"
            )
            schema = _read_json(schema_path)
            Draft202012Validator(schema).validate(_read_json(staging / "manifest.json"))
            staging.rename(output)
            return _read_json(output / "manifest.json")
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    finally:
        release_temp.cleanup()


def build_ch7_evidence_package(
    *,
    source_package: Path,
    release_archive: Path,
    issue6814_compact: Path,
    output: Path,
    portfolio_config: Path,
    config_path: Path | None = None,
    check_determinism: bool = False,
) -> dict[str, Any]:
    """Build the package, optionally proving byte-identical rebuilds."""

    config = _load_config(config_path)
    _validate_config(config)
    if check_determinism:
        with (
            tempfile.TemporaryDirectory(prefix="ch7-build-a-") as first_root,
            tempfile.TemporaryDirectory(prefix="ch7-build-b-") as second_root,
        ):
            first = Path(first_root) / "package"
            second = Path(second_root) / "package"
            first_manifest = _build_once(
                output=first,
                source_package=source_package,
                release_archive=release_archive,
                issue6814_compact=issue6814_compact,
                portfolio_config=portfolio_config,
                config=config,
            )
            second_manifest = _build_once(
                output=second,
                source_package=source_package,
                release_archive=release_archive,
                issue6814_compact=issue6814_compact,
                portfolio_config=portfolio_config,
                config=config,
            )
            if first_manifest != second_manifest or _tree_hash(first) != _tree_hash(second):
                raise Ch7EvidencePackageError("Chapter 7 package is not byte deterministic")
    return _build_once(
        output=output,
        source_package=source_package,
        release_archive=release_archive,
        issue6814_compact=issue6814_compact,
        portfolio_config=portfolio_config,
        config=config,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--release-archive", type=Path, required=True)
    parser.add_argument("--issue6814-compact", type=Path, required=True)
    parser.add_argument(
        "--portfolio-config",
        type=Path,
        default=Path("configs/analysis/ch7_worked_example_portfolio.v1.yaml"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-determinism", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build one package and return a typed CLI status code."""

    args = _parser().parse_args(argv)
    try:
        manifest = build_ch7_evidence_package(
            source_package=args.source_package,
            release_archive=args.release_archive,
            issue6814_compact=args.issue6814_compact,
            output=args.output,
            portfolio_config=args.portfolio_config,
            config_path=args.config,
            check_determinism=args.check_determinism,
        )
    except (Ch7EvidencePackageError, OSError, tarfile.TarError) as exc:
        print(f"ch7 evidence package unavailable: {exc}")
        return 2
    print(f"ch7 evidence package status: {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
