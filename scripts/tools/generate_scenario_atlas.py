"""Generate a compact scenario atlas with thumbnails and mechanism cards."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256
from robot_sf.benchmark.runner import load_scenario_matrix
from robot_sf.benchmark.scenario_thumbnails import resolve_scenario_label, save_scenario_thumbnails

if TYPE_CHECKING:
    from collections.abc import Sequence


MANIFEST_SCHEMA_VERSION = "scenario_atlas_manifest.v1"
GAPS_SCHEMA_VERSION = "scenario_atlas_coverage_gaps.v1"
CLAIM_BOUNDARY = (
    "Scenario atlas is a discoverability and inspection artifact, not benchmark evidence. "
    "Certification status, executed evidence, and missing coverage gaps must be interpreted "
    "before using a scenario in benchmark-facing claims."
)
ATLAS_COLUMNS = (
    "scenario_id",
    "family",
    "authored_intent",
    "certification_status",
    "executed_evidence",
    "hazards",
    "known_failure_modes",
    "source_link",
    "map_ref",
    "benchmark_ref",
    "thumbnail",
    "mechanism_card",
    "coverage_gaps",
)
OPTIONAL_COVERAGE = (
    ("contract", "missing_contract"),
    ("certificate", "missing_certificate"),
    ("hazard_mapping", "missing_hazard_mapping"),
)


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="Scenario matrix YAML path.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the atlas pack.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="scenario_atlas",
        help="Run identifier written into the manifest.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base seed for deterministic thumbnails.",
    )
    return parser


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    """Return the scenario identity used across atlas artifacts."""
    return resolve_scenario_label(dict(scenario))


def _metadata(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return scenario metadata mapping or an empty mapping."""
    raw = scenario.get("metadata")
    return raw if isinstance(raw, Mapping) else {}


def _atlas_meta(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return optional atlas-specific metadata mapping."""
    raw = scenario.get("atlas")
    return raw if isinstance(raw, Mapping) else {}


def _family(scenario_id: str, scenario: Mapping[str, Any]) -> str:
    """Infer a compact family label for atlas filtering."""
    metadata = _metadata(scenario)
    for key in ("family", "scenario_family", "archetype"):
        value = metadata.get(key)
        if value:
            return str(value)
    if scenario_id.startswith("classic_"):
        return "classic"
    if scenario_id.startswith("francis2023_"):
        return "francis2023"
    return "unspecified"


def _source_link(scenario_id: str, matrix: Path, scenario: Mapping[str, Any]) -> str:
    """Return a source pointer without copying source YAML into the atlas."""
    atlas_meta = _atlas_meta(scenario)
    source = str(atlas_meta.get("source_path") or _path_ref(matrix))
    return f"{source}#{scenario_id}"


def _map_ref(scenario: Mapping[str, Any], matrix: Path) -> str:
    """Return the scenario map reference when declared."""
    map_id = scenario.get("map_id")
    map_file = scenario.get("map_file")
    refs: list[str] = []
    if map_id:
        refs.append(f"map_id:{map_id}")
    if not map_file:
        return ";".join(refs)
    path = Path(str(map_file))
    if not path.is_absolute():
        path = matrix.parent / path
    refs.append(_path_ref(path))
    return ";".join(refs)


def _benchmark_ref(scenario: Mapping[str, Any], matrix: Path) -> str:
    """Return benchmark config links declared for the atlas row."""
    atlas_meta = _atlas_meta(scenario)
    raw = (
        atlas_meta.get("benchmark_config")
        or atlas_meta.get("benchmark_configs")
        or atlas_meta.get("benchmark_ref")
    )
    refs = _as_list(raw)
    if not refs and "configs/benchmarks/" in _path_ref(matrix):
        refs = [matrix]
    return ";".join(_path_ref(Path(str(ref))) for ref in refs)


def _coverage_gaps(scenario: Mapping[str, Any]) -> list[str]:
    """Return missing optional coverage records for a scenario."""
    atlas_meta = _atlas_meta(scenario)
    gaps = [gap_name for key, gap_name in OPTIONAL_COVERAGE if not atlas_meta.get(key)]
    return gaps


def _scenario_row(
    scenario: Mapping[str, Any],
    *,
    scenario_id: str,
    matrix: Path,
    thumbnail_path: str,
    card_path: str,
) -> dict[str, str]:
    """Build one CSV/Markdown atlas row."""
    metadata = _metadata(scenario)
    atlas_meta = _atlas_meta(scenario)
    intent = (
        metadata.get("authored_intent")
        or metadata.get("intended_mechanism")
        or metadata.get("description")
        or "not_declared"
    )
    hazards = _join(metadata.get("hazards"))
    failure_modes = _join(metadata.get("known_failure_modes"))
    certification = str(atlas_meta.get("certification_status") or "not_certified")
    evidence = str(atlas_meta.get("executed_evidence") or "not_executed")
    gaps = ";".join(_coverage_gaps(scenario))
    return {
        "scenario_id": scenario_id,
        "family": _family(scenario_id, scenario),
        "authored_intent": str(intent),
        "certification_status": certification,
        "executed_evidence": evidence,
        "hazards": hazards,
        "known_failure_modes": failure_modes,
        "source_link": _source_link(scenario_id, matrix, scenario),
        "map_ref": _map_ref(scenario, matrix),
        "benchmark_ref": _benchmark_ref(scenario, matrix),
        "thumbnail": thumbnail_path,
        "mechanism_card": card_path,
        "coverage_gaps": gaps,
    }


def _join(value: object) -> str:
    """Format list-like metadata for CSV cells."""
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        return ";".join(str(item) for item in value)
    return str(value)


def _as_list(value: object) -> list[object]:
    """Return scalar or list-like metadata as a list."""
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _write_csv(path: Path, rows: list[Mapping[str, str]]) -> None:
    """Write atlas rows as CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ATLAS_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[Mapping[str, str]]) -> None:
    """Write the human-readable atlas table."""
    lines = [
        "# Scenario Atlas",
        "",
        CLAIM_BOUNDARY,
        "",
        "| " + " | ".join(ATLAS_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in ATLAS_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_escape_markdown(str(row.get(column, ""))) for column in ATLAS_COLUMNS)
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_card(path: Path, row: Mapping[str, str], scenario: Mapping[str, Any]) -> None:
    """Write one scenario mechanism card."""
    metadata = _metadata(scenario)
    command = str(metadata.get("runnable_command") or "not_declared")
    text = (
        f"# {row['scenario_id']}\n\n"
        f"- authored_intent: {row['authored_intent']}\n"
        f"- family: {row['family']}\n"
        f"- hazards: {row['hazards'] or 'not_declared'}\n"
        f"- certification_status: {row['certification_status']}\n"
        f"- executed_evidence: {row['executed_evidence']}\n"
        f"- known_failure_modes: {row['known_failure_modes'] or 'not_declared'}\n"
        f"- source_link: {row['source_link']}\n"
        f"- map_ref: {row['map_ref'] or 'not_declared'}\n"
        f"- runnable_command: {command}\n"
        f"- coverage_gaps: {row['coverage_gaps'] or 'none'}\n\n"
        f"{CLAIM_BOUNDARY}\n"
    )
    path.write_text(text, encoding="utf-8")


def _write_gaps(path: Path, scenarios: list[tuple[str, Mapping[str, Any]]]) -> None:
    """Write explicit not-available records for missing optional atlas inputs."""
    records = [
        {
            "scenario_id": scenario_id,
            "gap": gap,
            "status": "not_available",
        }
        for scenario_id, scenario in scenarios
        for gap in _coverage_gaps(scenario)
    ]
    payload = {"schema_version": GAPS_SCHEMA_VERSION, "records": records}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_manifest(
    *,
    path: Path,
    run_id: str,
    matrix: Path,
    output: Path,
    generated: list[Path],
    command: str,
) -> None:
    """Write atlas provenance and output checksums."""
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "claim_boundary": CLAIM_BOUNDARY,
        "generation_command": command,
        "generation_commit": _git_commit(),
        "inputs": [{"path": _path_ref(matrix), "sha256": _sha256(matrix)}],
        "outputs": [
            {"path": item.relative_to(output).as_posix(), "sha256": _sha256(item)}
            for item in sorted(generated, key=lambda entry: entry.relative_to(output).as_posix())
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _escape_markdown(value: str) -> str:
    """Escape Markdown table separators."""
    return value.replace("|", "\\|").replace("\n", " ")


def _path_ref(path: Path) -> str:
    """Return a repo-relative path when possible, otherwise the original path."""
    resolved = path.resolve()
    try:
        repo_root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return path.as_posix()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _git_commit() -> str:
    """Return the current Git commit, or a stable placeholder outside Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _command(args: argparse.Namespace) -> str:
    """Return the command recorded in the manifest."""
    parts = [
        "uv",
        "run",
        "python",
        "scripts/tools/generate_scenario_atlas.py",
        "--matrix",
        args.matrix.as_posix(),
        "--output",
        args.output.as_posix(),
        "--run-id",
        args.run_id,
        "--base-seed",
        str(args.base_seed),
    ]
    return " ".join(shlex.quote(part) for part in parts)


def _relative_output_path(path: Path, output: Path) -> str:
    """Return an output-relative path with a clear error if it escapes output."""
    resolved_path = path.resolve()
    resolved_output = output.resolve()
    try:
        return resolved_path.relative_to(resolved_output).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Generated path '{resolved_path}' is outside output '{resolved_output}'."
        ) from exc


def _run(args: argparse.Namespace) -> dict[str, str]:
    """Generate the atlas and return key output paths."""
    matrix = args.matrix.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenario_matrix(matrix)
    thumb_dir = output / "thumbnails"
    card_dir = output / "mechanism_cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    thumbs = save_scenario_thumbnails(scenarios, out_dir=thumb_dir, base_seed=args.base_seed)

    rows: list[dict[str, str]] = []
    cards: list[Path] = []
    scenario_pairs = [
        (meta.scenario_id, scenario, Path(meta.png))
        for meta, scenario in zip(thumbs, scenarios, strict=False)
    ]
    if len(scenario_pairs) != len(scenarios) or len(scenario_pairs) != len(thumbs):
        raise ValueError(
            "Thumbnail generation count mismatch: "
            f"{len(scenarios)} scenarios produced {len(thumbs)} thumbnails."
        )
    for sid, scenario, thumb_path in scenario_pairs:
        card_path = card_dir / f"{sid}.md"
        row = _scenario_row(
            scenario,
            scenario_id=sid,
            matrix=matrix,
            thumbnail_path=_relative_output_path(thumb_path, output),
            card_path=_relative_output_path(card_path, output),
        )
        _write_card(card_path, row, scenario)
        rows.append(row)
        cards.append(card_path)

    csv_path = output / "scenario_atlas.csv"
    md_path = output / "scenario_atlas.md"
    gaps_path = output / "coverage_gaps.json"
    manifest_path = output / "atlas_manifest.json"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    _write_gaps(gaps_path, [(sid, scenario) for sid, scenario, _thumb_path in scenario_pairs])

    generated = [
        csv_path,
        md_path,
        gaps_path,
        *cards,
        *[Path(meta.png) for meta in thumbs],
    ]
    _write_manifest(
        path=manifest_path,
        run_id=args.run_id,
        matrix=matrix,
        output=output,
        generated=generated,
        command=_command(args),
    )
    return {
        "atlas": str(md_path),
        "manifest": str(manifest_path),
        "output": str(output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scenario atlas generator."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = _run(args)
    except Exception as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
