"""Compile benchmark report inputs into reusable paper artifact candidates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

os.environ.setdefault("MPLBACKEND", "Agg")

import yaml
from matplotlib import pyplot as plt

from robot_sf.benchmark.artifact_catalog import (
    ARTIFACT_CATALOG_SCHEMA_VERSION,
    sha256_file,
    validate_artifact_catalog,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


_OPTIONAL_REPORT_INPUTS = (
    "matrix_summary.json",
    "campaign_table.csv",
    "seed_episode_rows.csv",
    "seed_variability_by_scenario.csv",
    "statistical_sufficiency.json",
    "snqi_diagnostics.json",
    "comparability_matrix.json",
)
_NOT_AVAILABLE_SCHEMA = "benchmark_artifact_compiler.not_available.v1"
_CLAIM_BOUNDARY = (
    "Compiled diagnostic-only publication candidate; not standalone benchmark evidence. "
    "Planner fallback, degraded, and not_available rows remain explicit caveats."
)


@dataclass(frozen=True, slots=True)
class _FileRef:
    """Catalog-compatible path/checksum pair."""

    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return the catalog mapping representation."""
        return {"path": self.path, "sha256": self.sha256}


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="Benchmark campaign directory containing a reports/ subdirectory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for compiled tables, figures, catalog, and checksums.",
    )
    parser.add_argument(
        "--catalog-id",
        default="benchmark_campaign_artifacts",
        help="artifact_catalog.v1 catalog_id to write.",
    )
    return parser


def _read_campaign_rows(campaign_table: Path) -> list[dict[str, str]]:
    """Load campaign-table rows from CSV."""
    with campaign_table.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _copy_present_inputs(campaign_root: Path, output: Path) -> dict[str, _FileRef]:
    """Copy present report inputs into the compiler output and return source refs."""
    source_refs: dict[str, _FileRef] = {}
    source_reports = output / "sources" / "reports"
    source_reports.mkdir(parents=True, exist_ok=True)
    reports_root = campaign_root / "reports"

    for name in _OPTIONAL_REPORT_INPUTS:
        source = reports_root / name
        if not source.exists():
            continue
        destination = source_reports / name
        shutil.copy2(source, destination)
        source_refs[name] = _file_ref(output, destination)
    return source_refs


def _missing_inputs(campaign_root: Path) -> list[dict[str, str]]:
    """Return structured records for optional report inputs that are not available."""
    reports_root = campaign_root / "reports"
    return [
        {
            "input": f"reports/{name}",
            "status": "not_available",
            "reason": "optional input missing",
        }
        for name in _OPTIONAL_REPORT_INPUTS
        if not (reports_root / name).exists()
    ]


def _write_campaign_tables(rows: list[Mapping[str, str]], output: Path) -> dict[str, Path]:
    """Write CSV, Markdown, and LaTeX variants for the campaign table."""
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)

    csv_path = tables_dir / "campaign_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = tables_dir / "campaign_table.md"
    md_path.write_text(_markdown_table(rows, fieldnames), encoding="utf-8")

    tex_path = tables_dir / "campaign_table.tex"
    tex_path.write_text(_latex_table(rows, fieldnames), encoding="utf-8")
    return {"csv": csv_path, "md": md_path, "tex": tex_path}


def _write_not_available_inputs(records: list[Mapping[str, str]], output: Path) -> dict[str, Path]:
    """Write missing-input evidence as JSON and Markdown table."""
    payload = {
        "schema_version": _NOT_AVAILABLE_SCHEMA,
        "records": list(records),
    }
    json_path = output / "not_available_inputs.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table_path = output / "tables" / "not_available_inputs.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_markdown_table(records, ["input", "status", "reason"]), encoding="utf-8")
    return {"json": json_path, "md": table_path}


def _write_status_figure(rows: Iterable[Mapping[str, str]], output: Path) -> dict[str, Path]:
    """Write a planner-status summary figure in PNG and PDF formats."""
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(row.get("status", "unknown") or "unknown" for row in rows)
    labels = sorted(counts)
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = [_status_color(label) for label in labels]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Rows")
    ax.set_title("Planner status summary")
    ax.set_ylim(0, max(values or [1]) + 1)
    ax.grid(axis="y", color="#d0d7de", linewidth=0.8, alpha=0.8)
    for index, value in enumerate(values):
        ax.text(index, value + 0.05, str(value), ha="center", va="bottom")
    fig.tight_layout()

    png_path = figures_dir / "planner_status_summary.png"
    pdf_path = figures_dir / "planner_status_summary.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def _write_captions(output: Path) -> Path:
    """Write shared captions and claim-boundary text."""
    captions = (
        "# Benchmark Artifact Captions\n\n"
        "## fig_planner_status_summary\n\n"
        "diagnostic-only count of campaign-table rows by planner status. Fallback, degraded, "
        "and not_available rows are caveats rather than benchmark-success evidence.\n\n"
        "## tab_campaign_table\n\n"
        "Compiled campaign table preserving planner execution status and limitation labels.\n\n"
        "## tab_not_available_inputs\n\n"
        "Optional compiler inputs that were absent from the campaign report directory.\n"
    )
    path = output / "captions.md"
    path.write_text(captions, encoding="utf-8")
    return path


def _write_checksums(output: Path, paths: Iterable[Path]) -> Path:
    """Write a checksum manifest for generated artifacts except the manifest itself."""
    checksum_path = output / "checksums.sha256"
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(paths, key=lambda item: item.relative_to(output).as_posix())
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def _write_catalog(
    *,
    output: Path,
    catalog_id: str,
    source_refs: Mapping[str, _FileRef],
    table_paths: Mapping[str, Path],
    figure_paths: Mapping[str, Path],
    missing_paths: Mapping[str, Path],
    captions_path: Path,
    command: str,
) -> Path:
    """Write and validate an ``artifact_catalog.v1`` catalog."""
    caption_ref = _file_ref(output, captions_path).to_dict()
    campaign_sources = [
        ref.to_dict()
        for name, ref in sorted(source_refs.items())
        if name in {"campaign_table.csv", "matrix_summary.json", "statistical_sufficiency.json"}
    ]
    if not campaign_sources:
        campaign_sources = [_file_ref(output, table_paths["csv"]).to_dict()]
    missing_sources = (
        [_file_ref(output, missing_paths["json"]).to_dict()]
        if not source_refs
        else [ref.to_dict() for _, ref in sorted(source_refs.items())]
    )

    payload: dict[str, Any] = {
        "schema_version": ARTIFACT_CATALOG_SCHEMA_VERSION,
        "catalog_id": catalog_id,
        "artifacts": [
            {
                "artifact_id": "tab_campaign_table",
                "artifact_kind": "table",
                "source_kind": "benchmark_campaign",
                "source_files": campaign_sources,
                "outputs": {
                    key: _file_ref(output, path).to_dict() for key, path in table_paths.items()
                },
                "caption_file": caption_ref,
                "generation_command": command,
                "generation_commit": _git_commit(),
                "claim_boundary": _CLAIM_BOUNDARY,
            },
            {
                "artifact_id": "fig_planner_status_summary",
                "artifact_kind": "figure",
                "source_kind": "benchmark_campaign",
                "source_files": campaign_sources,
                "outputs": {
                    key: _file_ref(output, path).to_dict() for key, path in figure_paths.items()
                },
                "caption_file": caption_ref,
                "generation_command": command,
                "generation_commit": _git_commit(),
                "claim_boundary": _CLAIM_BOUNDARY,
            },
            {
                "artifact_id": "tab_not_available_inputs",
                "artifact_kind": "table",
                "source_kind": "benchmark_campaign",
                "source_files": missing_sources,
                "outputs": {
                    key: _file_ref(output, path).to_dict() for key, path in missing_paths.items()
                },
                "caption_file": caption_ref,
                "generation_command": command,
                "generation_commit": _git_commit(),
                "claim_boundary": _CLAIM_BOUNDARY,
            },
        ],
    }
    catalog_path = output / "artifact_catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    issues = validate_artifact_catalog(catalog_path)
    if issues:
        details = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        raise ValueError(f"generated artifact catalog failed validation: {details}")
    return catalog_path


def _fieldnames(rows: list[Mapping[str, str]]) -> list[str]:
    """Return stable field order from loaded campaign rows."""
    if not rows:
        return ["planner", "status"]
    return list(rows[0])


def _markdown_table(rows: Iterable[Mapping[str, str]], fieldnames: Sequence[str]) -> str:
    """Render rows as a compact GitHub-flavored Markdown table."""
    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join("---" for _ in fieldnames) + " |"
    body = [
        "| " + " | ".join(_escape_markdown(str(row.get(field, ""))) for field in fieldnames) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body]) + "\n"


def _latex_table(rows: Iterable[Mapping[str, str]], fieldnames: Sequence[str]) -> str:
    """Render rows as a small tabular environment."""
    columns = "l" * len(fieldnames)
    lines = [
        f"\\begin{{tabular}}{{{columns}}}",
        " \\toprule",
        " & ".join(_escape_latex(field) for field in fieldnames) + " \\\\",
        " \\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(_escape_latex(str(row.get(field, ""))) for field in fieldnames) + " \\\\"
        )
    lines.extend([" \\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _escape_markdown(value: str) -> str:
    """Escape Markdown table separators."""
    return value.replace("|", "\\|").replace("\n", " ")


def _escape_latex(value: str) -> str:
    """Escape common LaTeX special characters."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _status_color(status: str) -> str:
    """Return a readable fixed color for a planner status label."""
    return {
        "native": "#2da44e",
        "adapter": "#0969da",
        "fallback": "#bf8700",
        "degraded": "#8250df",
        "not_available": "#cf222e",
        "failed": "#a40e26",
    }.get(status, "#6e7781")


def _file_ref(output: Path, path: Path) -> _FileRef:
    """Build a catalog-relative file reference."""
    return _FileRef(path=path.relative_to(output).as_posix(), sha256=sha256_file(path))


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
    """Return the reproduction command recorded in the catalog."""
    return (
        "uv run python scripts/tools/compile_benchmark_artifacts.py "
        f"--campaign-root {args.campaign_root.as_posix()} "
        f"--output {args.output.as_posix()} "
        f"--catalog-id {args.catalog_id}"
    )


def _run(args: argparse.Namespace) -> dict[str, str]:
    """Compile artifacts and return key output paths."""
    campaign_root = args.campaign_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_refs = _copy_present_inputs(campaign_root, output)
    campaign_table = campaign_root / "reports" / "campaign_table.csv"
    if not campaign_table.exists():
        raise FileNotFoundError(f"required campaign table is missing: {campaign_table}")

    rows = _read_campaign_rows(campaign_table)
    table_paths = _write_campaign_tables(rows, output)
    missing_paths = _write_not_available_inputs(_missing_inputs(campaign_root), output)
    figure_paths = _write_status_figure(rows, output)
    captions_path = _write_captions(output)

    generated_paths = [
        *table_paths.values(),
        *missing_paths.values(),
        *figure_paths.values(),
        captions_path,
    ]
    catalog_path = _write_catalog(
        output=output,
        catalog_id=args.catalog_id,
        source_refs=source_refs,
        table_paths=table_paths,
        figure_paths=figure_paths,
        missing_paths=missing_paths,
        captions_path=captions_path,
        command=_command(args),
    )
    checksums_path = _write_checksums(output, [*generated_paths, catalog_path])
    return {
        "artifact_catalog": str(catalog_path),
        "checksums": str(checksums_path),
        "output": str(output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compiler CLI and return a POSIX exit code."""
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
