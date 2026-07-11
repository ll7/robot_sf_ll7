#!/usr/bin/env python3
"""Issue #5138: export per-family + per-cell breakdowns for the h600 campaigns.

Plain-language summary
----------------------
The extended-horizon (h600) campaigns publish only *aggregate*, per-planner
success. This script exports the per-scenario-family (and per-cell) success /
near-miss / collision / SNQI breakdowns for the two retained h600 campaign
bundles, using the same column schema as the release campaign's
``scenario_family_breakdown`` table, and it keeps the per-cell episode count
(``episodes`` = ``n``) on every row so any cell's sample size is computable.

Why it matters
--------------
Whether the universally-hard families (bottleneck, cross-trap, head-on-corridor)
stay at zero completion once waiting strategies become viable is the single
number that decides how strong the family-difficulty finding is. That number is
currently unknowable from the *exported* h600 artifacts, because the published
bundle only carries ``planner_metric_summary.csv`` (aggregate). The raw campaign
reports already contain ``scenario_family_breakdown.csv`` and
``scenario_breakdown.csv``; this script promotes them into the shared evidence
bundle with the release schema, a focused universally-hard-families summary, and
full provenance.

Claim boundary
--------------
This is **diagnostic-only, reporting over existing campaign bundles**. It runs
no new campaign, no Slurm job, no GPU, no training. It re-shapes existing
canonical campaign breakdown tables; it does not recompute any metric.

Caveats:
- Release 0.0.2 collision-count provenance is withdrawn by issue #3482. The
  collision columns are republished verbatim from the h600 campaign breakdowns
  for completeness only and are not used as paper/dissertation evidence here.
- The h600 legs are three-seed campaigns; the per-cell ``n`` is small (typically
  3). Treat per-cell success as diagnostic, not as a stable estimate.
- ``jerk_mean`` is absent from the h600 campaign breakdowns and is emitted as an
  empty field, matching the source.

Schema
------
The per-family export follows the release campaign's family-breakdown schema:

    run_label, job_id, scenario_matrix_hash, planner_key, algo, scenario_family,
    episodes, success_mean, collisions_mean, ped_collision_count_mean,
    obstacle_collision_count_mean, total_collision_count_mean,
    near_misses_mean, time_to_goal_norm_mean, path_efficiency_mean,
    comfort_exposure_mean, jerk_mean, snqi_mean

The per-cell export adds ``scenario_id`` (the cell granularity) and keeps
``episodes`` (per-cell ``n``) on every row.

Outputs are written into the h600 interpretation evidence bundle directory and
registered in ``source_manifest.json`` + ``SHA256SUMS`` so the F-C4(ii) gate's
checksum-coverage contract stays satisfied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import sha256_file

SCHEMA_VERSION = "issue_5138_h600_family_breakdown_export.v1"

DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_3810_h600_interpretation_2026-07")
REVIEW_MARKER = "AI-GENERATED NEEDS-REVIEW"

# Canonical h600 legs (job_id -> reports dir). #5164 pins these repository-
# accessible locations and verifies their checksums before this exporter may
# regenerate the diagnostic bundle.
DEFAULT_RUNS = (
    {
        "job_id": "13268",
        "run_label": "confirm",
        "reports_dir": Path(
            "docs/context/evidence/issue_3810_h600_interpretation_2026-07/source_reports/13268"
        ),
    },
    {
        "job_id": "13273",
        "run_label": "extended_roster",
        "reports_dir": Path(
            "docs/context/evidence/issue_3810_h600_interpretation_2026-07/source_reports/13273"
        ),
    },
)

# The three families the issue names as the family-difficulty finding's crux.
UNIVERSALLY_HARD_FAMILIES = ("bottleneck", "cross_trap", "head_on_corridor")

# Release-campaign family-breakdown schema. ``run_label`` / ``job_id`` /
# ``scenario_matrix_hash`` are prepended so both h600 legs are distinguishable in
# one table; the metric columns are exactly the release schema.
FAMILY_COLUMNS: tuple[str, ...] = (
    "run_label",
    "job_id",
    "scenario_matrix_hash",
    "planner_key",
    "algo",
    "scenario_family",
    "episodes",
    "success_mean",
    "collisions_mean",
    "ped_collision_count_mean",
    "obstacle_collision_count_mean",
    "total_collision_count_mean",
    "near_misses_mean",
    "time_to_goal_norm_mean",
    "path_efficiency_mean",
    "comfort_exposure_mean",
    "jerk_mean",
    "snqi_mean",
)

# Per-cell schema = family schema with ``scenario_id`` inserted after the family.
CELL_COLUMNS: tuple[str, ...] = (
    "run_label",
    "job_id",
    "scenario_matrix_hash",
    "planner_key",
    "algo",
    "scenario_family",
    "scenario_id",
    "episodes",
    "success_mean",
    "collisions_mean",
    "ped_collision_count_mean",
    "obstacle_collision_count_mean",
    "total_collision_count_mean",
    "near_misses_mean",
    "time_to_goal_norm_mean",
    "path_efficiency_mean",
    "comfort_exposure_mean",
    "jerk_mean",
    "snqi_mean",
)

# Universally-hard summary: family schema minus the long-tail motion metrics,
# plus an explicit ``zero_completion`` flag so the crux question is answerable
# at a glance.
HARD_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_label",
    "job_id",
    "scenario_matrix_hash",
    "planner_key",
    "algo",
    "scenario_family",
    "episodes",
    "success_mean",
    "near_misses_mean",
    "collisions_mean",
    "zero_completion",
)

GENERATED_OUTPUTS: tuple[str, ...] = (
    "h600_scenario_family_breakdown.csv",
    "h600_scenario_family_breakdown.md",
    "h600_scenario_cell_breakdown.csv",
    "h600_scenario_cell_breakdown.md",
    "h600_universally_hard_families_summary.csv",
    "h600_universally_hard_families_summary.md",
)

# Metric columns carried verbatim from the canonical breakdowns (already report-
# formatted strings). ``episodes`` is an int and is the per-cell ``n``.
FAMILY_METRIC_PASSTHROUGH: tuple[str, ...] = (
    "episodes",
    "success_mean",
    "collisions_mean",
    "ped_collision_count_mean",
    "obstacle_collision_count_mean",
    "total_collision_count_mean",
    "near_misses_mean",
    "time_to_goal_norm_mean",
    "path_efficiency_mean",
    "comfort_exposure_mean",
    "jerk_mean",
    "snqi_mean",
)


@dataclass(frozen=True)
class RunSource:
    """One h600 campaign leg to export."""

    job_id: str
    run_label: str
    reports_dir: Path


def _public_path(path: Path) -> str:
    """Return a repo-public path without local home/worktree prefixes."""

    resolved = path.resolve()
    for anchor in ("docs", "configs", "scripts", "tests", "output"):
        if anchor in resolved.parts:
            index = resolved.parts.index(anchor)
            return str(Path(*resolved.parts[index:]))
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of row dicts, preserving source field order."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path``."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_float_or_blank(value: str) -> float | None:
    """Parse a report-formatted metric string; blanks and NaN become ``None``."""

    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _campaign_meta(reports_dir: Path) -> dict[str, Any]:
    """Pull provenance fields from the campaign summary in ``reports_dir``."""

    summary_path = reports_dir / "campaign_summary.json"
    summary = _read_json(summary_path)
    campaign = summary.get("campaign") if isinstance(summary.get("campaign"), dict) else {}
    return {
        "scenario_matrix_hash": str(campaign.get("scenario_matrix_hash", "")),
        "campaign_id": str(campaign.get("campaign_id", "")),
        "git_hash": str(campaign.get("git_hash", "")),
        "created_at_utc": str(campaign.get("created_at_utc", "")),
    }


def _norm_row(
    raw: dict[str, str],
    *,
    run: RunSource,
    matrix_hash: str,
    passthrough: tuple[str, ...],
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Normalize one canonical breakdown row into the export schema.

    ``episodes`` is coerced to an int so the per-cell ``n`` is unambiguous; the
    metric columns are carried through verbatim (they are already report-
    formatted strings). Missing metric fields default to empty string.
    """

    row: dict[str, str] = {
        "run_label": run.run_label,
        "job_id": run.job_id,
        "scenario_matrix_hash": matrix_hash,
        "planner_key": str(raw.get("planner_key", "")),
        "algo": str(raw.get("algo", "")),
        "scenario_family": str(raw.get("scenario_family", "")),
    }
    if extra:
        row.update(extra)
    for field in passthrough:
        if field == "episodes":
            try:
                row["episodes"] = str(int(float(raw.get("episodes", "0") or 0)))
            except ValueError:
                row["episodes"] = "0"
        else:
            value = raw.get(field)
            row[field] = "" if value is None else str(value)
    return row


def _load_run(run: RunSource) -> dict[str, Any]:
    """Load the canonical family + cell breakdown rows for one h600 leg."""

    reports_dir = run.reports_dir
    family_path = reports_dir / "scenario_family_breakdown.csv"
    cell_path = reports_dir / "scenario_breakdown.csv"
    if not family_path.exists():
        raise FileNotFoundError(f"missing canonical family breakdown: {_public_path(family_path)}")
    if not cell_path.exists():
        raise FileNotFoundError(f"missing canonical cell breakdown: {_public_path(cell_path)}")
    meta = _campaign_meta(reports_dir)
    matrix_hash = meta["scenario_matrix_hash"]
    family_rows = [
        _norm_row(row, run=run, matrix_hash=matrix_hash, passthrough=FAMILY_METRIC_PASSTHROUGH)
        for row in _read_csv_rows(family_path)
    ]
    cell_rows = [
        _norm_row(
            row,
            run=run,
            matrix_hash=matrix_hash,
            passthrough=FAMILY_METRIC_PASSTHROUGH,
            extra={"scenario_id": str(row.get("scenario_id", ""))},
        )
        for row in _read_csv_rows(cell_path)
    ]
    provenance = {
        "job_id": run.job_id,
        "run_label": run.run_label,
        "reports_dir": _public_path(reports_dir),
        "campaign_id": meta["campaign_id"],
        "scenario_matrix_hash": matrix_hash,
        "git_hash": meta["git_hash"],
        "created_at_utc": meta["created_at_utc"],
        "source_files": {
            "scenario_family_breakdown.csv": {
                "path": _public_path(family_path),
                "sha256": sha256_file(family_path),
            },
            "scenario_breakdown.csv": {
                "path": _public_path(cell_path),
                "sha256": sha256_file(cell_path),
            },
            "seed_episode_rows.csv": {
                "path": _public_path(reports_dir / "seed_episode_rows.csv"),
                "sha256": sha256_file(reports_dir / "seed_episode_rows.csv"),
            },
        },
    }
    return {
        "family_rows": family_rows,
        "cell_rows": cell_rows,
        "provenance": provenance,
    }


def _sort_family(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Stable family-row sort: run, planner, family."""

    return sorted(rows, key=lambda r: (r["run_label"], r["planner_key"], r["scenario_family"]))


def _sort_cell(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Stable cell-row sort: run, planner, family, scenario_id."""

    return sorted(
        rows,
        key=lambda r: (
            r["run_label"],
            r["planner_key"],
            r["scenario_family"],
            r["scenario_id"],
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    """Write ``rows`` to ``path`` with a fixed column schema."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _fmt_cell(value: str) -> str:
    """Render a CSV cell value for Markdown (empty -> blank)."""

    return value if value else ""


def _write_markdown_table(
    path: Path,
    rows: list[dict[str, str]],
    columns: tuple[str, ...],
    *,
    title: str,
    intro: str,
) -> None:
    """Render ``rows`` as a GitHub-Flavored Markdown table with a header block."""

    lines: list[str] = [f"<!-- {REVIEW_MARKER} -->", "", f"## {title}", "", intro, ""]
    if not rows:
        lines.append("_No rows._")
        lines.append("")
    else:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for row in rows:
            cells = [str(_fmt_cell(row.get(col, ""))).replace("|", "\\|") for col in columns]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_hard_summary(family_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Focus the per-family table on the universally-hard families.

    Adds ``zero_completion`` (``yes`` when ``success_mean`` parses to ``0.0`` and
    ``episodes`` > 0) so the crux question — do these families stay at zero
    completion at h600 — is answerable directly from the export.
    """

    summary: list[dict[str, str]] = []
    for row in _sort_family(family_rows):
        if row["scenario_family"] not in UNIVERSALLY_HARD_FAMILIES:
            continue
        episodes = 0
        try:
            episodes = int(row.get("episodes", "0") or 0)
        except ValueError:
            episodes = 0
        success = _parse_float_or_blank(row.get("success_mean", ""))
        zero_completion = (
            "yes" if (episodes > 0 and success is not None and success <= 0.0) else "no"
        )
        summary.append(
            {
                "run_label": row["run_label"],
                "job_id": row["job_id"],
                "scenario_matrix_hash": row["scenario_matrix_hash"],
                "planner_key": row["planner_key"],
                "algo": row["algo"],
                "scenario_family": row["scenario_family"],
                "episodes": row["episodes"],
                "success_mean": row.get("success_mean", ""),
                "near_misses_mean": row.get("near_misses_mean", ""),
                "collisions_mean": row.get("collisions_mean", ""),
                "zero_completion": zero_completion,
            }
        )
    return summary


def _hard_summary_lede(hard_rows: list[dict[str, str]]) -> str:
    """One-paragraph plain-language reading of the universally-hard families."""

    if not hard_rows:
        return (
            "No universally-hard-family rows were found in the h600 breakdowns. "
            "This is unexpected for the classic-interactions matrix and should be "
            "investigated before the family-difficulty finding is read off this "
            "export."
        )
    # A family is "universally zero" for a leg only if EVERY planner arm with
    # episodes in that family reports zero completion.
    by_run_family: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in hard_rows:
        by_run_family.setdefault((row["run_label"], row["scenario_family"]), []).append(row)
    universally_zero: list[str] = []
    for (run_label, family), arms in sorted(by_run_family.items()):
        observed = [arm for arm in arms if int(arm.get("episodes", "0") or 0) > 0]
        if observed and all(arm["zero_completion"] == "yes" for arm in observed):
            universally_zero.append(f"{family} ({run_label})")
    if universally_zero:
        listing = ", ".join(universally_zero)
        return (
            "At least one universally-hard family still shows zero completion "
            f"across every planner arm: {listing}. This strengthens the "
            "family-difficulty finding for those cells. See the per-cell table for "
            "the cells that drive the zero and the arms that do clear."
        )
    return (
        "None of the universally-hard families (bottleneck, cross-trap, "
        "head-on-corridor) stays at zero completion across every planner arm at "
        "h600: at least one arm clears each family. This weakens a strict "
        "'universally impossible' reading of the family-difficulty finding; the "
        "per-cell table shows which speed/context cells remain hard. All values "
        "are diagnostic-only, three-seed h600 evidence."
    )


def _write_readme_fragment(hard_rows: list[dict[str, str]]) -> str:
    """Return the Markdown block appended to the bundle README."""

    return (
        "\n## Issue #5138 per-family + per-cell h600 breakdown export\n\n"
        "Per-scenario-family and per-cell success / near-miss / collision / SNQI "
        "breakdowns for the two retained h600 campaign legs (jobs 13268 "
        "`confirm` and 13273 `extended_roster`), promoted into this bundle so the "
        "family-difficulty question is answerable from the exported artifacts. "
        "The per-family table uses the release campaign's family-breakdown "
        "schema (with `run_label`/`job_id`/`scenario_matrix_hash` prepended); the "
        "per-cell table adds `scenario_id` (the cell granularity). Every row "
        "carries `episodes` (the per-cell `n`) so any cell's sample size is "
        "computable.\n\n"
        "- `h600_scenario_family_breakdown.csv` / `.md`: per-family breakdown, "
        "release-schema columns, both legs concatenated.\n"
        "- `h600_scenario_cell_breakdown.csv` / `.md`: per-cell (scenario_id) "
        "breakdown, both legs concatenated.\n"
        "- `h600_universally_hard_families_summary.csv` / `.md`: bottleneck, "
        "cross-trap, head-on-corridor per (leg, planner, family) with a "
        "`zero_completion` flag.\n\n"
        f"{_hard_summary_lede(hard_rows)}\n\n"
        "Claim boundary: diagnostic-only, reporting over existing campaign "
        "bundles; no new compute. Collision-count columns are republished verbatim "
        "from the h600 campaign breakdowns and are not paper/dissertation "
        "evidence (release 0.0.2 collision-count provenance withdrawn by issue "
        "#3482). The h600 legs are three-seed campaigns, so per-cell `n` is small "
        "(typically 3) and per-cell success is diagnostic, not a stable estimate. "
        "`jerk_mean` is absent from the h600 campaign breakdowns and is emitted "
        "as an empty field.\n"
    )


def _write_sha256sums(output_dir: Path) -> None:
    """Rewrite SHA256SUMS for every checksummed file in the bundle dir.

    Re-hashing the whole directory keeps the F-C4(ii) gate's coverage contract
    satisfied when new artifacts are added (the gate requires every ``.md`` /
    ``.json`` / ``.csv`` file in the dir to be listed with a matching digest).
    """

    lines: list[str] = []
    for path in sorted(output_dir.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        lines.append(f"{sha256_file(path)}  {_public_path(path)}")
    (output_dir / "SHA256SUMS").write_text(
        f"# {REVIEW_MARKER}\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )


def _update_source_manifest(
    output_dir: Path,
    *,
    provenance: list[dict[str, Any]],
    generated_outputs: list[str],
) -> dict[str, Any]:
    """Register the #5138 export block in the bundle source_manifest.json."""

    manifest_path = output_dir / "source_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["review_marker"] = REVIEW_MARKER
    manifest["generated_outputs"] = sorted(
        set(manifest.get("generated_outputs") or []) | set(generated_outputs)
    )
    manifest["issue_5138_family_breakdown_export"] = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": "diagnostic-only; reporting over existing h600 campaign bundles; no new compute",
        "provenance_portability": {
            "checksum_manifest": _public_path(output_dir / "SHA256SUMS"),
            "path_policy": "repo_relative",
            "source_manifest_paths": "repo_relative",
        },
        "runs": provenance,
        "generated_outputs": sorted(generated_outputs),
        "notes": {
            "per_cell_episode_count_column": "episodes",
            "collision_count_caveat": (
                "collision-count columns republished verbatim from h600 campaign "
                "breakdowns; not paper/dissertation evidence (release 0.0.2 "
                "collision-count provenance withdrawn by issue #3482)"
            ),
            "three_seed_caveat": (
                "h600 legs are three-seed campaigns; per-cell n is small "
                "(typically 3) and per-cell success is diagnostic, not a stable "
                "estimate"
            ),
        },
    }
    manifest["schema_version"] = manifest.get(
        "schema_version", "issue_4195_h600_aggregation.v1.source_manifest"
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def _append_readme_section(output_dir: Path, fragment: str) -> None:
    """Append the #5138 section to the bundle README (idempotent)."""

    readme_path = output_dir / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    marker_comment = f"<!-- {REVIEW_MARKER} -->"
    if marker_comment not in text:
        text = marker_comment + "\n\n" + text.lstrip()
    marker = "## Issue #5138 per-family + per-cell h600 breakdown export"
    if marker in text:
        # Replace the existing block (from the marker to EOF) so re-runs are clean.
        text = text.split(marker)[0].rstrip() + "\n"
    readme_path.write_text((text.rstrip() + "\n" + fragment).rstrip() + "\n", encoding="utf-8")


def build_export(
    output_dir: Path,
    runs: list[RunSource],
) -> dict[str, Any]:
    """Build the per-family + per-cell h600 breakdown export into ``output_dir``.

    Returns a small report dict with row counts and provenance for callers /
    tests. Raises ``FileNotFoundError`` if a required canonical breakdown is
    missing — this builder fails closed rather than emitting a partial export.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = [_load_run(run) for run in runs]
    family_rows = _sort_family([row for run in loaded for row in run["family_rows"]])
    cell_rows = _sort_cell([row for run in loaded for row in run["cell_rows"]])
    provenance = [run["provenance"] for run in loaded]
    hard_rows = _build_hard_summary(family_rows)

    # Per-family table (release schema + run provenance columns).
    family_csv = output_dir / "h600_scenario_family_breakdown.csv"
    family_md = output_dir / "h600_scenario_family_breakdown.md"
    _write_csv(family_csv, family_rows, FAMILY_COLUMNS)
    _write_markdown_table(
        family_md,
        family_rows,
        FAMILY_COLUMNS,
        title="h600 per-scenario-family breakdown (issue #5138)",
        intro=(
            "Per-family success / near-miss / collision / SNQI for the retained "
            "h600 legs (jobs 13268 `confirm`, 13273 `extended_roster`). Columns "
            "follow the release campaign's family-breakdown schema; `episodes` is "
            "the per-cell `n`. Diagnostic-only; see bundle README for caveats."
        ),
    )

    # Per-cell table (scenario_id granularity), per-cell n on every row.
    cell_csv = output_dir / "h600_scenario_cell_breakdown.csv"
    cell_md = output_dir / "h600_scenario_cell_breakdown.md"
    _write_csv(cell_csv, cell_rows, CELL_COLUMNS)
    _write_markdown_table(
        cell_md,
        cell_rows,
        CELL_COLUMNS,
        title="h600 per-cell (scenario_id) breakdown (issue #5138)",
        intro=(
            "Per-cell (scenario_id) success / near-miss / collision / SNQI for "
            "the retained h600 legs. `episodes` is the per-cell `n` (typically 3 "
            "for these three-seed campaigns). Diagnostic-only."
        ),
    )

    # Universally-hard families summary.
    hard_csv = output_dir / "h600_universally_hard_families_summary.csv"
    hard_md = output_dir / "h600_universally_hard_families_summary.md"
    _write_csv(hard_csv, hard_rows, HARD_SUMMARY_COLUMNS)
    _write_markdown_table(
        hard_md,
        hard_rows,
        HARD_SUMMARY_COLUMNS,
        title="h600 universally-hard families summary (issue #5138)",
        intro=(
            "Bottleneck, cross-trap, and head-on-corridor per (leg, planner, "
            "family). `zero_completion` = `yes` when a planner arm with "
            "episodes>0 reports success_mean of 0.0. This is the table that "
            "answers whether the universally-hard families stay at zero "
            "completion at h600. Diagnostic-only, three-seed evidence."
        ),
    )

    _update_source_manifest(
        output_dir,
        provenance=provenance,
        generated_outputs=list(GENERATED_OUTPUTS),
    )
    _append_readme_section(output_dir, _write_readme_fragment(hard_rows))
    _write_sha256sums(output_dir)

    return {
        "schema_version": SCHEMA_VERSION,
        "family_row_count": len(family_rows),
        "cell_row_count": len(cell_rows),
        "hard_summary_row_count": len(hard_rows),
        "runs": provenance,
        "generated_outputs": list(GENERATED_OUTPUTS),
    }


def _parse_runs(value: str) -> list[RunSource]:
    """Parse ``--runs`` as ``job_id:run_label:reports_dir`` triples."""

    runs: list[RunSource] = []
    for chunk in value.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"--runs entry must be job_id:run_label:reports_dir, got {chunk!r}"
            )
        job_id, run_label, reports_dir = parts
        runs.append(RunSource(job_id=job_id, run_label=run_label, reports_dir=Path(reports_dir)))
    return runs


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the #5138 h600 family-breakdown export."""

    parser = argparse.ArgumentParser(
        description="Export per-family + per-cell breakdowns for the h600 campaigns (issue #5138).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Evidence bundle directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    default_runs = "|".join(
        f"{r['job_id']}:{r['run_label']}:{r['reports_dir']}" for r in DEFAULT_RUNS
    )
    parser.add_argument(
        "--runs",
        type=_parse_runs,
        default=_parse_runs(default_runs),
        help=(
            "h600 legs as job_id:run_label:reports_dir triples joined by '|'. "
            f"Default: {default_runs}"
        ),
    )
    args = parser.parse_args(argv)

    report = build_export(args.output_dir, args.runs)
    print(
        "issue #5138 h600 family-breakdown export written to "
        f"{_public_path(args.output_dir)}: "
        f"{report['family_row_count']} family rows, "
        f"{report['cell_row_count']} cell rows, "
        f"{report['hard_summary_row_count']} universally-hard-family rows across "
        f"{len(report['runs'])} legs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
