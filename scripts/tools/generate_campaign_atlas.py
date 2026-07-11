"""Generate a discoverability atlas over the evidence-registry campaigns.

Scans ``docs/context/evidence/*/campaign/campaign_manifest.json`` and emits one
Markdown table row per campaign into ``docs/benchmarks/CAMPAIGN_ATLAS.md`` so any
published number is one hop from its producing campaign, commit, and hashes.

This is a discoverability surface only: it surfaces neutral provenance fields
(campaign id, date, git commit, scenario-matrix hash, episode count, report
links, manifest sha256 presence). It makes no benchmark claim. Malformed or
partial entries become a row flagged ``INCOMPLETE`` rather than raising.

The output is deterministic (sorted rows, no timestamps in the body) so
regeneration is diff-stable. ``--check`` exits nonzero when the committed atlas
is stale relative to the registry, mirroring the other generated-doc checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_EVIDENCE_ROOT = Path("docs/context/evidence")
DEFAULT_OUTPUT = Path("docs/benchmarks/CAMPAIGN_ATLAS.md")
FALLBACK_REMOTE = "https://github.com/ll7/robot_sf_ll7"

MANIFEST_REL = Path("campaign") / "campaign_manifest.json"
FLAT_MANIFEST_REL = Path("campaign_manifest.json")
SUMMARY_REL = Path("reports") / "campaign_summary.json"
SHA256_REL = Path("manifest.sha256")

# Canonical, priority-ordered "key report" files linked from each row. Only
# files that actually exist under the campaign's ``reports/`` dir are linked,
# and always in this fixed order so the output is deterministic.
KEY_REPORTS: tuple[str, ...] = (
    "campaign_report.md",
    "matrix_summary.json",
    "campaign_summary.json",
)

CLAIM_BOUNDARY = (
    "Campaign atlas is a discoverability and provenance index, not benchmark "
    "evidence. ``INCOMPLETE`` rows, missing episode counts, or absent manifest "
    "sha256 files must be resolved before treating a campaign as fully attested; "
    "consult the linked campaign manifest and reports for authoritative fields."
)

# Columns rendered in the Markdown table, in order.
COLUMNS: tuple[str, ...] = (
    "status",
    "campaign_id",
    "date",
    "git_commit",
    "scenario_matrix",
    "episodes",
    "key_reports",
    "manifest_sha256",
)


@dataclass(frozen=True)
class CampaignRow:
    """One resolved atlas row for a single campaign evidence bundle."""

    campaign_dir_name: str
    evidence_rel: str  # repo-relative path to the campaign evidence dir
    status: str  # "OK" or "INCOMPLETE"
    campaign_id: str
    date: str
    git_commit: str  # full hash, "" if unknown
    scenario_matrix: str  # repo-relative matrix path, "" if unknown
    scenario_matrix_hash: str
    episodes: str  # rendered count or "—"
    reports: tuple[str, ...]  # repo-relative report paths
    manifest_sha256_present: bool
    incomplete_reasons: tuple[str, ...]
    remote_base: str  # web base for commit links


# ---------------------------------------------------------------------------
# Tolerant parsing helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load JSON tolerantly; return ``(data, None)`` or ``(None, reason)``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"unreadable manifest ({exc})"
    except json.JSONDecodeError as exc:
        return None, f"malformed manifest json ({exc.msg})"
    if not isinstance(data, Mapping):
        return None, "manifest is not a JSON object"
    return dict(data), None


def _git_commit(manifest: Mapping[str, Any]) -> str:
    """Return the full git commit from the manifest's ``git`` block, else ``''``."""
    git_block = manifest.get("git")
    if isinstance(git_block, Mapping):
        commit = git_block.get("commit")
        if isinstance(commit, str) and commit.strip():
            return commit.strip()
    return ""


def _short_commit(commit: str) -> str:
    """Return a 12-char short commit hash, or the original if shorter."""
    return commit[:12] if commit else ""


def _remote_base(manifest: Mapping[str, Any] | None) -> str:
    """Return the normalized repository web base for building commit links."""
    candidates: list[str] = []
    if isinstance(manifest, Mapping):
        git_block = manifest.get("git")
        if isinstance(git_block, Mapping):
            remote = git_block.get("remote")
            if isinstance(remote, str) and remote.strip():
                candidates.append(remote.strip())
        repo_url = manifest.get("repository_url")
        if isinstance(repo_url, str) and repo_url.strip():
            candidates.append(repo_url.strip())
    for candidate in candidates:
        normalized = candidate.rstrip("/").removesuffix(".git")
        if normalized:
            return normalized
    return FALLBACK_REMOTE


def _date_from(manifest: Mapping[str, Any]) -> str:
    """Return the campaign date (date portion of created/started timestamp)."""
    for key in ("created_at_utc", "started_at_utc"):
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            # ISO-8601 timestamps: the date is the first 10 chars (YYYY-MM-DD).
            return value[:10] if len(value) >= 10 else value.strip()
    return ""


def _scenario_matrix(manifest: Mapping[str, Any]) -> str:
    """Return the scenario-matrix path declared by the manifest, else ``''``."""
    value = manifest.get("scenario_matrix")
    return str(value).strip() if isinstance(value, str) and value.strip() else ""


def _scenario_matrix_hash(manifest: Mapping[str, Any]) -> str:
    """Return the scenario-matrix hash declared by the manifest, else ``''``."""
    value = manifest.get("scenario_matrix_hash")
    return str(value).strip() if isinstance(value, str) and value.strip() else ""


def _episodes(manifest: Mapping[str, Any], summary_path: Path) -> str:
    """Resolve total episodes from the summary, then the manifest, else ``'—'``."""
    if summary_path.is_file():
        data, _reason = _load_json(summary_path)
        if isinstance(data, Mapping):
            campaign = data.get("campaign")
            if isinstance(campaign, Mapping):
                value = campaign.get("total_episodes")
                if isinstance(value, int) and value >= 0:
                    return str(value)
    value = manifest.get("total_episodes")
    if isinstance(value, int) and value >= 0:
        return str(value)
    return "—"


def _key_reports(reports_dir: Path, evidence_rel: str) -> tuple[str, ...]:
    """Return repo-relative POSIX paths for canonical key reports that exist."""
    found: list[str] = []
    if reports_dir.is_dir():
        for name in KEY_REPORTS:
            if (reports_dir / name).is_file():
                found.append(f"{evidence_rel}/reports/{name}")
    return tuple(found)


# ---------------------------------------------------------------------------
# Row resolution
# ---------------------------------------------------------------------------


def resolve_campaign(campaign_dir: Path, evidence_rel: str) -> CampaignRow:
    """Resolve one campaign evidence bundle into an atlas row.

    Never raises for user-data problems: a malformed manifest or missing fields
    produce an ``INCOMPLETE`` row instead.

    ``campaign_dir`` is an existing campaign directory and ``evidence_rel`` is
    the same path expressed relative to the repo root (used for links).

    Manifests may live under ``campaign/campaign_manifest.json`` (nested) or
    ``campaign_manifest.json`` (flat). Nested is preferred; flat is tolerated.
    """
    manifest_path = campaign_dir / MANIFEST_REL
    if not manifest_path.is_file():
        manifest_path = campaign_dir / FLAT_MANIFEST_REL
    summary_path = campaign_dir / SUMMARY_REL
    sha256_path = campaign_dir / SHA256_REL
    reports_dir = campaign_dir / "reports"

    data, reason = _load_json(manifest_path)
    if data is None:
        # Entirely unreadable/non-JSON manifest: still emit a row, flagged.
        return CampaignRow(
            campaign_dir_name=campaign_dir.name,
            evidence_rel=evidence_rel,
            status="INCOMPLETE",
            campaign_id=campaign_dir.name,
            date="",
            git_commit="",
            scenario_matrix="",
            scenario_matrix_hash="",
            episodes="—",
            reports=_key_reports(reports_dir, evidence_rel),
            manifest_sha256_present=sha256_path.is_file(),
            incomplete_reasons=(reason or "unknown manifest error",),
            remote_base=FALLBACK_REMOTE,
        )

    campaign_id = str(data.get("campaign_id") or "").strip()
    git_commit = _git_commit(data)
    scenario_matrix = _scenario_matrix(data)
    scenario_matrix_hash = _scenario_matrix_hash(data)
    date = _date_from(data)
    episodes = _episodes(data, summary_path)
    reports = _key_reports(reports_dir, evidence_rel)
    sha256_present = sha256_path.is_file()
    remote = _remote_base(data)

    reasons: list[str] = []
    if not campaign_id:
        reasons.append("missing campaign_id")
    if not git_commit:
        reasons.append("missing git commit")
    if not scenario_matrix:
        reasons.append("missing scenario_matrix")

    status = "OK" if not reasons else "INCOMPLETE"
    return CampaignRow(
        campaign_dir_name=campaign_dir.name,
        evidence_rel=evidence_rel,
        status=status,
        campaign_id=campaign_id or campaign_dir.name,
        date=date,
        git_commit=git_commit,
        scenario_matrix=scenario_matrix,
        scenario_matrix_hash=scenario_matrix_hash,
        episodes=episodes,
        reports=reports,
        manifest_sha256_present=sha256_present,
        incomplete_reasons=tuple(reasons),
        remote_base=remote,
    )


def scan_campaigns(evidence_root: Path) -> list[CampaignRow]:
    """Scan the evidence registry and return rows sorted by campaign id.

    A directory is a campaign candidate when it contains
    ``campaign/campaign_manifest.json`` (nested) or ``campaign_manifest.json``
    (flat). Directories without either are silently ignored (the registry also
    holds many non-campaign evidence bundles).

    ``evidence_root`` is interpreted relative to the repo root (the working
    directory when the tool runs), so all stored link paths stay repo-relative
    and rendered links resolve correctly against an atlas under the repo root.
    """
    rows: list[CampaignRow] = []
    if not evidence_root.is_dir():
        return rows
    for entry in sorted(evidence_root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        has_manifest = (entry / MANIFEST_REL).is_file() or (entry / FLAT_MANIFEST_REL).is_file()
        if not has_manifest:
            continue
        evidence_rel = entry.as_posix()
        rows.append(resolve_campaign(entry, evidence_rel))
    rows.sort(key=lambda row: (row.campaign_id, row.campaign_dir_name))
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _escape_cell(value: str) -> str:
    """Escape Markdown table separators and collapse newlines in a cell."""
    return value.replace("|", "\\|").replace("\n", " ")


def _link_target(rel_repo_path: str, atlas_output: Path) -> str:
    """Return a link target relative to the atlas file's directory.

    Both paths are repo-relative; the atlas is assumed to live under the repo
    root (e.g. ``docs/benchmarks/``). Falls back to the repo-relative path when
    no clean relative path exists (e.g. different drive).
    """
    try:
        rel = os.path.relpath(rel_repo_path, start=atlas_output.parent.as_posix())
    except ValueError:
        return rel_repo_path
    # Normalize for markdown URLs (forward slashes).
    return Path(rel).as_posix()


def _status_cell(row: CampaignRow) -> str:
    """Render the status column, annotating INCOMPLETE with its reasons."""
    if row.status == "OK":
        return "OK"
    reasons = "; ".join(row.incomplete_reasons) if row.incomplete_reasons else "unknown"
    return f"**INCOMPLETE** ({_escape_cell(reasons)})"


def _campaign_id_cell(row: CampaignRow, atlas_output: Path) -> str:
    """Render the campaign id as a link to the campaign evidence directory."""
    target = _link_target(row.evidence_rel, atlas_output)
    cid = _escape_cell(row.campaign_id)
    return f"[`{cid}`]({target})"


def _commit_cell(row: CampaignRow) -> str:
    """Render the git commit as a short hash linking to the commit on the web."""
    if not row.git_commit:
        return "—"
    short = _short_commit(row.git_commit)
    url = f"{row.remote_base}/commit/{row.git_commit}"
    return f"[`{short}`]({url})"


def _scenario_matrix_cell(row: CampaignRow) -> str:
    """Render the scenario matrix path plus its hash."""
    if not row.scenario_matrix:
        return "—"
    matrix = _escape_cell(row.scenario_matrix)
    if row.scenario_matrix_hash:
        return f"`{matrix}` `{row.scenario_matrix_hash}`"
    return f"`{matrix}`"


def _reports_cell(row: CampaignRow, atlas_output: Path) -> str:
    """Render key report links relative to the atlas, joined with ``<br>``."""
    if not row.reports:
        return "—"
    links = []
    for report_rel in row.reports:
        name = Path(report_rel).name
        target = _link_target(report_rel, atlas_output)
        links.append(f"[`{name}`]({target})")
    return "<br>".join(links)


def render_atlas(rows: Sequence[CampaignRow], *, atlas_output: Path) -> str:
    """Render the full deterministic Markdown atlas for the given rows."""
    ok_count = sum(1 for r in rows if r.status == "OK")
    incomplete_count = len(rows) - ok_count
    lines: list[str] = [
        "<!-- AI-GENERATED (robot_sf campaign atlas) - DO NOT EDIT; regenerate with",
        "     uv run python scripts/tools/generate_campaign_atlas.py",
        "     and commit the result. Deterministic: sorted rows, no timestamps. -->",
        "",
        "# Campaign Atlas",
        "",
        CLAIM_BOUNDARY,
        "",
        f"- Campaigns indexed: **{len(rows)}**",
        f"- OK: **{ok_count}**, INCOMPLETE: **{incomplete_count}**",
        "",
        "Each row points to a campaign bundle under `docs/context/evidence/`. "
        "Columns are neutral provenance fields: identifier, date, producing git "
        "commit, scenario matrix plus hash, episode count, key report links, and "
        "whether a `manifest.sha256` attests the bundle contents.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for row in rows:
        cells = {
            "status": _status_cell(row),
            "campaign_id": _campaign_id_cell(row, atlas_output),
            "date": row.date or "—",
            "git_commit": _commit_cell(row),
            "scenario_matrix": _scenario_matrix_cell(row),
            "episodes": row.episodes,
            "key_reports": _reports_cell(row, atlas_output),
            "manifest_sha256": "yes" if row.manifest_sha256_present else "no",
        }
        lines.append("| " + " | ".join(cells[col] for col in COLUMNS) + " |")
    lines.append("")
    lines.append(f"> {CLAIM_BOUNDARY}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT,
        help=f"Evidence registry root to scan (default: {DEFAULT_EVIDENCE_ROOT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output Markdown path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if the committed atlas differs from the generated one.",
    )
    return parser


def generate(evidence_root: Path, output: Path) -> tuple[str, int]:
    """Scan the registry and return ``(rendered_markdown, row_count)``."""
    rows = scan_campaigns(evidence_root)
    rendered = render_atlas(rows, atlas_output=output)
    return rendered, len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: write or check the campaign atlas."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    rendered, row_count = generate(args.evidence_root, args.output)

    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
        if current != rendered:
            print(
                f"ERROR: campaign atlas is stale or missing: {args.output}\n"
                "       regenerate with: uv run python scripts/tools/generate_campaign_atlas.py",
                file=sys.stderr,
            )
            return 1
        print(f"Campaign atlas is up to date: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote campaign atlas with {row_count} row(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
