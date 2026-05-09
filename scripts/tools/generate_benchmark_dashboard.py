#!/usr/bin/env python3
"""Generate a self-contained static dashboard from a benchmark bundle."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from robot_sf.common.artifact_paths import get_repository_root

DASHBOARD_SCHEMA_VERSION = "benchmark-static-dashboard.v1"
DEFAULT_TITLE = "Benchmark Dashboard"
REPORT_DOWNLOADS = (
    "campaign_summary_json",
    "campaign_table_csv",
    "campaign_table_md",
    "campaign_report_md",
    "matrix_summary_json",
    "scenario_breakdown_csv",
    "scenario_family_breakdown_csv",
    "snqi_diagnostics_json",
)
PRIMARY_METRICS = (
    "success_mean",
    "collisions_mean",
    "near_misses_mean",
    "time_to_goal_norm_mean",
    "path_efficiency_mean",
    "snqi_mean",
)


@dataclass(frozen=True)
class DashboardBundle:
    """Loaded benchmark bundle data needed by the dashboard."""

    bundle_root: Path
    summary_path: Path
    campaign: dict[str, Any]
    planner_rows: list[dict[str, Any]]
    warnings: list[str]
    artifacts: dict[str, Any]


def _utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _repo_relative(path: Path) -> str:
    """Return a repository-relative path where possible."""
    repo_root = get_repository_root().resolve()
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _slug(value: Any, *, fallback: str = "planner") -> str:
    """Create a stable URL-safe slug."""
    text = str(value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-._")
    return text or fallback


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_dashboard_bundle(bundle_root: Path) -> DashboardBundle:
    """Load the supported camera-ready campaign summary from a bundle root."""
    root = bundle_root.resolve()
    summary_path = root / "reports" / "campaign_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "Expected camera-ready campaign summary at "
            f"{summary_path}. Run the generator against a campaign bundle root."
        )
    payload = _read_json(summary_path)
    campaign = payload.get("campaign")
    planner_rows = payload.get("planner_rows")
    warnings = payload.get("warnings", [])
    artifacts = payload.get("artifacts", {})
    if not isinstance(campaign, dict):
        raise ValueError("campaign_summary.json is missing object field 'campaign'")
    if not isinstance(planner_rows, list):
        raise ValueError("campaign_summary.json is missing list field 'planner_rows'")
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
    if not isinstance(artifacts, dict):
        artifacts = {}
    rows = [row for row in planner_rows if isinstance(row, dict)]
    return DashboardBundle(
        bundle_root=root,
        summary_path=summary_path,
        campaign=campaign,
        planner_rows=rows,
        warnings=[str(item) for item in warnings],
        artifacts=artifacts,
    )


def _metric(row: dict[str, Any], key: str) -> float | None:
    """Return a numeric metric value from a planner row."""
    value = row.get(key)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed
    return None


def _format_value(value: Any, *, digits: int = 3) -> str:
    """Format scalar values for compact dashboard display."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float):
        return f"{float(value):.{digits}f}"
    if isinstance(value, str):
        try:
            return f"{float(value):.{digits}f}"
        except ValueError:
            return value
    return str(value)


def _status_class(row: dict[str, Any]) -> str:
    """Return CSS class for planner status."""
    if str(row.get("benchmark_success", "")).lower() == "true":
        return "ok"
    if str(row.get("availability_status", "")).lower() == "not_available":
        return "warn"
    return "fail"


def _copy_downloads(bundle: DashboardBundle, out_dir: Path) -> dict[str, str]:
    """Copy compact report downloads into the dashboard directory."""
    downloads: dict[str, str] = {}
    downloads_dir = out_dir / "downloads"
    for key in REPORT_DOWNLOADS:
        raw = bundle.artifacts.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        source = (get_repository_root() / raw).resolve()
        if not source.exists():
            source = (bundle.bundle_root / raw).resolve()
        if not source.exists() or not source.is_file():
            continue
        downloads_dir.mkdir(parents=True, exist_ok=True)
        target = downloads_dir / source.name
        shutil.copy2(source, target)
        downloads[key] = target.relative_to(out_dir).as_posix()
    return downloads


def _planner_identity(row: dict[str, Any]) -> str:
    """Return a stable planner display identifier."""
    return str(row.get("planner_key") or row.get("algo") or "planner")


def build_dashboard_payload(
    bundle: DashboardBundle, *, title: str, downloads: dict[str, str]
) -> dict:
    """Build the normalized dashboard data payload."""
    planner_rows = sorted(bundle.planner_rows, key=_planner_identity)
    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "title": title,
        "source": {
            "bundle_root": _repo_relative(bundle.bundle_root),
            "summary_path": _repo_relative(bundle.summary_path),
        },
        "campaign": bundle.campaign,
        "planner_rows": planner_rows,
        "warnings": bundle.warnings,
        "downloads": downloads,
    }


def _write_css(out_dir: Path) -> str:
    """Write local dashboard CSS."""
    css = """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --ink: #1f2933;
  --muted: #5d6875;
  --line: #d8dee6;
  --accent: #116d6e;
  --ok: #177245;
  --warn: #986500;
  --fail: #b42318;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
a { color: var(--accent); }
header, main { max-width: 1180px; margin: 0 auto; padding: 24px; }
header { padding-bottom: 8px; }
h1 { margin: 0 0 8px; font-size: 32px; letter-spacing: 0; }
h2 { margin: 28px 0 12px; font-size: 20px; letter-spacing: 0; }
.subtle { color: var(--muted); }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.card, .panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.label { color: var(--muted); font-size: 12px; text-transform: uppercase; }
.value { font-size: 24px; font-weight: 650; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; background: var(--panel); }
th, td { border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; }
th { color: var(--muted); font-size: 12px; text-transform: uppercase; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge { display: inline-block; border-radius: 999px; padding: 2px 8px; color: #fff; font-size: 12px; }
.ok { background: var(--ok); }
.warn { background: var(--warn); }
.fail { background: var(--fail); }
.bar { min-width: 90px; height: 8px; background: #e8edf2; border-radius: 999px; overflow: hidden; }
.bar span { display: block; height: 100%; background: var(--accent); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
.downloads { columns: 2 260px; }
.warning { border-left: 4px solid var(--warn); }
footer { color: var(--muted); padding: 24px; text-align: center; }
""".strip()
    asset_dir = out_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    path = asset_dir / "dashboard.css"
    path.write_text(css + "\n", encoding="utf-8")
    return path.relative_to(out_dir).as_posix()


def _card(label: str, value: Any) -> str:
    """Render one metric card."""
    return (
        '<div class="card">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(_format_value(value))}</div>'
        "</div>"
    )


def _success_bar(value: float | None) -> str:
    """Render a compact horizontal success bar."""
    width = 0.0 if value is None else max(0.0, min(1.0, value)) * 100.0
    return f'<div class="bar"><span style="width:{width:.1f}%"></span></div>'


def _html_shell(*, title: str, css_path: str, body: str) -> str:
    """Wrap page body in a complete HTML document."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <link rel="stylesheet" href="{escape(css_path)}">
</head>
<body>
{body}
</body>
</html>
"""


def _render_index(payload: dict[str, Any], *, css_path: str) -> str:
    """Render the main dashboard page."""
    campaign = payload["campaign"]
    rows = payload["planner_rows"]
    cards = "".join(
        [
            _card("Campaign", campaign.get("campaign_id") or campaign.get("name")),
            _card("Episodes", campaign.get("total_episodes")),
            _card("Runs", f"{campaign.get('successful_runs', 0)}/{campaign.get('total_runs', 0)}"),
            _card("Benchmark Success", campaign.get("benchmark_success")),
        ]
    )
    table_rows = []
    for row in rows:
        name = _planner_identity(row)
        slug = _slug(name)
        success = _metric(row, "success_mean")
        table_rows.append(
            "<tr>"
            f'<td><a href="planners/{escape(slug)}.html">{escape(name)}</a></td>'
            f"<td>{escape(str(row.get('algo', '')))}</td>"
            f'<td><span class="badge {_status_class(row)}">{escape(str(row.get("status", "unknown")))}</span></td>'
            f"<td>{escape(str(row.get('readiness_status', row.get('execution_mode', 'unknown'))))}</td>"
            f'<td class="num">{escape(_format_value(success))}</td>'
            f"<td>{_success_bar(success)}</td>"
            f'<td class="num">{escape(_format_value(row.get("collisions_mean")))}</td>'
            f'<td class="num">{escape(_format_value(row.get("near_misses_mean")))}</td>'
            f'<td class="num">{escape(_format_value(row.get("snqi_mean")))}</td>'
            "</tr>"
        )
    downloads = "".join(
        f'<li><a href="{escape(path)}">{escape(label)}</a></li>'
        for label, path in payload["downloads"].items()
    )
    warnings = "".join(
        f'<div class="panel warning">{escape(item)}</div>' for item in payload["warnings"]
    )
    body = f"""
<header>
  <h1>{escape(payload["title"])}</h1>
  <div class="subtle">Generated {escape(payload["generated_at_utc"])} from {escape(payload["source"]["summary_path"])}</div>
</header>
<main>
  <section class="cards">{cards}</section>
  <h2>Planner Summary</h2>
  <table>
    <thead><tr><th>Planner</th><th>Algo</th><th>Status</th><th>Readiness</th><th class="num">Success</th><th></th><th class="num">Collisions</th><th class="num">Near Misses</th><th class="num">SNQI</th></tr></thead>
    <tbody>{"".join(table_rows)}</tbody>
  </table>
  <h2>Downloads</h2>
  <div class="panel"><ul class="downloads">{downloads or "<li>No compact report downloads were found.</li>"}</ul></div>
  <h2>Warnings</h2>
  <div class="grid">{warnings or '<div class="panel">No campaign warnings reported.</div>'}</div>
</main>
<footer>Static dashboard; no backend or external assets required.</footer>
"""
    return _html_shell(title=str(payload["title"]), css_path=css_path, body=body)


def _render_planner_page(
    payload: dict[str, Any],
    row: dict[str, Any],
    *,
    css_path: str,
) -> str:
    """Render one per-planner page."""
    name = _planner_identity(row)
    metric_cards = "".join(
        _card(metric.replace("_", " "), row.get(metric)) for metric in PRIMARY_METRICS
    )
    detail_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in sorted(row.items())
        if not isinstance(value, dict | list)
    )
    body = f"""
<header>
  <h1>{escape(name)}</h1>
  <div class="subtle"><a href="../index.html">Back to summary</a></div>
</header>
<main>
  <section class="cards">{metric_cards}</section>
  <h2>Planner Details</h2>
  <table><tbody>{detail_rows}</tbody></table>
</main>
<footer>Source dashboard: {escape(payload["source"]["summary_path"])}</footer>
"""
    return _html_shell(title=f"{name} - {payload['title']}", css_path=css_path, body=body)


def write_dashboard(bundle: DashboardBundle, out_dir: Path, *, title: str) -> dict[str, Any]:
    """Write dashboard files and return the manifest payload."""
    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    downloads = _copy_downloads(bundle, out_dir)
    payload = build_dashboard_payload(bundle, title=title, downloads=downloads)
    css_path = _write_css(out_dir)

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "dashboard_data.json"
    data_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    index_path = out_dir / "index.html"
    index_path.write_text(_render_index(payload, css_path=css_path), encoding="utf-8")

    planner_dir = out_dir / "planners"
    planner_dir.mkdir(parents=True, exist_ok=True)
    planner_paths = []
    for row in payload["planner_rows"]:
        planner_path = planner_dir / f"{_slug(_planner_identity(row))}.html"
        planner_path.write_text(
            _render_planner_page(payload, row, css_path="../assets/dashboard.css"),
            encoding="utf-8",
        )
        planner_paths.append(planner_path.relative_to(out_dir).as_posix())

    files = [
        index_path.relative_to(out_dir).as_posix(),
        css_path,
        data_path.relative_to(out_dir).as_posix(),
        *planner_paths,
        *downloads.values(),
    ]
    manifest = {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "generated_at_utc": payload["generated_at_utc"],
        "source": payload["source"],
        "output_dir": _repo_relative(out_dir),
        "entrypoint": "index.html",
        "planner_pages": planner_paths,
        "files": sorted(set(files)),
        "self_contained": True,
    }
    manifest_path = out_dir / "dashboard_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    bundle = load_dashboard_bundle(args.bundle_root)
    manifest = write_dashboard(bundle, args.out, title=args.title)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
