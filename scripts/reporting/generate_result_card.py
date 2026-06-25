#!/usr/bin/env python3
"""Generate dissertation-oriented result cards from accepted evidence summaries."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

DECISIONS = ("promote", "diagnostic", "blocked", "stopped")
SCHEMA = "robot_sf.result_card.v1"


@dataclass(frozen=True)
class ResultCard:
    """Structured result-card payload rendered to Markdown and JSON."""

    title: str
    source_summary: str
    evidence_tier: str
    decision: str
    comparator: str
    claim_boundary: str
    metrics: dict[str, Any]
    commands: list[str]
    artifacts: list[str]
    caveats: list[str]
    non_transfer_notes: list[str]

    def to_json(self) -> dict[str, Any]:
        """Return the stable JSON representation."""
        return {
            "schema": SCHEMA,
            "title": self.title,
            "source_summary": self.source_summary,
            "evidence_tier": self.evidence_tier,
            "decision": self.decision,
            "comparator": self.comparator,
            "claim_boundary": self.claim_boundary,
            "metrics": self.metrics,
            "commands": self.commands,
            "artifacts": self.artifacts,
            "caveats": self.caveats,
            "non_transfer_notes": self.non_transfer_notes,
        }


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _flatten_numeric(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, bool):
        out[prefix] = value
    elif isinstance(value, int | float) and math.isfinite(float(value)):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, dict | int | float | bool):
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                _flatten_numeric(next_prefix, child, out)


def _parse_cli_metrics(cli_metrics: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in cli_metrics:
        if "=" not in metric:
            raise ValueError(f"Metric must be name=value: {metric}")
        key, raw_value = metric.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Metric name is empty: {metric}")
        try:
            value: Any = float(raw_value)
        except ValueError:
            value = raw_value.strip()
        else:
            if not math.isfinite(value):
                raise ValueError(f"Metric value must be finite: {metric}")
        metrics[key] = value
    return metrics


def _extract_top_level_metrics(summary: dict[str, Any], metrics: dict[str, Any]) -> None:
    for key, value in summary.items():
        if key in {"issue"} or isinstance(value, list):
            continue
        if key.endswith("_count") or key.endswith("_s") or key.endswith("_rate"):
            _flatten_numeric(key, value, metrics)
        elif isinstance(value, int | float | bool):
            _flatten_numeric(key, value, metrics)
        elif isinstance(value, dict) and (
            "count" in key or "summary" in key or "status" in key or "required" in key
        ):
            _flatten_numeric(key, value, metrics)


def _extract_variant_metrics(summary: dict[str, Any], metrics: dict[str, Any]) -> None:
    rows = summary.get("rows")
    if isinstance(rows, list):
        metrics.setdefault("row_count", len(rows))
        for row in rows:
            if not isinstance(row, dict):
                continue
            variant_results = row.get("variant_results")
            if not isinstance(variant_results, dict):
                continue
            for variant_name, variant in variant_results.items():
                if not isinstance(variant, dict):
                    continue
                closed_loop = variant.get("closed_loop_metrics")
                if isinstance(closed_loop, dict):
                    for metric_name, metric_value in closed_loop.items():
                        metric_key = f"variant.{variant_name}.{metric_name}"
                        _flatten_numeric(metric_key, metric_value, metrics)


def _extract_comparison_metrics(summary: dict[str, Any], metrics: dict[str, Any]) -> None:
    for prefix in ("baseline", "after", "change"):
        value = summary.get(prefix)
        if isinstance(value, dict):
            _flatten_numeric(prefix, value, metrics)


def _extract_metrics(summary: dict[str, Any], cli_metrics: list[str]) -> dict[str, Any]:
    metrics = _parse_cli_metrics(cli_metrics)
    _extract_top_level_metrics(summary, metrics)
    _extract_variant_metrics(summary, metrics)
    _extract_comparison_metrics(summary, metrics)
    return dict(sorted(metrics.items()))


def _extract_commands(summary: dict[str, Any], cli_commands: list[str]) -> list[str]:
    commands = list(cli_commands)
    for key in ("source_command", "command", "validation_command", "command_shape"):
        commands.extend(_string_list(summary.get(key)))
    commands.extend(_string_list(summary.get("commands")))
    validation = summary.get("validation")
    if isinstance(validation, list):
        for item in validation:
            if isinstance(item, dict):
                commands.extend(_string_list(item.get("command")))
    return _dedupe(commands)


def _has_exact_command(commands: list[str]) -> bool:
    vague_markers = ("see readme", "see docs", "n/a", "unknown", "not recorded")
    command_prefixes = ("uv ", "python ", "bash ", "scripts/", "./scripts/")
    for command in commands:
        normalized = command.strip().lower()
        if not normalized or any(marker in normalized for marker in vague_markers):
            continue
        if normalized.startswith(command_prefixes) or " --" in normalized:
            return True
    return False


def _extract_artifacts(
    summary: dict[str, Any], source_summary: Path, cli_artifacts: list[str]
) -> list[str]:
    artifacts = [source_summary.as_posix(), *cli_artifacts]
    for key in ("manifest", "episodes_source", "source", "trace"):
        artifacts.extend(_string_list(summary.get(key)))
    artifacts.extend(_string_list(summary.get("artifacts")))
    rows = summary.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                artifacts.extend(_string_list(row.get("trace")))
    return _dedupe(artifacts)


def _extract_caveats(summary: dict[str, Any], cli_caveats: list[str]) -> list[str]:
    caveats = [*cli_caveats]
    interpretation = summary.get("interpretation")
    if isinstance(interpretation, dict):
        caveats.extend(_string_list(interpretation.get("classification")))
        caveats.extend(_string_list(interpretation.get("caveat")))
    else:
        caveats.extend(_string_list(interpretation))
    caveats.extend(_string_list(summary.get("caveats")))
    rows = summary.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                caveats.extend(_string_list(row.get("caveats")))
    return _dedupe(caveats)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = value.strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _looks_local_output(path_text: str) -> bool:
    normalized = path_text.replace("\\", "/").strip()
    lower_normalized = normalized.lower()
    if "<local-output>" in normalized or "worktree-local" in lower_normalized:
        return True
    parts = PurePosixPath(normalized).parts
    return "output" in parts


def _validate_summary_status(summary: dict[str, Any], decision: str) -> None:
    if decision not in {"promote", "diagnostic"}:
        return
    status = str(summary.get("status", "")).strip().lower()
    if status and status not in {"pass", "passed", "ok", "success"}:
        raise ValueError(f"Summary status is not accepted for {decision}: {status}")
    validation_errors = summary.get("validation_errors")
    if validation_errors:
        raise ValueError("Summary has validation_errors; use blocked/stopped or fix evidence first")
    validation = summary.get("validation")
    if isinstance(validation, list):
        failed = [
            item
            for item in validation
            if isinstance(item, dict) and item.get("exit_code") not in {0, None}
        ]
        if failed:
            raise ValueError("Summary validation contains nonzero exit_code entries")


def build_result_card(args: argparse.Namespace) -> ResultCard:
    """Build and validate a result card from CLI arguments."""
    summary = _load_summary(args.summary_json)
    _validate_summary_status(summary, args.decision)
    claim_boundary = str(args.claim_boundary or summary.get("claim_boundary") or "").strip()
    if not claim_boundary:
        raise ValueError(
            "Missing claim boundary; provide --claim-boundary or summary.claim_boundary"
        )

    evidence_tier = args.evidence_tier.strip()
    if not evidence_tier:
        raise ValueError("--evidence-tier is required")

    comparator = args.comparator.strip()
    if not comparator:
        raise ValueError("--comparator is required")

    metrics = _extract_metrics(summary, args.metric)
    if not metrics:
        raise ValueError("No metrics found; provide at least one --metric name=value")

    title = args.title or str(
        summary.get("title") or f"Issue {summary.get('issue', 'unknown')} result card"
    )
    commands = _extract_commands(summary, args.command)
    if not commands:
        raise ValueError("Missing exact command provenance; provide --command or source_command")
    if not _has_exact_command(commands):
        raise ValueError("Command provenance must include an exact command, not only a doc pointer")

    artifacts = _extract_artifacts(summary, args.summary_json, args.artifact)
    local_only = [path for path in artifacts if _looks_local_output(path)]
    if local_only and not args.allow_local_output_with_durable_pointer:
        raise ValueError(
            "Local-only output references require --allow-local-output-with-durable-pointer: "
            + ", ".join(local_only)
        )

    caveats = _extract_caveats(summary, args.caveat)
    non_transfer_notes = _dedupe([*args.non_transfer_note])
    if not caveats:
        raise ValueError(
            "Missing caveats/non-transfer context; provide --caveat or summary caveats"
        )

    return ResultCard(
        title=title,
        source_summary=args.summary_json.as_posix(),
        evidence_tier=evidence_tier,
        decision=args.decision,
        comparator=comparator,
        claim_boundary=claim_boundary,
        metrics=metrics,
        commands=commands,
        artifacts=artifacts,
        caveats=caveats,
        non_transfer_notes=non_transfer_notes,
    )


def render_markdown(card: ResultCard) -> str:
    """Render a result card as dissertation-ready Markdown."""
    lines = [
        f"# {card.title}",
        "",
        "## Decision",
        "",
        f"- Evidence tier: `{card.evidence_tier}`",
        f"- Final decision: `{card.decision}`",
        f"- Comparator: {card.comparator}",
        "",
        "## Claim Boundary",
        "",
        card.claim_boundary,
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, value in card.metrics.items():
        lines.append(f"| `{key}` | {json.dumps(value, sort_keys=True)} |")
    lines.extend(["", "## Command Provenance", ""])
    lines.extend(f"- `{command}`" for command in card.commands)
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- `{artifact}`" for artifact in card.artifacts)
    lines.extend(["", "## Caveats And Non-Transfer Notes", ""])
    lines.extend(f"- {caveat}" for caveat in card.caveats)
    lines.extend(f"- {note}" for note in card.non_transfer_notes)
    return "\n".join(lines).rstrip() + "\n"


def render_latex_table(card: ResultCard) -> str:
    """Render a compact LaTeX tabular snippet for the metric list."""
    rows = ["\\begin{tabular}{ll}", "Metric & Value \\\\", "\\hline"]
    for key, value in card.metrics.items():
        rows.append(f"{key.replace('_', '\\_')} & {str(value).replace('_', '\\_')} \\\\")
    rows.append("\\end{tabular}")
    return "\n".join(rows) + "\n"


def write_outputs(card: ResultCard, output_dir: Path, include_latex: bool) -> dict[str, str]:
    """Write result-card artifacts and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "result_card.json"
    md_path = output_dir / "result_card.md"
    json_path.write_text(
        json.dumps(card.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(card), encoding="utf-8")
    outputs = {"json": json_path.as_posix(), "markdown": md_path.as_posix()}
    if include_latex:
        tex_path = output_dir / "result_card_table.tex"
        tex_path.write_text(render_latex_table(card), encoding="utf-8")
        outputs["latex_table"] = tex_path.as_posix()
    return outputs


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--evidence-tier", required=True)
    parser.add_argument("--decision", choices=DECISIONS, required=True)
    parser.add_argument("--comparator", required=True)
    parser.add_argument("--claim-boundary")
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--caveat", action="append", default=[])
    parser.add_argument("--non-transfer-note", action="append", default=[])
    parser.add_argument("--latex-table", action="store_true")
    parser.add_argument(
        "--allow-local-output-with-durable-pointer",
        action="store_true",
        help=(
            "Allow local output references only when the caller has also supplied a durable pointer "
            "or tracked evidence copy in --artifact."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        card = build_result_card(args)
        outputs = write_outputs(card, args.output_dir, args.latex_table)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"schema": SCHEMA, "outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
