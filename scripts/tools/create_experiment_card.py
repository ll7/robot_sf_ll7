#!/usr/bin/env python3
"""Generate a question-first experiment card draft for the experiment registry."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tools.validate_experiment_registry import (  # noqa: E402
    VALID_EVIDENCE_GRADES,
    VALID_PAPER_RELEVANCE,
    validate_registry,
)

REQUIRED_RECORD_FIELDS = (
    "experiment_id",
    "issue",
    "issue_url",
    "question",
    "hypothesis",
    "config",
    "command",
    "inputs",
    "outputs",
    "expected_artifacts",
    "evidence_grade",
    "paper_relevance",
    "status",
)

TODO_TRACKED_FIELDS = REQUIRED_RECORD_FIELDS + ("early_stop_criteria",)

TEMPLATE_NAMES = ("benchmark-analysis", "planner-ablation", "figure-table-pack")

ISSUE_URL_TEMPLATE = "https://github.com/ll7/robot_sf_ll7/issues/{issue}"

# Paths referenced by templates
_OUTPUT_EXPERIMENTS = "output/experiments"


@dataclass
class TemplateDef:
    """Holds template content before rendering into an experiment record."""

    question: str
    hypothesis: str
    config: list[str]
    command: str
    inputs: list[str] | list[dict[str, str]]
    outputs: list[dict[str, str]]
    expected_artifacts: list[dict[str, str]]
    early_stop_criteria: dict[str, str]
    evidence_grade: str
    paper_relevance: str
    status: str
    notes: str
    todo_fields: set[str] = field(default_factory=set)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--issue",
        type=str,
        required=True,
        help="GitHub issue number (e.g. 2103).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Stable experiment identifier (snake_case, e.g. issue_2103_example).",
    )
    parser.add_argument(
        "--template",
        choices=TEMPLATE_NAMES,
        required=True,
        help="Experiment card template to materialize.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output directory for this experiment card. "
            "Default: output/experiments/<experiment-id>."
        ),
    )
    parser.add_argument(
        "--issue-url",
        type=str,
        default="",
        help="Override the generated issue URL (default: derived from --issue).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Write the draft without running the registry validator.",
    )
    return parser


def _resolve_issue_url(issue: str, override: str) -> str:
    if override:
        return override
    return ISSUE_URL_TEMPLATE.format(issue=issue)


def _template_benchmark_analysis(experiment_id: str, output_root: Path) -> TemplateDef:
    output_path = output_root.as_posix()
    return TemplateDef(
        question=(
            "TODO: How does {candidate} compare against {baseline} on {scenario_set} "
            "in terms of {primary_metric} and {safety_metric}?"
        ),
        hypothesis=(
            "TODO: {candidate} will improve {primary_metric} by at least {expected_delta} "
            "without regressing {safety_metric} compared to {baseline}."
        ),
        config=[
            "TODO: path/to/candidate/config.yaml",
            "TODO: path/to/baseline/config.yaml",
            "TODO: path/to/eval/scenario_set.yaml",
        ],
        command=(
            "uv run python scripts/evaluation/run_benchmark.py \\\n"
            "  --candidate-config TODO \\\n"
            "  --baseline-config TODO \\\n"
            "  --scenario-set TODO \\\n"
            "  --seeds 111 112 113 \\\n"
            "  --output-root " + output_path + "\n"
            "# Then: uv run python scripts/tools/analyze_benchmark_comparison.py --input "
            + output_path
        ),
        inputs=[
            "TODO: path/to/candidate/config.yaml",
            "TODO: path/to/baseline/config.yaml",
        ],
        outputs=[
            {
                "path": output_path,
                "durable_reference": None,
                "evidence_role": "local staging only until promoted",
            },
        ],
        expected_artifacts=[
            {
                "name": "benchmark_summary_table",
                "artifact_id": "tab_benchmark_summary",
                "path": output_path + "/summary.yaml",
                "durable_reference_required": True,
            },
            {
                "name": "metric_comparison_figure",
                "artifact_id": "fig_metric_comparison",
                "path": output_path + "/metric_comparison.png",
                "durable_reference_required": True,
            },
            {
                "name": "episode_traces",
                "artifact_id": "episode_traces_archive",
                "path": output_path + "/traces/",
                "durable_reference_required": False,
            },
        ],
        early_stop_criteria=_early_stop_criteria_template(),
        evidence_grade="proposal",
        paper_relevance="exploratory",
        status="planned",
        notes=(
            "TODO: Replace placeholders in curly braces before running. "
            "Local output paths are not durable evidence until promoted "
            "to W&B, release storage, or the tracked evidence directory."
        ),
        todo_fields={"question", "hypothesis", "config", "command", "inputs"},
    )


def _template_planner_ablation(experiment_id: str, output_root: Path) -> TemplateDef:
    output_path = output_root.as_posix()
    return TemplateDef(
        question=(
            "TODO: Does {planner_variant} change {metric} compared to {baseline_planner} "
            "on the {scenario_set} benchmark?"
        ),
        hypothesis=(
            "TODO: {planner_variant} improves {metric} by at least {expected_delta} "
            "without reducing {safety_metric} relative to {baseline_planner}."
        ),
        config=[
            "TODO: path/to/planner_config.yaml",
            "TODO: path/to/baseline_planner_config.yaml",
            "TODO: path/to/scenarios.yaml",
        ],
        command=(
            "uv run python scripts/evaluation/ablate_planner.py \\\n"
            "  --planner-config TODO \\\n"
            "  --baseline-config TODO \\\n"
            "  --scenario-set TODO \\\n"
            "  --seeds 111 112 113 \\\n"
            "  --output-root " + output_path + "\n"
            "# Then: uv run python scripts/tools/render_planner_comparison.py --input "
            + output_path
        ),
        inputs=[
            "TODO: path/to/planner_config.yaml",
            "TODO: path/to/baseline_planner_config.yaml",
        ],
        outputs=[
            {
                "path": output_path,
                "durable_reference": None,
                "evidence_role": "local staging only until promoted",
            },
        ],
        expected_artifacts=[
            {
                "name": "planner_comparison_table",
                "artifact_id": "tab_planner_comparison",
                "path": output_path + "/comparison.yaml",
                "durable_reference_required": True,
            },
            {
                "name": "planner_status_figure",
                "artifact_id": "fig_planner_status_summary",
                "path": output_path + "/planner_status.png",
                "durable_reference_required": True,
            },
            {
                "name": "simulation_trace_bundle",
                "artifact_id": "trace_bundle_gz",
                "path": output_path + "/traces.tar.gz",
                "durable_reference_required": False,
            },
        ],
        early_stop_criteria=_early_stop_criteria_template(),
        evidence_grade="proposal",
        paper_relevance="exploratory",
        status="planned",
        notes=(
            "TODO: Replace placeholders in curly braces before running. "
            "Planner ablation must include at least three seeds per condition. "
            "Document fallback/degraded modes in the final comparison table."
        ),
        todo_fields={"question", "hypothesis", "config", "command", "inputs"},
    )


def _template_figure_table_pack(experiment_id: str, output_root: Path) -> TemplateDef:
    output_path = output_root.as_posix()
    record_path = (output_root / f"{experiment_id}.yaml").as_posix()
    return TemplateDef(
        question=(
            "TODO: What publication-ready figures and tables can be produced from "
            "{source_experiment_id} to support {target_claim}?"
        ),
        hypothesis=(
            "TODO: The existing results in {source_experiment_id} support {target_claim} "
            "when rendered as {figure_kind} and {table_kind}."
        ),
        config=[
            "TODO: path/to/source_experiment_config.yaml",
            "TODO: path/to/visualization_config.yaml",
        ],
        command=(
            "uv run python scripts/tools/render_figure_table_pack.py \\\n"
            "  --source-experiment-id TODO \\\n"
            "  --output-root " + output_path + " \\\n"
            "  --figures TODO \\\n"
            "  --tables TODO\n"
            "# Then validate: uv run python scripts/tools/validate_experiment_registry.py \\\n"
            "#   " + record_path
        ),
        inputs=[
            "TODO: path/to/source_experiment_record.yaml",
        ],
        outputs=[
            {
                "path": output_path,
                "durable_reference": None,
                "evidence_role": "local staging only until promoted",
            },
        ],
        expected_artifacts=[
            {
                "name": "publication_figure",
                "artifact_id": "fig_benchmark_outcome_matrix",
                "path": output_path + "/figures/outcome_matrix.pdf",
                "durable_reference_required": True,
            },
            {
                "name": "summary_table",
                "artifact_id": "tab_campaign_table",
                "path": output_path + "/tables/campaign_table.csv",
                "durable_reference_required": True,
            },
            {
                "name": "artifact_catalog",
                "artifact_id": "artifact_catalog_yaml",
                "path": output_path + "/artifact_catalog.yaml",
                "durable_reference_required": True,
            },
        ],
        early_stop_criteria={},
        evidence_grade="proposal",
        paper_relevance="exploratory",
        status="planned",
        notes=(
            "TODO: Replace placeholders in curly braces before running. "
            "All figures and tables must carry durable_reference before "
            "paper_relevance can be promoted to paper_facing."
        ),
        todo_fields={"question", "hypothesis", "config", "command", "inputs"},
    )


_TEMPLATES = {
    "benchmark-analysis": _template_benchmark_analysis,
    "planner-ablation": _template_planner_ablation,
    "figure-table-pack": _template_figure_table_pack,
}


def _early_stop_criteria_template() -> dict[str, str]:
    """Return the predeclared early-stop block for Slurm training launch packets."""
    return {
        "metric": "TODO: primary progress metric, e.g. eval/success_rate",
        "threshold": "TODO: minimum acceptable value or trend",
        "check_cadence": "TODO: eval interval or wall-clock cadence",
        "minimum_runtime_or_timesteps": "TODO: do not cancel before this floor",
        "cancel_condition": "TODO: exact condition that justifies cancellation",
        "diagnostic_preservation_action": (
            "TODO: manifest/log/artifact/context note to preserve before cancellation is complete"
        ),
    }


def _build_record(
    experiment_id: str,
    issue: str,
    issue_url: str,
    template_name: str,
    output_root: Path,
) -> dict[str, Any]:
    builder = _TEMPLATES[template_name]
    tpl = builder(experiment_id, output_root)

    record: dict[str, Any] = {
        "schema_version": "experiment-record.v1",
        "experiment_id": experiment_id,
        "issue": issue,
        "issue_url": issue_url,
        "question": tpl.question,
        "hypothesis": tpl.hypothesis,
        "config": tpl.config,
        "command": tpl.command,
        "inputs": tpl.inputs,
        "outputs": tpl.outputs,
        "expected_artifacts": tpl.expected_artifacts,
        "early_stop_criteria": tpl.early_stop_criteria,
        "evidence_grade": tpl.evidence_grade,
        "paper_relevance": tpl.paper_relevance,
        "status": tpl.status,
        "notes": tpl.notes,
    }

    return record


def _contains_todo(value: Any) -> bool:
    """Return True when a scaffold value still contains a TODO placeholder."""
    if isinstance(value, str):
        return "TODO" in value
    if isinstance(value, list):
        return any(_contains_todo(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_todo(item) for item in value.values())
    return False


def _find_todo_fields(record: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for field_name in TODO_TRACKED_FIELDS:
        value = record.get(field_name)
        if value is None:
            fields.append(field_name)
        elif _contains_todo(value):
            fields.append(field_name)
    return fields


def _write_record_yaml(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, default_flow_style=False, sort_keys=False)


def _validate_generated_record(record: dict[str, Any]) -> list[str]:
    """Return scaffold validation errors for a single generated record."""
    errors: list[str] = []
    for field_name in REQUIRED_RECORD_FIELDS:
        value = record.get(field_name)
        if value in (None, "", [], {}):
            errors.append(f"missing required field {field_name!r}")
    if record.get("schema_version") != "experiment-record.v1":
        errors.append("schema_version must be 'experiment-record.v1'")
    if record.get("evidence_grade") not in VALID_EVIDENCE_GRADES:
        errors.append(f"evidence_grade must be one of {sorted(VALID_EVIDENCE_GRADES)}")
    if record.get("paper_relevance") not in VALID_PAPER_RELEVANCE:
        errors.append(f"paper_relevance must be one of {sorted(VALID_PAPER_RELEVANCE)}")
    return errors


def _write_checklist(path: Path, todo_fields: list[str], template_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Experiment Card Checklist - {path.stem}",
        "",
        f"**Template**: {template_name}",
        f"**Generated**: {datetime.now(UTC).date().isoformat()}",
        "",
        "## Validation Checklist",
        "",
        "- [ ] Replace all TODO placeholders with concrete values",
    ]
    if todo_fields:
        lines.append("")
        lines.append("### Fields Requiring Attention")
        for f in todo_fields:
            lines.append(f"  - [ ] `{f}` - contains TODO placeholder")
    lines.extend(
        [
            "",
            "### Registry Validation",
            "",
            "```bash",
            "uv run python scripts/tools/validate_experiment_registry.py experiments/registry.yaml",
            "```",
            "",
            "### Evidence Promotion",
            "",
            "- [ ] Run experiment and verify outputs match `expected_artifacts`",
            "- [ ] Fill `early_stop_criteria` before submitting long Slurm training jobs",
            "- [ ] Promote durable artifacts (W&B / release storage / docs/context/evidence/)",
            "- [ ] Add `durable_reference` to each artifact in the record",
            "- [ ] Update `evidence_grade` (proposal to observed) and `paper_relevance` if applicable",
            "- [ ] Register the record in `experiments/registry.yaml`",
            "",
            "### Local vs Durable Reminder",
            "",
            "  Local `output/` paths are disposable worktree artifacts.",
            "  Until `durable_reference` is set, these are not paper-facing evidence.",
            "",
            "### Slurm Early-Stop Criteria",
            "",
            "Before submitting a long Slurm training job, predeclare:",
            "",
            "- [ ] metric",
            "- [ ] threshold",
            "- [ ] check cadence",
            "- [ ] minimum runtime or timesteps",
            "- [ ] cancel condition",
            "- [ ] diagnostic preservation action",
            "",
            "A cancelled run can be useful diagnostic evidence only when the stop rule was",
            "predeclared and the logs, manifest, config, commit, and relevant outputs are preserved.",
            "",
            "### Update Flow",
            "",
            "1. Edit the YAML record directly when values change",
            "2. Re-validate with `validate_experiment_registry.py`",
            "3. Update the registry index (`experiments/registry.yaml`) when ready",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_record_for_cli(record: dict[str, Any]) -> list[str]:
    """Validate a generated record using a temporary registry index."""
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        record_path = tmp_path / "record.yaml"
        registry_path = tmp_path / "registry.yaml"
        _write_record_yaml(record_path, record)
        _write_record_yaml(
            registry_path,
            {
                "schema_version": "experiment-registry.v1",
                "records": [record_path.name],
            },
        )
        return validate_registry(registry_path)


def main(argv: list[str] | None = None) -> int:
    """Run the experiment card scaffold CLI."""
    args = _build_parser().parse_args(argv)
    issue_url = _resolve_issue_url(args.issue, args.issue_url)
    output_dir = args.output_root or Path(_OUTPUT_EXPERIMENTS) / args.experiment_id

    record = _build_record(
        experiment_id=args.experiment_id,
        issue=args.issue,
        issue_url=issue_url,
        template_name=args.template,
        output_root=output_dir,
    )

    record_path = output_dir / f"{args.experiment_id}.yaml"
    checklist_path = output_dir / "CHECKLIST.md"

    _write_record_yaml(record_path, record)
    todo_fields = _find_todo_fields(record)
    _write_checklist(checklist_path, todo_fields, args.template)

    print(f"Wrote experiment card: {record_path}")
    print(f"Wrote checklist:       {checklist_path}")

    if todo_fields:
        print(f"\n{len(todo_fields)} field(s) contain TODO placeholders - edit before registering:")
        for field in todo_fields:
            print(f"  - {field}")

    if not args.skip_validation:
        print()
        errors = _validate_generated_record(record)
        errors.extend(_validate_record_for_cli(record))
        if errors:
            print(f"Validation found {len(errors)} issue(s):", file=sys.stderr)
            for err in errors:
                print(f"  {err}", file=sys.stderr)
            return 2
        print("Generated card passes registry validation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
