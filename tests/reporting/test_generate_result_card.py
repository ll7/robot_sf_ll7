"""Tests for the result-card generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reporting import generate_result_card as cards

REPO_ROOT = Path(__file__).resolve().parents[2]
FORECAST_SUMMARY = (
    REPO_ROOT / "docs/context/evidence/issue_3164_frozen_forecast_policy/summary.json"
)
RUNTIME_SUMMARY = REPO_ROOT / "docs/context/evidence/issue_2799_signalized_runtime/summary.json"
PERFORMANCE_SUMMARY = (
    REPO_ROOT / "docs/context/evidence/issue_3142_fast_pysf_force_optimization_summary.json"
)


def test_generates_forecast_result_card_with_latex(tmp_path: Path) -> None:
    """Existing forecast evidence should render Markdown, JSON, and optional LaTeX."""
    exit_code = cards.main(
        [
            str(FORECAST_SUMMARY),
            "--output-dir",
            str(tmp_path),
            "--evidence-tier",
            "diagnostic-replay",
            "--decision",
            "diagnostic",
            "--comparator",
            "shared frozen no-forecast replay baseline",
            "--command",
            "uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/evidence/issue_3164_frozen_forecast_policy/summary.json",
            "--latex-table",
        ]
    )

    payload = json.loads((tmp_path / "result_card.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "result_card.md").read_text(encoding="utf-8")
    latex = (tmp_path / "result_card_table.tex").read_text(encoding="utf-8")

    assert exit_code == 0
    assert payload["schema"] == cards.SCHEMA
    assert payload["decision"] == "diagnostic"
    assert payload["evidence_tier"] == "diagnostic-replay"
    assert "claim_boundary" in payload
    assert payload["metrics"]["fixture_count"] == 4
    assert payload["metrics"]["variant.none.runtime_s"] == 1.1
    assert "## Claim Boundary" in markdown
    assert "\\begin{tabular}" in latex


def test_render_latex_table_escapes_underscores() -> None:
    """LaTeX metric cells escape underscores without f-string expressions."""
    card = cards.ResultCard(
        title="Synthetic",
        source_summary="summary.json",
        evidence_tier="smoke",
        decision="diagnostic",
        comparator="baseline",
        claim_boundary="Synthetic renderer regression test.",
        metrics={"metric_key": "value_with_underscore"},
        commands=["python script.py"],
        artifacts=["summary.json"],
        caveats=["Synthetic fixture."],
        non_transfer_notes=[],
    )

    latex = cards.render_latex_table(card)

    assert "metric\\_key & value\\_with\\_underscore \\\\" in latex


def test_generates_runtime_result_card_from_count_metrics(tmp_path: Path) -> None:
    """Runtime evidence summaries expose useful count/status metrics."""
    cards.main(
        [
            str(RUNTIME_SUMMARY),
            "--output-dir",
            str(tmp_path),
            "--evidence-tier",
            "runtime-smoke",
            "--decision",
            "promote",
            "--comparator",
            "signalized row denominator contract",
            "--caveat",
            "Runtime rows prove denominator semantics only, not compliance.",
            "--command",
            "uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/signalized_runtime_smoke_issue_2799.yaml --output-root output/benchmarks/issue_2799_signalized_runtime --label issue_2799_signalized_runtime_red --skip-publication-bundle",
            "--command",
            "uv run python scripts/tools/generate_signalized_runtime_metrics_report.py --episodes-jsonl <campaign-root>/runs/goal__differential_drive/episodes.jsonl",
            "--artifact",
            "docs/context/evidence/issue_2799_signalized_runtime/README.md",
            "--allow-local-output-with-durable-pointer",
        ]
    )

    payload = json.loads((tmp_path / "result_card.json").read_text(encoding="utf-8"))

    assert payload["metrics"]["total_rows"] == 4
    assert payload["metrics"]["observable_count"] == 2
    assert payload["metrics"]["all_required_runtime_row_classes_present.red_required_stop"] is True
    assert payload["commands"][0].startswith(
        "uv run python scripts/tools/run_camera_ready_benchmark.py"
    )
    assert payload["commands"][-1] == "see README.md fenced reproduction command"


def test_generates_performance_result_card_from_comparison_metrics(tmp_path: Path) -> None:
    """The current #3142 profiling summary should render with explicit claim boundary override."""
    cards.main(
        [
            str(PERFORMANCE_SUMMARY),
            "--output-dir",
            str(tmp_path),
            "--evidence-tier",
            "performance-smoke",
            "--decision",
            "diagnostic",
            "--comparator",
            "same command shape baseline versus after optimization",
            "--claim-boundary",
            "Single before/after smoke comparison; diagnostic support only, not a robust speedup claim.",
        ]
    )

    payload = json.loads((tmp_path / "result_card.json").read_text(encoding="utf-8"))

    assert payload["metrics"]["baseline.step_samples"] == 20
    assert payload["metrics"]["after.pedestrian_count"] == 17
    assert payload["metrics"]["after.steady_steps_per_sec"] > 0
    assert "uv run pytest fast-pysf/tests" in payload["commands"]
    assert all("output/issue-3142" not in artifact for artifact in payload["artifacts"])


def test_fails_closed_without_claim_boundary(tmp_path: Path) -> None:
    """Claim-free summaries must not produce dissertation-ready result cards."""
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"title": "Missing claim", "source_command": "cmd", "metric_count": 1}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        cards.main(
            [
                str(summary),
                "--output-dir",
                str(tmp_path / "out"),
                "--evidence-tier",
                "smoke",
                "--decision",
                "diagnostic",
                "--comparator",
                "baseline",
                "--caveat",
                "Fixture only.",
            ]
        )


def test_fails_closed_without_metrics(tmp_path: Path) -> None:
    """A card without metrics cannot satisfy the evidence contract."""
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "claim_boundary": "Only a textual note.",
                "source_command": "cmd",
                "caveats": ["No metrics."],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        cards.main(
            [
                str(summary),
                "--output-dir",
                str(tmp_path / "out"),
                "--evidence-tier",
                "smoke",
                "--decision",
                "diagnostic",
                "--comparator",
                "baseline",
            ]
        )


def test_rejects_local_output_artifact_without_durable_pointer(tmp_path: Path) -> None:
    """Local output paths should not leak into durable cards by default."""
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "claim_boundary": "Diagnostic only.",
                "source_command": "cmd",
                "episodes_source": "output/benchmarks/run/episodes.jsonl",
                "metric_count": 1,
                "caveats": ["Local output only."],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        cards.main(
            [
                str(summary),
                "--output-dir",
                str(tmp_path / "out"),
                "--evidence-tier",
                "smoke",
                "--decision",
                "diagnostic",
                "--comparator",
                "baseline",
            ]
        )


def test_rejects_vague_command_pointer(tmp_path: Path) -> None:
    """A README pointer alone is not exact command provenance."""
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "claim_boundary": "Diagnostic only.",
                "source_command": "see README.md reproduction command",
                "metric_count": 1,
                "caveats": ["No exact command in this summary."],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        cards.main(
            [
                str(summary),
                "--output-dir",
                str(tmp_path / "out"),
                "--evidence-tier",
                "smoke",
                "--decision",
                "diagnostic",
                "--comparator",
                "baseline",
            ]
        )


def test_rejects_non_finite_cli_metric(tmp_path: Path) -> None:
    """Explicit metrics should not smuggle NaN or infinity into result cards."""
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "claim_boundary": "Diagnostic only.",
                "source_command": "uv run python scripts/example.py --flag",
                "caveats": ["Synthetic fixture."],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        cards.main(
            [
                str(summary),
                "--output-dir",
                str(tmp_path / "out"),
                "--evidence-tier",
                "smoke",
                "--decision",
                "diagnostic",
                "--comparator",
                "baseline",
                "--metric",
                "bad=NaN",
            ]
        )


def test_ignores_non_mapping_variant_results(tmp_path: Path) -> None:
    """Malformed variant_results should be ignored instead of crashing."""
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "claim_boundary": "Diagnostic only.",
                "source_command": "uv run python scripts/example.py --flag",
                "metric_count": 1,
                "caveats": ["Synthetic fixture."],
                "rows": [{"variant_results": ["not", "a", "mapping"]}],
            }
        ),
        encoding="utf-8",
    )

    cards.main(
        [
            str(summary),
            "--output-dir",
            str(tmp_path / "out"),
            "--evidence-tier",
            "smoke",
            "--decision",
            "diagnostic",
            "--comparator",
            "baseline",
        ]
    )

    payload = json.loads((tmp_path / "out/result_card.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["row_count"] == 1


def test_local_output_detection_uses_path_components() -> None:
    """Only real output path components should trigger local-output rejection."""
    assert cards._looks_local_output("output/benchmarks/run.json")
    assert cards._looks_local_output("/tmp/worktree/output/benchmarks/run.json")
    assert not cards._looks_local_output("docs/context/evidence/not_output_summary.json")


def test_main_reports_invalid_json_as_cli_error(tmp_path: Path) -> None:
    """Invalid JSON input should be reported through argparse rather than a traceback."""
    summary = tmp_path / "summary.json"
    summary.write_text("{not json", encoding="utf-8")

    with pytest.raises(SystemExit):
        cards.main(
            [
                str(summary),
                "--output-dir",
                str(tmp_path / "out"),
                "--evidence-tier",
                "smoke",
                "--decision",
                "diagnostic",
                "--comparator",
                "baseline",
            ]
        )
