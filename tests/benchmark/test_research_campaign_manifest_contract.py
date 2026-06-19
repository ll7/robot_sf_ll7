"""Contract checks for the research campaign manifest example."""

from pathlib import Path, PurePosixPath

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST = REPO_ROOT / "configs/benchmarks/research_campaign_manifest.example.yaml"


def _load_example_manifest() -> dict[str, object]:
    return yaml.safe_load(EXAMPLE_MANIFEST.read_text(encoding="utf-8"))


def test_research_campaign_manifest_example_covers_required_sections() -> None:
    """Example manifest should expose the issue #3062 campaign handoff sections."""
    manifest = _load_example_manifest()

    assert manifest["schema_version"] == "research-campaign-manifest.v0.1"
    for section in (
        "campaign",
        "scenario_suite",
        "planners",
        "seed_policy",
        "metrics",
        "row_status_policy",
        "outputs",
        "durable_evidence",
        "validation",
    ):
        assert section in manifest


def test_research_campaign_manifest_example_preserves_row_status_contract() -> None:
    """Fallback/degraded/unavailable rows must stay visible and non-successful."""
    manifest = _load_example_manifest()
    row_status_policy = manifest["row_status_policy"]

    assert "successful_evidence" in row_status_policy["success_values"]
    for caveated_status in ("fallback", "degraded", "not_available", "failed", "blocked"):
        assert caveated_status in row_status_policy["allowed_values"]
        assert caveated_status not in row_status_policy["success_values"]
    assert set(row_status_policy["fail_closed_values"]) == {"not_available", "failed", "blocked"}


def test_research_campaign_manifest_example_separates_local_and_durable_outputs() -> None:
    """The example must not treat local output paths as durable evidence."""
    manifest = _load_example_manifest()
    outputs = manifest["outputs"]
    durable_evidence = manifest["durable_evidence"]
    durable_plan = durable_evidence["plan"]

    local_root = PurePosixPath(outputs["local_root"])
    durable_path = PurePosixPath(durable_plan["path"])

    assert local_root.parts[:2] == ("output", "benchmarks")
    assert outputs["disposable"] is True
    assert durable_plan["kind"] == "tracked_context_evidence"
    assert durable_path.parts[:3] == ("docs", "context", "evidence")
    assert durable_plan["required_before_claim"] is True
    assert durable_path.parts[0] != "output"


def test_research_campaign_manifest_example_names_summary_expectations() -> None:
    """Summary JSON and table contracts should cover provenance, rows, and artifacts."""
    manifest = _load_example_manifest()
    outputs = manifest["outputs"]
    summary_json = outputs["summary_json"]
    summary_table = outputs["summary_table"]
    metric_ids = manifest["metrics"]["ids"]

    for field in (
        "campaign_id",
        "source_manifest",
        "git_commit",
        "planner_rows",
        "row_status_summary",
        "artifact_paths",
        "caveats",
    ):
        assert field in summary_json["required_fields"]

    for column in ("scenario_id", "planner_id", "adapter_mode", "seed", "row_status"):
        assert column in summary_table["required_columns"]

    for metric_id in metric_ids:
        assert metric_id in summary_table["required_columns"]
