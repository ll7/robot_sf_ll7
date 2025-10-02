from robot_sf.training.multi_extractor_models import (
    ExtractorRunRecord,
    HardwareProfile,
    TrainingRunSummary,
)
from robot_sf.training.multi_extractor_summary import write_summary_artifacts


def test_summary_markdown_contains_required_sections(tmp_path):
    base_dir = tmp_path / "run_outputs"
    base_dir.mkdir()

    hardware = HardwareProfile(
        platform="Ubuntu 22.04",
        arch="x86_64",
        python_version="3.11.6",
        workers=4,
        gpu_model="RTX 4090",
        cuda_version="12.2",
    )

    extractor = ExtractorRunRecord(
        config_name="vision_transformer",
        status="failed",
        start_time="2025-10-02T08:00:00Z",
        end_time="2025-10-02T08:01:00Z",
        duration_seconds=60.0,
        hardware_profile=hardware,
        worker_mode="vectorized",
        training_steps=32,
        metrics={},
        artifacts={},
        reason="out of memory",
    )

    summary = TrainingRunSummary(
        run_id="20251002-gpu",
        created_at="2025-10-02T08:00:00Z",
        output_root=str(base_dir),
        hardware_overview=[hardware],
        extractor_results=[extractor],
        aggregate_metrics={"best_mean_reward": 0.0, "total_wall_time": 60.0},
        notes=["GPU OOM encountered"],
    )

    paths = write_summary_artifacts(summary=summary, destination=base_dir)
    summary_md = paths["markdown"]

    text = summary_md.read_text(encoding="utf-8")

    required_phrases = [
        "# Multi-Extractor Training Summary",
        "Run ID: 20251002-gpu",
        "## Hardware Overview",
        "| Platform | Arch | GPU Model | CUDA Version | Python | Workers |",
        "## Extractor Results",
        "| Extractor Name | Status | Worker Mode | Duration (s) | Best Metric | Notes |",
        "## Failures & Skips",
        "out of memory",
        "## Aggregated Metrics",
        "best_mean_reward",
        "## Reproducibility",
        "summary.json",
    ]

    for phrase in required_phrases:
        assert phrase in text, f"Missing required Markdown section fragment: {phrase}"
