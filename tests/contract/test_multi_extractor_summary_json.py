import json
from pathlib import Path

from jsonschema import Draft7Validator

from robot_sf.training.multi_extractor_models import (
    ExtractorRunRecord,
    HardwareProfile,
    TrainingRunSummary,
)
from robot_sf.training.multi_extractor_summary import write_summary_artifacts

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "contracts" / "training_summary.schema.json"


def test_summary_json_matches_contract(tmp_path):
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)

    base_dir = tmp_path / "run_outputs"
    base_dir.mkdir()

    hardware = HardwareProfile(
        platform="macOS 15",
        arch="arm64",
        python_version="3.11.6",
        workers=1,
        gpu_model=None,
        cuda_version=None,
    )

    extractor = ExtractorRunRecord(
        config_name="mlp_small",
        status="success",
        start_time="2025-10-02T12:00:00Z",
        end_time="2025-10-02T12:05:00Z",
        duration_seconds=300.0,
        hardware_profile=hardware,
        worker_mode="single-thread",
        training_steps=128,
        metrics={"best_mean_reward": 123.45},
        artifacts={"checkpoint": "extractors/mlp_small/final.zip"},
        reason=None,
    )

    summary = TrainingRunSummary(
        run_id="20251002-abcdef",
        created_at="2025-10-02T12:00:00Z",
        output_root=str(base_dir),
        hardware_overview=[hardware],
        extractor_results=[extractor],
        aggregate_metrics={"best_mean_reward": 123.45, "total_wall_time": 300.0},
        notes=["All extractors converged"],
    )

    paths = write_summary_artifacts(summary=summary, destination=base_dir)
    summary_json = paths["json"]

    assert summary_json.exists(), "Expected summary.json to be written"

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    errors = sorted(error.message for error in validator.iter_errors(payload))
    assert not errors, f"Schema violations: {errors}"
