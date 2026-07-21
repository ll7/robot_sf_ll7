"""End-to-end contract coverage for the currently merged issue #3574 tooling.

The fixture exercises the pre-run manifest, trace readiness bridge, and report
writer together.  It is deliberately synthetic: it proves integration only and
does not supply benchmark evidence about heterogeneous populations.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from robot_sf.benchmark.heterogeneous_population_ablation import (
    build_mean_matched_harness_manifest,
)
from robot_sf.benchmark.pedestrian_control_trace import PEDESTRIAN_CONTROL_TRACE_LABELS_KEY

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/benchmarks/issue_3574_mean_matched_harness_smoke.yaml"
STATE_PATH = REPO_ROOT / "docs/context/issue_3574_state.yaml"
HARNESS_NOTE_PATH = REPO_ROOT / "docs/context/issue_3574_mean_matched_harness.md"
SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/build_heterogeneous_population_ablation_report.py"
OLD_SOURCE_ARTIFACT_HEAD = "da438cd7a90e6c63549e9bb4d62be4f57c9c1a2b"
PRE_CHANGE_ANALYSIS_SHA256 = "5133126aa5baf0f3b1da089cabc332ead97d9b75f060cf02f56d89c9ec559465"


def _load_report_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_heterogeneous_population_ablation_report", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_records(
    manifest: dict[str, Any], *, null_first_response_law_fraction: bool = False
) -> list[dict[str, Any]]:
    """Construct complete records matching the tracked pre-run manifest."""

    records: list[dict[str, Any]] = []
    for row in manifest["manifest_rows"]:
        pedestrians = []
        for label in row["arm_population"][PEDESTRIAN_CONTROL_TRACE_LABELS_KEY]:
            pedestrians.append(
                {
                    **label,
                    "steps": [
                        {
                            "step": 0,
                            "clearance_m": 0.8,
                            "near_field_exposure_s": 0.1,
                        },
                        {
                            "step": 1,
                            "clearance_m": 1.2,
                            "near_field_exposure_s": 0.0,
                        },
                    ],
                }
            )
        response_law_fraction = row.get("response_law_fraction")
        if response_law_fraction is None:
            response_law_fraction = 0.0
        if null_first_response_law_fraction and not records:
            response_law_fraction = None
        records.append(
            {
                "scenario_id": row["scenario_id"],
                "planner": row["planner"],
                "seed": row["seed"],
                "population_arm": row["population_arm"],
                "response_law_fraction": response_law_fraction,
                "metrics": {"mean_clearance": 1.0},
                "algorithm_metadata": {
                    "pedestrian_control_trace": {
                        "schema_version": "pedestrian-control-trace.v1",
                        "near_field_clearance_threshold_m": 1.0,
                        "pedestrian_count": len(pedestrians),
                        "pedestrians": pedestrians,
                    }
                },
            }
        )
    return records


def _run_cli(module: ModuleType, argv: list[str]) -> int:
    old_argv = sys.argv
    sys.argv = argv
    try:
        return int(module.main())
    finally:
        sys.argv = old_argv


def _single_planner_manifest(manifest: dict[str, Any], planner: str) -> dict[str, Any]:
    """Return a self-consistent manifest slice for one independently completed planner shard."""
    shard_manifest = json.loads(json.dumps(manifest))
    shard_manifest["manifest_rows"] = [
        row for row in shard_manifest["manifest_rows"] if row["planner"] == planner
    ]
    shard_manifest["row_count"] = len(shard_manifest["manifest_rows"])
    return shard_manifest


def _write_shard_inputs(root: Path, manifest: dict[str, Any], planner: str) -> tuple[Path, Path]:
    """Write one planner's manifest and records as independent finalizer inputs."""
    shard_dir = root / "inputs" / planner
    shard_dir.mkdir(parents=True)
    shard_manifest = _single_planner_manifest(manifest, planner)
    manifest_path = shard_dir / "manifest.json"
    manifest_path.write_text(json.dumps(shard_manifest), encoding="utf-8")
    records_path = shard_dir / "episode_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _fixture_records(shard_manifest)),
        encoding="utf-8",
    )
    return manifest_path, records_path


def _emit_shard_receipt(
    *,
    root: Path,
    manifest: dict[str, Any],
    planner: str,
    output_root: Path,
    durable_root: Path,
    source_artifact_head: str = OLD_SOURCE_ARTIFACT_HEAD,
) -> tuple[Path, Path, Path]:
    """Run the supported shard CLI and return receipt plus source paths."""
    manifest_path, records_path = _write_shard_inputs(root, manifest, planner)
    argv = [
        "build_heterogeneous_population_ablation_report.py",
        "--manifest",
        str(manifest_path),
        "--records",
        str(records_path),
        "--output-dir",
        str(output_root),
        "--durable-dir",
        str(durable_root),
        "--mode",
        "shard",
    ]
    argv.extend(["--source-artifact-head", source_artifact_head])
    code = _run_cli(_load_report_cli(), argv)
    assert code == 0
    return (
        output_root / "shards" / planner / "shard_receipt.json",
        manifest_path,
        records_path,
    )


def _run_finalize(
    receipt_paths: list[Path],
    output_dir: Path,
    *,
    source_roots: list[Path] | None = None,
) -> int:
    """Run the supported receipt finalizer with optional relocated source roots."""
    argv = [
        "build_heterogeneous_population_ablation_report.py",
        "--mode",
        "finalize",
        "--output-dir",
        str(output_dir),
    ]
    for receipt_path in receipt_paths:
        argv.extend(["--shard-receipt", str(receipt_path)])
    for source_root in source_roots or []:
        argv.extend(["--source-root", str(source_root)])
    return _run_cli(_load_report_cli(), argv)


def _current_head() -> str:
    module = _load_report_cli()
    return module.subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_report_metric_direction_contract_is_explicit_and_fail_closed() -> None:
    """Each supported trace metric has a declared safety direction."""

    module = _load_report_cli()
    assert module.metric_higher_is_safer("clearance_m") is True
    assert module.metric_higher_is_safer("near_field_exposure_s") is False
    with pytest.raises(ValueError, match="No higher_is_safer direction"):
        module.metric_higher_is_safer("unknown_safety_metric")


def test_csv_row_leaves_trace_metrics_blank_when_metadata_is_missing() -> None:
    """Malformed optional trace metadata does not crash CSV rendering."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = _single_planner_manifest(
        build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH)), "goal"
    )
    record = _fixture_records(manifest)[0]
    record["algorithm_metadata"] = None

    row = _load_report_cli()._csv_row(record)

    assert row["mean_clearance_m"] == ""
    assert row["cvar_clearance_m"] == ""


def test_tracked_manifest_metrics_flow_into_per_archetype_report(tmp_path: Path) -> None:
    """Every declared trace metric receives an aligned per-archetype report."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records_path = tmp_path / "episode_records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in _fixture_records(manifest, null_first_response_law_fraction=True)
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    durable_dir = tmp_path / "durable"

    code = _run_cli(
        _load_report_cli(),
        [
            "build_heterogeneous_population_ablation_report.py",
            "--manifest",
            str(manifest_path),
            "--records",
            str(records_path),
            "--output-dir",
            str(output_dir),
            "--durable-dir",
            str(durable_dir),
        ],
    )

    assert code == 0
    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest["status"] == "ready"
    assert updated_manifest["claim_boundary"] == "captured_runtime_ready"
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["integration_readiness"]["ready"] is True
    # This fixture intentionally exercises a missing response-law value, which leaves the
    # rank bootstrap blocked while preserving the existing multi-planner report behavior.
    assert summary["rank_sensitivity"]["status"] == "blocked"
    reports = summary["per_archetype_metric_reports"]
    assert sorted(reports) == ["clearance_m", "near_field_exposure_s"]
    for metric_key in reports:
        assert len(reports[metric_key]) == 36
        first_report = next(iter(reports[metric_key].values()))
        assert first_report["metric_key"] == metric_key
        assert first_report["higher_is_safer"] is (metric_key == "clearance_m"), (
            f"Expected {metric_key!r} to declare the correct CVaR safety direction; "
            f"got {first_report['higher_is_safer']!r}"
        )
        assert first_report["arms"]["heterogeneous"]["ready"] is True
        assert first_report["arms"]["mean_matched_homogeneous"]["ready"] is True

    assert summary["ablation_reports"] == reports["clearance_m"]
    assert (durable_dir / "summary.json").exists()
    assert (output_dir / "analysis.md").exists()
    assert not (output_dir / "shard_receipt.json").exists()


def test_single_planner_shard_emits_non_comparative_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed planner shard is recoverable without re-running its simulations."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = _single_planner_manifest(
        build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH)), "goal"
    )
    manifest["expected_head"] = OLD_SOURCE_ARTIFACT_HEAD
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records_path = tmp_path / "episode_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _fixture_records(manifest)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    durable_dir = tmp_path / "durable"
    module = _load_report_cli()
    monkeypatch.setattr(
        module,
        "compute_bootstrap_rank_sensitivity",
        lambda *args, **kwargs: pytest.fail("shard mode must not calculate cross-planner ranks"),
    )

    code = _run_cli(
        module,
        [
            "build_heterogeneous_population_ablation_report.py",
            "--manifest",
            str(manifest_path),
            "--records",
            str(records_path),
            "--output-dir",
            str(output_dir),
            "--durable-dir",
            str(durable_dir),
            "--mode",
            "shard",
        ],
    )

    assert code == 0
    shard_output_dir = output_dir / "shards" / "goal"
    shard_durable_dir = durable_dir / "shards" / "goal"
    receipt = json.loads((shard_output_dir / "shard_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "validated"
    assert receipt["artifact_kind"] == "single_planner_shard_validation_receipt"
    assert receipt["evidence_status"] == "diagnostic-only"
    assert receipt["planner"] == "goal"
    assert receipt["planner_count"] == 1
    assert receipt["integration_readiness"]["ready"] is True
    assert receipt["cross_planner_rank_comparison"] == {
        "claim_made": False,
        "minimum_planners_for_final_combined_report": 2,
        "reason": "single_planner_shard",
        "status": "not_run",
    }
    provenance = receipt["provenance"]
    assert provenance["manifest"]["resolved_path"] == str(manifest_path.resolve())
    assert provenance["records"]["resolved_path"] == str(records_path.resolve())
    assert len(provenance["manifest"]["sha256"]) == 64
    assert len(provenance["records"]["sha256"]) == 64
    assert provenance["config"]["available"] is True
    assert provenance["source_artifact_head"] == OLD_SOURCE_ARTIFACT_HEAD
    assert provenance["receipt_builder_head"] == _current_head()
    assert provenance["source_artifact_head"] != provenance["receipt_builder_head"]
    assert provenance["command_identity"]["mode"] == "shard"
    assert provenance["command_identity"]["schema_version"] == receipt["schema_version"]
    assert receipt["streaming_stats"] == {
        "processing_model": "sequential_whole_shard",
        "max_live_raw_shards": 1,
        "max_live_raw_records": len(_fixture_records(manifest)),
        "raw_shards_retained": 0,
        "raw_records_retained": 0,
        "reduced_rank_record_count": len(_fixture_records(manifest)),
    }
    assert (
        json.loads((shard_durable_dir / "shard_receipt.json").read_text(encoding="utf-8"))
        == receipt
    )
    markdown = (shard_output_dir / "shard_receipt.md").read_text(encoding="utf-8")
    assert "No cross-planner rank or comparison claim was made." in markdown
    assert "not paper-grade evidence" in markdown
    assert (shard_output_dir / "ablation_results.csv").exists()
    assert not (shard_output_dir / "summary.json").exists()
    assert not (shard_output_dir / "rank_sensitivity.json").exists()
    assert not (shard_output_dir / "analysis.md").exists()
    assert not (output_dir / "shard_receipt.json").exists()


def test_shard_preserves_established_blocked_manifest_readiness(tmp_path: Path) -> None:
    """Shard mode neither rewrites nor bypasses blocked_pending_control_trace."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = _single_planner_manifest(
        build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH)), "goal"
    )
    manifest["status"] = "blocked_pending_control_trace"
    manifest["expected_head"] = OLD_SOURCE_ARTIFACT_HEAD
    records = _fixture_records(manifest)
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "episode_records.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    module = _load_report_cli()
    expected_readiness = module.assess_mean_matched_episode_records(manifest, records)

    code = _run_cli(
        module,
        [
            "build_heterogeneous_population_ablation_report.py",
            "--manifest",
            str(manifest_path),
            "--records",
            str(records_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "shard",
        ],
    )

    assert code == 2
    actual_readiness = json.loads(
        (output_dir / "integration_readiness.json").read_text(encoding="utf-8")
    )
    assert actual_readiness == expected_readiness
    assert actual_readiness["status"] == "blocked"
    assert actual_readiness["ready"] is False
    assert actual_readiness["blockers"]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == (
        "blocked_pending_control_trace"
    )
    assert not (output_dir / "shards" / "goal" / "shard_receipt.json").exists()


def test_final_combined_report_rejects_single_planner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final rank-comparison path stays fail-closed for incomplete planner coverage."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = _single_planner_manifest(
        build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH)), "goal"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records_path = tmp_path / "episode_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _fixture_records(manifest)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    durable_dir = tmp_path / "durable"
    module = _load_report_cli()
    monkeypatch.setattr(
        module,
        "compute_bootstrap_rank_sensitivity",
        lambda *args, **kwargs: pytest.fail("combined gate must run before rank bootstrap"),
    )

    code = _run_cli(
        module,
        [
            "build_heterogeneous_population_ablation_report.py",
            "--manifest",
            str(manifest_path),
            "--records",
            str(records_path),
            "--output-dir",
            str(output_dir),
            "--durable-dir",
            str(durable_dir),
            "--mode",
            "combined",
        ],
    )

    assert code == 2
    blocker = json.loads((output_dir / "combined_report_blocked.json").read_text(encoding="utf-8"))
    assert blocker["status"] == "blocked"
    assert blocker["reason"] == "cross_planner_rank_comparison_requires_at_least_two_planners"
    assert "at least two distinct planners" in blocker["error"]
    assert not (output_dir / "summary.json").exists()
    assert not (output_dir / "rank_sensitivity.json").exists()
    assert not (output_dir / "analysis.md").exists()


def test_two_shards_share_output_root_without_overwriting(tmp_path: Path) -> None:
    """Planner-qualified shard paths keep independent receipts collision-free."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH))
    output_root = tmp_path / "shared-output"
    durable_root = tmp_path / "shared-durable"

    goal_receipt, _, _ = _emit_shard_receipt(
        root=tmp_path,
        manifest=manifest,
        planner="goal",
        output_root=output_root,
        durable_root=durable_root,
    )
    orca_receipt, _, _ = _emit_shard_receipt(
        root=tmp_path,
        manifest=manifest,
        planner="orca",
        output_root=output_root,
        durable_root=durable_root,
    )

    assert goal_receipt != orca_receipt
    assert json.loads(goal_receipt.read_text(encoding="utf-8"))["planner"] == "goal"
    assert json.loads(orca_receipt.read_text(encoding="utf-8"))["planner"] == "orca"
    assert (output_root / "shards" / "goal" / "ablation_results.csv").exists()
    assert (output_root / "shards" / "orca" / "ablation_results.csv").exists()
    assert not (output_root / "shard_receipt.json").exists()


def test_finalize_three_verified_shards_builds_existing_rank_report(tmp_path: Path) -> None:
    """Three independent receipts finalize directly without manual JSONL concatenation."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH))
    manifest["expected_head"] = OLD_SOURCE_ARTIFACT_HEAD
    shard_output = tmp_path / "shard-output"
    shard_durable = tmp_path / "shard-durable"
    receipt_paths = [
        _emit_shard_receipt(
            root=tmp_path,
            manifest=manifest,
            planner=planner,
            output_root=shard_output,
            durable_root=shard_durable,
        )[0]
        for planner in ("goal", "orca", "social_force")
    ]
    final_output = tmp_path / "final-output"
    final_durable = tmp_path / "final-durable"
    argv = [
        "build_heterogeneous_population_ablation_report.py",
        "--mode",
        "finalize",
        "--output-dir",
        str(final_output),
        "--durable-dir",
        str(final_durable),
    ]
    for receipt_path in receipt_paths:
        argv.extend(["--shard-receipt", str(receipt_path)])

    code = _run_cli(_load_report_cli(), argv)

    assert code == 0
    summary = json.loads((final_output / "summary.json").read_text(encoding="utf-8"))
    assert summary["integration_readiness"]["ready"] is True
    assert summary["integration_readiness"]["expected_row_count"] == 72
    assert summary["rank_sensitivity"]["status"] == "ready"
    assert summary["rank_sensitivity"]["planners"] == ["goal", "orca", "social_force"]
    finalization = summary["finalization_provenance"]
    assert finalization["status"] == "verified"
    assert finalization["planners"] == ["goal", "orca", "social_force"]
    assert finalization["receipt_count"] == 3
    assert all(len(row["receipt"]["sha256"]) == 64 for row in finalization["receipts"])
    assert finalization["source_artifact_head"] == OLD_SOURCE_ARTIFACT_HEAD
    assert finalization["receipt_builder_heads"] == [_current_head()]
    assert all(
        row["source_artifact_head"] == OLD_SOURCE_ARTIFACT_HEAD for row in finalization["receipts"]
    )
    assert all(row["receipt_builder_head"] == _current_head() for row in finalization["receipts"])
    assert finalization["command_identity"]["mode"] == "finalize"
    assert finalization["command_identity"]["schema_version"] == finalization["schema_version"]
    assert finalization["streaming_stats"]["processing_model"] == "sequential_whole_shard"
    assert finalization["streaming_stats"]["max_live_raw_shards"] == 1
    assert finalization["streaming_stats"]["raw_records_retained"] == 0
    assert finalization["streaming_stats"]["combined_raw_record_list_allocated"] is False
    assert (final_output / "rank_sensitivity.json").exists()
    assert (final_output / "analysis.md").exists()
    assert (final_output / "finalization_provenance.json").exists()
    assert not (final_output / "combined_episode_records.jsonl").exists()

    direct_bundle = tmp_path / "direct-input"
    direct_bundle.mkdir()
    direct_manifest = direct_bundle / "manifest.json"
    direct_records = direct_bundle / "episode_records.jsonl"
    direct_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    direct_records.write_text(
        "".join(json.dumps(record) + "\n" for record in _fixture_records(manifest)),
        encoding="utf-8",
    )
    direct_output = tmp_path / "direct-output"
    assert (
        _run_cli(
            _load_report_cli(),
            [
                "build_heterogeneous_population_ablation_report.py",
                "--manifest",
                str(direct_manifest),
                "--records",
                str(direct_records),
                "--output-dir",
                str(direct_output),
                "--durable-dir",
                str(tmp_path / "direct-durable"),
                "--mode",
                "combined",
            ],
        )
        == 0
    )
    analysis_bytes = (final_output / "analysis.md").read_bytes()
    assert analysis_bytes == (direct_output / "analysis.md").read_bytes()
    assert hashlib.sha256(analysis_bytes).hexdigest() == PRE_CHANGE_ANALYSIS_SHA256
    analysis = analysis_bytes.decode("utf-8")
    for prior_section in (
        "## Claim Boundary",
        "## Executive Summary",
        "## Per-Archetype Trace Metrics",
        "## Rank-Order Sensitivity Analysis",
        "## Detailed Ablation Results",
        "## Non-Reactive Mixture Sweeps Caveats",
    ):
        assert prior_section in analysis
    assert "a larger sample of seeds and scenarios is required" in analysis

    reverse_output = tmp_path / "reverse-final-output"
    reverse_argv = [
        "build_heterogeneous_population_ablation_report.py",
        "--mode",
        "finalize",
        "--output-dir",
        str(reverse_output),
    ]
    for receipt_path in reversed(receipt_paths):
        reverse_argv.extend(["--shard-receipt", str(receipt_path)])
    assert _run_cli(_load_report_cli(), reverse_argv) == 0
    assert (reverse_output / "summary.json").read_bytes() == (
        final_output / "summary.json"
    ).read_bytes()
    reverse_summary = json.loads((reverse_output / "summary.json").read_text(encoding="utf-8"))
    assert (
        reverse_summary["finalization_provenance"]["command_identity"]
        == finalization["command_identity"]
    )
    assert reverse_summary["rank_sensitivity"] == summary["rank_sensitivity"]


def test_finalize_rejects_tampered_receipt(tmp_path: Path) -> None:
    """Receipt fields cannot be changed after validation and still authorize ranking."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH))
    output_root = tmp_path / "shard-output"
    durable_root = tmp_path / "shard-durable"
    goal_receipt, _, _ = _emit_shard_receipt(
        root=tmp_path,
        manifest=manifest,
        planner="goal",
        output_root=output_root,
        durable_root=durable_root,
    )
    orca_receipt, _, _ = _emit_shard_receipt(
        root=tmp_path,
        manifest=manifest,
        planner="orca",
        output_root=output_root,
        durable_root=durable_root,
    )
    tampered = json.loads(goal_receipt.read_text(encoding="utf-8"))
    tampered["planner"] = "orca"
    goal_receipt.write_text(json.dumps(tampered), encoding="utf-8")
    final_output = tmp_path / "final-output"

    code = _run_cli(
        _load_report_cli(),
        [
            "build_heterogeneous_population_ablation_report.py",
            "--mode",
            "finalize",
            "--shard-receipt",
            str(goal_receipt),
            "--shard-receipt",
            str(orca_receipt),
            "--output-dir",
            str(final_output),
        ],
    )

    assert code == 2
    blocker = json.loads((final_output / "finalization_blocked.json").read_text(encoding="utf-8"))
    assert blocker["reason"] == "shard_receipt_verification_failed"
    assert "planner does not match" in blocker["error"]
    assert not (final_output / "summary.json").exists()


def test_finalize_rejects_tampered_records_source(tmp_path: Path) -> None:
    """A post-receipt records mutation is rejected by SHA-256 verification."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH))
    output_root = tmp_path / "shard-output"
    durable_root = tmp_path / "shard-durable"
    goal_receipt, _, goal_records = _emit_shard_receipt(
        root=tmp_path,
        manifest=manifest,
        planner="goal",
        output_root=output_root,
        durable_root=durable_root,
    )
    orca_receipt, _, _ = _emit_shard_receipt(
        root=tmp_path,
        manifest=manifest,
        planner="orca",
        output_root=output_root,
        durable_root=durable_root,
    )
    goal_records.write_text(goal_records.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    final_output = tmp_path / "final-output"

    code = _run_cli(
        _load_report_cli(),
        [
            "build_heterogeneous_population_ablation_report.py",
            "--mode",
            "finalize",
            "--shard-receipt",
            str(goal_receipt),
            "--shard-receipt",
            str(orca_receipt),
            "--output-dir",
            str(final_output),
        ],
    )

    assert code == 2
    blocker = json.loads((final_output / "finalization_blocked.json").read_text(encoding="utf-8"))
    assert blocker["reason"] == "shard_receipt_verification_failed"
    assert "digest verification failed" in blocker["error"]
    assert not (final_output / "summary.json").exists()


def test_finalize_rejects_23_vs_24_normalized_cells(tmp_path: Path) -> None:
    """Each planner shard must cover the exact same cells after planner removal."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH))
    output_root = tmp_path / "shard-output"
    durable_root = tmp_path / "shard-durable"
    goal_receipt, _, _ = _emit_shard_receipt(
        root=tmp_path,
        manifest=manifest,
        planner="goal",
        output_root=output_root,
        durable_root=durable_root,
        source_artifact_head=OLD_SOURCE_ARTIFACT_HEAD,
    )
    short_orca = _single_planner_manifest(manifest, "orca")
    short_orca["manifest_rows"] = short_orca["manifest_rows"][:-1]
    short_orca["row_count"] = len(short_orca["manifest_rows"])
    orca_receipt, _, _ = _emit_shard_receipt(
        root=tmp_path,
        manifest=short_orca,
        planner="orca",
        output_root=output_root,
        durable_root=durable_root,
        source_artifact_head=OLD_SOURCE_ARTIFACT_HEAD,
    )
    final_output = tmp_path / "final-output"

    assert _run_finalize([goal_receipt, orca_receipt], final_output) == 2
    blocker = json.loads((final_output / "finalization_blocked.json").read_text())
    assert "normalized cell coverage mismatch" in blocker["error"]
    assert "missing=1" in blocker["error"]


def test_finalize_rejects_population_parameter_mismatch(tmp_path: Path) -> None:
    """Parity covers the complete manifest row, not only its campaign coordinates."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH))
    output_root = tmp_path / "shard-output"
    durable_root = tmp_path / "shard-durable"
    goal_receipt, _, _ = _emit_shard_receipt(
        root=tmp_path,
        manifest=manifest,
        planner="goal",
        output_root=output_root,
        durable_root=durable_root,
    )

    changed_orca = _single_planner_manifest(manifest, "orca")
    labels = changed_orca["manifest_rows"][0]["arm_population"]["pedestrian_control_trace_labels"]
    labels[0]["desired_speed_factor"] = float(labels[0]["desired_speed_factor"]) + 0.1
    orca_receipt, _, _ = _emit_shard_receipt(
        root=tmp_path,
        manifest=changed_orca,
        planner="orca",
        output_root=output_root,
        durable_root=durable_root,
    )

    final_output = tmp_path / "final-output"
    assert _run_finalize([goal_receipt, orca_receipt], final_output) == 2
    blocker = json.loads((final_output / "finalization_blocked.json").read_text(encoding="utf-8"))
    assert "normalized cell coverage mismatch" in blocker["error"]
    assert "missing=1" in blocker["error"]
    assert "extra=1" in blocker["error"]


def test_source_and_builder_heads_have_independent_finalization_contracts(
    tmp_path: Path,
) -> None:
    """Source heads must match; builder heads remain independent provenance."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH))
    module = _load_report_cli()
    assert (
        module._source_artifact_head(manifest, OLD_SOURCE_ARTIFACT_HEAD) == OLD_SOURCE_ARTIFACT_HEAD
    )
    assert module._receipt_builder_head() == _current_head()
    output_root = tmp_path / "shard-output"
    durable_root = tmp_path / "shard-durable"
    goal_receipt, _, _ = _emit_shard_receipt(
        root=tmp_path,
        manifest=manifest,
        planner="goal",
        output_root=output_root,
        durable_root=durable_root,
        source_artifact_head=OLD_SOURCE_ARTIFACT_HEAD,
    )
    orca_receipt, _, _ = _emit_shard_receipt(
        root=tmp_path,
        manifest=manifest,
        planner="orca",
        output_root=output_root,
        durable_root=durable_root,
        source_artifact_head=OLD_SOURCE_ARTIFACT_HEAD,
    )
    orca_payload = json.loads(orca_receipt.read_text(encoding="utf-8"))
    orca_payload["provenance"]["receipt_builder_head"] = "older-receipt-builder"
    orca_payload["provenance"]["command_identity"]["receipt_builder_head"] = "older-receipt-builder"
    orca_receipt.write_text(json.dumps(orca_payload), encoding="utf-8")
    independent_output = tmp_path / "independent-output"
    assert _run_finalize([goal_receipt, orca_receipt], independent_output) == 0
    independent = json.loads(
        (independent_output / "finalization_provenance.json").read_text(encoding="utf-8")
    )
    assert independent["source_artifact_head"] == OLD_SOURCE_ARTIFACT_HEAD
    assert independent["receipt_builder_heads"] == [
        _current_head(),
        "older-receipt-builder",
    ]

    orca_payload["provenance"]["source_artifact_head"] = "different-source-head"
    orca_payload["provenance"]["command_identity"]["source_artifact_head"] = "different-source-head"
    orca_receipt.write_text(json.dumps(orca_payload), encoding="utf-8")
    mismatch_output = tmp_path / "mismatch-output"
    assert _run_finalize([goal_receipt, orca_receipt], mismatch_output) == 2
    mismatch = json.loads(
        (mismatch_output / "finalization_blocked.json").read_text(encoding="utf-8")
    )
    assert "source artifact heads must be present and equal" in mismatch["error"]

    missing_payload = json.loads(goal_receipt.read_text(encoding="utf-8"))
    missing_payload["provenance"]["source_artifact_head"] = None
    goal_receipt.write_text(json.dumps(missing_payload), encoding="utf-8")
    missing_output = tmp_path / "missing-output"
    assert _run_finalize([goal_receipt, orca_receipt], missing_output) == 2
    missing = json.loads((missing_output / "finalization_blocked.json").read_text(encoding="utf-8"))
    assert "has no source artifact head" in missing["error"]


def test_finalize_resolves_relocated_sources_from_controller_roots(tmp_path: Path) -> None:
    """Portable receipt references remain usable after source bundles move."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH))
    output_root = tmp_path / "shard-output"
    durable_root = tmp_path / "shard-durable"
    receipts = [
        _emit_shard_receipt(
            root=tmp_path,
            manifest=manifest,
            planner=planner,
            output_root=output_root,
            durable_root=durable_root,
            source_artifact_head=OLD_SOURCE_ARTIFACT_HEAD,
        )[0]
        for planner in ("goal", "orca")
    ]
    relocated = tmp_path / "relocated"
    (tmp_path / "inputs").rename(relocated)
    final_output = tmp_path / "final-output"

    assert (
        _run_finalize(
            receipts,
            final_output,
            source_roots=[relocated / "goal", relocated / "orca"],
        )
        == 0
    )
    summary = json.loads((final_output / "summary.json").read_text(encoding="utf-8"))
    assert summary["integration_readiness"]["ready"] is True


@pytest.mark.parametrize(
    "planner",
    [".", "..", "../goal", "goal/orca", r"goal\orca", "/absolute", " goal"],
)
def test_planner_artifact_key_rejects_unsafe_components(planner: str) -> None:
    """Planner-qualified receipt paths cannot escape or alias their output root."""

    module = _load_report_cli()
    with pytest.raises(ValueError, match="unsafe planner artifact component"):
        module._planner_artifact_key(planner)


def test_streaming_source_hash_and_parse_share_open_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing a path after open cannot split parsed bytes from the recorded digest."""

    module = _load_report_cli()
    source = tmp_path / "source.json"
    original = b'{"value": "original"}\n'
    source.write_bytes(original)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b'{"value": "replacement"}\n')

    def replace_after_open(path: Path, label: str) -> None:
        if label == "toctou":
            replacement.replace(path)

    monkeypatch.setattr(module, "_AFTER_SOURCE_OPEN_HOOK", replace_after_open)
    payload, identity = module._read_json_source(
        source,
        argument=str(source),
        bundle_root=tmp_path,
        label="toctou",
    )

    assert payload == {"value": "original"}
    assert identity["sha256"] == module._sha256_bytes(original)
    assert json.loads(source.read_text(encoding="utf-8")) == {"value": "replacement"}

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = _single_planner_manifest(
        build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH)), "goal"
    )
    manifest["manifest_rows"] = manifest["manifest_rows"][:1]
    manifest["row_count"] = 1
    record = _fixture_records(manifest)[0]
    original_line = (json.dumps(record) + "\n").encode()
    records_source = tmp_path / "records.jsonl"
    records_source.write_bytes(original_line)
    tampered_record = {**record, "planner": "tampered"}
    records_replacement = tmp_path / "records-replacement.jsonl"
    records_replacement.write_text(json.dumps(tampered_record) + "\n", encoding="utf-8")

    def replace_records_after_open(path: Path, label: str) -> None:
        if label == "records-toctou":
            records_replacement.replace(path)

    monkeypatch.setattr(module, "_AFTER_SOURCE_OPEN_HOOK", replace_records_after_open)
    reduced, records_identity = module._load_reduce_records(
        records_source,
        argument=str(records_source),
        bundle_root=tmp_path,
        manifest=manifest,
        label="records-toctou",
    )
    assert reduced["integration_readiness"]["ready"] is True
    assert records_identity["sha256"] == module._sha256_bytes(original_line)
    assert json.loads(records_source.read_text(encoding="utf-8"))["planner"] == "tampered"


def test_manifest_rewrite_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest write failure blocks report generation and preserves the source manifest."""

    module = _load_report_cli()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"status": "pending_runtime_capture"}), encoding="utf-8")
    records_path = tmp_path / "episode_records.jsonl"
    records_path.write_text("", encoding="utf-8")
    output_dir = tmp_path / "output"

    monkeypatch.setattr(
        module,
        "assess_mean_matched_episode_records",
        lambda manifest, records: {"ready": True, "trace_metric_keys": []},
    )

    def fail_rewrite(path: Path, manifest: dict[str, Any]) -> None:
        raise OSError("simulated disk-full failure")

    monkeypatch.setattr(module, "_rewrite_manifest_status", fail_rewrite)
    code = _run_cli(
        module,
        [
            "build_heterogeneous_population_ablation_report.py",
            "--manifest",
            str(manifest_path),
            "--records",
            str(records_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert code == 2
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == (
        "pending_runtime_capture"
    )
    failure = json.loads((output_dir / "manifest_rewrite_failure.json").read_text(encoding="utf-8"))
    assert failure == {
        "error": "simulated disk-full failure",
        "manifest_path": str(manifest_path),
        "reason": "manifest_rewrite_failed",
        "status": "blocked",
    }
    assert not (output_dir / "summary.json").exists()


def test_tracked_manifest_declares_required_response_law_sweep() -> None:
    """The committed #3574 matrix makes all required fractions runnable."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["response_law_fractions"] == [0.0, 0.1, 0.25, 0.5]


def test_durable_notes_match_tracked_manifest_row_count() -> None:
    """Issue #3574's execution notes name the committed matrix size."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    state = yaml.safe_load(STATE_PATH.read_text(encoding="utf-8"))
    manifest = build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH))

    assert isinstance(manifest, dict), (
        f"Expected the generated manifest to be a dictionary; got {type(manifest).__name__}"
    )
    documented_row_count = manifest.get("row_count")
    assert documented_row_count == 72, (
        f"Expected the generated manifest row count to be 72; got {documented_row_count!r}"
    )
    assert f"{documented_row_count}_row" in state["next_empirical_action"]
    assert f"{documented_row_count}-row tracked manifest" in HARNESS_NOTE_PATH.read_text(
        encoding="utf-8"
    )
