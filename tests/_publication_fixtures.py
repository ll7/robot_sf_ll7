"""Synthetic calibrated publication fixtures; never admitted benchmark evidence."""

# evidence-writer-exempt: only synthetic pytest tmp_path inputs, not benchmark evidence.

from __future__ import annotations

import hashlib
import json
from itertools import product
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import yaml


def calibrated_asset_contents() -> dict[str, str]:
    """Build synthetic calibrated metadata; no calibration or simulation is executed."""
    from robot_sf.benchmark.metric_definitions import metric_definitions_sha256
    from robot_sf.benchmark.robot_force_contract import declared_force_source_contract
    from robot_sf.benchmark.snqi.evaluation_seeds import SEALED_EVALUATION_SEEDS_SHA256
    from robot_sf.benchmark.snqi.v2_spec import FAMILY, SIMULATED_FORCE, WEIGHTS

    arms = [f"synthetic-arm-{i}" for i in range(14)]
    scenarios = [f"synthetic-scenario-{i}" for i in range(48)]
    seeds = [1001, 1002]
    grid_hash = hashlib.sha256(
        json.dumps(sorted(product(arms, scenarios, seeds)), separators=(",", ":")).encode()
    ).hexdigest()
    files = {f"runs/{arm}__differential_drive/episodes.jsonl": "a" * 64 for arm in arms}
    anchors = {
        "version": "SNQI-v2.0",
        "status": "frozen",
        "metric_schema_version": "robot-sf-metrics.v2",
        "metric_definitions_sha256": metric_definitions_sha256(),
        "evaluation_seeds_sha256": SEALED_EVALUATION_SEEDS_SHA256,
        "anchors": {
            term: {
                "lower": 0,
                "upper": value,
                "type": "normative" if term in ("T", "N") else "calibration_p95",
            }
            for term, value in {"T": 3, "N": 0.25, "F": 10, "J": 2, "K": 4}.items()
        },
        "force_decision": {
            "source": SIMULATED_FORCE,
            "selected_source_contract": declared_force_source_contract(SIMULATED_FORCE),
            "spearman_rho_F_N": 0.8,
            "threshold_absolute_rho": 0.90,
            "N": "clip(near_misses/steps/0.25)",
            "selected_source_coverage": 1344,
        },
        "calibration": {
            "arms": arms,
            "scenarios": scenarios,
            "seeds": seeds,
            "episode_count": 1344,
            "quantile_method": "linear",
            "benchmark_execution": "nonfallback",
            "command_mode_counts": {arm: {"native": 96} for arm in arms},
            "source_commit": "a" * 40,
            "run_id": "synthetic-test-only",
            "grid_sha256": grid_hash,
            "split_id": f"snqi-v2-dev1001-1002-{grid_hash[:12]}",
            "scenario_horizons": dict.fromkeys(scenarios, 600),
            "episode_files_sha256": files,
            "producer_sidecars_sha256": {name + ".provenance.json": "b" * 64 for name in files},
            "episodes_sha256": hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "episodes_hash_rule": "sha256(sorted compact JSON relative-path-to-file-sha256 map)",
            "campaign_config_hash": "c" * 64,
            "campaign_config_identity": "c" * 16,
            "campaign_manifest_sha256": "d" * 64,
        },
    }
    return {
        "weights": json.dumps(
            {
                "version": "SNQI-v2.0",
                "weights": {
                    "w_" + term: {"value": value, "rationale": "synthetic test fixture"}
                    for term, value in WEIGHTS.items()
                },
            }
        ),
        "anchors": json.dumps(anchors),
        "family": yaml.safe_dump(FAMILY),
    }


def stage_publication_custody(run: Path) -> None:
    """Retain full synthetic scorer inputs for generic export/preflight tests."""
    contents = calibrated_asset_contents()
    provenance = {"snqi_v2_version": "SNQI-v2"}
    for role, extension in (("weights", "json"), ("anchors", "json"), ("family", "yaml")):
        relative = f"publication_inputs/{role}.{extension}"
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents[role])
        provenance[f"snqi_v2_{role}_path"] = relative
        provenance[f"snqi_v2_{role}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = run / "reports/snqi_v2_diagnostics.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "schema_version": "snqi-v2-diagnostics.v1",
                "episode_count": 1,
                "provenance": provenance,
            }
        )
    )


def publication_campaign(run: Path) -> dict[str, Any]:
    """Declare a complete synthetic campaign for metadata/preflight export tests.

    Returns:
        A campaign declaration independent of the production custody implementation.
    """
    pins = json.loads((run / "reports/snqi_v2_diagnostics.json").read_text())["provenance"]
    anchors = json.loads((run / pins["snqi_v2_anchors_path"]).read_text())
    return {
        "schema_version": "benchmark-camera-ready-campaign.v2",
        "legacy_snqi": "excluded",
        "metrics": pins,
        "metric_list": ["success", "snqi_v2"],
        "scenario_horizons": anchors["calibration"]["scenario_horizons"],
    }
