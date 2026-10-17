"""Real retained seed-1001 native envelope and source-bound receipt checks."""

# evidence-writer-exempt: pytest tmp_path fixtures intentionally corrupt proof and
# native envelopes; durable results are produced through the actual gate/CLI.

from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path

import pytest

from robot_sf.benchmark.scenario_generation.persistence_gate import (
    ScenarioPersistenceValidationError,
    validate_persistence_record,
    verify_window_replay_receipt,
)
from scripts.tools import evaluate_pipeline_persistence_gate as gate

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/context/evidence/hardcase_positive_control"
RECEIPT = EVIDENCE / "admission/window_receipt.json"


def test_native_integer_steps_does_not_hide_the_seed1001_trace() -> None:
    """Native episode count must not shadow its nested detailed trace."""
    source = json.loads(gzip.decompress((EVIDENCE / "source.jsonl.gz").read_bytes()))
    assert source["steps"] == 600
    assert (
        gate._extract_steps(source)
        == source["algorithm_metadata"]["simulation_step_trace"]["steps"]
    )


@pytest.mark.new_feature_no_base(reason="base has no source-bound window receipt contract")
def test_retained_seed1001_window_receipt_verifies_actual_replays() -> None:
    """Verify independently retained native records, not only a status template."""
    result = verify_window_replay_receipt(RECEIPT, repo_root=ROOT)
    assert result["exact_replay"]["status"] == "pass"
    assert (
        result["exact_replay"]["replay_digest"]
        == "37488c586d0a7fd203b9385377434516f108edb8bef011a6003a3236e135fed3"
    )
    assert result["event"]["near_miss_samples"] == 11
    assert result["event"]["first_near_miss_step"] == 374
    assert result["event"]["min_clearance_m"] == 0.47727323760374385
    assert result["replay_scope"]["kind"] == "source_bound_window"
    assert result["source_episode_id"] not in result["replay_episode_ids"]


@pytest.mark.new_feature_no_base(reason="base has no source-bound window receipt contract")
@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("missing", "parity proof/schema invalid"),
        ("digest", "source trace/bounds mismatch"),
        ("repeated_run", "independent replay run identities"),
        ("locator", "window artifact locator mismatch"),
    ],
)
def test_window_receipt_refuses_missing_false_or_duplicate_proof(
    tmp_path: Path, mutation: str, reason: str
) -> None:
    """Even a declared pass is rejected when its actual evidence cannot prove parity."""
    receipt = deepcopy(json.loads(RECEIPT.read_text()))
    if mutation == "missing":
        del receipt["parity_proof"]
    elif mutation == "digest":
        receipt["parity_proof"]["source_window_trace_sha256"] = "0" * 64
    elif mutation == "repeated_run":
        receipt["bindings"]["replay_manifests"][1] = receipt["bindings"]["replay_manifests"][0]
    else:
        receipt["bindings"]["planner_config"]["resolved_path"] = "other_config.yaml"
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ScenarioPersistenceValidationError, match=reason):
        verify_window_replay_receipt(path, repo_root=ROOT)


@pytest.mark.new_feature_no_base(reason="base persistence has no explicit window comparison scope")
def test_native_window_gate_accepts_parity_without_promoting_unknown_persistence(
    tmp_path: Path,
) -> None:
    """Window parity passes while the unchanged nine-cell robustness gate rejects."""
    source = json.loads(gzip.decompress((EVIDENCE / "source.jsonl.gz").read_bytes()))
    episodes = tmp_path / "source.jsonl"
    episodes.write_text(json.dumps(source) + "\n")
    results = tmp_path / "results.json"
    results.write_bytes(
        gzip.decompress((EVIDENCE / "followup/packet/gate_replay_results.json.gz").read_bytes())
    )
    records = gate.evaluate_pipeline_candidates(
        manifest_path=None,
        candidate_path=EVIDENCE / "entry.json",
        episodes_path=episodes,
        config_path=ROOT / "configs/analysis/issue_5600_persistence_gate.yaml",
        output_dir=tmp_path / "gate",
        replay_results_path=results,
        window_receipt_path=RECEIPT,
        receipt_root=ROOT,
    )
    record = records[0]
    validate_persistence_record(record)
    assert record["schema_version"] == "generated_scenario_persistence.v2"
    assert record["planner"] == "social_force"
    assert (
        record["exact_replay"]["status"] == record["critical_event_reproduced"]["status"] == "pass"
    )
    assert record["promotion"]["verdict"] == "reject"
    assert all(c["verdict"] == "unknown" for c in record["perturbation_persistence"]["cells"])
