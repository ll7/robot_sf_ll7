"""Focused no-planner tests for the issue #5303 diagnostic handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from robot_sf.benchmark.issue_5303_search_promotion_analysis import (
    OUTCOME_ROW_SCHEMA_VERSION,
    analyze_issue_5303_search_promotion,
)
from scripts.tools.compare_adversarial_samplers import parse_args

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "configs/adversarial/issue_5303_search_promotion_contract.yaml"


def _immutable_sha256(row: dict[str, object]) -> str:
    payload = dict(row)
    payload.pop("immutable_record_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _diagnostic_row(*, arm: str, seed: int, index: int, candidate_hash: str) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": OUTCOME_ROW_SCHEMA_VERSION,
        "row_id": f"{arm}:{seed}:{index:04d}:search",
        "arm": arm,
        "method": arm,
        "search_seed": seed,
        "candidate_index": index,
        "normalized_candidate_config_sha256": candidate_hash,
        "candidate": {"scenario_seed": seed + index},
        "scenario_family": "classic_group_crossing_medium",
        "scenario_config_path": "configs/adversarial/issue_5303_classic_group_crossing_medium.yaml",
        "scenario_config_sha256": "scenario-hash",
        "search_space_path": "configs/adversarial/crossing_ttc_space.yaml",
        "search_space_sha256": "space-hash",
        "target_planner_config_path": (
            "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml"
        ),
        "target_planner_config_sha256": "target-hash",
        "neutral_reference_planner_config_path": (
            "configs/policy_search/candidates/scenario_adaptive_orca_v1.yaml"
        ),
        "neutral_reference_planner_config_sha256": "reference-hash",
        "execution_stage": "search",
        "execution_seed": seed + index,
        "seed_lineage": {"search_seed": seed},
        "execution_mode": "native",
        "readiness_status": "ready",
        "availability_status": "available",
        "constraints_first_outcome": {"status": "observed"},
        "objective_value": 0.0,
        "primary_failure_mechanism": None,
        "stable_attribution_evidence": "not_collected_diagnostic_only",
        "certification": {"status": "passed"},
        "recertification_lineage": "issue_6139_frozen_input",
        "deterministic_replay": "not_run_diagnostic_only",
        "confirmation_target": "not_run_diagnostic_only",
        "confirmation_reference": "not_run_diagnostic_only",
        "second_execution_context": "not_run_diagnostic_only",
        "execution_commit": "unit-test",
        "execution_context_label": "diagnostic_native_context_a",
        "admission_decision": "not_admitted_diagnostic_only",
        "exclusion_reason": "diagnostic_only_no_replay_reference_or_second_context",
    }
    row["immutable_record_sha256"] = _immutable_sha256(row)
    return row


def _write_complete_outcomes(path: Path, *, duplicate_first_optuna_hash: bool = False) -> None:
    lines: list[str] = []
    for arm in ("optuna", "random"):
        for seed in (530301, 530302, 530303):
            for index in range(64):
                candidate_hash = f"{arm}-{seed}-{index}"
                if (
                    duplicate_first_optuna_hash
                    and arm == "optuna"
                    and seed == 530301
                    and index == 1
                ):
                    candidate_hash = "optuna-530301-0"
                row = _diagnostic_row(
                    arm=arm,
                    seed=seed,
                    index=index,
                    candidate_hash=candidate_hash,
                )
                lines.append(json.dumps(row, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_diagnostic_cli_requires_the_frozen_execution_bindings() -> None:
    """The runner parser accepts the complete native diagnostic command shape."""
    args = parse_args(
        [
            "--policy",
            "hybrid_rule_local_planner",
            "--algo-config",
            "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml",
            "--reference-algo-config",
            "configs/policy_search/candidates/scenario_adaptive_orca_v1.yaml",
            "--scenario-template",
            "configs/adversarial/issue_5303_classic_group_crossing_medium.yaml",
            "--scenario-family",
            "classic_group_crossing_medium",
            "--search-space",
            "configs/adversarial/crossing_ttc_space.yaml",
            "--sampler",
            "random",
            "--sampler",
            "optuna",
            "--budget",
            "64",
            "--seed",
            "530301",
            "--seed",
            "530302",
            "--seed",
            "530303",
            "--objective",
            "constraints_first_lexicographic_v1",
            "--horizon",
            "100",
            "--dt",
            "0.1",
            "--require-certification",
            "--benchmark-profile",
            "experimental",
            "--issue-5303-diagnostic-only",
            "--execution-context-label",
            "diagnostic_native_context_a",
            "--output-dir",
            "output/adversarial/issue_5303_search_promotion",
            "--out-json",
            "output/adversarial/issue_5303_search_promotion/report.json",
            "--out-md",
            "output/adversarial/issue_5303_search_promotion/comparison_table.md",
            "--outcomes-jsonl",
            "output/adversarial/issue_5303_search_promotion/outcomes.jsonl",
        ]
    )

    assert args.policy == "hybrid_rule_local_planner"
    assert args.require_certification is True
    assert args.issue_5303_diagnostic_only is True
    assert args.samplers == ["random", "optuna"]
    assert args.seed == [530301, 530302, 530303]


def test_diagnostic_analysis_retains_duplicate_attempt_in_primary_denominator(
    tmp_path: Path,
) -> None:
    """Global within-arm duplicate collapse changes only the unique endpoint, not 192 attempts."""
    outcomes = tmp_path / "outcomes.jsonl"
    _write_complete_outcomes(outcomes, duplicate_first_optuna_hash=True)

    result = analyze_issue_5303_search_promotion(
        outcomes,
        contract_path=CONTRACT_PATH,
        repo_root=REPO_ROOT,
    )

    assert result.ready, result.blockers
    assert result.decision == "inconclusive"
    assert result.accounting["expected_attempts_per_arm"] == 192
    assert result.accounting["observed_attempts_per_arm"] == {"optuna": 192, "random": 192}
    assert result.accounting["global_within_arm_normalized_hash_duplicates"]["optuna"] == 1
    assert result.accounting["unique_normalized_hashes_per_arm"]["optuna"] == 191
    assert result.warnings


def test_diagnostic_analysis_fails_closed_on_a_missing_attempt(tmp_path: Path) -> None:
    """A missing row remains in the denominator and blocks a readiness result."""
    outcomes = tmp_path / "outcomes.jsonl"
    _write_complete_outcomes(outcomes)
    lines = outcomes.read_text(encoding="utf-8").splitlines()
    outcomes.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    result = analyze_issue_5303_search_promotion(
        outcomes,
        contract_path=CONTRACT_PATH,
        repo_root=REPO_ROOT,
    )

    assert result.ready is False
    assert result.decision == "inconclusive"
    assert result.accounting["expected_attempts_per_arm"] == 192
    assert result.accounting["missing_attempts_per_arm"]["random"] == 1
    assert any("missing scheduled attempts" in blocker for blocker in result.blockers)


def test_diagnostic_analysis_keeps_attrition_in_denominator_and_fails_closed(
    tmp_path: Path,
) -> None:
    """Fallback/degraded availability is counted, never silently removed as complete-case data."""
    outcomes = tmp_path / "outcomes.jsonl"
    _write_complete_outcomes(outcomes)
    rows = [json.loads(line) for line in outcomes.read_text(encoding="utf-8").splitlines()]
    rows[0]["availability_status"] = "fallback"
    rows[0]["immutable_record_sha256"] = _immutable_sha256(rows[0])
    outcomes.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = analyze_issue_5303_search_promotion(
        outcomes,
        contract_path=CONTRACT_PATH,
        repo_root=REPO_ROOT,
    )

    assert result.ready is False
    assert result.accounting["observed_attempts_per_arm"]["optuna"] == 192
    assert result.accounting["recorded_fallback_degraded_or_unavailable_rows"]["optuna"] == 1
    assert any("remains in the primary denominator" in blocker for blocker in result.blockers)


def test_diagnostic_analysis_fails_closed_on_malformed_outcome_lines(tmp_path: Path) -> None:
    """Blank, non-JSON, and non-object lines cannot disappear from accounting diagnostics."""
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text("\nnot-json\n[]\n", encoding="utf-8")

    result = analyze_issue_5303_search_promotion(
        outcomes,
        contract_path=CONTRACT_PATH,
        repo_root=REPO_ROOT,
    )

    assert result.ready is False
    assert result.accounting["observed_attempts_per_arm"] == {"optuna": 0, "random": 0}
    assert any("blank line" in blocker for blocker in result.blockers)
    assert any("not JSON" in blocker for blocker in result.blockers)
    assert any("must be an object" in blocker for blocker in result.blockers)


def test_diagnostic_analysis_fails_closed_on_invalid_record_content(tmp_path: Path) -> None:
    """Invalid records stay counted but prevent an apparently complete diagnostic result."""
    outcomes = tmp_path / "outcomes.jsonl"
    _write_complete_outcomes(outcomes)
    rows = [json.loads(line) for line in outcomes.read_text(encoding="utf-8").splitlines()]

    rows[0]["schema_version"] = "unsupported"
    rows[0]["execution_stage"] = "replay"
    rows[0]["method"] = "random"
    rows[0]["immutable_record_sha256"] = _immutable_sha256(rows[0])

    rows[1]["immutable_record_sha256"] = "not-the-row-hash"

    rows[2]["normalized_candidate_config_sha256"] = ""
    rows[2]["immutable_record_sha256"] = _immutable_sha256(rows[2])

    rows[3]["admission_decision"] = "admitted"
    rows[3]["exclusion_reason"] = "not-diagnostic"
    rows[3]["certification"] = {"status": "failed"}
    rows[3]["immutable_record_sha256"] = _immutable_sha256(rows[3])

    outcomes.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    result = analyze_issue_5303_search_promotion(
        outcomes,
        contract_path=CONTRACT_PATH,
        repo_root=REPO_ROOT,
    )

    assert result.ready is False
    assert result.accounting["observed_attempts_per_arm"]["optuna"] == 192
    assert result.accounting["immutable_hash_failure_count"] == 1
    assert result.accounting["recorded_invalid_or_unevaluable_rows"]["optuna"] == 1
    assert any("unsupported schema_version" in blocker for blocker in result.blockers)
    assert any("method must match" in blocker for blocker in result.blockers)
    assert any("lacks a normalized candidate hash" in blocker for blocker in result.blockers)
    assert any("must remain not admitted" in blocker for blocker in result.blockers)
    assert any("invalid or unevaluable" in blocker for blocker in result.blockers)


def test_contract_schema_matches_diagnostic_record_fixture() -> None:
    """The committed schema explicitly covers each per-attempt field emitted in tests."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    row = _diagnostic_row(arm="optuna", seed=530301, index=0, candidate_hash="candidate")
    assert set(contract["outcome_row_schema"]["required_fields"]) == set(row)
