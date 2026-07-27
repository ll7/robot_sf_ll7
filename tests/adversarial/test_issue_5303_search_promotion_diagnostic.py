"""Focused no-planner tests for the issue #5303 diagnostic handoff."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.issue_5303_search_promotion_analysis import (
    OUTCOME_ROW_SCHEMA_VERSION,
    analyze_issue_5303_search_promotion,
)
from scripts.tools import compare_adversarial_samplers

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "configs/adversarial/issue_5303_search_promotion_contract.yaml"
FROZEN_CONTRACT = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
assert isinstance(FROZEN_CONTRACT, dict)
FROZEN_INPUTS = {
    entry["id"]: entry
    for entry in FROZEN_CONTRACT["input_provenance"]["required_inputs"]
    if isinstance(entry, dict) and isinstance(entry.get("id"), str)
}


def _frozen_input(input_id: str) -> tuple[str, str]:
    """Return one path/hash pair from the committed frozen contract."""
    entry = FROZEN_INPUTS[input_id]
    assert isinstance(entry, dict)
    path = entry.get("path")
    digest = entry.get("sha256")
    assert isinstance(path, str)
    assert isinstance(digest, str)
    return path, digest


def _canonical_sha256(payload: dict[str, object]) -> str:
    """Return the canonical digest used for candidate and immutable-row hashes."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _immutable_sha256(row: dict[str, object]) -> str:
    payload = dict(row)
    payload.pop("immutable_record_sha256", None)
    return _canonical_sha256(payload)


def _diagnostic_row(
    *,
    arm: str,
    seed: int,
    index: int,
    candidate: dict[str, object] | None = None,
) -> dict[str, object]:
    scenario_config_path, scenario_config_sha256 = _frozen_input("diagnostic_scenario_template")
    search_space_path, search_space_sha256 = _frozen_input("search_space")
    target_config_path, target_config_sha256 = _frozen_input("target_planner_config")
    reference_config_path, reference_config_sha256 = _frozen_input(
        "neutral_reference_planner_config"
    )
    family_split = FROZEN_CONTRACT["family_split"]
    step3_execution = FROZEN_CONTRACT["step3_execution"]
    assert isinstance(family_split, dict)
    assert isinstance(step3_execution, dict)
    candidate = candidate or {
        "scenario_seed": seed + index,
        "fixture_id": f"{arm}-{seed}-{index}",
    }
    candidate_seed = candidate["scenario_seed"]
    row: dict[str, object] = {
        "schema_version": OUTCOME_ROW_SCHEMA_VERSION,
        "row_id": f"{arm}:{seed}:{index:04d}:search",
        "arm": arm,
        "method": arm,
        "search_seed": seed,
        "candidate_index": index,
        "normalized_candidate_config_sha256": _canonical_sha256(candidate),
        "candidate": candidate,
        "scenario_family": family_split["fresh_outcome_family"],
        "scenario_config_path": scenario_config_path,
        "scenario_config_sha256": scenario_config_sha256,
        "search_space_path": search_space_path,
        "search_space_sha256": search_space_sha256,
        "target_planner_config_path": target_config_path,
        "target_planner_config_sha256": target_config_sha256,
        "neutral_reference_planner_config_path": reference_config_path,
        "neutral_reference_planner_config_sha256": reference_config_sha256,
        "execution_stage": "search",
        "execution_seed": candidate_seed,
        "seed_lineage": {
            "search_seed": seed,
            "candidate_scenario_seed": candidate_seed,
            "deterministic_replay_seed": None,
            "confirmation_seeds": [],
            "second_context_seed": None,
        },
        "execution_mode": step3_execution["required_execution_mode"],
        "readiness_status": "ready",
        "availability_status": "available",
        "constraints_first_outcome": {"status": "observed"},
        "objective": step3_execution["diagnostic_objective"],
        "objective_value": 0.0,
        "primary_failure_mechanism": None,
        "stable_attribution_evidence": "not_collected_diagnostic_only",
        "certification": {"status": "passed"},
        "recertification_lineage": "issue_6139_frozen_input",
        "deterministic_replay": "not_run_diagnostic_only",
        "confirmation_target": "not_run_diagnostic_only",
        "confirmation_reference": "not_run_diagnostic_only",
        "second_execution_context": "not_run_diagnostic_only",
        "execution_commit": "a" * 40,
        "execution_context_label": "diagnostic_adapter_context_a",
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
                candidate = None
                if (
                    duplicate_first_optuna_hash
                    and arm == "optuna"
                    and seed == 530301
                    and index == 1
                ):
                    candidate = {
                        "scenario_seed": 530301,
                        "fixture_id": "optuna-530301-0",
                    }
                row = _diagnostic_row(
                    arm=arm,
                    seed=seed,
                    index=index,
                    candidate=candidate,
                )
                lines.append(json.dumps(row, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_diagnostic_cli_requires_the_frozen_execution_bindings() -> None:
    """The runner parser accepts the exact frozen diagnostic command."""
    step3_execution = FROZEN_CONTRACT["step3_execution"]
    assert isinstance(step3_execution, dict)
    command = step3_execution["diagnostic_search_command"]
    assert isinstance(command, str)
    command_parts = shlex.split(command)
    assert command_parts[:4] == [
        "uv",
        "run",
        "python",
        "scripts/tools/compare_adversarial_samplers.py",
    ]
    args = compare_adversarial_samplers.parse_args(command_parts[4:])

    assert args.policy == "hybrid_rule_local_planner"
    assert args.require_certification is True
    assert args.issue_5303_diagnostic_only is True
    assert args.samplers == ["random", "optuna"]
    assert args.seed == [530301, 530302, 530303]
    assert args.execution_context_label == "diagnostic_adapter_context_a"
    assert args.warm_start_record == [
        "issue5305_classic_cross_trap_medium_fbbd96687d61",
        "issue5305_classic_cross_trap_medium_fe24f0ff86a1",
    ]


def test_diagnostic_context_records_frozen_execution_statuses(tmp_path: Path) -> None:
    """Rows retain verified command bindings when episode records omit status metadata."""
    scenario = tmp_path / "scenario.yaml"
    search_space = tmp_path / "space.yaml"
    scenario.write_text("scenarios: []\n", encoding="utf-8")
    search_space.write_text("variables: {}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config": {
                    "scenario_template": str(scenario),
                    "search_space_path": str(search_space),
                },
                "candidates": [{"candidate": {"scenario_seed": 1}}],
            }
        ),
        encoding="utf-8",
    )
    comparison_row = compare_adversarial_samplers.SamplerComparisonRow(
        objective="constraints_first_lexicographic_v1",
        sampler="optuna",
        budget=64,
        seed=530301,
        manifest_path=str(manifest),
        best_bundle_path=None,
        best_objective_value=None,
        best_valid_objective=None,
        num_candidates=1,
        num_valid_candidates=1,
        num_invalid_candidates=0,
        num_failed_evaluations=0,
        invalid_candidate_rate=0.0,
        first_failure_iteration=None,
        certified_valid_failure_count=0,
        replayable_valid_failure_count=0,
        replay_success_rate=None,
        fallback_candidate_count=0,
        degraded_candidate_count=0,
        held_out_family_yield=None,
        held_out_family_status="not_admitted_diagnostic_only",
        caveats=(),
    )
    context = compare_adversarial_samplers.Issue5303DiagnosticContext(
        scenario_family="classic_group_crossing_medium",
        target_planner_config=REPO_ROOT
        / "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml",
        neutral_reference_planner_config=REPO_ROOT
        / "configs/policy_search/candidates/scenario_adaptive_orca_v1.yaml",
        execution_mode="adapter",
        readiness_status="ready",
        availability_status="available",
        execution_context_label="diagnostic_adapter_context_a",
        execution_commit="a" * 40,
    )

    row = compare_adversarial_samplers.build_issue_5303_search_outcome_rows(
        rows=[comparison_row], context=context
    )[0]

    assert row["execution_mode"] == "adapter"
    assert row["readiness_status"] == "ready"
    assert row["availability_status"] == "available"


def test_diagnostic_runner_checks_preflight_before_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale contract aborts before the diagnostic runner can start a search."""
    step3_execution = FROZEN_CONTRACT["step3_execution"]
    assert isinstance(step3_execution, dict)
    command = step3_execution["diagnostic_search_command"]
    assert isinstance(command, str)
    command_parts = shlex.split(command)

    def stale_preflight(*_args: object, **_kwargs: object) -> object:
        return type("Preflight", (), {"ready": False, "blockers": ("stale input hash",)})()

    monkeypatch.setattr(
        compare_adversarial_samplers,
        "preflight_issue_5303_contract",
        stale_preflight,
    )
    monkeypatch.setattr(
        compare_adversarial_samplers,
        "run_sampler_comparison",
        lambda **_kwargs: pytest.fail("search must not start after a failed preflight"),
    )

    with pytest.raises(RuntimeError, match="preflight failed before diagnostic execution"):
        compare_adversarial_samplers.main(command_parts[4:])


def test_diagnostic_runner_rejects_argument_drift_before_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The diagnostic flag cannot run a command that drifts from the frozen budget."""
    step3_execution = FROZEN_CONTRACT["step3_execution"]
    assert isinstance(step3_execution, dict)
    command = step3_execution["diagnostic_search_command"]
    assert isinstance(command, str)
    command_parts = shlex.split(command)
    budget_index = command_parts.index("--budget")
    command_parts[budget_index + 1] = "63"

    monkeypatch.setattr(
        compare_adversarial_samplers,
        "run_sampler_comparison",
        lambda **_kwargs: pytest.fail("search must not start after frozen-command drift"),
    )

    with pytest.raises(RuntimeError, match="mismatched frozen bindings: budget"):
        compare_adversarial_samplers.main(command_parts[4:])


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
    assert result.accounting["frozen_contract_preflight_ready"] is True
    assert (
        result.accounting["frozen_contract_sha256"]
        == hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    )
    assert result.warnings


def test_diagnostic_analysis_recomputes_candidate_hash_before_deduplication(
    tmp_path: Path,
) -> None:
    """A self-consistent row cannot use a false candidate hash to evade deduplication."""
    outcomes = tmp_path / "outcomes.jsonl"
    _write_complete_outcomes(outcomes)
    rows = [json.loads(line) for line in outcomes.read_text(encoding="utf-8").splitlines()]
    rows[0]["normalized_candidate_config_sha256"] = "0" * 64
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
    assert result.accounting["normalized_candidate_hash_failure_count"] == 1
    assert any(
        "candidate hash does not match candidate content" in item for item in result.blockers
    )


def test_diagnostic_analysis_rejects_self_hashed_wrong_frozen_bindings(tmp_path: Path) -> None:
    """A self-hash cannot substitute for the frozen input and execution bindings."""
    outcomes = tmp_path / "outcomes.jsonl"
    _write_complete_outcomes(outcomes)
    rows = [json.loads(line) for line in outcomes.read_text(encoding="utf-8").splitlines()]
    wrong_values = {
        "scenario_family": "classic_cross_trap_medium",
        "scenario_config_path": "configs/scenarios/templates/crossing_ttc.yaml",
        "scenario_config_sha256": "0" * 64,
        "search_space_path": "configs/adversarial/not_the_frozen_space.yaml",
        "search_space_sha256": "1" * 64,
        "target_planner_config_path": "configs/policy_search/candidates/not_the_target.yaml",
        "target_planner_config_sha256": "2" * 64,
        "neutral_reference_planner_config_path": "configs/policy_search/candidates/not_the_ref.yaml",
        "neutral_reference_planner_config_sha256": "3" * 64,
        "objective": "worst_case_snqi",
        "execution_mode": "not_native",
    }

    for field, wrong_value in wrong_values.items():
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0][field] = wrong_value
        tampered_rows[0]["immutable_record_sha256"] = _immutable_sha256(tampered_rows[0])
        tampered_outcomes = tmp_path / f"wrong_{field}.jsonl"
        tampered_outcomes.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in tampered_rows) + "\n",
            encoding="utf-8",
        )

        result = analyze_issue_5303_search_promotion(
            tampered_outcomes,
            contract_path=CONTRACT_PATH,
            repo_root=REPO_ROOT,
        )

        assert result.ready is False
        assert result.accounting["frozen_row_binding_failure_counts"][field] == 1
        assert any(f"frozen {field!r} binding" in blocker for blocker in result.blockers)


def test_diagnostic_analysis_rejects_a_contract_not_matching_its_manifest(tmp_path: Path) -> None:
    """Acceptance rechecks the manifest instead of trusting a supplied contract file."""
    outcomes = tmp_path / "outcomes.jsonl"
    _write_complete_outcomes(outcomes)
    changed_contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    changed_contract["step3_execution"]["required_execution_mode"] = "degraded"
    changed_contract_path = tmp_path / "changed_contract.yaml"
    changed_contract_path.write_text(yaml.safe_dump(changed_contract), encoding="utf-8")

    result = analyze_issue_5303_search_promotion(
        outcomes,
        contract_path=changed_contract_path,
        repo_root=REPO_ROOT,
    )

    assert result.ready is False
    assert result.accounting["frozen_contract_preflight_ready"] is False
    assert any("frozen contract preflight failed" in blocker for blocker in result.blockers)


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
    assert any(
        "candidate hash does not match candidate content" in blocker for blocker in result.blockers
    )
    assert any("must remain not admitted" in blocker for blocker in result.blockers)
    assert any("invalid or unevaluable" in blocker for blocker in result.blockers)


def test_contract_schema_matches_diagnostic_record_fixture() -> None:
    """The committed schema explicitly covers each per-attempt field emitted in tests."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    row = _diagnostic_row(arm="optuna", seed=530301, index=0)
    assert set(contract["outcome_row_schema"]["required_fields"]) == set(row)
