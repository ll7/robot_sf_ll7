"""Focused tests for the issue #5303 search-promotion frozen-contract preflight.

These tests prove three things together:

1. The side-effect-free preflight reproduces the frozen contract hash and asserts the
   frozen design, the recomputed power analysis, and the honest diagnostic declaration,
   by reading only the contract, the #6139 recertification receipt, and the manifest.
2. The preflight is genuinely side-effect-free: it imports no adversarial execution
   surface (samplers/search/runtime/qd/warm_start/transfer_matrix/campaign/replay),
   no subprocess, and no network module, and it detects contract tampering.
3. The honest scientific conclusion holds: with exactly three search seeds per method,
   the seed-clustered permutation null cannot reach p<=0.05 two-sided, so the future run
   is preregistered as diagnostic/inconclusive without weakening any threshold.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.benchmark.issue_5303_search_promotion_preregistration import (
    CONTRACT_SCHEMA_VERSION,
    DEFAULT_CONTRACT_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RECEIPT_PATH,
    SCHEMA_VERSION,
    Issue5303PreflightResult,
    _min_permutation_p_values,
    _warm_start_space_errors,
    dump_preflight_payload,
    preflight_issue_5303_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_MODULE_PATH = REPO_ROOT / "robot_sf/benchmark/issue_5303_search_promotion_preregistration.py"
CONTRACT_PATH = REPO_ROOT / DEFAULT_CONTRACT_PATH
RECEIPT_PATH = REPO_ROOT / DEFAULT_RECEIPT_PATH
MANIFEST_PATH = REPO_ROOT / DEFAULT_MANIFEST_PATH

# Any import of these fragments would let the preflight touch adversarial execution
# surfaces or side effects, which the side-effect-free contract forbids.
FORBIDDEN_IMPORT_FRAGMENTS = (
    "robot_sf.adversarial",
    "subprocess",
    "socket",
    "urllib",
    "requests",
    "http.client",
    "asyncio",
    "multiprocessing",
    "concurrent.futures",
)
FORBIDDEN_SOURCE_TOKENS = ("os.system", "os.popen", "popen", "__import__", "eval(", "exec(")

SEMANTIC_DRIFT_MUTATIONS: dict[str, tuple[tuple[str, ...], object]] = {
    "entry_gate_state": (("entry_gate", "blocking_issue_state"), "open"),
    "control_role": (("controls", "rejection_controls", "0", "role"), "optional"),
    "method_sampler": (
        ("methods", "entries", "0", "sampler_class"),
        "not_the_frozen_sampler",
    ),
    "candidate_space_certification": (
        ("candidate_space_and_feasibility", "require_certification"),
        False,
    ),
    "gate_rule": (("counted_weak_point_gates", "gates", "0", "rule"), "changed"),
    "estimand_gate": (("estimand", "counted_only_when_all_seven_gates_pass"), False),
    "uncertainty_resamples": (("uncertainty", "resamples"), 1),
    "null_both_required": (("null_tests", "both_required"), False),
    "attrition_reason": (("missing_invalid_attrition", "reason_recorded"), False),
    "input_hash_algorithm": (("input_provenance", "algorithm"), "sha256"),
    "diagnostic_entrypoint": (
        ("step3_execution", "diagnostic_search_command"),
        "uv run python scripts/tools/wrong_runner.py",
    ),
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_nested_contract_value(
    contract: dict[str, Any], path: tuple[str, ...], value: object
) -> None:
    """Set one nested test mutation without duplicating branch-heavy setup code."""
    target: Any = contract
    for key in path[:-1]:
        target = target[int(key)] if isinstance(target, list) else target[key]
    final_key = path[-1]
    target[int(final_key) if isinstance(target, list) else final_key] = value


def _preflight_rehashed_contract(
    tmp_path: Path,
    contract: dict[str, Any],
    name: str,
    *,
    receipt_path: Path = RECEIPT_PATH,
) -> Issue5303PreflightResult:
    """Run preflight against a temporary contract whose manifest hash was refreshed."""
    contract_path = tmp_path / f"{name}.yaml"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["contract_sha256"] = _sha256_file(contract_path)
    manifest_path = tmp_path / f"{name}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return preflight_issue_5303_contract(
        contract_path,
        receipt_path=receipt_path,
        manifest_path=manifest_path,
        repo_root=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# 1. Frozen contract checks
# ---------------------------------------------------------------------------


def test_preflight_passes_on_frozen_contract() -> None:
    """The committed frozen contract passes every check with no blockers."""
    result = preflight_issue_5303_contract(repo_root=REPO_ROOT)
    assert result.ready, "blockers:\n  " + "\n  ".join(result.blockers)
    assert result.blocked is False
    assert not result.blockers
    failed = [name for name, ok in result.checks.items() if not ok]
    assert failed == [], f"failed checks: {failed}"


def test_warm_start_space_errors_fail_closed_for_malformed_inputs(tmp_path: Path) -> None:
    """The side-effect-free warm-start check reports malformed and out-of-space records."""
    archive_path = tmp_path / "archive.json"
    search_space_path = tmp_path / "space.yaml"

    missing_errors = _warm_start_space_errors(
        archive_path=archive_path,
        record_ids=("record",),
        search_space_path=search_space_path,
    )
    assert missing_errors and "could not be loaded" in missing_errors[0]

    archive_path.write_text(json.dumps({"entries": {}}), encoding="utf-8")
    search_space_path.write_text("variables: {}\nconstraints: {}\n", encoding="utf-8")
    assert _warm_start_space_errors(
        archive_path=archive_path,
        record_ids=("record",),
        search_space_path=search_space_path,
    ) == ["warm-start archive must contain an entries list"]

    archive_path.write_text(json.dumps({"entries": []}), encoding="utf-8")
    search_space_path.write_text("variables: []\nconstraints: {}\n", encoding="utf-8")
    assert _warm_start_space_errors(
        archive_path=archive_path,
        record_ids=("record",),
        search_space_path=search_space_path,
    ) == ["warm-start search space must contain variables and constraints mappings"]

    archive_path.write_text(
        json.dumps(
            {
                "entries": [
                    {"archive_id": "missing_candidate", "candidate": None},
                    {"archive_id": "bad_pose", "candidate": {"start": {}, "goal": []}},
                    {
                        "archive_id": "bad_values",
                        "candidate": {
                            "start": {"x": "not-a-number", "y": "nan"},
                            "goal": {"x": 1.0, "y": 1.0},
                            "spawn_time_s": -1.0,
                            "pedestrian_speed_mps": 0.0,
                            "pedestrian_delay_s": -1.0,
                            "scenario_seed": 1.5,
                        },
                    },
                    {
                        "archive_id": "short_distance",
                        "candidate": {
                            "start": {"x": 1.0, "y": 1.0},
                            "goal": {"x": 1.0, "y": 1.0},
                            "spawn_time_s": 1.0,
                            "pedestrian_speed_mps": 1.0,
                            "pedestrian_delay_s": 1.0,
                            "scenario_seed": 42,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    search_space_path.write_text(
        yaml.safe_dump(
            {
                "variables": {
                    "start_x": {"min": 0.0, "max": 2.0},
                    "start_y": {"min": 0.0, "max": 2.0},
                    "goal_x": {"min": "nan", "max": 2.0},
                    "goal_y": {"min": "invalid", "max": 2.0},
                    "spawn_time_s": {"min": 0.0, "max": 2.0},
                    "pedestrian_speed_mps": {"min": 0.1, "max": 2.0},
                    "pedestrian_delay_s": {"min": 0.0, "max": 2.0},
                    "scenario_seed": {"min": 0, "max": 100},
                },
                "constraints": {"min_start_goal_distance_m": 10.0},
            }
        ),
        encoding="utf-8",
    )
    errors = _warm_start_space_errors(
        archive_path=archive_path,
        record_ids=("missing_candidate", "bad_pose", "bad_values", "short_distance", "absent"),
        search_space_path=search_space_path,
    )
    assert any("has no candidate mapping" in error for error in errors)
    assert any("invalid start/goal poses" in error for error in errors)
    assert any("non-numeric start_x" in error for error in errors)
    assert any("non-finite start_y" in error for error in errors)
    assert any("invalid bounds for goal_y" in error for error in errors)
    assert any("outside" in error for error in errors)
    assert any("scenario_seed must be an integer" in error for error in errors)
    assert any("must be non-negative" in error for error in errors)
    assert any("must be positive" in error for error in errors)
    assert any("distance" in error for error in errors)


def test_preflight_schema_and_contract_schema_versions() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert SCHEMA_VERSION == "issue_5303_search_promotion_preflight.v1"
    assert contract["schema_version"] == CONTRACT_SCHEMA_VERSION


def test_contract_hash_matches_manifest() -> None:
    """The preflight recomputes the contract SHA-256 and it matches the frozen manifest."""
    result = preflight_issue_5303_contract(repo_root=REPO_ROOT)
    recomputed = _sha256_file(CONTRACT_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert result.metadata["contract_file_sha256"] == recomputed
    assert manifest["contract_sha256"] == recomputed
    assert result.checks["contract_hash_matches_manifest"]


def test_receipt_hashes_cross_check() -> None:
    """The receipt and actual archive raw-file hashes match the frozen entry gate."""
    result = preflight_issue_5303_contract(repo_root=REPO_ROOT)
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    archive_path = REPO_ROOT / contract["entry_gate"]["certified_archive_path"]
    assert result.metadata["receipt_file_sha256"] == _sha256_file(RECEIPT_PATH)
    assert (
        result.metadata["receipt_file_sha256"]
        == contract["entry_gate"]["recertification_receipt_file_sha256"]
    )
    assert (
        receipt["recertification_sha256"]
        == contract["entry_gate"]["recertification_self_declared_sha256"]
    )
    assert receipt["archive_sha256"] == contract["entry_gate"]["certified_archive_sha256"]
    assert result.metadata["certified_archive_file_sha256"] == _sha256_file(archive_path)
    assert (
        result.metadata["certified_archive_file_sha256"]
        == contract["entry_gate"]["certified_archive_sha256"]
    )
    assert result.checks["receipt_file_hash_matches_contract"]
    assert result.checks["receipt_self_declared_hash_matches_contract"]
    assert result.checks["certified_archive_exists"]
    assert result.checks["certified_archive_file_hash_matches_contract"]
    assert result.checks["archive_hash_consistent"]


def test_explicit_receipt_override_is_checked_instead_of_contract_path(tmp_path: Path) -> None:
    """The public receipt override must select the supplied file and fail closed on drift."""
    overridden_receipt = tmp_path / "overridden_receipt.json"
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    receipt["recertification_sha256"] = "0" * 64
    overridden_receipt.write_text(json.dumps(receipt), encoding="utf-8")

    result = preflight_issue_5303_contract(
        CONTRACT_PATH,
        receipt_path=overridden_receipt,
        manifest_path=MANIFEST_PATH,
        repo_root=REPO_ROOT,
    )

    assert result.ready is False
    assert result.metadata["receipt_file_sha256"] == _sha256_file(overridden_receipt)
    assert result.checks["receipt_file_hash_matches_contract"] is False
    assert result.checks["receipt_self_declared_hash_matches_contract"] is False


def test_eligible_records_match_receipt_family_split() -> None:
    """The contract's eligible IDs match the receipt's corrected eligible records exactly."""
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    eligible_by_family: dict[str, list[str]] = {}
    for record in receipt["records"]:
        if record["after"]["benchmark_eligibility"] == "eligible":
            eligible_by_family.setdefault(record["scenario_family"], []).append(
                record["archive_id"]
            )
    eligible_by_family = {k: sorted(v) for k, v in eligible_by_family.items()}

    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    split = contract["family_split"]
    assert (
        sorted(split["fit_family_eligible_records"])
        == eligible_by_family["classic_cross_trap_medium"]
    )
    assert (
        sorted(split["fresh_outcome_family_eligible_records"])
        == eligible_by_family["classic_group_crossing_medium"]
    )
    assert len(eligible_by_family["classic_cross_trap_medium"]) == 2
    assert len(eligible_by_family["classic_group_crossing_medium"]) == 6
    assert sum(len(v) for v in eligible_by_family.values()) == 8

    result = preflight_issue_5303_contract(repo_root=REPO_ROOT)
    assert result.checks["fit_family_eligible_ids_match_receipt"]
    assert result.checks["fresh_family_eligible_ids_match_receipt"]
    assert result.checks["no_excluded_ids_in_eligible_sets"]


def test_frozen_design_fields() -> None:
    """The matched budget, seeds, methods, time cap, and objective ordering are frozen."""
    result = preflight_issue_5303_contract(repo_root=REPO_ROOT)
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["budget"]["candidate_budget_per_search_seed_per_method"] == 64
    assert contract["budget"]["search_seeds_per_method"] == 3
    assert len(contract["budget"]["search_seeds"]) == 3
    assert [m["name"] for m in contract["methods"]["entries"]] == ["optuna", "random"]
    assert contract["target_planner"]["name"] == "scenario_adaptive_hybrid_orca_v2_collision_guard"
    assert contract["neutral_reference_planner"]["name"] == "scenario_adaptive_orca_v1"
    provenance_ids = {entry["id"] for entry in contract["input_provenance"]["required_inputs"]}
    assert "diagnostic_runner" in provenance_ids
    assert "adversarial_search_runner" in provenance_ids
    assert {"preflight_module", "contract_check_cli"} <= provenance_ids
    for check_name in (
        "candidate_budget_64_per_seed",
        "search_seeds_exactly_three",
        "total_candidates_per_method_frozen",
        "methods_exactly_optuna_and_random",
        "methods_and_warm_start_frozen",
        "target_planner_config_frozen",
        "family_split_inputs_frozen",
        "simulator_time_cap_frozen",
        "objective_constraints_first",
        "objective_hard_constraint_veto",
        "objective_runner_registered",
        "counted_weak_point_gates_all_seven",
        "counted_weak_point_gate_semantics_frozen",
        "gates_fail_closed",
        "input_provenance_complete",
        "input_provenance_algorithm",
        "input_provenance_hashes",
        "entry_gate_bindings_frozen",
        "controls_frozen",
        "method_entries_frozen",
        "candidate_space_and_feasibility_frozen",
        "objective_definition_frozen",
        "estimand_definition_frozen",
        "uncertainty_definition_frozen",
        "null_tests_both_required",
        "missing_invalid_attrition_policy_frozen",
        "power_declaration_frozen",
        "future_run_boundary_frozen",
        "intention_to_search_primary_denominator",
        "missing_invalid_stay_primary_denominator",
        "outcome_row_schema_complete",
        "promotion_campaign_stopped",
        "diagnostic_run_requires_separate_authorization",
        "step3_execution_declared_diagnostic_only",
        "step3_runner_static_support",
        "step3_runner_outcome_writer_support",
        "step3_runner_row_schema_matches_contract",
        "step3_analysis_static_support",
        "step3_execution_command_complete",
        "step3_warm_start_search_space_compatible",
        "step3_analysis_command_complete",
    ):
        assert result.checks[check_name], check_name


# ---------------------------------------------------------------------------
# 2. Side-effect-free proof
# ---------------------------------------------------------------------------


def _imported_module_names(source: str) -> set[str]:
    """Return all top-level module names imported by a Python source string."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
                names.add(node.module.split(".")[0])
            if node.level == 0 and node.module:
                names.add(node.module)
    return names


def test_preflight_imports_no_adversarial_or_side_effect_modules() -> None:
    """The preflight source imports no adversarial execution surface or side-effect module."""
    source = PREFLIGHT_MODULE_PATH.read_text(encoding="utf-8")
    imported = _imported_module_names(source)
    for fragment in FORBIDDEN_IMPORT_FRAGMENTS:
        hits = {name for name in imported if fragment in name}
        assert not hits, f"preflight imports forbidden module(s) matching {fragment!r}: {hits}"


def test_preflight_source_has_no_forbidden_side_effect_tokens() -> None:
    """The preflight source contains no shell/network/dynamic-exec tokens."""
    source = PREFLIGHT_MODULE_PATH.read_text(encoding="utf-8")
    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source, f"preflight source contains forbidden token {token!r}"


def test_preflight_runtime_does_not_import_forbidden_modules() -> None:
    """Running the preflight does not pull any forbidden module into sys.modules."""
    forbidden_present_before = {name for name in sys.modules if "robot_sf.adversarial" in name}
    preflight_issue_5303_contract(repo_root=REPO_ROOT)
    forbidden_present_after = {name for name in sys.modules if "robot_sf.adversarial" in name}
    assert forbidden_present_after == forbidden_present_before, (
        "preflight execution imported an adversarial module: "
        f"{forbidden_present_after - forbidden_present_before}"
    )


def test_check_command_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI check command reproduces the contract hash and exits zero."""
    from scripts.tools.check_issue_5303_search_promotion_contract import main

    exit_code = main(["--repo-root", str(REPO_ROOT)])
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["ready"] is True
    assert payload["blocked"] is False


def test_preflight_detects_contract_field_tampering(tmp_path: Path) -> None:
    """Editing a frozen field after freezing fails the preflight (tamper-evident)."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["budget"]["candidate_budget_per_search_seed_per_method"] = 32  # tampered
    tampered_contract = tmp_path / "tampered_contract.yaml"
    tampered_contract.write_text(yaml.safe_dump(contract), encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tampered_manifest = tmp_path / "contract_frozen.json"
    # Manifest still records the ORIGINAL contract hash, so a field edit must mismatch.
    tampered_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    result = preflight_issue_5303_contract(
        tampered_contract,
        receipt_path=RECEIPT_PATH,
        manifest_path=tampered_manifest,
        repo_root=REPO_ROOT,
    )
    assert result.ready is False
    assert result.checks["contract_hash_matches_manifest"] is False
    assert result.checks["candidate_budget_64_per_seed"] is False


@pytest.mark.parametrize(
    ("section", "field", "expected_blocker"),
    (
        ("family_split", "fit_family_eligible_records", "fit_family_eligible_records"),
        (
            "family_split",
            "fresh_outcome_family_eligible_records",
            "fresh_outcome_family_eligible_records",
        ),
        ("methods", "entries", "methods must be exactly"),
        ("controls", "rejection_controls", "doorway rejection-control seeds"),
        ("objective", "tiers", "objective tiers must be constraints-first"),
        ("decision_rule", "outcomes", "decision_rule must retain"),
    ),
)
def test_preflight_returns_blocked_for_malformed_contract_lists(
    tmp_path: Path,
    section: str,
    field: str,
    expected_blocker: str,
) -> None:
    """Malformed list-shaped contract fields fail closed instead of raising TypeError."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(contract, dict)
    nested = contract[section]
    assert isinstance(nested, dict)
    nested[field] = None
    contract_path = tmp_path / f"malformed_{section}_{field}.yaml"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")

    result = preflight_issue_5303_contract(contract_path, repo_root=REPO_ROOT)

    assert result.ready is False
    assert any(expected_blocker in blocker for blocker in result.blockers)


def test_preflight_returns_blocked_for_malformed_receipt_records(tmp_path: Path) -> None:
    """A parseable receipt with malformed records is reported as blocked, not raised."""
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert isinstance(receipt, dict)
    receipt["records"] = None
    receipt_path = tmp_path / "malformed_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(contract, dict)
    entry_gate = contract["entry_gate"]
    assert isinstance(entry_gate, dict)
    entry_gate["recertification_receipt_path"] = str(receipt_path)
    contract_path = tmp_path / "contract_with_malformed_receipt.yaml"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")

    result = preflight_issue_5303_contract(contract_path, repo_root=REPO_ROOT)

    assert result.ready is False
    assert any(
        "receipt.records must be a list of mappings" in blocker for blocker in result.blockers
    )


def test_preflight_fails_closed_for_malformed_receipt_archive_id(tmp_path: Path) -> None:
    """An excluded receipt row with an unhashable archive ID must block without raising."""
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert isinstance(receipt, dict)
    records = receipt["records"]
    assert isinstance(records, list)
    excluded_record = next(
        record for record in records if record["after"]["benchmark_eligibility"] != "eligible"
    )
    excluded_record["archive_id"] = ["not", "hashable"]
    receipt_path = tmp_path / "receipt_with_unhashable_archive_id.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(contract, dict)
    entry_gate = contract["entry_gate"]
    assert isinstance(entry_gate, dict)
    entry_gate["recertification_receipt_path"] = str(receipt_path)
    result = _preflight_rehashed_contract(
        tmp_path,
        contract,
        "contract_with_unhashable_archive_id",
        receipt_path=receipt_path,
    )

    assert result.ready is False
    assert result.checks["receipt_record_archive_ids_shape"] is False
    assert any("string archive_id" in blocker for blocker in result.blockers)


def test_preflight_fails_closed_for_missing_diagnostic_runner(tmp_path: Path) -> None:
    """A missing frozen runner path blocks the handoff instead of raising on source read."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(contract, dict)
    step3 = contract["step3_execution"]
    assert isinstance(step3, dict)
    step3["runner_ref"] = "scripts/tools/missing_issue_5303_runner.py"

    result = _preflight_rehashed_contract(tmp_path, contract, "contract_with_missing_runner")

    assert result.ready is False
    assert result.checks["step3_runner_static_support"] is False
    assert result.checks["step3_warm_start_wiring"] is False
    assert any("frozen runner" in blocker for blocker in result.blockers)


def test_preflight_fails_closed_for_unhashable_diagnostic_boundary(tmp_path: Path) -> None:
    """A malformed non-promotion boundary blocks without passing through ``set``."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(contract, dict)
    future_run = contract["future_run_declaration"]
    assert isinstance(future_run, dict)
    diagnostic_run = future_run["separately_justified_diagnostic_search_run"]
    assert isinstance(diagnostic_run, dict)
    diagnostic_run["never_authorizes"] = [{"not": "hashable"}]

    result = _preflight_rehashed_contract(
        tmp_path, contract, "contract_with_unhashable_diagnostic_boundary"
    )

    assert result.ready is False
    assert result.checks["diagnostic_run_requires_separate_authorization"] is False
    assert any("diagnostic binding" in blocker for blocker in result.blockers)


def test_preflight_rejects_diagnostic_execution_authorization(tmp_path: Path) -> None:
    """The historical command binding cannot authorize a diagnostic run."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(contract, dict)
    future_run = contract["future_run_declaration"]
    assert isinstance(future_run, dict)
    diagnostic_run = future_run["separately_justified_diagnostic_search_run"]
    assert isinstance(diagnostic_run, dict)
    diagnostic_run["authorized"] = True

    result = _preflight_rehashed_contract(
        tmp_path,
        contract,
        "contract_with_unauthorized_diagnostic_flip",
    )

    assert result.ready is False
    assert result.checks["diagnostic_run_requires_separate_authorization"] is False
    assert any("pending separate review" in blocker for blocker in result.blockers)


@pytest.mark.parametrize(
    ("path", "check_name"),
    (
        (("entry_gate", "entry_gate_satisfied"), "entry_gate_satisfied"),
        (("counted_weak_point_gates", "fail_closed"), "gates_fail_closed"),
    ),
)
def test_preflight_requires_literal_true_for_gate_flags(
    tmp_path: Path,
    path: tuple[str, ...],
    check_name: str,
) -> None:
    """Truthy strings must not satisfy frozen boolean gate flags."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(contract, dict)
    _set_nested_contract_value(contract, path, "false")

    result = _preflight_rehashed_contract(tmp_path, contract, f"contract_with_string_{check_name}")

    assert result.ready is False
    assert result.checks[check_name] is False


def test_preflight_detects_threshold_weakening(tmp_path: Path) -> None:
    """Weakening the positive-gate threshold to make the gate 'testable' fails the preflight."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["positive_gate"]["both_null_tests_p_le_0_05"] = False  # weakened
    contract["power_analysis"]["positive_gate_robustly_testable"] = True  # dishonest
    contract["future_run_declaration"]["status"] = "promote"  # dishonest
    tampered_contract = tmp_path / "weakened_contract.yaml"
    tampered_contract.write_text(yaml.safe_dump(contract), encoding="utf-8")
    # Recompute manifest hash so only the field-value checks fail, not the hash check.
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["contract_sha256"] = hashlib.sha256(tampered_contract.read_bytes()).hexdigest()
    tampered_manifest = tmp_path / "contract_frozen.json"
    tampered_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    result = preflight_issue_5303_contract(
        tampered_contract,
        receipt_path=RECEIPT_PATH,
        manifest_path=tampered_manifest,
        repo_root=REPO_ROOT,
    )
    assert result.ready is False
    assert result.checks["positive_gate_thresholds_kept"] is False
    assert result.checks["positive_gate_not_robustly_testable"] is False
    assert result.checks["future_run_diagnostic_inconclusive"] is False


@pytest.mark.parametrize(
    ("mutation", "check_name"),
    [
        ("search_seed", "search_seeds_exactly_three"),
        ("null_test_name", "null_tests_two_seed_permutations"),
        ("hard_veto", "objective_hard_constraint_veto"),
        ("confirmation_threshold", "counted_weak_point_gate_semantics_frozen"),
    ],
)
def test_preflight_rejects_rehashed_frozen_design_drift(
    tmp_path: Path,
    mutation: str,
    check_name: str,
) -> None:
    """Rehashing cannot hide drift in a seed, test, veto, or confirmation gate."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    if mutation == "search_seed":
        contract["budget"]["search_seeds"] = [530301, 530302, 530399]
    elif mutation == "null_test_name":
        contract["null_tests"]["tests"][0]["name"] = "invented_permutation"
    elif mutation == "hard_veto":
        contract["objective"]["tiers"][0]["veto"] = False
    elif mutation == "confirmation_threshold":
        contract["counted_weak_point_gates"]["confirmation"]["mechanism_threshold_seeds"] = "3_of_5"
    else:  # pragma: no cover - the parametrization above is exhaustive.
        raise AssertionError(f"unknown mutation: {mutation}")
    tampered_contract = tmp_path / "rehashed_drift.yaml"
    tampered_contract.write_text(yaml.safe_dump(contract), encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["contract_sha256"] = hashlib.sha256(tampered_contract.read_bytes()).hexdigest()
    tampered_manifest = tmp_path / "contract_frozen.json"
    tampered_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    result = preflight_issue_5303_contract(
        tampered_contract,
        receipt_path=RECEIPT_PATH,
        manifest_path=tampered_manifest,
        repo_root=REPO_ROOT,
    )

    assert result.checks["contract_hash_matches_manifest"] is True
    assert result.checks[check_name] is False
    assert result.ready is False


def test_preflight_detects_handoff_input_hash_tampering(tmp_path: Path) -> None:
    """A stale runner/input hash cannot pass the side-effect-free readiness gate."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["input_provenance"]["required_inputs"][0]["sha256"] = "0" * 64
    tampered_contract = tmp_path / "stale_input_hash.yaml"
    tampered_contract.write_text(yaml.safe_dump(contract), encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["contract_sha256"] = hashlib.sha256(tampered_contract.read_bytes()).hexdigest()
    tampered_manifest = tmp_path / "contract_frozen.json"
    tampered_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    result = preflight_issue_5303_contract(
        tampered_contract,
        receipt_path=RECEIPT_PATH,
        manifest_path=tampered_manifest,
        repo_root=REPO_ROOT,
    )

    assert result.ready is False
    assert result.checks["contract_hash_matches_manifest"] is True
    assert result.checks["input_provenance_hashes"] is False


def test_preflight_detects_certified_archive_file_tampering(tmp_path: Path) -> None:
    """A receipt's stale self-report cannot hide a changed certified archive file."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    archive_path = REPO_ROOT / contract["entry_gate"]["certified_archive_path"]
    tampered_archive = tmp_path / "archive.json"
    tampered_archive.write_bytes(archive_path.read_bytes() + b"\narchive-tamper\n")
    contract["entry_gate"]["certified_archive_path"] = str(tampered_archive)
    tampered_contract = tmp_path / "archive_tamper_contract.yaml"
    tampered_contract.write_text(yaml.safe_dump(contract), encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["contract_sha256"] = hashlib.sha256(tampered_contract.read_bytes()).hexdigest()
    tampered_manifest = tmp_path / "contract_frozen.json"
    tampered_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    result = preflight_issue_5303_contract(
        tampered_contract,
        receipt_path=RECEIPT_PATH,
        manifest_path=tampered_manifest,
        repo_root=REPO_ROOT,
    )

    assert result.ready is False
    assert result.checks["certified_archive_exists"] is True
    assert result.checks["certified_archive_file_hash_matches_contract"] is False
    assert result.checks["archive_hash_consistent"] is False
    assert any("certified archive file SHA-256" in blocker for blocker in result.blockers)


def test_preflight_detects_incomplete_diagnostic_command(tmp_path: Path) -> None:
    """The preflight rejects a handoff that omits a declared output artifact flag."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    command = contract["step3_execution"]["diagnostic_search_command"]
    contract["step3_execution"]["diagnostic_search_command"] = command.replace(
        " --out-json output/adversarial/issue_5303_search_promotion/report.json", ""
    )
    tampered_contract = tmp_path / "incomplete_command.yaml"
    tampered_contract.write_text(yaml.safe_dump(contract), encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["contract_sha256"] = hashlib.sha256(tampered_contract.read_bytes()).hexdigest()
    tampered_manifest = tmp_path / "contract_frozen.json"
    tampered_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    result = preflight_issue_5303_contract(
        tampered_contract,
        receipt_path=RECEIPT_PATH,
        manifest_path=tampered_manifest,
        repo_root=REPO_ROOT,
    )

    assert result.ready is False
    assert result.checks["contract_hash_matches_manifest"] is True
    assert result.checks["step3_execution_command_complete"] is False


@pytest.mark.parametrize("suffix", (" unexpected", " --unknown-option value"))
def test_preflight_rejects_extra_or_unknown_diagnostic_command_tokens(
    tmp_path: Path, suffix: str
) -> None:
    """Static command parsing must agree with the actual diagnostic CLI parser."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(contract, dict)
    step3 = contract["step3_execution"]
    assert isinstance(step3, dict)
    command = step3["diagnostic_search_command"]
    assert isinstance(command, str)
    step3["diagnostic_search_command"] = command + suffix

    result = _preflight_rehashed_contract(tmp_path, contract, "command_with_extra_tokens")

    assert result.ready is False
    assert result.checks["step3_command_parses"] is False
    assert result.checks["step3_execution_command_complete"] is False


@pytest.mark.parametrize(
    ("mutation", "check_name"),
    [
        ("entry_gate_state", "entry_gate_bindings_frozen"),
        ("control_role", "controls_frozen"),
        ("method_sampler", "method_entries_frozen"),
        ("candidate_space_certification", "candidate_space_and_feasibility_frozen"),
        ("gate_rule", "counted_weak_point_gate_semantics_frozen"),
        ("estimand_gate", "estimand_definition_frozen"),
        ("uncertainty_resamples", "uncertainty_definition_frozen"),
        ("null_both_required", "null_tests_both_required"),
        ("attrition_reason", "missing_invalid_attrition_policy_frozen"),
        ("input_hash_algorithm", "input_provenance_algorithm"),
        ("diagnostic_entrypoint", "step3_diagnostic_command_entrypoint"),
    ],
)
def test_preflight_rejects_rehashed_contract_semantic_drift(
    tmp_path: Path, mutation: str, check_name: str
) -> None:
    """Rehashing cannot hide drift in any machine-readable promotion binding."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    try:
        path, value = SEMANTIC_DRIFT_MUTATIONS[mutation]
    except KeyError as exc:  # pragma: no cover - the parametrization is exhaustive.
        raise AssertionError(f"unknown mutation: {mutation}") from exc
    _set_nested_contract_value(contract, path, value)

    tampered_contract = tmp_path / f"rehashed_{mutation}.yaml"
    tampered_contract.write_text(yaml.safe_dump(contract), encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["contract_sha256"] = hashlib.sha256(tampered_contract.read_bytes()).hexdigest()
    tampered_manifest = tmp_path / f"manifest_{mutation}.json"
    tampered_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    result = preflight_issue_5303_contract(
        tampered_contract,
        receipt_path=RECEIPT_PATH,
        manifest_path=tampered_manifest,
        repo_root=REPO_ROOT,
    )

    assert result.checks["contract_hash_matches_manifest"] is True
    assert result.checks[check_name] is False
    assert result.ready is False


def test_preflight_fails_closed_for_missing_contract_and_writes_requested_payload(
    tmp_path: Path,
) -> None:
    """Missing contracts block before parsing, while explicit output remains machine-readable."""
    missing_result = preflight_issue_5303_contract(
        tmp_path / "missing_contract.yaml",
        receipt_path=RECEIPT_PATH,
        manifest_path=MANIFEST_PATH,
        repo_root=REPO_ROOT,
    )

    assert missing_result.ready is False
    assert missing_result.checks == {"contract_exists": False}
    assert any("contract not found" in blocker for blocker in missing_result.blockers)

    output = tmp_path / "preflight" / "payload.json"
    result = preflight_issue_5303_contract(repo_root=REPO_ROOT)
    dump_preflight_payload(result, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert payload["metadata"]["contract_file_sha256"] == _sha256_file(CONTRACT_PATH)


# ---------------------------------------------------------------------------
# 3. Honest power analysis and diagnostic declaration
# ---------------------------------------------------------------------------


def test_permutation_power_math() -> None:
    """C(6,3)=20 arrangements; min two-sided p = 2/20 = 0.10; min one-sided p = 1/20 = 0.05."""
    assert math.comb(6, 3) == 20
    arrangements, one_sided, two_sided = _min_permutation_p_values(3)
    assert arrangements == 20
    assert one_sided == pytest.approx(1 / 20)
    assert two_sided == pytest.approx(2 / 20)
    assert two_sided == pytest.approx(0.10)
    assert two_sided > 0.05
    assert one_sided == pytest.approx(0.05)


def test_power_analysis_fields_match_recomputed_math() -> None:
    result = preflight_issue_5303_contract(repo_root=REPO_ROOT)
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    power = contract["power_analysis"]
    assert power["permutation_arrangements_C_6_3"] == 20
    assert power["min_two_sided_permutation_p"] == pytest.approx(0.10)
    assert power["min_one_sided_permutation_p"] == pytest.approx(0.05)
    assert power["two_sided_can_reject_at_threshold"] is False
    assert power["positive_gate_robustly_testable"] is False
    assert result.checks["power_arrangements_C_6_3"]
    assert result.checks["power_min_two_sided_p"]
    assert result.checks["power_two_sided_cannot_reject"]
    assert result.checks["positive_gate_not_robustly_testable"]


def test_future_run_is_diagnostic_inconclusive_without_weakening_thresholds() -> None:
    """The run is preregistered diagnostic/inconclusive; thresholds stay as proposed."""
    result = preflight_issue_5303_contract(repo_root=REPO_ROOT)
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["future_run_declaration"]["status"] == "diagnostic_inconclusive"
    assert contract["future_run_declaration"]["thresholds_not_weakened"] is True
    assert contract["future_run_declaration"]["declare_before_outcomes"] is True
    assert contract["positive_gate"]["thresholds_weakened"] is False
    assert contract["positive_gate"]["admitted_weak_points_floor"] == 2
    assert contract["positive_gate"]["both_null_tests_p_le_0_05"] is True
    assert result.checks["future_run_diagnostic_inconclusive"]
    assert result.checks["diagnostic_thresholds_not_weakened"]
    assert result.checks["positive_gate_not_weakened"]


def test_reimporting_preflight_module_is_idempotent() -> None:
    """Re-importing the module does not mutate global sampler/optimizer registries."""
    importlib.reload(
        importlib.import_module("robot_sf.benchmark.issue_5303_search_promotion_preregistration")
    )
    test_preflight_passes_on_frozen_contract()
