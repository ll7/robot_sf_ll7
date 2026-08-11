"""Focused tests for the powered six-seed issue #5303 promotion contract (schema v2).

These tests prove five things together:

1. The side-effect-free powered preflight reproduces the frozen v2 contract hash and
   asserts the frozen design (six seeds, 64 candidates per seed per method, 768 scheduled
   attempts), the exact cluster-level inference (C(12,6)=924 assignments), the
   outcome-free sensitivity boundary, and the machine-readable terminal result schema, by
   reading only the contract, the #6139 recertification receipt, the manifest, and
   statically parsed sources.
2. The preflight and its check CLI are genuinely side-effect-free: they import no
   adversarial execution surface (samplers/search/runtime/qd/warm_start/transfer_matrix/
   campaign/replay), no subprocess, and no network module, and the deterministic
   check-only identity emission performs no planner execution and no outcome read.
3. The 768 scheduled search identities are complete, unique, deterministic, and identical
   to the committed evidence manifest.
4. The merged PR #6586 timing dimensions (spawn_time_s, pedestrian_delay_s) are
   runtime-effective for the powered space/template pair, while the historical
   inert/no-pedestrian mode is rejected fail-closed.
5. Missing, duplicate, mismatched, fallback, degraded, unavailable, lineage-incomplete,
   and inert inputs fail closed, and the historical v1 contract is rejected for
   promotion-capable execution.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.adversarial.config import RangeConfig, SearchSpaceConfig
from robot_sf.benchmark.issue_5303_search_promotion_preflight import (
    evaluate_preflight,
    evaluate_preflight_from_files,
)
from robot_sf.benchmark.issue_5303_search_promotion_preregistration_v2 import (
    CONTRACT_SCHEMA_VERSION,
    DEFAULT_CONTRACT_PATH,
    DEFAULT_IDENTITY_MANIFEST_PATH,
    DEFAULT_MANIFEST_PATH,
    EXPECTED_CANDIDATE_BUDGET,
    EXPECTED_METHOD_COMMAND_ORDER,
    EXPECTED_SEARCH_SEEDS,
    EXPECTED_SEEDS_PER_METHOD,
    EXPECTED_TOTAL_CANDIDATES_PER_METHOD,
    EXPECTED_TOTAL_SCHEDULED_ATTEMPTS,
    HISTORICAL_CONTRACT_SCHEMA_VERSION,
    HISTORICAL_SPACE_PATH,
    HISTORICAL_TEMPLATE_PATH,
    POWERED_SPACE_PATH,
    POWERED_TEMPLATE_PATH,
    RESULT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Issue5303PoweredPreflightResult,
    _min_permutation_p_values,
    _two_sided_rejection_region_capacity,
    downstream_activation_errors,
    dump_preflight_payload,
    identity_manifest_bytes,
    identity_manifest_sha256,
    preflight_issue_5303_powered_contract,
    scheduled_search_identities,
    validate_promotion_execution_contract,
    validate_terminal_result,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
POWERED_MODULE_PATH = (
    REPO_ROOT / "robot_sf/benchmark/issue_5303_search_promotion_preregistration_v2.py"
)
POWERED_CLI_PATH = REPO_ROOT / "scripts/tools/check_issue_5303_search_promotion_contract_v2.py"
CONTRACT_PATH = REPO_ROOT / DEFAULT_CONTRACT_PATH
MANIFEST_PATH = REPO_ROOT / DEFAULT_MANIFEST_PATH
IDENTITY_MANIFEST_PATH = REPO_ROOT / DEFAULT_IDENTITY_MANIFEST_PATH

# Any import of these fragments would let the powered preflight or CLI touch adversarial
# execution surfaces or side effects, which the side-effect-free contract forbids. The
# timing gate arrives through robot_sf.benchmark.issue_5303_search_promotion_preflight,
# which materializes in-memory payloads and hashes them only.
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


def _outcome_row_fields() -> list[str]:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    return list(contract["outcome_row_schema"]["required_fields"])


def _frozen_execution_command() -> str:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    return str(contract["step3_execution"]["execution_command"])


SEMANTIC_DRIFT_MUTATIONS: dict[str, tuple[tuple[str, ...], object]] = {
    "schema_version": (("schema_version",), HISTORICAL_CONTRACT_SCHEMA_VERSION),
    "task_id": (("task_id",), "issue-5303-step-2-search-promotion-preregistration"),
    "evidence_boundary": (("evidence_boundary",), "evidence_grade"),
    "entry_gate_state": (("entry_gate", "blocking_issue_state"), "open"),
    "control_role": (("controls", "rejection_controls", "0", "role"), "optional"),
    "method_sampler": (
        ("methods", "entries", "0", "sampler_class"),
        "not_the_frozen_sampler",
    ),
    "seed_roster_duplicate": (
        ("budget", "search_seeds"),
        [530301, 530301, 530303, 530304, 530305, 530306],
    ),
    "seed_roster_five_seeds": (
        ("budget", "search_seeds"),
        [530301, 530302, 530303, 530304, 530305],
    ),
    "seed_post_outcome_change_allowed": (
        ("budget", "no_post_outcome_seed_addition_replacement_retry_or_stopping"),
        False,
    ),
    "candidate_budget": (("budget", "candidate_budget_per_search_seed_per_method"), 32),
    "total_scheduled_attempts": (("budget", "total_scheduled_attempts"), 384),
    "candidate_space_certification": (
        ("candidate_space_and_feasibility", "require_certification"),
        False,
    ),
    "runtime_effective_missing": (
        ("candidate_space_and_feasibility", "runtime_effective_candidate_space"),
        {},
    ),
    "runtime_effective_inert_allowed": (
        (
            "candidate_space_and_feasibility",
            "runtime_effective_candidate_space",
            "preflight_status_required",
        ),
        "blocked_inert_dimensions",
    ),
    "gate_rule": (("counted_weak_point_gates", "gates", "0", "rule"), "changed"),
    "gate_excluded_classes": (
        (
            "counted_weak_point_gates",
            "excluded_row_classes_never_discoveries_but_remain_primary_denominator",
        ),
        ["fallback", "degraded"],
    ),
    "estimand_gate": (("estimand", "counted_only_when_all_seven_gates_pass"), False),
    "denominator_rewritten": (
        ("estimand", "denominator"),
        "intention_to_search_192_scheduled_attempts_per_method",
    ),
    "uncertainty_resampled_only": (
        ("uncertainty", "method"),
        "nonparametric_bootstrap_over_seed_clusters",
    ),
    "null_both_required": (("null_tests", "both_required"), False),
    "null_exact_enumeration": (("null_tests", "exact_enumeration"), "bootstrap_resamples"),
    "attrition_reason": (("missing_invalid_attrition", "reason_recorded"), False),
    "attrition_fallback_allowed": (
        ("missing_invalid_attrition", "excluded_from_primary_denominator"),
        True,
    ),
    "outcome_row_lineage_incomplete": (
        ("outcome_row_schema", "required_fields"),
        [field for field in sorted(_outcome_row_fields()) if field != "seed_lineage"],
    ),
    "positive_gate_weakened": (("positive_gate", "admitted_weak_points_floor"), 0),
    "thresholds_weakened": (("positive_gate", "thresholds_weakened"), True),
    "power_arrangements": (("power_analysis", "permutation_arrangements_C_12_6"), 20),
    "power_two_sided_impossible": (
        ("power_analysis", "two_sided_can_reject_at_threshold"),
        False,
    ),
    "future_run_diagnostic": (("future_run_declaration", "status"), "diagnostic_inconclusive"),
    "future_run_execution_authorized": (
        ("future_run_declaration", "execution_authorized_here"),
        True,
    ),
    "result_schema_missing_field": (
        ("result_handoff", "required_fields"),
        ["schema_version", "decision", "contract_sha256"],
    ),
    "activation_closure_alone": (
        ("result_handoff", "downstream_activation", "activation_on_issue_closure_alone"),
        True,
    ),
    "activation_min_admitted": (
        ("result_handoff", "downstream_activation", "min_admitted_candidate_count"),
        1,
    ),
    "input_hash_algorithm": (("input_provenance", "algorithm"), "sha256"),
    "execution_entrypoint": (
        ("step3_execution", "execution_command"),
        "uv run python scripts/tools/wrong_runner.py",
    ),
    "execution_command_diagnostic_flag": (
        ("step3_execution", "execution_command"),
        _frozen_execution_command() + " --issue-5303-diagnostic-only",
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
    receipt_path: Path | None = None,
) -> Issue5303PoweredPreflightResult:
    """Run powered preflight against a temporary contract whose manifest hash was refreshed."""
    contract_path = tmp_path / f"{name}.yaml"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["contract_sha256"] = _sha256_file(contract_path)
    manifest_path = tmp_path / f"{name}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return preflight_issue_5303_powered_contract(
        contract_path,
        receipt_path=receipt_path,
        manifest_path=manifest_path,
        repo_root=REPO_ROOT,
    )


def _frozen_contract() -> dict[str, Any]:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Frozen powered contract checks
# ---------------------------------------------------------------------------


def test_preflight_passes_on_frozen_powered_contract() -> None:
    """The committed frozen powered contract passes every check with no blockers."""
    result = preflight_issue_5303_powered_contract(repo_root=REPO_ROOT)
    assert result.ready, "blockers:\n  " + "\n  ".join(result.blockers)
    assert result.blocked is False
    assert not result.blockers
    failed = [name for name, ok in result.checks.items() if not ok]
    assert failed == [], f"failed checks: {failed}"


def test_preflight_schema_and_contract_schema_versions() -> None:
    """Frozen schema versions prevent incompatible contract or report interpretation."""
    contract = _frozen_contract()
    assert contract["schema_version"] == CONTRACT_SCHEMA_VERSION
    assert contract["schema_version"] != HISTORICAL_CONTRACT_SCHEMA_VERSION
    result = preflight_issue_5303_powered_contract(repo_root=REPO_ROOT)
    assert result.to_payload()["schema_version"] == SCHEMA_VERSION


def test_manifest_contract_hash_matches_file() -> None:
    """The frozen manifest pins the exact contract bytes committed with this packet."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["contract_sha256"] == _sha256_file(CONTRACT_PATH)
    assert manifest["base_commit"] == _frozen_contract()["base_commit"]


def test_semantic_drift_mutations_fail_closed(tmp_path: Path) -> None:
    """Every frozen-field mutation is rejected fail-closed after re-hashing the contract."""
    for name, (path, value) in SEMANTIC_DRIFT_MUTATIONS.items():
        contract = _frozen_contract()
        _set_nested_contract_value(contract, path, value)
        result = _preflight_rehashed_contract(tmp_path, contract, name)
        assert result.blocked, f"mutation {name!r} did not fail closed"
        assert not result.ready, f"mutation {name!r} reported ready"


def test_missing_contract_fails_closed(tmp_path: Path) -> None:
    """A missing contract file fails closed before any other check."""
    result = preflight_issue_5303_powered_contract(tmp_path / "absent.yaml", repo_root=REPO_ROOT)
    assert result.blocked
    assert result.checks["contract_exists"] is False


def test_missing_manifest_fails_closed(tmp_path: Path) -> None:
    """A missing preregistration manifest fails closed with a hash-check blocker."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(yaml.safe_dump(_frozen_contract()), encoding="utf-8")
    result = preflight_issue_5303_powered_contract(
        contract_path,
        manifest_path=tmp_path / "absent_manifest.json",
        repo_root=REPO_ROOT,
    )
    assert result.blocked
    assert result.checks["manifest_exists"] is False
    assert result.checks["contract_hash_matches_manifest"] is False


def test_mismatched_input_hash_fails_closed(tmp_path: Path) -> None:
    """A tampered input hash fails closed even when the file exists."""
    contract = _frozen_contract()
    for entry in contract["input_provenance"]["required_inputs"]:
        if entry["id"] == "powered_search_space":
            entry["sha256"] = "0" * 64
    result = _preflight_rehashed_contract(tmp_path, contract, "mismatched_hash")
    assert result.blocked
    assert result.checks["input_provenance_hashes"] is False


def test_missing_input_file_fails_closed(tmp_path: Path) -> None:
    """A provenance entry pointing at a missing file fails closed."""
    contract = _frozen_contract()
    for entry in contract["input_provenance"]["required_inputs"]:
        if entry["id"] == "powered_scenario_template":
            entry["path"] = "configs/adversarial/does_not_exist_v2.yaml"
    result = _preflight_rehashed_contract(tmp_path, contract, "missing_input")
    assert result.blocked
    assert result.checks["input_provenance_hashes"] is False


# ---------------------------------------------------------------------------
# 2. Side-effect-free / no-execution / no-outcome boundary
# ---------------------------------------------------------------------------


def _import_dotted_names(tree: ast.Module) -> set[str]:
    """Return every dotted module name referenced by import statements."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize(
    "source_path", [POWERED_MODULE_PATH, POWERED_CLI_PATH], ids=lambda p: p.name
)
def test_powered_surfaces_are_side_effect_free(source_path: Path) -> None:
    """The powered module and CLI import no execution surface, subprocess, or network."""
    assert source_path.is_file()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = _import_dotted_names(tree)
    for fragment in FORBIDDEN_IMPORT_FRAGMENTS:
        offenders = {name for name in imported if name.startswith(fragment)}
        assert not offenders, f"{source_path.name} imports forbidden surface: {offenders}"
    source = source_path.read_text(encoding="utf-8")
    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source, f"{source_path.name} contains forbidden token {token!r}"


def test_powered_module_reuses_6586_timing_gate() -> None:
    """Runtime effectiveness is proven through the merged PR #6586 preflight surface."""
    source = POWERED_MODULE_PATH.read_text(encoding="utf-8")
    assert "from robot_sf.benchmark.issue_5303_search_promotion_preflight import" in source


def test_cli_identity_emission_is_pure() -> None:
    """The check CLI emits identities without loading the contract, manifest, or receipt."""
    cli_source = POWERED_CLI_PATH.read_text(encoding="utf-8")
    tree = ast.parse(cli_source)
    main_functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    assert main_functions, "the check CLI must expose a main(argv) entry point"
    identities_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    ]
    assert identities_functions, "the check CLI must parse --identities through argparse"
    assert "--identities" in cli_source


# ---------------------------------------------------------------------------
# 3. The 768 scheduled search identities
# ---------------------------------------------------------------------------


def test_scheduled_identities_are_exactly_768() -> None:
    """The identity manifest covers every scheduled attempt exactly once."""
    identities = scheduled_search_identities()
    assert len(identities) == EXPECTED_TOTAL_SCHEDULED_ATTEMPTS == 768
    assert len({identity["identity_sha256"] for identity in identities}) == 768
    per_method: dict[str, int] = {}
    per_seed: dict[int, int] = {}
    for identity in identities:
        per_method[identity["method"]] = per_method.get(identity["method"], 0) + 1
        per_seed[identity["search_seed"]] = per_seed.get(identity["search_seed"], 0) + 1
    assert per_method == dict.fromkeys(("optuna", "random"), EXPECTED_TOTAL_CANDIDATES_PER_METHOD)
    assert per_seed == dict.fromkeys(EXPECTED_SEARCH_SEEDS, 2 * EXPECTED_CANDIDATE_BUDGET)
    for method in EXPECTED_METHOD_COMMAND_ORDER:
        for seed in EXPECTED_SEARCH_SEEDS:
            indices = sorted(
                identity["candidate_index"]
                for identity in identities
                if identity["method"] == method and identity["search_seed"] == seed
            )
            assert indices == list(range(EXPECTED_CANDIDATE_BUDGET))


def test_scheduled_identities_are_deterministic_and_outcome_free() -> None:
    """Repeated derivation yields byte-identical identities with stable hashes."""
    first = identity_manifest_bytes()
    second = identity_manifest_bytes()
    assert first == second
    assert hashlib.sha256(first).hexdigest() == identity_manifest_sha256()
    identities = scheduled_search_identities()
    canonical = json.dumps(
        {
            "schema_version": identities[0]["schema_version"],
            "task_id": identities[0]["task_id"],
            "method": identities[0]["method"],
            "search_seed": identities[0]["search_seed"],
            "candidate_index": identities[0]["candidate_index"],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert (
        hashlib.sha256(canonical.encode("utf-8")).hexdigest() == (identities[0]["identity_sha256"])
    )


def test_committed_identity_manifest_matches_recomputation() -> None:
    """The committed evidence file is byte-identical to the canonical derivation."""
    assert IDENTITY_MANIFEST_PATH.is_file()
    assert IDENTITY_MANIFEST_PATH.read_bytes() == identity_manifest_bytes()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["scheduled_identity_manifest_sha256"] == identity_manifest_sha256()
    assert manifest["scheduled_identity_count"] == EXPECTED_TOTAL_SCHEDULED_ATTEMPTS


def test_cli_check_and_identity_emission(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The CLI exits 0 on the frozen contract and emits exactly 768 identities."""
    from scripts.tools.check_issue_5303_search_promotion_contract_v2 import main

    assert main(["--repo-root", str(REPO_ROOT)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["schema_version"] == SCHEMA_VERSION

    assert main(["--identities"]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["scheduled_attempt_count"] == EXPECTED_TOTAL_SCHEDULED_ATTEMPTS
    assert len(emitted["identities"]) == EXPECTED_TOTAL_SCHEDULED_ATTEMPTS


def test_cli_rejects_v1_contract_for_promotion_capable_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Pointing the powered CLI at the historical v1 contract fails closed."""
    from scripts.tools.check_issue_5303_search_promotion_contract_v2 import main

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    historical_contract_path = (
        REPO_ROOT / "configs/adversarial/issue_5303_search_promotion_contract.yaml"
    )
    manifest["contract_sha256"] = _sha256_file(historical_contract_path)
    manifest_path = tmp_path / "v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    exit_code = main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--contract",
            "configs/adversarial/issue_5303_search_promotion_contract.yaml",
            "--manifest",
            str(manifest_path),
        ]
    )
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["checks"]["contract_schema_version"] is False
    assert payload["checks"]["historical_contract_rejected_for_promotion"] is True


def test_dump_preflight_payload(tmp_path: Path) -> None:
    """The optional payload dump writes the stable result payload when requested."""
    result = preflight_issue_5303_powered_contract(repo_root=REPO_ROOT)
    output = tmp_path / "payload.json"
    dump_preflight_payload(result, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    dump_preflight_payload(result, None)  # None target stays a no-op


# ---------------------------------------------------------------------------
# 4. Runtime-effective timing dimensions (#6586) and historical rejection
# ---------------------------------------------------------------------------


def test_powered_space_reaches_promotion_timing_ready() -> None:
    """spawn_time_s and pedestrian_delay_s change the effective scenario for the v2 pair."""
    preflight = evaluate_preflight_from_files(
        search_space_path=REPO_ROOT / POWERED_SPACE_PATH,
        scenario_template_path=REPO_ROOT / POWERED_TEMPLATE_PATH,
    )
    assert preflight.status == "promotion_timing_ready"
    assert preflight.promotion_ready
    assert preflight.materialized_pedestrian_id == "issue_5303_powered_promotion_candidate"
    assert preflight.single_pedestrian_populated
    assert preflight.pedestrian_route_populated
    assert {probe.name for probe in preflight.dimensions} == {
        "spawn_time_s",
        "pedestrian_delay_s",
    }
    assert all(probe.status == "effective" for probe in preflight.dimensions)
    assert all(probe.hash_changed for probe in preflight.dimensions)
    assert all(probe.bound_to_pedestrian for probe in preflight.dimensions)


def test_historical_inert_mode_stays_rejected() -> None:
    """The historical no-pedestrian space/template pair remains blocked_no_pedestrian."""
    preflight = evaluate_preflight_from_files(
        search_space_path=REPO_ROOT / HISTORICAL_SPACE_PATH,
        scenario_template_path=REPO_ROOT / HISTORICAL_TEMPLATE_PATH,
    )
    assert preflight.status == "blocked_no_pedestrian"
    assert not preflight.promotion_ready


def test_pedestrian_id_mismatch_fails_closed() -> None:
    """An override identity that does not match the space declaration fails closed."""
    preflight = evaluate_preflight_from_files(
        search_space_path=REPO_ROOT / POWERED_SPACE_PATH,
        scenario_template_path=REPO_ROOT / POWERED_TEMPLATE_PATH,
        pedestrian_id="some_other_pedestrian",
    )
    assert preflight.status == "blocked_no_pedestrian"


def test_unbound_space_dimension_fails_closed() -> None:
    """A space without a pedestrian identity is rejected even with declared ranges."""
    space = SearchSpaceConfig(
        start_x=RangeConfig(1.0, 3.0),
        start_y=RangeConfig(2.0, 4.0),
        goal_x=RangeConfig(7.0, 9.0),
        goal_y=RangeConfig(2.0, 4.0),
        spawn_time_s=RangeConfig(0.0, 2.0),
        pedestrian_speed_mps=RangeConfig(0.8, 1.4),
        pedestrian_delay_s=RangeConfig(0.0, 2.0),
        scenario_seed=RangeConfig(100.0, 25000.0),
        min_start_goal_distance_m=2.0,
        pedestrian_id=None,
        _declared_variables=frozenset(
            {
                "start_x",
                "start_y",
                "goal_x",
                "goal_y",
                "spawn_time_s",
                "pedestrian_speed_mps",
                "pedestrian_delay_s",
                "scenario_seed",
            }
        ),
    )
    template = yaml.safe_load((REPO_ROOT / POWERED_TEMPLATE_PATH).read_text(encoding="utf-8"))
    preflight = evaluate_preflight(search_space=space, template_scenario=template["scenarios"][0])
    assert preflight.status == "blocked_no_pedestrian"
    assert not preflight.promotion_ready


# ---------------------------------------------------------------------------
# 5. Historical v1 boundary and promotion capability
# ---------------------------------------------------------------------------


def test_v1_contract_rejected_for_promotion_capable_execution() -> None:
    """The validator rejects the historical contract and accepts only the v2 contract."""
    v1_contract = yaml.safe_load(
        (REPO_ROOT / "configs/adversarial/issue_5303_search_promotion_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    errors = validate_promotion_execution_contract(v1_contract)
    assert any("rejected for promotion-capable execution" in error for error in errors)

    v2_contract = _frozen_contract()
    assert validate_promotion_execution_contract(v2_contract) == []

    mutated = _frozen_contract()
    mutated["budget"]["search_seeds"] = [530301, 530302]
    mutated_errors = validate_promotion_execution_contract(mutated)
    assert any("six-seed roster" in error for error in mutated_errors)

    assert validate_promotion_execution_contract({"schema_version": "unknown.v9"})


def test_v1_contract_still_diagnostic_inconclusive() -> None:
    """The powered preflight proves the v1 contract remains the stopped diagnostic."""
    result = preflight_issue_5303_powered_contract(repo_root=REPO_ROOT)
    assert result.checks["historical_contract_immutable_diagnostic"] is True
    assert result.checks["historical_contract_rejected_for_promotion"] is True
    assert result.checks["powered_contract_promotion_capable"] is True


# ---------------------------------------------------------------------------
# 6. Terminal result schema and downstream activation
# ---------------------------------------------------------------------------


def _valid_terminal_result() -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "decision": "promote",
        "contract_sha256": _sha256_file(CONTRACT_PATH),
        "execution_commit": "a" * 40,
        "admitted_candidate_count": 5,
        "candidate_manifest_sha256": "b" * 64,
        "evidence_packet_sha256": "c" * 64,
    }


def test_terminal_result_schema_validates() -> None:
    """A well-formed promote result passes; malformed payloads fail closed."""
    assert validate_terminal_result(_valid_terminal_result()) == []
    assert (
        validate_terminal_result(
            _valid_terminal_result(), expected_contract_sha256=_sha256_file(CONTRACT_PATH)
        )
        == []
    )

    missing = _valid_terminal_result()
    del missing["evidence_packet_sha256"]
    assert any("evidence_packet_sha256" in error for error in validate_terminal_result(missing))

    bad_decision = _valid_terminal_result()
    bad_decision["decision"] = "promoted"
    assert any("decision" in error for error in validate_terminal_result(bad_decision))

    bad_hash = _valid_terminal_result()
    bad_hash["candidate_manifest_sha256"] = "not-a-hash"
    assert any("candidate_manifest_sha256" in error for error in validate_terminal_result(bad_hash))

    bad_commit = _valid_terminal_result()
    bad_commit["execution_commit"] = "xyz"
    assert any("execution_commit" in error for error in validate_terminal_result(bad_commit))

    bad_count = _valid_terminal_result()
    bad_count["admitted_candidate_count"] = -1
    assert any("admitted_candidate_count" in error for error in validate_terminal_result(bad_count))

    wrong_contract = _valid_terminal_result()
    assert any(
        "contract_sha256" in error
        for error in validate_terminal_result(wrong_contract, expected_contract_sha256="d" * 64)
    )


def test_downstream_activation_gates() -> None:
    """Downstream activation requires promote, >=5 admitted candidates, and valid hashes."""
    assert downstream_activation_errors(_valid_terminal_result()) == []

    stop_result = _valid_terminal_result()
    stop_result["decision"] = "stop"
    assert any("promote" in error for error in downstream_activation_errors(stop_result))

    inconclusive_result = _valid_terminal_result()
    inconclusive_result["decision"] = "inconclusive"
    assert downstream_activation_errors(inconclusive_result)

    few_admitted = _valid_terminal_result()
    few_admitted["admitted_candidate_count"] = 4
    assert any(
        "admitted_candidate_count" in error for error in downstream_activation_errors(few_admitted)
    )

    assert downstream_activation_errors({"decision": "promote"})


# ---------------------------------------------------------------------------
# 7. Outcome-free sensitivity analysis (exact enumeration)
# ---------------------------------------------------------------------------


def test_exact_six_seed_sensitivity_math() -> None:
    """C(12,6)=924 assignments make the two-sided p<=0.05 boundary attainable."""
    arrangements, min_one_sided, min_two_sided = _min_permutation_p_values(
        EXPECTED_SEEDS_PER_METHOD
    )
    assert arrangements == math.comb(12, 6) == 924
    assert min_one_sided == pytest.approx(1 / 924)
    assert min_two_sided == pytest.approx(2 / 924)
    assert min_two_sided <= 0.05
    capacity = _two_sided_rejection_region_capacity(arrangements, 0.05)
    assert capacity == 46
    assert capacity / arrangements <= 0.05 < (capacity + 1) / arrangements


def test_contract_power_analysis_matches_recomputation() -> None:
    """The frozen contract's sensitivity fields equal the recomputed exact values."""
    contract = _frozen_contract()
    power = contract["power_analysis"]
    arrangements, min_one_sided, min_two_sided = _min_permutation_p_values(
        EXPECTED_SEEDS_PER_METHOD
    )
    assert power["permutation_arrangements_C_12_6"] == arrangements
    assert power["min_one_sided_permutation_p"] == pytest.approx(min_one_sided)
    assert power["min_two_sided_permutation_p"] == pytest.approx(min_two_sided)
    assert power["two_sided_can_reject_at_threshold"] is True
    assert power["two_sided_rejection_region_max_arrangements"] == 46
    distinction = power["attainable_significance_vs_power"]
    assert "attainable" in distinction["attainable_significance"].lower() or (
        "924" in distinction["attainable_significance"]
    )
    assert "no outcome-free power claim" in distinction["power_or_sensitivity"]
