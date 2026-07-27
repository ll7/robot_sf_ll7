"""Tests for the v2 row-level independent planner-outcome contract (issue #3275)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from robot_sf.adversarial.independent_outcomes import (
    OUTCOME_SCHEMA_VERSION,
    AdmissionSpec,
    build_independent_outcome_evaluation,
    load_independent_outcomes,
    payload_sha256,
)

_PLANNER = "social_force"
_PLANNER_CFG = "dfdebd497e19a046e41cb2b1e7d7a7f54cd592ac0a465e4149efff19efa16735"
_EVAL_FAMILY = "classic_cross_trap_medium"
_EXECUTION_COMMIT = "ecf997d392a4f2c1a4fb5a56e8101acb030b7e2f"


def _sha256(label: str) -> str:
    """Return a deterministic test-only SHA-256 digest."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _replay() -> dict[str, Any]:
    """A passing deterministic-replay lineage block."""
    signature = _sha256("replay-signature")
    return {
        "exact_signature_match": True,
        "original_signature_sha256": signature,
        "replay_signature_sha256": signature,
    }


def _confirmation(*, confirmed: int = 3, attempts: int = 5) -> dict[str, Any]:
    """A passing 3-of-5 independent-seed confirmation block."""
    return {
        "confirmed_count": confirmed,
        "attempt_count": attempts,
        "stable_attribution": True,
    }


def _row(  # noqa: PLR0913
    *,
    row_id: str,
    manifest_id: str,
    arm: str,
    rank: int,
    failure: bool,
    candidate_pool_index: int | None = None,
    scenario_seed: int = 99001,
    execution_seed: int = 70001,
    record_sha256: str | None = None,
    admission_status: str = "admitted",
    exclusion_reason: str | None = None,
    execution_mode: str = "native",
    scenario_cert: str = "passed",
    candidate_cert: str = "passed",
    replay: dict[str, Any] | None = None,
    confirmation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one v2 outcome row with overridable admission-relevant fields."""
    default_pool_index = rank - 1 if arm == "proposal" else 10_000 + rank - 1
    return {
        "row_id": row_id,
        "candidate_manifest_id": manifest_id,
        "candidate_manifest_sha256": _sha256(f"manifest-{manifest_id}"),
        "selection_arm": arm,
        "selection_rank": rank,
        "candidate_pool_seed": 42,
        "candidate_pool_index": (
            default_pool_index if candidate_pool_index is None else candidate_pool_index
        ),
        "target_planner_id": _PLANNER,
        "target_planner_config_sha256": _PLANNER_CFG,
        "scenario_family": _EVAL_FAMILY,
        "scenario_seed": scenario_seed,
        "execution_seed": execution_seed,
        "execution_commit": _EXECUTION_COMMIT,
        "execution_command": ["python", "-m", "robot_sf.run_eval"],
        "execution_config_lineage": {"config": "eval.yaml", "sha256": "cfg-hash"},
        "execution_mode": execution_mode,
        "primary_failure": "collision" if failure else "none",
        "termination_reason": "collision" if failure else "goal_reached",
        "independent_failure_outcome": failure,
        "scenario_certification_status": scenario_cert,
        "candidate_certification_status": candidate_cert,
        "replay_lineage": replay if replay is not None else _replay(),
        "confirmation_lineage": confirmation if confirmation is not None else _confirmation(),
        "record_sha256": record_sha256 or _sha256(f"record-{manifest_id}"),
        "admission_status": admission_status,
        "exclusion_reason": exclusion_reason,
    }


def _packet(rows: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    """Build a v2 packet wrapping ``rows``."""
    payload: dict[str, Any] = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "source": "unit-test-fixture",
        "artifact": "docs/context/evidence/unit-test.json",
        "outcome_source": "planner_execution",
        "objective": "certified_failure_outcome",
        "target_planner_id": _PLANNER,
        "target_planner_config_sha256": _PLANNER_CFG,
        "eval_archive_sha256": "eval-hash",
        "rows": rows,
    }
    payload.update(overrides)
    return payload


def _balanced_packet(*, proposal_failures: int, random_failures: int, per_arm: int = 4) -> dict:
    """Build a 3-of-5-compatible packet with chosen candidate failure counts."""
    rows: list[dict[str, Any]] = []
    for rank in range(per_arm):
        rows.extend(
            _row(
                row_id=f"prop_{rank}_{seed}",
                manifest_id=f"prop_cand_{rank}",
                arm="proposal",
                rank=rank + 1,
                failure=rank < proposal_failures,
                scenario_seed=99_000 + rank,
                execution_seed=70_000 + rank * 10 + seed,
                confirmation=_confirmation(confirmed=5 if rank < proposal_failures else 0),
            )
            for seed in range(5)
        )
    for rank in range(per_arm):
        rows.extend(
            _row(
                row_id=f"rand_{rank}_{seed}",
                manifest_id=f"rand_cand_{rank}",
                arm="random",
                rank=rank + 1,
                failure=rank < random_failures,
                scenario_seed=199_000 + rank,
                execution_seed=170_000 + rank * 10 + seed,
                confirmation=_confirmation(confirmed=5 if rank < random_failures else 0),
            )
            for seed in range(5)
        )
    return _packet(rows)


def _spec_for_packet(  # noqa: C901
    packet: dict[str, Any], *, budget_per_arm: int
) -> AdmissionSpec:
    """Build an external frozen manifest binding for a test packet.

    The helper treats every row's selected manifest as predeclared, padding a
    short arm with unexecuted IDs so the production evaluator must reject an
    incomplete packet after checking row-level failures. A deliberately
    over-budget packet remains over-budget in the binding and therefore fails
    before it can produce a decision.
    """
    ids_by_arm: dict[str, list[str]] = {"proposal": [], "random": []}
    execution_seeds: dict[str, list[int]] = {}
    candidate_pool_indices: dict[str, int] = {}
    scenario_seeds: dict[str, int] = {}
    record_hashes: dict[str, str] = {}
    next_padding_pool_index = 100_000
    for row in packet.get("rows", []):
        if not isinstance(row, dict) or row.get("selection_arm") not in ids_by_arm:
            continue
        arm = str(row["selection_arm"])
        manifest_id = str(row.get("candidate_manifest_id", ""))
        if manifest_id and manifest_id not in ids_by_arm[arm]:
            ids_by_arm[arm].append(manifest_id)
        execution_seed = row.get("execution_seed")
        if manifest_id and isinstance(execution_seed, int) and not isinstance(execution_seed, bool):
            execution_seeds.setdefault(manifest_id, []).append(execution_seed)
        candidate_pool_index = row.get("candidate_pool_index")
        if manifest_id and manifest_id not in candidate_pool_indices:
            if (
                isinstance(candidate_pool_index, int)
                and not isinstance(candidate_pool_index, bool)
                and candidate_pool_index >= 0
            ):
                candidate_pool_indices[manifest_id] = candidate_pool_index
            else:
                candidate_pool_indices[manifest_id] = next_padding_pool_index
                next_padding_pool_index += 1
        scenario_seed = row.get("scenario_seed")
        if manifest_id and manifest_id not in scenario_seeds:
            scenario_seeds[manifest_id] = (
                scenario_seed
                if isinstance(scenario_seed, int) and not isinstance(scenario_seed, bool)
                else 0
            )
        record_sha256 = row.get("record_sha256")
        if manifest_id and manifest_id not in record_hashes:
            record_hashes[manifest_id] = (
                record_sha256
                if isinstance(record_sha256, str) and record_sha256
                else _sha256(f"record-{manifest_id}")
            )
    for manifest_index, seed_values in enumerate(execution_seeds.values()):
        next_seed = 900_000 + manifest_index * 10
        while len(seed_values) < 5:
            if next_seed not in seed_values:
                seed_values.append(next_seed)
            next_seed += 1
    for arm in ("proposal", "random"):
        padding_index = 0
        while len(ids_by_arm[arm]) < budget_per_arm:
            manifest_id = f"unexecuted_{arm}_{padding_index}"
            padding_index += 1
            if manifest_id in ids_by_arm[arm]:
                continue
            ids_by_arm[arm].append(manifest_id)
            execution_seeds[manifest_id] = [
                80_000 + padding_index * 10 + seed_offset for seed_offset in range(5)
            ]
            candidate_pool_indices[manifest_id] = next_padding_pool_index
            next_padding_pool_index += 1
            scenario_seeds[manifest_id] = 0
            record_hashes[manifest_id] = _sha256(f"record-{manifest_id}")
    expected_ids = ids_by_arm["proposal"] + ids_by_arm["random"]
    return AdmissionSpec(
        expected_target_planner_id=_PLANNER,
        expected_eval_family=_EVAL_FAMILY,
        confirmation_threshold="3_of_5",
        expected_target_planner_config_sha256=_PLANNER_CFG,
        expected_candidate_manifest_sha256_by_id={
            manifest_id: _sha256(f"manifest-{manifest_id}") for manifest_id in expected_ids
        },
        expected_candidate_pool_index_by_manifest_id={
            manifest_id: candidate_pool_indices[manifest_id] for manifest_id in expected_ids
        },
        expected_scenario_seed_by_manifest_id={
            manifest_id: scenario_seeds[manifest_id] for manifest_id in expected_ids
        },
        expected_record_sha256_by_manifest={
            manifest_id: record_hashes[manifest_id] for manifest_id in expected_ids
        },
        expected_candidate_manifest_ids_by_arm={
            arm: tuple(manifest_ids) for arm, manifest_ids in ids_by_arm.items()
        },
        expected_execution_seeds_by_manifest_id={
            manifest_id: tuple(seed_values) for manifest_id, seed_values in execution_seeds.items()
        },
        expected_candidate_pool_seed=42,
        expected_execution_commit=_EXECUTION_COMMIT,
    )


def _evaluate(
    packet: dict[str, Any] | None,
    *,
    budget_per_arm: int,
    minimally_important: float = 0.20,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate a packet against an exact test-only external binding."""
    spec_packet = packet if packet is not None else _packet([])
    return build_independent_outcome_evaluation(
        packet,
        budget_per_arm=budget_per_arm,
        minimally_important=minimally_important,
        admission_spec=_spec_for_packet(spec_packet, budget_per_arm=budget_per_arm),
        **kwargs,
    )


def test_load_independent_outcomes_missing_is_not_available() -> None:
    """Absence of a packet is a fail-closed not-available state."""
    state, reason, payload = load_independent_outcomes(None)
    assert state == "not_available"
    assert "No independent outcome path" in reason
    assert payload is None


def test_load_independent_outcomes_malformed_file_blocks(tmp_path: Path) -> None:
    """Supplied malformed payloads block instead of falling back."""
    path = tmp_path / "outcomes.json"
    path.write_text("{not json", encoding="utf-8")

    state, reason, payload = load_independent_outcomes(path)

    assert state == "blocked"
    assert "Failed to load independent outcomes" in reason
    assert payload is None


def test_v1_flat_array_packet_is_rejected_as_deprecated() -> None:
    """The deprecated flat-array v1 contract is no longer admitted."""
    legacy = {
        "schema_version": "adversarial_independent_outcomes.v1",
        "outcome_source": "planner_execution",
        "objective": "certified_failure_outcome",
        "proposal_outcomes": [10.0],
        "random_outcomes": [0.0],
    }
    result = _evaluate(legacy, budget_per_arm=4)
    assert result["status"] == "blocked"
    assert "deprecated" in result["reason"]
    assert result["independent_outcomes_available"] is False


def test_archive_nearness_objective_is_circular_and_blocked() -> None:
    """Archive-nearness objective cannot open the claim gate."""
    packet = _packet([], objective="archive_nearness")
    result = _evaluate(packet, budget_per_arm=4)
    assert result["status"] == "blocked"
    assert "circular" in result["reason"]


def test_target_planner_mismatch_blocks() -> None:
    """A packet for the wrong target planner fails closed."""
    packet = _packet([], target_planner_id="goal")
    result = _evaluate(packet, budget_per_arm=4)
    assert result["status"] == "blocked"
    assert "target_planner_id mismatch" in result["reason"]


def test_fallback_execution_mode_row_fails_closed() -> None:
    """A fallback execution row cannot be admitted (fail closed, not degraded)."""
    packet = _packet(
        [
            _row(
                row_id="r0",
                manifest_id="c0",
                arm="proposal",
                rank=1,
                failure=True,
                execution_mode="fallback",
            )
        ]
    )
    result = _evaluate(packet, budget_per_arm=4)
    assert result["status"] == "blocked"
    assert "execution_mode" in result["reason"]
    assert "native" in result["reason"]


def test_missing_required_field_fails_closed() -> None:
    """A row missing a required field fails closed with the field name."""
    row = _row(row_id="r0", manifest_id="c0", arm="proposal", rank=1, failure=True)
    del row["record_sha256"]
    packet = _packet([row])
    result = _evaluate(packet, budget_per_arm=4)
    assert result["status"] == "blocked"
    assert "record_sha256" in result["reason"]


def test_malformed_execution_lineage_fields_fail_closed() -> None:
    """Required row-level execution lineage must be well typed and non-empty."""
    malformed_values = (
        ("row_id", "", "row_id"),
        ("execution_command", ["python", ""], "execution_command"),
        ("execution_config_lineage", {}, "execution_config_lineage"),
        ("selection_rank", True, "selection_rank"),
        ("candidate_pool_seed", True, "candidate_pool_seed"),
        ("scenario_seed", True, "scenario_seed"),
        ("primary_failure", "", "primary_failure"),
        ("termination_reason", "", "termination_reason"),
    )
    for field_name, value, reason_fragment in malformed_values:
        row = _row(row_id="r0", manifest_id="c0", arm="proposal", rank=1, failure=True)
        row[field_name] = value

        result = _evaluate(_packet([row]), budget_per_arm=1)

        assert result["status"] == "blocked"
        assert reason_fragment in result["reason"]


def test_candidate_manifest_hash_requires_external_binding() -> None:
    """Rows cannot self-attest manifest lineage when no frozen binding is supplied."""
    packet = _packet([_row(row_id="r0", manifest_id="c0", arm="proposal", rank=1, failure=True)])
    unbound_spec = replace(
        _spec_for_packet(packet, budget_per_arm=1),
        expected_candidate_manifest_sha256_by_id=None,
    )

    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=1, minimally_important=0.20, admission_spec=unbound_spec
    )

    assert result["status"] == "blocked"
    assert "manifest SHA-256 binding" in result["reason"]


def test_candidate_manifest_hash_mismatch_fails_closed() -> None:
    """A row hash must match the separate frozen manifest-hash binding."""
    row = _row(row_id="r0", manifest_id="c0", arm="proposal", rank=1, failure=True)
    row["candidate_manifest_sha256"] = "wrong-manifest-hash"
    result = _evaluate(_packet([row]), budget_per_arm=1)

    assert result["status"] == "blocked"
    assert "candidate_manifest_sha256 mismatch" in result["reason"]


def test_candidate_pool_index_requires_external_binding() -> None:
    """Rows cannot self-attest their shared-pool position without a frozen binding."""
    packet = _balanced_packet(proposal_failures=1, random_failures=0, per_arm=1)
    unbound_spec = replace(
        _spec_for_packet(packet, budget_per_arm=1),
        expected_candidate_pool_index_by_manifest_id=None,
    )

    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=1, minimally_important=0.20, admission_spec=unbound_spec
    )

    assert result["status"] == "blocked"
    assert "candidate-pool index binding" in result["reason"]


def test_candidate_pool_index_mismatch_fails_closed() -> None:
    """A row cannot substitute a different shared-pool index after manifest freeze."""
    packet = _balanced_packet(proposal_failures=1, random_failures=0, per_arm=1)
    spec = _spec_for_packet(packet, budget_per_arm=1)
    proposal_id = packet["rows"][0]["candidate_manifest_id"]
    expected_indices = dict(spec.expected_candidate_pool_index_by_manifest_id or {})
    expected_indices[proposal_id] = 99
    drifted_spec = replace(spec, expected_candidate_pool_index_by_manifest_id=expected_indices)

    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=1, minimally_important=0.20, admission_spec=drifted_spec
    )

    assert result["status"] == "blocked"
    assert "candidate_pool_index mismatch" in result["reason"]


def test_scenario_seed_requires_external_binding() -> None:
    """Rows cannot self-attest the candidate scenario seed across confirmation runs."""
    packet = _balanced_packet(proposal_failures=1, random_failures=0, per_arm=1)
    unbound_spec = replace(
        _spec_for_packet(packet, budget_per_arm=1),
        expected_scenario_seed_by_manifest_id=None,
    )

    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=1, minimally_important=0.20, admission_spec=unbound_spec
    )

    assert result["status"] == "blocked"
    assert "scenario-seed binding" in result["reason"]


def test_scenario_seed_mismatch_fails_closed() -> None:
    """All execution rows must use the manifest's single frozen scenario seed."""
    packet = _balanced_packet(proposal_failures=1, random_failures=0, per_arm=1)
    spec = _spec_for_packet(packet, budget_per_arm=1)
    proposal_id = packet["rows"][0]["candidate_manifest_id"]
    expected_seeds = dict(spec.expected_scenario_seed_by_manifest_id or {})
    expected_seeds[proposal_id] += 1
    drifted_spec = replace(spec, expected_scenario_seed_by_manifest_id=expected_seeds)

    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=1, minimally_important=0.20, admission_spec=drifted_spec
    )

    assert result["status"] == "blocked"
    assert "scenario_seed mismatch" in result["reason"]


def test_record_hash_requires_external_binding() -> None:
    """Rows cannot self-attest record lineage without a frozen record-hash binding."""
    packet = _balanced_packet(proposal_failures=1, random_failures=0, per_arm=1)
    unbound_spec = replace(
        _spec_for_packet(packet, budget_per_arm=1),
        expected_record_sha256_by_manifest=None,
    )

    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=1, minimally_important=0.20, admission_spec=unbound_spec
    )

    assert result["status"] == "blocked"
    assert "record SHA-256 binding" in result["reason"]


def test_record_hash_mismatch_fails_closed() -> None:
    """A row record hash must match its separate frozen external binding."""
    packet = _balanced_packet(proposal_failures=1, random_failures=0, per_arm=1)
    spec = _spec_for_packet(packet, budget_per_arm=1)
    packet["rows"][0]["record_sha256"] = "wrong-record-hash"

    result = build_independent_outcome_evaluation(
        packet, budget_per_arm=1, minimally_important=0.20, admission_spec=spec
    )

    assert result["status"] == "blocked"
    assert "record_sha256 mismatch" in result["reason"]


def test_replay_signature_mismatch_fails_closed() -> None:
    """A replay signature that differs from the original signature fails closed."""
    row = _row(
        row_id="r0",
        manifest_id="c0",
        arm="proposal",
        rank=1,
        failure=True,
        replay={
            "exact_signature_match": True,
            "original_signature_sha256": "abc",
            "replay_signature_sha256": "xyz",
        },
    )
    packet = _packet([row])
    result = _evaluate(packet, budget_per_arm=4)
    assert result["status"] == "blocked"
    assert "signature" in result["reason"]


def test_confirmation_below_threshold_is_a_valid_candidate_non_failure() -> None:
    """Two confirmed failures out of five produce one valid candidate non-failure."""
    rows = [
        *[
            _row(
                row_id=f"p0_{seed}",
                manifest_id="cp0",
                arm="proposal",
                rank=1,
                failure=seed <= 2,
                scenario_seed=99_001,
                execution_seed=seed,
                confirmation=_confirmation(confirmed=2),
            )
            for seed in range(1, 6)
        ],
        *[
            _row(
                row_id=f"r0_{seed}",
                manifest_id="cr0",
                arm="random",
                rank=1,
                failure=False,
                scenario_seed=199_001,
                execution_seed=100 + seed,
                confirmation=_confirmation(confirmed=0),
            )
            for seed in range(1, 6)
        ],
    ]

    result = _evaluate(_packet(rows), budget_per_arm=1)

    assert result["status"] == "complete"
    assert result["proposal"]["outcomes"] == [False]
    assert result["proposal_failure_yield"] == 0.0


def test_execution_seed_binding_requires_all_five_confirmation_attempts() -> None:
    """A 3-of-5 packet cannot bind and execute only the three confirming seeds."""
    packet = _balanced_packet(proposal_failures=1, random_failures=0, per_arm=1)
    spec = _spec_for_packet(packet, budget_per_arm=1)
    short_seed_binding = {
        manifest_id: tuple(seed_values[:3])
        for manifest_id, seed_values in (spec.expected_execution_seeds_by_manifest_id or {}).items()
    }

    result = build_independent_outcome_evaluation(
        packet,
        budget_per_arm=1,
        minimally_important=0.20,
        admission_spec=replace(
            spec,
            expected_execution_seeds_by_manifest_id=short_seed_binding,
        ),
    )

    assert result["status"] == "blocked"
    assert "must contain exactly 5 seeds" in result["reason"]


def test_confirmation_count_must_match_observed_seed_outcomes() -> None:
    """Candidate-level confirmation metadata cannot disagree with its five rows."""
    packet = _balanced_packet(proposal_failures=1, random_failures=0, per_arm=1)
    packet["rows"][0]["confirmation_lineage"]["confirmed_count"] = 4

    result = _evaluate(packet, budget_per_arm=1)

    assert result["status"] == "blocked"
    assert "confirmation count mismatch" in result["reason"]


def test_scenario_certification_not_passed_fails_closed() -> None:
    """A candidate whose scenario certification did not pass fails closed."""
    row = _row(
        row_id="r0",
        manifest_id="c0",
        arm="proposal",
        rank=1,
        failure=True,
        scenario_cert="stress_only",
    )
    packet = _packet([row])
    result = _evaluate(packet, budget_per_arm=4)
    assert result["status"] == "blocked"
    assert "scenario_certification_status" in result["reason"]


def test_wrong_eval_family_fails_closed() -> None:
    """A row from the fit family (not the held-out eval family) fails closed."""
    row = _row(
        row_id="r0",
        manifest_id="c0",
        arm="proposal",
        rank=1,
        failure=True,
    )
    row["scenario_family"] = "classic_group_crossing_medium"
    packet = _packet([row])
    result = _evaluate(packet, budget_per_arm=4)
    assert result["status"] == "blocked"
    assert "scenario_family" in result["reason"]


def test_excluded_row_with_reason_blocks_an_incomplete_predeclared_arm() -> None:
    """An excluded candidate cannot silently shrink the frozen arm denominator."""
    rows = [
        *[
            _row(
                row_id=f"p0_{seed}",
                manifest_id="cp0",
                arm="proposal",
                rank=1,
                failure=True,
                execution_seed=seed,
                confirmation=_confirmation(confirmed=5),
            )
            for seed in range(5)
        ],
        _row(
            row_id="p1",
            manifest_id="cp1",
            arm="proposal",
            rank=2,
            failure=False,
            admission_status="excluded",
            exclusion_reason="candidate_pool_collision_disjoint_by_candidate",
        ),
        *[
            _row(
                row_id=f"r0_{seed}",
                manifest_id="cr0",
                arm="random",
                rank=1,
                failure=False,
                execution_seed=100 + seed,
                confirmation=_confirmation(confirmed=0),
            )
            for seed in range(5)
        ],
    ]
    packet = _packet(rows)
    result = _evaluate(packet, budget_per_arm=2)
    assert result["status"] == "blocked"
    assert "complete predeclared manifest set" in result["reason"]


def test_three_of_five_confirmation_counts_as_one_candidate_failure() -> None:
    """Mixed seeds honor the frozen 3-of-5 rule instead of requiring 5-of-5."""
    rows = [
        *[
            _row(
                row_id=f"p0_{seed}",
                manifest_id="cp0",
                arm="proposal",
                rank=1,
                failure=seed <= 3,
                scenario_seed=99_001,
                execution_seed=seed,
            )
            for seed in range(1, 6)
        ],
        *[
            _row(
                row_id=f"r0_{seed}",
                manifest_id="cr0",
                arm="random",
                rank=1,
                failure=False,
                scenario_seed=199_001,
                execution_seed=100 + seed,
                confirmation=_confirmation(confirmed=0),
            )
            for seed in range(1, 6)
        ],
    ]
    packet = _packet(rows)
    result = _evaluate(packet, budget_per_arm=1)

    assert result["status"] == "complete"
    assert result["proposal"]["outcomes"] == [True]
    assert result["random"]["outcomes"] == [False]


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    [("primary_failure", "deadlock"), ("termination_reason", "timeout")],
)
def test_different_confirming_attributions_fail_closed(
    field: str,
    drifted_value: str,
) -> None:
    """Confirming seeds must retain the same failure mechanism and termination."""
    rows = [
        *[
            _row(
                row_id=f"p0_{seed}",
                manifest_id="cp0",
                arm="proposal",
                rank=1,
                failure=True,
                scenario_seed=99_001,
                execution_seed=seed,
                confirmation=_confirmation(confirmed=5),
            )
            for seed in range(1, 6)
        ],
        *[
            _row(
                row_id=f"r0_{seed}",
                manifest_id="cr0",
                arm="random",
                rank=1,
                failure=False,
                scenario_seed=199_001,
                execution_seed=100 + seed,
                confirmation=_confirmation(confirmed=0),
            )
            for seed in range(1, 6)
        ],
    ]
    rows[1][field] = drifted_value

    result = _evaluate(_packet(rows), budget_per_arm=1)

    assert result["status"] == "blocked"
    assert "unstable attribution" in result["reason"]


def test_candidate_manifest_id_in_both_arms_fails_closed() -> None:
    """A candidate manifest cannot be aggregated into both disjoint arms."""
    packet = _packet(
        [
            _row(row_id="p0", manifest_id="shared", arm="proposal", rank=1, failure=True),
            _row(row_id="r0", manifest_id="shared", arm="random", rank=1, failure=False),
        ]
    )
    result = _evaluate(packet, budget_per_arm=1)

    assert result["status"] == "blocked"
    assert "overlap" in result["reason"]


def test_complete_packet_decision_follows_execution() -> None:
    """A strong proposal-favors-execution packet yields a continue/underpowered decision."""
    # 4/4 proposal failures vs 0/4 random: large effect, but k=4 is underpowered
    # for delta=0.20 (min detectable at k=4 is 0.75), so the decision is inconclusive.
    packet = _balanced_packet(proposal_failures=4, random_failures=0, per_arm=4)
    result = _evaluate(packet, budget_per_arm=4)
    assert result["status"] == "complete"
    assert result["independent_outcomes_available"] is True
    assert result["proposal_failure_yield"] == 1.0
    assert result["random_failure_yield"] == 0.0
    assert result["comparison"]["yield_improvement"] == 1.0
    assert result["comparison"]["null_rejected"] is True
    assert result["comparison"]["powered"] is False
    assert result["decision"]["status"] == "inconclusive"
    assert result["decision"]["reason"] == "underpowered_for_minimally_important_effect"


def test_underpowered_random_better_than_proposal_is_inconclusive() -> None:
    """An underpowered random-favoring result cannot produce a stop decision."""
    # 0/6 proposal vs 6/6 random is still underpowered for the frozen 0.20 effect.
    packet = _balanced_packet(proposal_failures=0, random_failures=6, per_arm=6)
    result = _evaluate(packet, budget_per_arm=6)
    assert result["status"] == "complete"
    assert result["comparison"]["powered"] is False
    assert result["decision"]["status"] == "inconclusive"
    assert result["decision"]["reason"] == "underpowered_for_minimally_important_effect"


def test_over_budget_packet_cannot_be_complete_or_drive_a_decision() -> None:
    """A 30-per-arm packet cannot manufacture power for a frozen 12-arm study."""
    packet = _balanced_packet(proposal_failures=30, random_failures=0, per_arm=30)

    result = _evaluate(packet, budget_per_arm=12, n_permutations=10)

    assert result["status"] == "blocked"
    assert result["independent_outcomes_available"] is False
    assert (
        "predeclared proposal manifest count 30 != frozen candidate budget 12" in result["reason"]
    )
    assert "comparison" not in result


def test_execution_seed_must_match_the_external_manifest_lineage() -> None:
    """A row cannot substitute a different execution seed after manifest freeze."""
    packet = _balanced_packet(proposal_failures=1, random_failures=0, per_arm=1)
    spec = _spec_for_packet(packet, budget_per_arm=1)
    proposal_id = packet["rows"][0]["candidate_manifest_id"]
    expected_seeds = dict(spec.expected_execution_seeds_by_manifest_id or {})
    expected_seeds[proposal_id] = tuple(range(999_999, 1_000_004))
    drifted_spec = replace(spec, expected_execution_seeds_by_manifest_id=expected_seeds)

    result = build_independent_outcome_evaluation(
        packet,
        budget_per_arm=1,
        minimally_important=0.20,
        admission_spec=drifted_spec,
    )

    assert result["status"] == "blocked"
    assert "execution_seed is not predeclared" in result["reason"]


def test_opposite_sign_regressions_decision_follows_execution() -> None:
    """Decision follows execution independent of which arm it favors.

    The runner supplies archive-nearness only as a diagnostic; the decision is
    driven solely by the admitted execution rows. This test encodes the
    opposite-sign guarantee: the decision tracks execution, and the
    diagnostic archive-nearness namespace is never consulted here.
    """
    # k=30 is the smallest frozen-effect budget whose Fisher boundary is <= 0.20.
    # Case A: execution favors proposal.
    exec_proposal_better = _balanced_packet(proposal_failures=30, random_failures=0, per_arm=30)
    res_a = _evaluate(
        exec_proposal_better,
        budget_per_arm=30,
        n_permutations=10,
    )
    assert res_a["comparison"]["yield_improvement"] > 0.0

    # Case B: execution favors random (opposite sign).
    exec_random_better = _balanced_packet(proposal_failures=0, random_failures=30, per_arm=30)
    res_b = _evaluate(
        exec_random_better,
        budget_per_arm=30,
        n_permutations=10,
    )
    assert res_b["comparison"]["yield_improvement"] < 0.0
    assert res_b["decision"]["status"] == "stop"
    assert res_a["decision"]["status"] == "continue"
    assert res_a["comparison"]["yield_improvement"] * res_b["comparison"]["yield_improvement"] < 0.0


def test_eval_archive_hash_mismatch_fails_closed() -> None:
    """Outcome packets must match the held-out eval split they claim to score."""
    packet = _balanced_packet(proposal_failures=1, random_failures=0, per_arm=2)
    result = _evaluate(
        packet,
        budget_per_arm=2,
        expected_eval_archive_sha256="expected-eval-hash",
    )
    assert result["status"] == "blocked"
    assert result["expected_eval_archive_sha256"] == "expected-eval-hash"
    assert result["observed_eval_archive_sha256"] == "eval-hash"


def test_load_independent_outcomes_reads_json_payload(tmp_path: Path) -> None:
    """Readable JSON objects are passed through for report integration."""
    path = tmp_path / "outcomes.json"
    path.write_text(
        json.dumps(_balanced_packet(proposal_failures=1, random_failures=0)), encoding="utf-8"
    )

    state, reason, payload = load_independent_outcomes(path)

    assert state == "active"
    assert "loaded successfully" in reason
    assert payload is not None
    assert payload["outcome_source"] == "planner_execution"
    assert payload_sha256(payload)
