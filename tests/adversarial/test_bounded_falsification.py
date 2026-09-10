"""Contract tests for the default-disabled issue #8571 falsification slice."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import scripts.adversarial.run_issue_8571_bounded_falsification_slice as preflight_cli
from robot_sf.adversarial.bounded_falsification import (
    CANONICAL_CMA_ES_OWNER,
    CLAIM_BOUNDARY,
    VERTICAL_SLICE_SCHEMA_VERSION,
    BoundedFalsificationError,
    build_bounded_falsification_preflight,
    validate_bounded_falsification_preflight,
    write_bounded_falsification_preflight,
)
from robot_sf.benchmark.research_answerability import (
    compute_adversarial_falsification_packet_digest,
)
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/adversarial/issue_8570_bounded_falsification_answerability_v1.yaml"


def test_preflight_prepares_equal_budget_controls_without_compute() -> None:
    """The admitted slice preserves source custody while keeping all native work disabled."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)

    assert report["schema_version"] == VERTICAL_SLICE_SCHEMA_VERSION
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    assert report["packet"]["issue"] == 8570
    assert report["source"]["base_commit"] == "c61b0f93683e1f9d2c83125f1d830b3d0162f3d1"
    assert report["zero_overlay_equivalence"]["status"] == "passed"
    assert report["execution"] == {
        "campaign_launched": False,
        "compute_authorized_by_packet": False,
        "default_disabled": True,
        "optimizer_instantiated": False,
        "planner_executed": False,
        "simulator_executed": False,
    }

    assert len(report["arms"]["random"]) == 3
    assert len(report["arms"]["halton"]) == 3
    assert all(row["candidate_budget"] == 64 for row in report["arms"]["random"])
    assert all(row["candidate_budget"] == 64 for row in report["arms"]["halton"])
    assert report["arms"]["cma_es"]["execution_status"] == "declared_not_executed"
    assert report["arms"]["cma_es"]["candidate_budget_per_seed"] == 64

    assert len(report["outcome_rows"]) == 3 * 2 * 64
    assert report["outcome_summary"]["blocked"] == 3 * 2 * 64
    assert report["outcome_summary"]["invalid"] == 0
    assert report["outcome_summary"]["result"] == 0
    assert report["outcome_summary"]["null"] == 0
    assert all(row["simulation_executed"] is False for row in report["outcome_rows"])
    assert all(row["native_outcome_digest"] is None for row in report["outcome_rows"])
    assert all(row["replay_digest"] is None for row in report["outcome_rows"])


def test_packet_preserves_issue_references_and_cma_es_custody() -> None:
    """YAML parsing keeps issue references and binds the declared primary owner."""
    packet = yaml.safe_load(PACKET.read_text(encoding="utf-8"))

    assert packet["source"]["inputs"]["search_space"]["role"] == (
        "frozen #7340 search-space bytes and bounds"
    )
    assert packet["source"]["inputs"]["scenario_template"]["role"] == (
        "frozen #7340 scenario-template bytes and actor binding"
    )
    assert packet["answerability"]["question"]["research_question"].endswith(
        "the frozen #7340 source contract?"
    )
    assert packet["compute_authorization"]["blocking_reasons"][0].endswith(
        "in #7340 template mode; metadata or provenance cannot authorize compute."
    )
    assert packet["search_methods"]["primary"]["owner"] == CANONICAL_CMA_ES_OWNER

    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    assert report["arms"]["cma_es"]["owner"] == CANONICAL_CMA_ES_OWNER
    assert report["arms"]["cma_es"]["implementation_owner"] == CANONICAL_CMA_ES_OWNER


def test_preflight_validator_rejects_cma_es_owner_mismatch() -> None:
    """The primary arm cannot carry separate or ambiguous owner identities."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    report["arms"]["cma_es"]["implementation_owner"] = "robot_sf.adversarial.cma_me"

    with pytest.raises(BoundedFalsificationError, match="CMA-ES arm"):
        validate_bounded_falsification_preflight(report)


def test_preflight_is_deterministic_and_persisted_report_round_trips(tmp_path: Path) -> None:
    """Repeated source-bound preparation emits byte-stable candidate and gate provenance."""
    first = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    second = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    assert first == second

    output = tmp_path / "preflight.json"
    persisted = write_bounded_falsification_preflight(
        PACKET,
        output,
        repo_root=REPO_ROOT,
    )
    assert persisted == first
    assert json.loads(output.read_text(encoding="utf-8")) == first


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report.pop("feasibility"),
        lambda report: report["feasibility"].pop("owner"),
    ],
)
def test_preflight_validator_rejects_missing_feasibility_fields(mutate) -> None:
    """The report must include every canonical feasibility declaration field."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    mutate(report)

    with pytest.raises(BoundedFalsificationError, match="feasibility"):
        validate_bounded_falsification_preflight(report)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("feasibility", "owner", "forged.owner"),
        ("feasibility", "native_predicates_executed", True),
        ("feasibility", "simulator_validity", "valid"),
        ("native_outcomes", "status", "available"),
    ],
)
def test_preflight_validator_rejects_forged_feasibility_and_native_claims(
    section: str, field: str, value: object
) -> None:
    """Feasibility and native execution claims must be recomputed from the packet."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    report[section][field] = value

    with pytest.raises(BoundedFalsificationError, match=section):
        validate_bounded_falsification_preflight(report)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report.update({"unexpected": True}),
        lambda report: report["feasibility"].update({"unexpected": True}),
        lambda report: report["arms"]["random"][0].update({"unexpected": True}),
        lambda report: report["outcome_rows"][0].update({"unexpected": True}),
    ],
)
def test_preflight_validator_rejects_unknown_report_and_nested_fields(mutate) -> None:
    """Unknown fields cannot extend the builder-emitted report or ledger schemas."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    mutate(report)

    with pytest.raises(BoundedFalsificationError, match="unknown fields"):
        validate_bounded_falsification_preflight(report)


def test_preflight_validator_rejects_claim_result_rows() -> None:
    """Preparation must not be relabeled as an available result or null outcome."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    report["outcome_rows"][0]["status"] = "result"
    report["outcome_summary"]["blocked"] -= 1
    report["outcome_summary"]["result"] += 1

    with pytest.raises(BoundedFalsificationError, match="result or null"):
        validate_bounded_falsification_preflight(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("simulation_executed", True),
        ("native_outcome_digest", "a" * 64),
        ("replay_digest", "b" * 64),
    ],
)
def test_preflight_validator_rejects_outcome_execution_or_digests(
    field: str, value: object
) -> None:
    """No-result rows cannot carry native execution or replay evidence."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    report["outcome_rows"][0][field] = value

    with pytest.raises(BoundedFalsificationError, match=field):
        validate_bounded_falsification_preflight(report)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("execution", "simulator_executed", True, "simulator_executed"),
        ("execution", "planner_executed", True, "planner_executed"),
        ("execution", "optimizer_instantiated", True, "optimizer_instantiated"),
        ("execution", "campaign_launched", True, "campaign_launched"),
        ("execution", "default_disabled", False, "default_disabled"),
        ("execution", "compute_authorized_by_packet", True, "compute_authorized_by_packet"),
        ("gate", "authorized", True, "gate"),
        ("gate", "status", "not_requested", "gate"),
        ("gate", "blocking_reasons", [], "blocking_reasons"),
        ("native_outcomes", "rows", 1, "native_outcomes"),
        ("native_outcomes", "digest", "c" * 64, "native_outcomes"),
        ("replay", "rows", 1, "replay"),
        ("replay", "digest", "d" * 64, "replay"),
    ],
)
def test_preflight_validator_rejects_report_level_compute_claims(
    section: str, field: str, value: object, message: str
) -> None:
    """Report-level declarations cannot reopen the default-disabled compute path."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    report[section][field] = value

    with pytest.raises(BoundedFalsificationError, match=message):
        validate_bounded_falsification_preflight(report)


def test_preflight_validator_rejects_executed_cma_es_arm() -> None:
    """The declared CMA-ES arm must remain uninstantiated in this preflight."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    report["arms"]["cma_es"]["execution_status"] = "executed"

    with pytest.raises(BoundedFalsificationError, match="CMA-ES"):
        validate_bounded_falsification_preflight(report)


def test_preflight_validator_binds_coordinated_gate_mutation_to_packet() -> None:
    """A coordinated report-field mutation cannot authorize the canonical blocked packet."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    report["gate"].update({"authorized": True, "status": "not_requested"})
    report["execution"]["compute_authorized_by_packet"] = True

    with pytest.raises(BoundedFalsificationError, match="canonical packet"):
        validate_bounded_falsification_preflight(report)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("native_outcomes", "status", "available"),
        ("replay", "status", "completed"),
    ],
)
def test_preflight_validator_rejects_noncanonical_execution_status(
    section: str, field: str, value: str
) -> None:
    """Disabled native and replay sections cannot be relabeled as completed evidence."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    report[section][field] = value

    with pytest.raises(BoundedFalsificationError, match=section):
        validate_bounded_falsification_preflight(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "failed"),
        ("source_digest", "a" * 64),
        ("materialized_digest", "b" * 64),
        ("patch_digest", "c" * 64),
    ],
)
def test_preflight_validator_rejects_zero_overlay_identity_mutation(field: str, value: str) -> None:
    """The zero-overlay verdict and all source digests must match the recomputed source."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    report["zero_overlay_equivalence"][field] = value

    with pytest.raises(BoundedFalsificationError, match="zero_overlay_equivalence"):
        validate_bounded_falsification_preflight(report)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["arms"].pop("random"),
        lambda report: report["arms"].update({"halton": []}),
        lambda report: report["arms"].update({"random": report["arms"]["random"][:-1]}),
    ],
)
def test_preflight_validator_rejects_missing_empty_or_shrunk_control_arm(mutate) -> None:
    """Every canonical control seed and candidate row must remain in the arm ledger."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    mutate(report)

    with pytest.raises(BoundedFalsificationError, match="arms|ledger"):
        validate_bounded_falsification_preflight(report)


def test_preflight_validator_rejects_empty_outcome_ledger() -> None:
    """An internally consistent empty outcome ledger cannot replace the canonical rows."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    report["outcome_rows"] = []
    report["outcome_summary"] = dict.fromkeys(
        ("result", "null", "inconclusive", "invalid", "unavailable", "blocked"), 0
    )

    with pytest.raises(BoundedFalsificationError, match="outcome ledger"):
        validate_bounded_falsification_preflight(report)


def test_preflight_validator_rejects_shrunk_outcome_ledger() -> None:
    """Removing a row while repairing the summary still fails canonical ledger comparison."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    report["outcome_rows"].pop()
    report["outcome_summary"]["blocked"] -= 1

    with pytest.raises(BoundedFalsificationError, match="outcome ledger"):
        validate_bounded_falsification_preflight(report)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["packet"].update({"packet_id": "forged-packet"}),
            "packet identity",
        ),
        (
            lambda report: report["source"].update({"scenario_template_digest": "d" * 64}),
            "source identity",
        ),
        (
            lambda report: report["arms"]["random"][0].update({"search_seed": 123}),
            "candidate ledger",
        ),
        (
            lambda report: report["arms"]["random"][0]["preparation"]["candidates"][0][
                "candidate"
            ].update({"candidate_id": "forged-candidate"}),
            "candidate ledger",
        ),
        (
            lambda report: report["arms"]["random"][0]["preparation"]["candidates"][0][
                "overlay"
            ].update({"materialized_digest": "e" * 64}),
            "candidate ledger",
        ),
        (
            lambda report: report["arms"]["random"][0]["preparation"]["candidates"][0][
                "overlay"
            ].update({"source_digest": "f" * 64}),
            "candidate ledger",
        ),
        (
            lambda report: report["arms"]["random"][0]["preparation"]["provenance"].update(
                {"source_digest": "0" * 64}
            ),
            "candidate ledger",
        ),
        (
            lambda report: report["outcome_rows"][0].update(
                {"candidate_id": "forged-outcome-candidate"}
            ),
            "outcome ledger",
        ),
        (
            lambda report: report["outcome_rows"][0].update(
                {"overlay_materialized_digest": "f" * 64}
            ),
            "outcome ledger",
        ),
        (
            lambda report: report["outcome_rows"][0]["candidate"].update(
                {"candidate_id": "forged-nested-candidate"}
            ),
            "outcome ledger",
        ),
    ],
)
def test_preflight_validator_rejects_forged_packet_source_candidate_or_overlay_identity(
    mutation, message: str
) -> None:
    """Ledger identities are checked against canonical recomputation, not report assertions."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    mutation(report)

    with pytest.raises(BoundedFalsificationError, match=message):
        validate_bounded_falsification_preflight(report)


def test_preflight_validator_rejects_invalid_outcome_without_rejection_metadata() -> None:
    """An invalid row must retain the pre-simulation rejection record that explains it."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    report["outcome_rows"][0]["status"] = "invalid"
    report["outcome_summary"]["blocked"] -= 1
    report["outcome_summary"]["invalid"] += 1

    with pytest.raises(BoundedFalsificationError, match="rejection metadata"):
        validate_bounded_falsification_preflight(report)


def test_preflight_validator_accepts_explicit_source_binding_arguments() -> None:
    """The source-bound validator remains compatible with the public one-argument fixture call."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)

    validate_bounded_falsification_preflight(
        report,
        packet_path=PACKET,
        repo_root=REPO_ROOT,
    )


def test_preflight_rejects_rehashed_noncanonical_packet(tmp_path: Path) -> None:
    """Recomputing a digest must not authorize a mutated packet outside the committed identity."""
    packet = yaml.safe_load(PACKET.read_text(encoding="utf-8"))
    packet["seed_policy"]["search_seeds"] = [857011, 857012, 857013]
    packet["budget"]["max_steps_per_rollout"] = 7
    packet["compute_ceiling"]["max_steps_per_rollout"] = 7
    packet["outcome_vocabulary"]["blocked"] = "forged blocked meaning"
    packet["self_digest"] = compute_adversarial_falsification_packet_digest(packet)
    forged_packet = tmp_path / "forged-packet.yaml"
    forged_packet.write_text(yaml.safe_dump(packet, sort_keys=False), encoding="utf-8")

    with pytest.raises(BoundedFalsificationError, match="canonical packet"):
        build_bounded_falsification_preflight(forged_packet, repo_root=REPO_ROOT)


def test_generated_route_overlays_fail_closed_as_proposal_only(tmp_path: Path) -> None:
    """Inline route proposals must never be mistaken for loader-ready runtime inputs."""
    report = build_bounded_falsification_preflight(PACKET, repo_root=REPO_ROOT)
    overlay = report["arms"]["random"][0]["preparation"]["candidates"][0]["overlay"]
    assert overlay["provenance"]["route_overrides_contract"]["status"] == "proposal_only"
    assert overlay["provenance"]["route_overrides_contract"]["loadable"] is False
    assert overlay["provenance"]["route_overrides_contract"]["payload_sha256"]

    materialized = deepcopy(overlay["materialized"])
    materialized["scenarios"][0]["map_file"] = str(
        REPO_ROOT / "maps/svg_maps/classic_station_platform.svg"
    )
    scenario_path = tmp_path / "proposal.yaml"
    scenario_path.write_text(yaml.safe_dump(materialized, sort_keys=False), encoding="utf-8")
    scenario = load_scenarios(scenario_path)[0]

    assert "route_overrides_file" not in scenario
    with pytest.raises(ValueError, match="proposal-only"):
        build_robot_config_from_scenario(scenario, scenario_path=scenario_path)


@pytest.mark.parametrize(
    ("fail_on_blocked", "expected_exit"),
    [(False, 0), (True, 2)],
)
def test_cli_blocked_exit_mode_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_on_blocked: bool,
    expected_exit: int,
) -> None:
    """Ledger generation may succeed by default, while admission mode fails closed."""
    blocked_report = {
        "schema_version": VERTICAL_SLICE_SCHEMA_VERSION,
        "gate": {"status": "blocked"},
        "outcome_summary": {"blocked": 1},
        "execution": {"simulator_executed": False},
    }
    monkeypatch.setattr(
        preflight_cli,
        "write_bounded_falsification_preflight",
        lambda *args, **kwargs: blocked_report,
    )
    argv = [
        "run_issue_8571_bounded_falsification_slice.py",
        "--output",
        str(tmp_path / "report.json"),
    ]
    if fail_on_blocked:
        argv.append("--fail-on-blocked")
    monkeypatch.setattr(sys, "argv", argv)

    assert preflight_cli.main() == expected_exit
