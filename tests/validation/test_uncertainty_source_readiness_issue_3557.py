"""Tests for issue #3557 uncertainty-source readiness inventory."""

from __future__ import annotations

from scripts.validation.check_uncertainty_source_readiness_issue_3557 import (
    CLASS_PROBABILITY_AMBIGUITY,
    COVARIANCE_INFLATION,
    DEFAULT_SOURCE_SPECS,
    EXISTENCE_DEGRADATION,
    MISSING_CONDITION_BUILDER,
    MISSING_SCENARIO_HOOK,
    MISSING_SURROGATE_OUTPUT,
    SOURCE_READY,
    TRACKING_NOISE,
    VISIBILITY_OCCLUSION,
    SourceReadinessSpec,
    inspect_uncertainty_source_readiness,
    main,
)


def test_default_inventory_classifies_current_and_missing_sources() -> None:
    """Default report shows only hardcoded #3471 source currently runnable."""

    report = inspect_uncertainty_source_readiness()
    by_source = {row.source: row for row in report.sources}

    assert tuple(by_source) == tuple(spec.source for spec in DEFAULT_SOURCE_SPECS)
    assert by_source[EXISTENCE_DEGRADATION].status == SOURCE_READY
    for source in (
        VISIBILITY_OCCLUSION,
        COVARIANCE_INFLATION,
        CLASS_PROBABILITY_AMBIGUITY,
    ):
        assert by_source[source].status == MISSING_SCENARIO_HOOK
        assert by_source[source].condition_builder.present
        assert not by_source[source].scenario_hook.present
        assert by_source[source].expected_surrogate_outputs.present
    assert by_source[TRACKING_NOISE].status == MISSING_CONDITION_BUILDER
    assert not by_source[TRACKING_NOISE].condition_builder.present
    assert not by_source[TRACKING_NOISE].scenario_hook.present
    assert by_source[TRACKING_NOISE].expected_surrogate_outputs.present


def test_report_payload_is_inventory_not_generalization_claim() -> None:
    """Machine-readable payload keeps the claim boundary explicit."""

    payload = inspect_uncertainty_source_readiness().as_dict()

    assert payload["issue"] == 3557
    assert payload["ready"] is False
    assert payload["ready_sources"] == [EXISTENCE_DEGRADATION]
    assert "does not run the episode harness" in payload["claim_boundary"]
    assert "does not claim" in payload["claim_boundary"]


def test_synthetic_missing_surrogate_output_is_classified() -> None:
    """A source with builder and hook but missing output contract is blocked accordingly."""

    report = inspect_uncertainty_source_readiness(
        (
            SourceReadinessSpec(
                source="synthetic_missing_output",
                condition_builder_owner=(
                    "scripts.validation.run_scenario_belief_episode_safety_issue_3471:"
                    "build_belief_for_mode"
                ),
                scenario_hook_owner="scripts.validation.run_scenario_belief_episode_safety_issue_3471:run_episode",
                surrogate_output_owner=None,
            ),
        )
    )

    row = report.sources[0]
    assert row.status == MISSING_SURROGATE_OUTPUT
    assert row.condition_builder.present
    assert row.scenario_hook.present
    assert not row.expected_surrogate_outputs.present


def test_cli_emits_inventory_without_running_when_sources_blocked(capsys) -> None:
    """CLI emits inventory without treating missing prerequisites as execution failure."""

    exit_code = main(["--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"blocked_sources"' in captured.out
    assert VISIBILITY_OCCLUSION in captured.out


def test_cli_can_fail_closed_for_gate_usage(capsys) -> None:
    """Explicit gate mode exits nonzero while source prerequisites are missing."""

    exit_code = main(["--fail-on-blocked"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert '"ready": false' in captured.out
