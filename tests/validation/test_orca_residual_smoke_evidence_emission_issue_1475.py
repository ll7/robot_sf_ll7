"""Contract-completeness tests for ORCA-residual smoke evidence emission (issue #1475).

These tests pin the public-repo seam that makes the four required smoke-evidence
fields populate from real rollout signals:

* ``residual_clipping_rate``
* ``guard_veto_rate``
* ``fallback_degraded_status``
* ``artifact_pointer_status``

The historical fail-closed regression (SLURM job 12913) produced episode rows whose
``algorithm_metadata`` carried ``shield_stats`` but no ``residual_clipping_stats`` and
whose shield ``last_decision.action_adaptation`` (a ``direct_policy_command``
pass-through) lacked a ``residual_clipped`` key, so the residual extractor returned
``None`` and the stage classified ``missing_required_smoke_evidence``.

This is a contract-completeness fix: it guarantees the four fields are non-null
diagnostics, NOT that the smoke lane succeeds (``success_rate`` may still be 0.0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.planner.guarded_ppo import (
    GuardedPPOAdapter,
    build_guarded_ppo_config,
)
from robot_sf.planner.safety_shield import new_shield_stats, update_shield_stats
from scripts.validation.run_policy_search_candidate import (
    _attach_orca_residual_smoke_evidence,
    _row_residual_clipping_rate,
)


def _far_field_observation() -> dict[str, Any]:
    """Return a structured socnav observation with no pedestrians and a distant goal.

    Such an observation drives a guarded-PPO pass-through (``ppo_clear``) so the
    emitted action adaptation is a ``direct_policy_command`` with no residual applied.
    """
    return {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [10.0, 0.0], "next": [12.0, 0.0]},
        "pedestrians": {"positions": [], "velocities": [], "count": [0]},
    }


def _pass_through_adapter() -> GuardedPPOAdapter:
    """Return a guarded-PPO adapter configured for the residual smoke contract."""
    config = build_guarded_ppo_config(
        {
            "prior_residual_mode": True,
            "prior_residual_max_linear_delta": 0.35,
            "prior_residual_max_angular_delta": 0.35,
            # No prior adapter wired here, so far-field safe PPO commands pass through
            # as ``direct_policy_command`` decisions (the historical null path).
            "prior_policy": "none",
        }
    )
    return GuardedPPOAdapter(config=config, fallback_adapter=None, prior_adapter=None)


def test_pass_through_decision_emits_residual_clipped_signal() -> None:
    """A direct pass-through decision must still carry a ``residual_clipped`` flag.

    Before the fix this metadata was absent for ``direct_policy_command`` steps, which
    starved the residual-clipping extractor of evidence on any rollout where the PPO
    command was already safe.
    """
    adapter = _pass_through_adapter()
    decision = adapter.choose_command_decision(_far_field_observation(), (0.8, 0.0))

    metadata = decision.to_metadata()
    fallback_state = metadata["fallback_controller_state"]
    action_adaptation = fallback_state["action_adaptation"]

    # No bounded residual was applied on this pass-through, so the residual was
    # genuinely not clipped: the key must be present and False (a real signal).
    assert "residual_clipped" in action_adaptation
    assert action_adaptation["residual_clipped"] is False


def _representative_row(residual_clipped: bool) -> dict[str, Any]:
    """Build an episode row mirroring the guarded-PPO smoke rollout structure.

    The shield stats are accumulated through the production ``update_shield_stats``
    helper so the row carries the same ``last_decision`` shape that the map runner
    writes into ``algorithm_metadata`` during a real rollout.
    """
    adapter = _pass_through_adapter()
    decision = adapter.choose_command_decision(_far_field_observation(), (0.8, 0.0))
    # Force the residual-clipped flag for the clipped variant without fabricating the
    # rest of the rollout-derived decision payload.
    decision_metadata = decision.to_metadata()
    decision_metadata["fallback_controller_state"]["action_adaptation"]["residual_clipped"] = bool(
        residual_clipped
    )

    shield_stats = new_shield_stats()
    update_shield_stats(shield_stats, decision)
    shield_stats["last_decision"] = decision_metadata

    return {
        "status": "completed",
        "termination_reason": "max_steps",
        "metrics": {
            "shield_intervention_rate": 0.0,
        },
        "algorithm_metadata": {
            "shield_stats": shield_stats,
        },
    }


def test_residual_extractor_resolves_from_shield_last_decision() -> None:
    """The residual extractor must recover a rate from the shield ``last_decision``.

    This is the fallback path used when no aggregate ``residual_clipping_stats`` block
    is present (e.g. an older runtime), and it is the exact null path from job 12913.
    """
    not_clipped_rate, not_clipped_source = _row_residual_clipping_rate(
        _representative_row(residual_clipped=False)
    )
    assert not_clipped_rate == 0.0
    assert not_clipped_source.endswith("action_adaptation")

    clipped_rate, _clipped_source = _row_residual_clipping_rate(
        _representative_row(residual_clipped=True)
    )
    assert clipped_rate == 1.0


def test_attach_smoke_evidence_populates_all_required_fields(tmp_path: Path) -> None:
    """All four required smoke-evidence fields must be non-null with no missing fields.

    Contract-completeness boundary: this asserts the evidence block is complete, not
    that the lane passed. A degraded run would still classify degraded via the same
    helper; here the representative rows are non-degraded.
    """
    rows = [_representative_row(residual_clipped=False)]
    jsonl_path = tmp_path / "smoke.jsonl"
    jsonl_path.write_text("{}\n", encoding="utf-8")

    summary: dict[str, Any] = {}
    _attach_orca_residual_smoke_evidence(
        summary,
        rows,
        jsonl_path,
        missing_jsonl=False,
    )

    for field in (
        "residual_clipping_rate",
        "guard_veto_rate",
        "fallback_degraded_status",
        "artifact_pointer_status",
    ):
        assert summary.get(field) is not None, f"{field} should be populated"
        assert summary.get(field) != "", f"{field} should not be blank"

    evidence = summary["orca_residual_smoke_evidence"]
    assert evidence["missing_required_fields"] == []
    assert summary["fallback_degraded_status"] == "clear"
    assert summary["artifact_pointer_status"] == "local_jsonl_present"


def test_attach_smoke_evidence_preserves_degraded_classification(tmp_path: Path) -> None:
    """A genuinely degraded row must still classify degraded (fail-closed intact)."""
    row = _representative_row(residual_clipped=False)
    row["status"] = "fallback_degraded"

    jsonl_path = tmp_path / "smoke.jsonl"
    jsonl_path.write_text("{}\n", encoding="utf-8")

    summary: dict[str, Any] = {}
    _attach_orca_residual_smoke_evidence(summary, [row], jsonl_path, missing_jsonl=False)

    assert summary["fallback_degraded_status"] == "degraded"
    # The contract block stays complete even when the run is degraded.
    assert summary["orca_residual_smoke_evidence"]["missing_required_fields"] == []
