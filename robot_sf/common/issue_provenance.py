"""Central issue-number provenance for code-owned diagnostic contracts.

These identifiers are breadcrumbs to the issue contracts that introduced a
runtime or report field. They are not mutable runtime configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class IssueProvenance:
    """Immutable metadata for an issue-backed code contract."""

    issue: int
    purpose: str


SCENARIO_BELIEF_DESIGN_PARENT = IssueProvenance(
    issue=1966,
    purpose="ScenarioBelief design parent carried in debug projections.",
)
LIVE_FORECAST_REPLAY_GATE_CONTRACT = IssueProvenance(
    issue=2941,
    purpose="Native live forecast replay gate report contract.",
)

SCENARIO_BELIEF_DESIGN_PARENT_ISSUE = SCENARIO_BELIEF_DESIGN_PARENT.issue
LIVE_FORECAST_REPLAY_GATE_CONTRACT_ISSUE = LIVE_FORECAST_REPLAY_GATE_CONTRACT.issue

ISSUE_PROVENANCE_BY_KEY = MappingProxyType(
    {
        "scenario_belief_design_parent": SCENARIO_BELIEF_DESIGN_PARENT,
        "live_forecast_replay_gate_contract": LIVE_FORECAST_REPLAY_GATE_CONTRACT,
    }
)

__all__ = [
    "ISSUE_PROVENANCE_BY_KEY",
    "LIVE_FORECAST_REPLAY_GATE_CONTRACT",
    "LIVE_FORECAST_REPLAY_GATE_CONTRACT_ISSUE",
    "SCENARIO_BELIEF_DESIGN_PARENT",
    "SCENARIO_BELIEF_DESIGN_PARENT_ISSUE",
    "IssueProvenance",
]
