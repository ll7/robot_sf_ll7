"""Fail-closed preflight for the control-action-latency fidelity sweep (issue #5034).

Issue #5034 asks to *execute* the control-action-latency sweep that issue #4977
scoped and PR #5026 wires into the fidelity-sensitivity study config. Executing
the campaign is only meaningful once the study config actually carries a
``control_action_latency`` axis whose variants exercise the required
action-latency step set. Until then, running the existing fidelity campaign
would sweep the other fidelity axes and could be *mistaken for* the latency
sweep this issue requires.

This module is the guard that prevents that confusion. It inspects the study
config and **fails closed** unless a ``control_action_latency`` axis is present
and its variants cover the required action-latency step set (0, 1, 3 steps, i.e.
the 0/100/300 ms-equivalent delays named in the issue). It runs no episode and
promotes no claim: the emitted packet is a launch/readiness artifact and is not
benchmark evidence, not simulator-realism evidence, not sim-to-real evidence,
and not paper-facing evidence.

The axis schema mirrors the surface PR #5026 introduces::

    - key: control_action_latency
      variants:
        - key: zero_step_nominal      # baseline, patch.sim_config.action_latency_steps: 0
        - key: one_step_100ms         # patch.sim_config.action_latency_steps: 1
        - key: three_step_300ms       # patch.sim_config.action_latency_steps: 3

so the moment PR #5026 lands the same config flips this preflight from
``blocked`` to ``ready`` with no further change here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "control-action-latency-sweep-preflight.v1"

#: Axis key the control-action-latency sweep requires (matches PR #5026).
AXIS_KEY = "control_action_latency"

#: Action-latency step variants the issue-#5034 sweep must exercise: the
#: 0/100/300 ms-equivalent delays (1 step == one control period).
REQUIRED_LATENCY_STEPS: tuple[int, ...] = (0, 1, 3)

#: Location of the latency step inside each variant patch (PR #5026 schema).
LATENCY_STEP_PATCH_KEYS: tuple[str, str] = ("sim_config", "action_latency_steps")

#: The unmerged wiring this sweep depends on.
DEPENDENCY_PR = "#5026"
PARENT_ISSUE = 4977
ISSUE = 5034

DECISION_READY = "ready"
DECISION_BLOCKED = "blocked"

EVIDENCE_STATUS = "launch_readiness_preflight_not_benchmark_evidence"

CLAIM_BOUNDARY = (
    "Launch/readiness preflight only: verifies the fidelity-sensitivity study config carries "
    "a control_action_latency axis whose variants cover the required action-latency step set "
    "before the sweep is executed or promoted as the control-action-latency sweep. "
    "ready is a pre-registration gate, not an execution-ready or runtime-availability signal; "
    "it is not benchmark evidence, not simulator-realism evidence, not sim-to-real evidence, "
    "and not paper-facing evidence."
)


def _find_axis(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the ``control_action_latency`` axis mapping, or ``None`` if absent."""
    axes = config.get("axes")
    if not isinstance(axes, Sequence):
        return None
    for axis in axes:
        if isinstance(axis, Mapping) and axis.get("key") == AXIS_KEY:
            return axis
    return None


def _variant_latency_step(variant: Mapping[str, Any]) -> int | None:
    """Extract ``patch.sim_config.action_latency_steps`` from a variant, if present.

    Returns:
        The integer action-latency step, or ``None`` when the variant does not
        encode one at the expected patch location, so malformed variants fail
        closed rather than silently counting toward coverage.
    """
    patch = variant.get("patch")
    node: Any = patch
    for key in LATENCY_STEP_PATCH_KEYS:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    if isinstance(node, bool):  # guard against YAML true/false coercion
        return None
    if isinstance(node, int):
        return node
    return None


def _collect_latency_steps(axis: Mapping[str, Any]) -> tuple[list[int], int]:
    """Return the observed latency steps and the baseline-variant count for an axis."""
    steps: list[int] = []
    baseline_count = 0
    variants = axis.get("variants")
    if isinstance(variants, Sequence):
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            baseline_count += int(bool(variant.get("baseline", False)))
            step = _variant_latency_step(variant)
            if step is not None:
                steps.append(step)
    return steps, baseline_count


def check_control_action_latency_axis(
    config: Mapping[str, Any],
    *,
    config_path: str,
    git_head: str,
    date: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed control-action-latency sweep preflight packet.

    Args:
        config: Raw fidelity-sensitivity study config mapping.
        config_path: Repo-relative path of the config, recorded for provenance.
        git_head: Git head recorded for provenance.
        date: Optional ISO date string recorded for provenance.

    Returns:
        JSON-serializable packet. ``decision`` is ``ready`` only when the
        ``control_action_latency`` axis is present and covers every required
        action-latency step with exactly one baseline variant; otherwise it is
        ``blocked`` and ``blockers`` states the exact unmet prerequisite.
    """
    blockers: list[str] = []
    axis = _find_axis(config)
    observed_steps: list[int] = []
    missing_steps: list[int] = list(REQUIRED_LATENCY_STEPS)

    if axis is None:
        axis_present = False
        blockers.append(
            f"control_action_latency axis absent from {config_path}; depends on PR "
            f"{DEPENDENCY_PR} (parent issue #{PARENT_ISSUE}), which wires the axis and the "
            "sim_config.action_latency_steps field. Merge that PR before this sweep can be "
            "executed or promoted as the control-action-latency sweep."
        )
    else:
        axis_present = True
        observed_steps, baseline_count = _collect_latency_steps(axis)
        observed_set = set(observed_steps)
        missing_steps = [s for s in REQUIRED_LATENCY_STEPS if s not in observed_set]
        if missing_steps:
            blockers.append(
                f"control_action_latency axis in {config_path} is missing required "
                f"action-latency step variants {missing_steps} (required "
                f"{list(REQUIRED_LATENCY_STEPS)}, observed {sorted(observed_set)}); the sweep "
                "must exercise the 0/100/300 ms-equivalent delays."
            )
        if baseline_count != 1:
            blockers.append(
                f"control_action_latency axis in {config_path} must mark exactly one baseline "
                f"variant (found {baseline_count})."
            )

    decision = DECISION_BLOCKED if blockers else DECISION_READY

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "parent_issue": PARENT_ISSUE,
        "depends_on_pr": DEPENDENCY_PR,
        "decision": decision,
        "ready": decision == DECISION_READY,
        "axis_key": AXIS_KEY,
        "axis_present": axis_present,
        "required_latency_steps": list(REQUIRED_LATENCY_STEPS),
        "observed_latency_steps": sorted(set(observed_steps)),
        "missing_latency_steps": missing_steps,
        "blockers": blockers,
        "evidence_status": EVIDENCE_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "config_path": config_path,
        "git_head": git_head,
        "date": date,
        "next_command_template": (
            "uv run python scripts/benchmark/run_fidelity_sensitivity_campaign.py "
            f"--config {config_path} --fixed-scope-plan-only --require-launchable "
            "--plan-out output/fidelity_latency_plan"
        ),
    }


def write_control_action_latency_preflight(
    packet: Mapping[str, Any], output_dir: str | Path
) -> Path:
    """Write a deterministic JSON preflight packet.

    Returns:
        Path to the written ``control_action_latency_sweep_preflight.json`` file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    packet_path = out / "control_action_latency_sweep_preflight.json"
    packet_path.write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return packet_path
