"""Human-facing diagnostics for fail-closed Issue #5592 artifact validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def format_fail_closed_warning(
    *,
    tool: str,
    reason: str,
    input_paths: Sequence[Path],
    output_path: Path | None = None,
) -> str:
    """Format a prominent, actionable warning for an ineligible artifact."""
    lines = [
        "=" * 78,
        "WARNING: ISSUE #5592 ARTIFACT REJECTED - NOT ELIGIBLE FOR EVIDENCE",
        "=" * 78,
        f"Tool: {tool}",
        "",
        "This is an intentional fail-closed safety boundary.",
        "The input is not valid for ranking or cross-matrix comparison.",
        "Do NOT interpret, publish, compare, or promote it as benchmark,",
        "generalization, or paper-facing evidence.",
        "",
        "Reason:",
        f"  {reason}",
        "",
        "Inputs examined:",
        *(f"  - {path}" for path in input_paths),
    ]
    if output_path is not None:
        lines.extend(
            [
                "",
                f"Intended output: {output_path}",
                "  This invocation did not produce a valid output; do not consume a partial or",
                "  stale file at this path.",
            ]
        )
    lines.extend(
        [
            "",
            "RECOMMENDED FIX - REQUIRED BEFORE RERUN:",
            "  1. Repair or regenerate the source artifact from the frozen #5592 contract;",
            "     do not hand-edit a rank or bypass this validation message.",
            "  2. For episode aggregates, verify the frozen 12-planner roster occurs exactly",
            "     once; for ranking inputs, verify the matching roster signature and a unique",
            "     1..4 permutation. Where applicable, required metrics must be finite, rates",
            "     must be in [0, 1], and native metadata must declare status=ok",
            "     and planner_kinematics.execution_mode=native; availability_status",
            "     must be available when present.",
            "  3. Preserve exact ties as tie_not_identifiable; never invent a performance",
            "     ordering to make the downstream comparison pass.",
            "  4. Re-run this tool and keep the result blocked/diagnostic-only until this",
            "     warning disappears on a clean, provenance-preserving run.",
            "",
            "Current disposition: BLOCKED. Exit code remains non-zero by design.",
            "=" * 78,
        ]
    )
    return "\n".join(lines)
