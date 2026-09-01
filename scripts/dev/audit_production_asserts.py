#!/usr/bin/env python3
"""Build a deterministic inventory of production ``assert`` statements.

This is the issue-scoped audit for #7330.  It deliberately reads tracked
source only, records the exact repository commit, and fails closed when an
assertion is not covered by the reviewed classification table below.  It does
not change production code or interpret benchmark output.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "production-assert-inventory.v1"
ORIGINAL_ISSUE_ASSERT_COUNT = 62
HISTORICAL_RECONCILIATION_REFS = (
    "#6479",
    "#6516",
    "#6529",
    "#7210",
    "#7213",
    "#7216",
    "#7218",
    "#7221",
    "#7223",
    "#7228",
    "#7231",
    "#7234",
    "#7238",
)


class InventoryError(RuntimeError):
    """Raised when the audit cannot produce a complete reviewed inventory."""


@dataclass(frozen=True)
class AssertionRow:
    """Source-backed representation of one production assertion."""

    path: str
    line: int
    end_line: int
    qualified_scope: str
    expression: str
    message: str | None
    control_flow: tuple[str, ...]


@dataclass(frozen=True)
class Review:
    """Reviewed classification and ownership disposition for one assertion."""

    classification: str
    rationale: str
    ownership_status: str
    ownership_references: tuple[str, ...]


def _review(
    rationale: str,
    *,
    ownership_status: str,
    ownership_references: tuple[str, ...],
) -> Review:
    """Create the sole classification used by the current residual set."""
    return Review(
        classification="genuine_internal_invariant",
        rationale=rationale,
        ownership_status=ownership_status,
        ownership_references=ownership_references,
    )


_PR_6529_REFS = ("PR #6529", "#6479")
_NEW_RESIDUAL_REFS = ("#6479", "#6516", "#6529")


# Keys use normalized AST expressions and qualified scopes instead of line
# numbers so harmless line movement cannot silently change a classification.
REVIEWED_ASSERTIONS: dict[tuple[str, str, str], Review] = {
    (
        "robot_sf/baselines/social_force.py",
        "SocialForcePlanner._compute_total_force",
        "self._wrapper is not None",
    ): _review(
        "Planner setup initializes the force wrapper before total-force computation; this is a private state/type-narrowing invariant.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/benchmark/map_runner/map_runner_episode.py",
        "_setup_and_run_step_loop",
        "state is not None",
    ): _review(
        "Setup failures propagate through the finally block, so reaching the result builder is a postcondition that narrows the step-loop state.",
        ownership_status="unowned_residual",
        ownership_references=_NEW_RESIDUAL_REFS,
    ),
    (
        "robot_sf/data/external/ind.py",
        "_resolve_dataset_paths",
        "tracks_meta is not None",
    ): _review(
        "The preceding missing-file branch rejects the recording before construction; the assertion narrows the three resolved sibling paths for the typed constructor.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/data/external/ind.py",
        "_resolve_dataset_paths",
        "recording_meta is not None",
    ): _review(
        "The preceding missing-file branch rejects the recording before construction; the assertion narrows the three resolved sibling paths for the typed constructor.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/data/external/ind.py",
        "_resolve_dataset_paths",
        "background is not None",
    ): _review(
        "The preceding missing-file branch rejects the recording before construction; the assertion narrows the three resolved sibling paths for the typed constructor.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/planner/chance_constrained_mpc_provider.py",
        "_aggregate_horizon_risk",
        "cvar_alpha is not None",
    ): _review(
        "The non-joint formulation supplies CVaR alpha before rolling-window aggregation; this is an internal formulation postcondition.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/planner/crowdnav_height.py",
        "CrowdNavHeightAdapter.act",
        "self._hidden_state is not None",
    ): _review(
        "The act path resets recurrent state before inference; the assertion narrows the initialized private state.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/planner/socnav_orca.py",
        "ORCAPlannerAdapter._rvo2_simulator_for",
        "self._rvo2_robot_id is not None",
    ): _review(
        "A matching immutable scene signature implies the cached simulator and robot identifier were initialized together.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/planner/socnav_prediction.py",
        "PredictionPlannerAdapter._raise_cached_error",
        "self._load_error is not None",
    ): _review(
        "The cached-error helper is reachable only after model initialization recorded an exception; the assertion narrows that private state before re-raising it.",
        ownership_status="unowned_residual",
        ownership_references=_NEW_RESIDUAL_REFS,
    ),
    (
        "robot_sf/planner/socnav_prediction.py",
        "PredictionPlannerAdapter._predict_with_baseline",
        "self._baseline_predictor is not None",
    ): _review(
        "The baseline prediction branch initializes its predictor before use; this is a private lifecycle invariant rather than caller validation.",
        ownership_status="unowned_residual",
        ownership_references=_NEW_RESIDUAL_REFS,
    ),
    (
        "robot_sf/planner/socnav_sacadrl.py",
        "SACADRLPlannerAdapter._raise_cached_error",
        "self._load_error is not None",
    ): _review(
        "The cached-error helper is reachable only after model initialization recorded an exception; the assertion narrows that private state before re-raising it.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/planner/sonic_crowdnav.py",
        "SonicCrowdNavAdapter.act",
        "self._hidden_state is not None",
    ): _review(
        "The act path resets recurrent state before inference; the assertion narrows the initialized private state.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/render/sim_view_text_overlay.py",
        "SimViewTextOverlay._get_pedestrian_info_lines",
        "state.ego_ped_pose is not None",
    ): _review(
        "The pedestrian-data predicate guarantees the pose needed by the display calculation; this is a local postcondition/type narrowing.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/render/sim_view_text_overlay.py",
        "SimViewTextOverlay._get_pedestrian_info_lines",
        "state.ego_ped_action is not None",
    ): _review(
        "The pedestrian-data predicate guarantees the action needed by the display fields; this is a local postcondition/type narrowing.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/sim/simulator.py",
        "Simulator._apply_residual_adversary",
        "adversary is not None",
    ): _review(
        "An active configuration builds the residual adversary before this point; the assertion narrows the private capability used by the simulator.",
        ownership_status="unowned_residual",
        ownership_references=_NEW_RESIDUAL_REFS,
    ),
    (
        "robot_sf/telemetry/tensorboard_adapter.py",
        "TensorBoardAdapter.consume_snapshot",
        "self._writer is not None",
    ): _review(
        "The availability branch starts the writer before consumption; the assertion is a static-checker type narrowing for telemetry state.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/training/hybrid_replay_buffer.py",
        "HybridReplayBuffer.sample",
        "self._offline is not None",
    ): _review(
        "A positive offline sample count is admitted only when the offline partition is available; the assertion narrows that initialized private buffer.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/training/orca_residual_lineage_packet.py",
        "<module>",
        "set(_SMOKE_DERIVED_DIAGNOSTIC_FIELDS).issubset(_REQUIRED_DIAGNOSTICS)",
    ): _review(
        "The module-level contract keeps smoke-derived fields a subset of the canonical diagnostics vocabulary; it is an internal schema invariant.",
        ownership_status="completed_historical_review",
        ownership_references=_PR_6529_REFS,
    ),
    (
        "robot_sf/prediction/oracle_transition_trace.py",
        "OracleTransitionTraceV1.__post_init__",
        "pre_index is not None",
    ): _review(
        "The route-index completeness branch narrows all waypoint boundary indices before validating the one-step advance invariant.",
        ownership_status="unowned_residual",
        ownership_references=_NEW_RESIDUAL_REFS,
    ),
    (
        "robot_sf/prediction/oracle_transition_trace.py",
        "OracleTransitionTraceV1.__post_init__",
        "post_behavior_index is not None",
    ): _review(
        "The route-index completeness branch narrows all waypoint boundary indices before validating the one-step advance invariant.",
        ownership_status="unowned_residual",
        ownership_references=_NEW_RESIDUAL_REFS,
    ),
    (
        "robot_sf/prediction/oracle_transition_trace.py",
        "OracleTransitionTraceV1.__post_init__",
        "post_integration_index is not None",
    ): _review(
        "The route-index completeness branch narrows all waypoint boundary indices before validating the one-step advance invariant.",
        ownership_status="unowned_residual",
        ownership_references=_NEW_RESIDUAL_REFS,
    ),
    (
        "robot_sf/prediction/oracle_transition_trace.py",
        "_fold_force_stage",
        "stage.delta_force_xy is not None",
    ): _review(
        "The force-stage constructor requires additive deltas before this internal fold narrows the optional field for arithmetic.",
        ownership_status="unowned_residual",
        ownership_references=_NEW_RESIDUAL_REFS,
    ),
    (
        "robot_sf/prediction/oracle_transition_trace.py",
        "_fold_force_stage",
        "stage.operation_kind in {ForceOperationKind.REPLACEMENT, ForceOperationKind.TRANSFORMED}",
    ): _review(
        "The preceding operation dispatch exhausts the known force-stage kinds; this assertion protects the replacement/transform branch from future enum drift.",
        ownership_status="unowned_residual",
        ownership_references=_NEW_RESIDUAL_REFS,
    ),
    (
        "robot_sf/prediction/oracle_transition_trace.py",
        "_fold_force_stage",
        "stage.result_force_xy is not None",
    ): _review(
        "The force-stage constructor requires a recorded result for replacement and transformed stages before this internal fold returns it.",
        ownership_status="unowned_residual",
        ownership_references=_NEW_RESIDUAL_REFS,
    ),
}


def _git(repo_root: Path, *args: str) -> str:
    """Run one read-only git command and return stripped stdout."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_symbolic_ref(repo_root: Path) -> str:
    """Return the current symbolic ref, or an empty string for detached HEAD."""
    command = ["git", "-C", str(repo_root), "symbolic-ref", "--short", "-q", "HEAD"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode == 1:
        return ""
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout.strip()


def _tracked_python_files(repo_root: Path, source_root: Path) -> list[Path]:
    """Return tracked Python files below ``source_root`` in lexical order."""
    relative_root = source_root.relative_to(repo_root).as_posix()
    if relative_root == ".":
        pathspecs = ("*.py", "**/*.py")
    else:
        pathspecs = (f"{relative_root}/*.py", f"{relative_root}/**/*.py")
    output = _git(repo_root, "ls-files", "--", *pathspecs)
    paths = {
        (repo_root / relative_path).resolve()
        for relative_path in output.splitlines()
        if relative_path
    }
    return sorted(path for path in paths if path.is_file())


def _control_flow_label(node: ast.AST) -> str | None:
    """Return a stable description for one control-flow ancestor."""
    match node:
        case ast.If(test=test) | ast.While(test=test):
            keyword = "if" if isinstance(node, ast.If) else "while"
            return f"{keyword} {ast.unparse(test)}"
        case ast.For(target=target, iter=iter_node) | ast.AsyncFor(target=target, iter=iter_node):
            keyword = "for" if isinstance(node, ast.For) else "async for"
            return f"{keyword} {ast.unparse(target)} in {ast.unparse(iter_node)}"
        case ast.Try():
            return "try"
        case ast.ExceptHandler(type=None):
            return "except"
        case ast.ExceptHandler(type=exception_type):
            return f"except {ast.unparse(exception_type)}"
        case ast.With(items=items) | ast.AsyncWith(items=items):
            keyword = "with" if isinstance(node, ast.With) else "async with"
            rendered_items = ", ".join(ast.unparse(item) for item in items)
            return f"{keyword} {rendered_items}"
        case ast.Match(subject=subject):
            return f"match {ast.unparse(subject)}"
        case _:
            return None


def _parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Build parent links for deterministic scope and control-flow lookup."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _assertion_row(
    path: Path,
    source_root: Path,
    node: ast.Assert,
    parents: dict[int, ast.AST],
) -> AssertionRow:
    """Project one AST assertion into the inventory schema."""
    ancestors: list[ast.AST] = []
    current = parents.get(id(node))
    while current is not None:
        ancestors.append(current)
        current = parents.get(id(current))

    scopes: list[str] = []
    for ancestor in reversed(ancestors):
        if isinstance(ancestor, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(ancestor.name)
    qualified_scope = ".".join(scopes) if scopes else "<module>"

    control_flow = [
        label
        for ancestor in reversed(ancestors)
        if (label := _control_flow_label(ancestor)) is not None
    ]
    relative_path = path.relative_to(source_root.parent).as_posix()
    return AssertionRow(
        path=relative_path,
        line=node.lineno,
        end_line=node.end_lineno or node.lineno,
        qualified_scope=qualified_scope,
        expression=ast.unparse(node.test),
        message=ast.unparse(node.msg) if node.msg is not None else None,
        control_flow=tuple(control_flow),
    )


def _collect_assertions(repo_root: Path, source_root: Path) -> list[AssertionRow]:
    """Parse every tracked source file and return assertions in source order."""
    rows: list[AssertionRow] = []
    for path in _tracked_python_files(repo_root, source_root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise InventoryError(f"cannot parse tracked source {path}: {exc}") from exc
        parents = _parent_map(tree)
        rows.extend(
            _assertion_row(path, source_root, node, parents)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert)
        )
    return sorted(rows, key=lambda row: (row.path, row.line, row.expression))


def _row_key(row: AssertionRow) -> tuple[str, str, str]:
    """Return the stable lookup key for a reviewed assertion."""
    return row.path, row.qualified_scope, row.expression


def _reviewed_rows(rows: list[AssertionRow]) -> list[dict[str, Any]]:
    """Attach reviewed classifications, refusing unknown source assertions."""
    unknown = sorted({_row_key(row) for row in rows if _row_key(row) not in REVIEWED_ASSERTIONS})
    if unknown:
        details = "; ".join(" | ".join(key) for key in unknown)
        raise InventoryError(f"unreviewed production assertion(s): {details}")

    reviewed: list[dict[str, Any]] = []
    for row in rows:
        review = REVIEWED_ASSERTIONS[_row_key(row)]
        reviewed.append(
            {
                "path": row.path,
                "line": row.line,
                "end_line": row.end_line,
                "scope": row.qualified_scope,
                "expression": row.expression,
                "message": row.message,
                "local_context": {"control_flow": list(row.control_flow)},
                "classification": review.classification,
                "rationale": review.rationale,
                "ownership": {
                    "status": review.ownership_status,
                    "references": list(review.ownership_references),
                },
                "recommended_action": "retain_assert_as_internal_invariant",
            }
        )
    return reviewed


def build_inventory(repo_root: Path, source_root: Path) -> dict[str, Any]:
    """Build the complete JSON-compatible inventory for one repository commit."""
    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    try:
        source_root.relative_to(repo_root)
    except ValueError as exc:
        raise InventoryError("source root must be inside the repository root") from exc

    rows = _collect_assertions(repo_root, source_root)
    reviewed = _reviewed_rows(rows)
    classifications = Counter(row["classification"] for row in reviewed)
    ownership = Counter(row["ownership"]["status"] for row in reviewed)
    tracked_files = _tracked_python_files(repo_root, source_root)
    dirty = bool(_git(repo_root, "status", "--porcelain", "--untracked-files=all"))

    return {
        "schema": SCHEMA,
        "source": {
            "repository": repo_root.name,
            "root": source_root.relative_to(repo_root).as_posix(),
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "ref": _git_symbolic_ref(repo_root) or "DETACHED",
            "clean": not dirty,
            "tracked_python_file_count": len(tracked_files),
        },
        "historical_reconciliation": {
            "original_issue_6479_assertion_count": ORIGINAL_ISSUE_ASSERT_COUNT,
            "merged_pr_6529": {
                "historical_converted_count": 11,
                "historical_internal_invariant_count": 23,
                "current_rows_reconciled": ownership["completed_historical_review"],
            },
            "benchmark_sibling_issue_6516": {
                "historical_scope": "benchmark-only assert-to-raise hardening",
                "current_rows_reopened": 0,
            },
            "narrower_children": list(HISTORICAL_RECONCILIATION_REFS[3:]),
            "open_owner_result": "no_current_row_is_owned_by_an_open_issue_or_pr",
        },
        "assertions": reviewed,
        "counts": {
            "assertion_count": len(reviewed),
            "classification": dict(sorted(classifications.items())),
            "ownership": dict(sorted(ownership.items())),
        },
        "recommendation": {
            "code": "close_parent_residuals_internal_only",
            "reason": (
                "All current residual assertions are reviewed genuine internal invariants; "
                "no caller/configuration/evidence guard remains and no conversion child is selected."
            ),
            "next_conversion_child": None,
        },
        "scope_boundary": {
            "production_changes": False,
            "benchmark_execution": False,
            "evidence_interpretation": False,
            "slurm_or_external_work": False,
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the inventory as deterministic reviewable Markdown."""
    source = payload["source"]
    counts = payload["counts"]
    reconciliation = payload["historical_reconciliation"]
    lines = [
        "# Production assert inventory (issue #7330)",
        "",
        "Read-only AST inventory of tracked `robot_sf/**/*.py` source.",
        "",
        f"- Commit: `{source['commit']}`",
        f"- Ref: `{source['ref']}`",
        f"- Clean: `{str(source['clean']).lower()}`",
        f"- Tracked Python files: `{source['tracked_python_file_count']}`",
        "- Tests, generated output, and non-package vendored paths are excluded by the source root.",
        "",
        "## Counts",
        "",
        f"- Original #6479 claim: `{reconciliation['original_issue_6479_assertion_count']}` assertions",
        f"- Current exact-source count: `{counts['assertion_count']}` assertions",
        "",
        "| Classification | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{name}` | {count} |" for name, count in counts["classification"].items())
    lines.extend(
        [
            "",
            "| Ownership disposition | Count |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| `{name}` | {count} |" for name, count in counts["ownership"].items())
    lines.extend(
        [
            "",
            "## Historical reconciliation",
            "",
            f"- PR #6529: 11 historical conversions and 23 retained invariants; `{reconciliation['merged_pr_6529']['current_rows_reconciled']}` current rows match its retained-invariant review.",
            "- Issue #6516: benchmark-only hardening was reviewed separately; no current row is reopened from that completed scope.",
            f"- Narrower children checked: {', '.join(reconciliation['narrower_children'])}.",
            f"- Open-owner result: `{reconciliation['open_owner_result']}`.",
            "",
            "## Current rows",
            "",
            "| Location | Expression | Classification | Ownership | Rationale |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["assertions"]:
        location = f"`{row['path']}:{row['line']}` ({row['scope']})"
        expression = f"`{row['expression']}`"
        rationale = row["rationale"].replace("|", "\\|")
        lines.append(
            f"| {location} | {expression} | `{row['classification']}` | "
            f"`{row['ownership']['status']}` | {rationale} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"`{payload['recommendation']['code']}`",
            "",
            payload["recommendation"]["reason"],
            "",
            "This audit changes no production behavior, benchmark evidence, Slurm state, or issue state.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, default=Path("robot_sf"))
    parser.add_argument("--json", dest="json_path", type=Path, required=True)
    parser.add_argument("--markdown", dest="markdown_path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the read-only inventory command."""
    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    source_root = args.root if args.root.is_absolute() else repo_root / args.root
    try:
        payload = build_inventory(repo_root, source_root)
    except InventoryError as exc:
        print(f"inventory failed closed: {exc}", file=sys.stderr)
        return 2

    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    markdown_text = render_markdown(payload)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json_text, encoding="utf-8")
    args.markdown_path.write_text(markdown_text, encoding="utf-8")
    print(
        f"inventoried {payload['counts']['assertion_count']} assertions at "
        f"{payload['source']['commit']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
