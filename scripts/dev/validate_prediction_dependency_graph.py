"""Validate the prediction-lane dependency graph artifact.

This helper keeps the forecast-lane issue graph machine-checkable before agents
start or continue blocked learned-prediction work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ALLOWED_EXECUTION_STATES = {
    "ready",
    "blocked",
    "running",
    "parent",
    "closed_scaffold",
    "completed",
}
ALLOWED_GATE_STATUSES = {"passed", "required", "blocked", "running", "not_applicable"}


def _load_graph(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise SystemExit(f"prediction dependency graph validation failed: cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"prediction dependency graph validation failed: invalid JSON in {path}: {error}") from error


def _validate_graph_header(data: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(data.get("schema_version"), str):
        errors.append("schema_version must be a string")
    if not isinstance(data.get("lane"), str):
        errors.append("lane must be a string")
    if not isinstance(data.get("nodes"), list) or not data["nodes"]:
        errors.append("nodes must be a non-empty list")
    if not isinstance(data.get("execution_order"), list):
        errors.append("execution_order must be a list")
    if not isinstance(data.get("required_issue_set"), list):
        errors.append("required_issue_set must be a list")
    return errors


def _validate_gate(node_issue: int, gate: dict, known: set[int]) -> list[str]:
    errors: list[str] = []
    gate_id = gate.get("id")
    if not isinstance(gate_id, str):
        errors.append(f"issue {node_issue}: gate id must be a string")
        return errors

    status = gate.get("status")
    if status not in ALLOWED_GATE_STATUSES:
        errors.append(
            f"issue {node_issue}: gate {gate_id} status '{status}' is not in "
            f"{sorted(ALLOWED_GATE_STATUSES)}"
        )

    deps = gate.get("depends_on", [])
    if not isinstance(deps, list):
        errors.append(f"issue {node_issue}: gate {gate_id} depends_on must be a list")
        return errors
    for dep in deps:
        if not isinstance(dep, int):
            errors.append(f"issue {node_issue}: gate {gate_id} has non-int depends_on value {dep!r}")
        elif dep not in known:
            errors.append(f"issue {node_issue}: gate {gate_id} depends on unknown issue {dep}")
    return errors


def _validate_node(node: dict, seen: set[int], known: set[int]) -> list[str]:  # noqa: C901,PLR0912
    errors: list[str] = []
    issue = node.get("issue")
    if not isinstance(issue, int):
        return ["node issue must be an integer"]
    if issue <= 0:
        errors.append(f"issue {issue}: issue must be > 0")
    if issue in seen:
        errors.append(f"duplicate issue entry for {issue}")
    seen.add(issue)

    for field in ("title", "node_type"):
        if not isinstance(node.get(field), str):
            errors.append(f"issue {issue}: {field} must be a string")

    issue_state = node.get("issue_state")
    if issue_state not in {"open", "closed"}:
        errors.append(f"issue {issue}: issue_state '{issue_state}' must be 'open' or 'closed'")

    execution_state = node.get("execution_state")
    if execution_state not in ALLOWED_EXECUTION_STATES:
        errors.append(
            f"issue {issue}: execution_state '{execution_state}' is not "
            f"in {sorted(ALLOWED_EXECUTION_STATES)}"
        )

    for key in ("depends_on", "blocked_by"):
        refs = node.get(key, [])
        if not isinstance(refs, list):
            errors.append(f"issue {issue}: {key} must be a list")
            continue
        for ref in refs:
            if not isinstance(ref, int):
                errors.append(f"issue {issue}: {key} contains non-int value {ref!r}")
                continue
            if ref == issue:
                errors.append(f"issue {issue}: {key} contains self-reference")
            elif ref not in known:
                errors.append(f"issue {issue}: {key} references unknown issue {ref}")

    gates = node.get("evidence_gates", [])
    if not isinstance(gates, list):
        errors.append(f"issue {issue}: evidence_gates must be a list")
    else:
        for gate in gates:
            if not isinstance(gate, dict):
                errors.append(f"issue {issue}: gate entry must be an object")
            else:
                errors.extend(_validate_gate(issue, gate, known))

    return errors


def _validate_acyclic(nodes_by_issue: dict[int, dict]) -> list[str]:
    visit_state: dict[int, int] = {}
    errors: list[str] = []

    def walk(issue: int) -> None:
        state = visit_state.get(issue, 0)
        if state == 1:
            raise RuntimeError(f"dependency cycle detected at issue {issue}")
        if state == 2:
            return
        visit_state[issue] = 1
        for dep in nodes_by_issue[issue].get("depends_on", []):
            walk(dep)
        visit_state[issue] = 2

    for issue in nodes_by_issue:
        walk(issue)
    return errors


def validate(path: Path) -> int:  # noqa: C901,PLR0912
    """Validate graph payload and return non-zero on violations."""
    data = _load_graph(path)
    issues = data.get("nodes", [])

    errors = _validate_graph_header(data)
    if errors:
        print("prediction dependency graph validation failed: " + "; ".join(errors), file=sys.stderr)
        return 1

    required_set = data.get("required_issue_set", [])
    known: set[int] = {
        node.get("issue") for node in issues if isinstance(node, dict) and isinstance(node.get("issue"), int)
    }
    data_map = {node.get("issue"): node for node in issues if isinstance(node, dict) and isinstance(node.get("issue"), int)}

    seen: set[int] = set()
    node_errors: list[str] = []
    for node in issues:
        if isinstance(node, dict):
            node_errors.extend(_validate_node(node, seen, known))
        else:
            node_errors.append("all nodes must be objects")

    missing = [issue for issue in required_set if not isinstance(issue, int) or issue not in known]
    if missing:
        node_errors.append(
            "required_issue_set references missing or invalid issues: "
            + ", ".join(str(issue) for issue in missing)
        )

    order = data.get("execution_order", [])
    if len(set(order)) != len(order):
        node_errors.append("execution_order contains duplicates")
    for issue in order:
        if issue not in known:
            node_errors.append(f"execution_order references unknown issue {issue}")
    for issue in known:
        if issue not in order:
            node_errors.append(f"execution_order is missing issue {issue}")
    order_position: dict[int, int] = {issue: index for index, issue in enumerate(order)}
    for issue in known:
        depends_on = data_map.get(issue, {}).get("depends_on", [])
        if not isinstance(depends_on, list):
            continue
        position = order_position[issue]
        for dep in depends_on:
            if dep in order_position and order_position[dep] > position:
                node_errors.append(
                    f"execution_order violation: {dep} appears after {issue} though it is a dependency"
                )

    if not node_errors:
        try:
            node_errors.extend(_validate_acyclic(data_map))
        except RuntimeError as error:
            node_errors.append(str(error))

    if node_errors:
        print("prediction dependency graph validation failed: " + "; ".join(node_errors), file=sys.stderr)
        return 1

    print(f"prediction dependency graph validation passed for {len(known)} issue nodes")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("docs/context/prediction_lane_dependency_graph.json"),
        help="Path to prediction-lane dependency graph JSON file",
    )
    return parser.parse_args()


def main() -> int:
    """Entry-point for CLI invocation."""
    args = _parse_args()
    return validate(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
