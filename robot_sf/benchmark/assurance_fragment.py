"""Assurance Case Fragment exporter.

Generates a machine-readable claim-argument-evidence tree in GSN-flavored JSON
along with Markdown and SVG representations.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.common.artifact_paths import get_repository_root


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of the file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    """Return a repository-relative path as a string."""
    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def build_assurance_fragment(  # noqa: C901, PLR0915
    campaign_summary: dict[str, Any],
    repo_root: Path,
    release_gate_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a GSN-flavored assurance fragment from campaign summary and config.

    Returns:
        JSON-serializable GSN assurance fragment payload.
    """
    campaign_meta = campaign_summary.get("campaign", {})
    campaign_name = campaign_meta.get("name", "unnamed_campaign")
    campaign_id = campaign_meta.get("campaign_id", "unnamed_campaign_id")
    scenario_matrix = campaign_meta.get("scenario_matrix", "unknown_matrix")
    scenario_matrix_hash = campaign_meta.get("scenario_matrix_hash", "")
    git_hash = campaign_meta.get("git_hash", "unknown")
    campaign_meta.get("benchmark_success", False)

    nodes: dict[str, Any] = {}

    # G0: Root Goal
    g0_id = "G_root"
    nodes[g0_id] = {
        "id": g0_id,
        "type": "goal",
        "text": f"Campaign '{campaign_name}' meets safety, comfort, and performance requirements on scenario matrix '{scenario_matrix}'.",
        "children": ["S_campaign", "C_matrix", "C_git"],
    }

    # C_matrix: Context for Matrix
    nodes["C_matrix"] = {
        "id": "C_matrix",
        "type": "context",
        "text": f"Scenario Matrix: {scenario_matrix} (SHA-256: {scenario_matrix_hash})",
    }

    # C_git: Context for git commit
    nodes["C_git"] = {
        "id": "C_git",
        "type": "context",
        "text": f"Git Commit SHA: {git_hash}",
    }

    # S_campaign: Decompose campaign by planners
    s_campaign_id = "S_campaign"
    planner_rows = campaign_summary.get("planner_rows", [])
    planner_goal_ids = [f"G_{row.get('planner_key')}" for row in planner_rows if row.get("planner_key")]

    # We also link the final summary JSON as a direct solution/evidence for this strategy
    s_campaign_children = list(planner_goal_ids) + ["Sn_campaign_summary"]
    nodes[s_campaign_id] = {
        "id": s_campaign_id,
        "type": "strategy",
        "text": "Argue by demonstrating each evaluated planner satisfies individual gate and performance criteria.",
        "children": s_campaign_children,
    }

    # Sn_campaign_summary: Solution node for summary
    summary_rel = campaign_summary.get("artifacts", {}).get("campaign_summary_json", "")
    summary_sha = ""
    if summary_rel:
        summary_path = repo_root / summary_rel
        if summary_path.exists():
            summary_sha = _sha256_file(summary_path)

    nodes["Sn_campaign_summary"] = {
        "id": "Sn_campaign_summary",
        "type": "solution",
        "text": f"Campaign summary report: {summary_rel}",
        "metadata": {
            "path": summary_rel,
            "sha256": summary_sha,
        },
    }

    # Map planner run entries for episodes files
    run_entries = campaign_summary.get("runs", [])
    planner_to_episodes: dict[str, tuple[str, str]] = {}
    for entry in run_entries:
        pkey = entry.get("planner_key")
        ep_path_str = entry.get("episodes_path")
        if pkey and ep_path_str:
            ep_path = repo_root / ep_path_str
            if ep_path.exists():
                sha = _sha256_file(ep_path)
                planner_to_episodes[pkey] = (ep_path_str, sha)

    # Process each planner row
    for row in planner_rows:
        pkey = row.get("planner_key")
        if not pkey:
            continue
        algo = row.get("algo", "unknown_algo")
        kinematics = row.get("kinematics", "unknown_kinematics")
        episodes_count = row.get("episodes", 0)
        success_mean = row.get("success_mean", "0.0")
        collisions_mean = row.get("collisions_mean", "0.0")
        snqi_mean = row.get("snqi_mean", "nan")
        readiness_tier = row.get("readiness_tier", "unknown_tier")
        execution_mode = row.get("execution_mode", "unknown_mode")
        p_success = row.get("benchmark_success", "false")

        p_goal_id = f"G_{pkey}"
        p_strat_id = f"S_{pkey}"
        p_context_id = f"C_{pkey}_context"
        p_train_id = f"A_{pkey}_train"
        p_deploy_id = f"A_{pkey}_deploy"

        p_children = [p_strat_id, p_context_id, p_train_id, p_deploy_id]

        nodes[p_goal_id] = {
            "id": p_goal_id,
            "type": "goal",
            "text": f"Planner '{pkey}' ({algo}) meets gate requirements under '{kinematics}' kinematics.",
            "children": p_children,
        }

        nodes[p_context_id] = {
            "id": p_context_id,
            "type": "context",
            "text": f"Planner Readiness: {readiness_tier}, Execution Mode: {execution_mode}",
        }

        # AMLAS-style placeholders for learned components
        nodes[p_train_id] = {
            "id": p_train_id,
            "type": "assumption",
            "text": f"AMLAS Training Data: training data assumptions for learned components of '{pkey}' stated, not verified.",
        }
        nodes[p_deploy_id] = {
            "id": p_deploy_id,
            "type": "assumption",
            "text": f"AMLAS Deployment Match: deployment-context and ODD match for '{pkey}' stated, not verified.",
        }

        strat_children = [f"G_{pkey}_success", f"G_{pkey}_safety"]
        if snqi_mean != "nan":
            strat_children.append(f"G_{pkey}_snqi")

        nodes[p_strat_id] = {
            "id": p_strat_id,
            "type": "strategy",
            "text": f"Demonstrate performance metrics satisfy release criteria and benchmark thresholds for '{pkey}'.",
            "children": strat_children,
        }

        # Find episodes file
        ep_rel, ep_sha = planner_to_episodes.get(pkey, ("", ""))
        ep_solution_id = f"Sn_{pkey}_episodes"

        nodes[ep_solution_id] = {
            "id": ep_solution_id,
            "type": "solution",
            "text": f"Episode logs for '{pkey}': {ep_rel}",
            "metadata": {
                "path": ep_rel,
                "sha256": ep_sha,
                "episode_count": episodes_count,
            },
        }

        nodes[f"G_{pkey}_success"] = {
            "id": f"G_{pkey}_success",
            "type": "goal",
            "text": f"Planner achieves success rate mean of {success_mean} (benchmark_success={p_success}).",
            "children": [ep_solution_id],
        }

        nodes[f"G_{pkey}_safety"] = {
            "id": f"G_{pkey}_safety",
            "type": "goal",
            "text": f"Planner achieves collision rate mean of {collisions_mean}.",
            "children": [ep_solution_id],
        }

        if snqi_mean != "nan":
            nodes[f"G_{pkey}_snqi"] = {
                "id": f"G_{pkey}_snqi",
                "type": "goal",
                "text": f"Planner achieves SNQI mean score of {snqi_mean}.",
                "children": [ep_solution_id],
            }

    # Add release gate report details if available
    if release_gate_report:
        gate_rel = _repo_relative(Path(release_gate_report.get("provenance", {}).get("input", {}).get("path", "")), repo_root)
        gate_sha = release_gate_report.get("provenance", {}).get("input", {}).get("sha256", "")

        nodes["Sn_release_gates"] = {
            "id": "Sn_release_gates",
            "type": "solution",
            "text": f"Release gate evaluation report: {gate_rel}",
            "metadata": {
                "path": gate_rel,
                "sha256": gate_sha,
            },
        }

        # Link each planner goal to the release gates solution
        for row in release_gate_report.get("matrix_rows", []):
            pkey = row.get("planner_key")
            if pkey:
                p_strat_id = f"S_{pkey}"
                if p_strat_id in nodes:
                    g_gate_id = f"G_{pkey}_gates"
                    nodes[g_gate_id] = {
                        "id": g_gate_id,
                        "type": "goal",
                        "text": f"Release gates safety: {row.get('safety_status')}, comfort: {row.get('comfort_status')} (overall={row.get('overall_status')}).",
                        "children": ["Sn_release_gates"],
                    }
                    if g_gate_id not in nodes[p_strat_id]["children"]:
                        nodes[p_strat_id]["children"].append(g_gate_id)

    return {
        "schema_version": "assurance_fragment.v1",
        "campaign_id": campaign_id,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "nodes": nodes,
    }


def validate_assurance_fragment(payload: dict[str, Any]) -> None:
    """Validate assurance fragment payload against JSON Schema."""
    schema_path = get_repository_root() / "robot_sf/benchmark/schemas/assurance_fragment.schema.v1.json"
    if not schema_path.exists():
        return
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    Draft202012Validator(schema).validate(payload)


def render_assurance_fragment_to_markdown(payload: dict[str, Any]) -> str:  # noqa: C901
    """Render the GSN tree to Markdown format, including a Mermaid diagram.

    Returns:
        The Markdown content as a string.
    """
    nodes = payload["nodes"]
    campaign_id = payload["campaign_id"]
    gen_time = payload["generated_at_utc"]

    lines = [
        f"# Assurance Case Fragment for Campaign: `{campaign_id}`",
        f"Generated at: {gen_time}",
        "",
        "## Goal Structuring Notation (GSN) Diagram",
        "",
        "```mermaid",
        "graph TD",
        "  %% GSN node styling",
        "  classDef goal fill:#d4edda,stroke:#28a745,stroke-width:2px;",
        "  classDef strategy fill:#fff3cd,stroke:#ffc107,stroke-width:2px;",
        "  classDef context fill:#e2e3e5,stroke:#6c757d,stroke-width:2px;",
        "  classDef assumption fill:#cce5ff,stroke:#007bff,stroke-width:2px;",
        "  classDef solution fill:#f8d7da,stroke:#dc3545,stroke-width:2px;",
        "",
    ]

    # Declare nodes
    for node_id, node in sorted(nodes.items()):
        text = node["text"].replace('"', '\\"')
        ntype = node["type"]
        if ntype == "goal":
            lines.append(f'  {node_id}["Goal: {text}"]:::goal')
        elif ntype == "strategy":
            lines.append(f'  {node_id}[/"Strategy: {text}"/]:::strategy')
        elif ntype == "assumption":
            lines.append(f'  {node_id}(("Assumption: {text}")):::assumption')
        elif ntype == "context":
            lines.append(f'  {node_id}(["Context: {text}"]):::context')
        elif ntype == "solution":
            lines.append(f'  {node_id}[["Solution: {text}"]]:::solution')

    lines.append("")
    # Declare connections
    for node_id, node in sorted(nodes.items()):
        children = node.get("children", [])
        for child_id in children:
            if child_id in nodes:
                lines.append(f"  {node_id} --> {child_id}")

    lines.extend([
        "```",
        "",
        "## GSN Hierarchical Text Tree",
        "",
    ])

    visited = set()

    def print_node(node_id, indent=0):
        if node_id not in nodes or node_id in visited:
            return
        visited.add(node_id)
        node = nodes[node_id]
        indent_str = "  " * indent
        ntype = node["type"].upper()
        lines.append(f"{indent_str}- **{ntype} ({node_id})**: {node['text']}")
        for child_id in node.get("children", []):
            print_node(child_id, indent + 1)

    if "G_root" in nodes:
        print_node("G_root")

    return "\n".join(lines) + "\n"


def _wrap_text(text: str, max_chars: int = 24) -> list[str]:
    """Helper to wrap text into lines of at most max_chars length.

    Returns:
        A list of wrapped text lines.
    """
    words = text.split()
    lines = []
    current_line: list[str] = []
    current_len = 0
    for word in words:
        if current_len + len(word) + (1 if current_line else 0) > max_chars:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)
        else:
            current_line.append(word)
            current_len += len(word) + (1 if len(current_line) > 1 else 0)
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def render_assurance_fragment_to_svg(payload: dict[str, Any]) -> str:  # noqa: C901, PLR0912, PLR0915
    """Render GSN tree to a standalone SVG vector image.

    Returns:
        The SVG XML content as a string.
    """
    nodes = payload["nodes"]

    # Simple BFS to identify levels of the GSN tree starting from G_root
    levels: list[list[str]] = []
    queue = [("G_root", 0)]
    node_depth: dict[str, int] = {}
    while queue:
        node_id, depth = queue.pop(0)
        if node_id in node_depth:
            if depth <= node_depth[node_id]:
                continue
        node_depth[node_id] = depth
        for child in nodes.get(node_id, {}).get("children", []):
            if child in nodes:
                queue.append((child, depth + 1))

    depth_to_nodes: dict[int, list[str]] = defaultdict(list)
    for node_id, depth in node_depth.items():
        depth_to_nodes[depth].append(node_id)

    for depth in sorted(depth_to_nodes.keys()):
        levels.append(sorted(depth_to_nodes[depth]))

    if not levels:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><text x="10" y="50">Empty GSN Tree</text></svg>'

    # Layout constants
    box_width = 200
    box_height = 80
    dx = 240
    dy = 160

    max_cols = max(len(level) for level in levels)
    width = max(800, max_cols * dx + 100)
    height = len(levels) * dy + 100

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '  <defs>',
        '    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#444" />',
        '    </marker>',
        '  </defs>',
        '  <rect width="100%" height="100%" fill="#fafafa" />',
    ]

    # Calculate coordinates for each node
    node_coords: dict[str, tuple[float, float]] = {}
    for L_idx, level_nodes in enumerate(levels):
        y = L_idx * dy + 50
        num_nodes = len(level_nodes)
        start_x = (width - num_nodes * dx) / 2 + (dx - box_width) / 2
        for i, node_id in enumerate(level_nodes):
            x = start_x + i * dx
            node_coords[node_id] = (x, y)

    # Draw connection lines first (so boxes render over them)
    for node_id, coords in node_coords.items():
        parent_x, parent_y = coords
        children = nodes.get(node_id, {}).get("children", [])
        for child_id in children:
            if child_id in node_coords:
                child_x, child_y = node_coords[child_id]
                x1 = parent_x + box_width / 2
                y1 = parent_y + box_height
                x2 = child_x + box_width / 2
                y2 = child_y
                svg_lines.append(
                    f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="#555" stroke-width="2" marker-end="url(#arrow)" />'
                )

    # Draw GSN node boxes
    for node_id, coords in node_coords.items():
        x, y = coords
        node = nodes[node_id]
        ntype = node["type"]
        text = node["text"]

        # Color palettes matching Markdown styling
        if ntype == "goal":
            stroke_color = "#28a745"
            fill_color = "#d4edda"
            box_shape = f'<rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" rx="0" ry="0" fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" />'
        elif ntype == "strategy":
            stroke_color = "#ffc107"
            fill_color = "#fff3cd"
            # Draw parallelogram
            points = f"{x+15},{y} {x+box_width},{y} {x+box_width-15},{y+box_height} {x},{y+box_height}"
            box_shape = f'<polygon points="{points}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" />'
        elif ntype == "assumption":
            stroke_color = "#007bff"
            fill_color = "#cce5ff"
            # Draw ellipse/oval
            cx = x + box_width / 2
            cy = y + box_height / 2
            rx = box_width / 2
            ry = box_height / 2
            box_shape = f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" />'
        elif ntype == "context":
            stroke_color = "#6c757d"
            fill_color = "#e2e3e5"
            box_shape = f'<rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" rx="15" ry="15" fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" />'
        else: # solution
            stroke_color = "#dc3545"
            fill_color = "#f8d7da"
            box_shape = f'<rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" rx="0" ry="0" fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" />'

        svg_lines.append(f"  <!-- Node {node_id} -->")
        svg_lines.append(f"  {box_shape}")

        # Add Node ID header text
        header_y = y + 16
        svg_lines.append(
            f'  <text x="{x + box_width/2:.1f}" y="{header_y:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#333">'
            f'{ntype.upper()} ({node_id})</text>'
        )

        # Wrap text and render lines
        wrapped_lines = _wrap_text(text, max_chars=22)
        start_text_y = y + 32
        for line_idx, line in enumerate(wrapped_lines[:3]): # max 3 lines to fit box
            line_y = start_text_y + line_idx * 14
            escaped_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            svg_lines.append(
                f'  <text x="{x + box_width/2:.1f}" y="{line_y:.1f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="10" fill="#333">{escaped_line}</text>'
            )

    svg_lines.append("</svg>\n")
    return "\n".join(svg_lines)


def write_assurance_fragment(
    reports_dir: Path,
    payload: dict[str, Any],
    repo_root: Path,
) -> dict[str, Path]:
    """Write the assurance fragment GSN JSON, Markdown, and SVG files.

    Returns:
        Dict mapping from file suffix/format to written Path.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    json_path = reports_dir / "assurance_fragment.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written["json"] = json_path

    md_path = reports_dir / "assurance_fragment.md"
    md_content = render_assurance_fragment_to_markdown(payload)
    md_path.write_text(md_content, encoding="utf-8")
    written["markdown"] = md_path

    svg_path = reports_dir / "assurance_fragment.svg"
    svg_content = render_assurance_fragment_to_svg(payload)
    svg_path.write_text(svg_content, encoding="utf-8")
    written["svg"] = svg_path

    return written
