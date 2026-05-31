#!/usr/bin/env python3
"""Generate the repo-local skills README from the typed skill registry."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / ".agents" / "skills"
README = SKILLS_ROOT / "README.md"
REGISTRY = SKILLS_ROOT / "skills.yaml"

CATEGORY_TITLES = {
    "github-issue": "Issue Lifecycle",
    "github-pr": "PR Lifecycle",
    "validation": "Validation And Cleanup",
    "benchmark-evidence": "Benchmark And Experiment Evidence",
    "campaign-analysis": "Campaign Analysis",
    "context-docs": "Context And Docs",
    "research-iteration": "Research Iteration",
    "slurm": "SLURM And Campaign Submission",
    "domain-utility": "Domain-Specific Utilities",
    "general": "General",
}


def _load_registry() -> dict[str, Any]:
    """Return the parsed typed registry."""
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def _yes(value: bool) -> str:
    """Render booleans compactly for generated tables."""
    return "yes" if value else "no"


def _cell(value: object) -> str:
    """Render a Markdown table cell safely."""
    return str(value).replace("\n", " ").replace("|", "\\|")


def render_readme(registry: dict[str, Any]) -> str:
    """Render README text from registry metadata."""
    skills: dict[str, dict[str, Any]] = registry["skills"]
    by_category: dict[str, list[str]] = defaultdict(list)
    for name, metadata in skills.items():
        by_category[metadata.get("category", "general")].append(name)

    lines = [
        "# Robot SF Skills",
        "",
        "This directory contains repo-local skills for Coding Agents. Use this README as the",
        "generated routing index; read the specific `SKILL.md` before applying a skill.",
        "",
        "## Quick Routing",
        "",
        "| User intent | Primary skill | Secondary skill |",
        "| --- | --- | --- |",
        "| Not sure which skill applies | `skill-picker` | none |",
        "| Take the next eligible issue to PR | `goal-issue-implementation` | `gh-issue-autopilot` |",
        "| Execute one selected issue to ready PR | `gh-issue-autopilot` | `implementation-verification`, `gh-pr-opener` |",
        "| Clarify or repair issue contracts | `issue-contract-maintainer` | legacy aliases only when explicitly named |",
        "| Fix PR review comments | `gh-pr-comment-fixer` | `pr-ready-check` |",
        "| Open a ready PR | `gh-pr-opener` | `artifact-provenance` |",
        "| Verify branch claims | `implementation-verification` | `pr-ready-check` |",
        "| Run the standard readiness gate | `pr-ready-check` | none |",
        "| Review benchmark output | `analyze-camera-ready-benchmark` | `benchmark-row-status`, `artifact-provenance` |",
        "| Classify benchmark rows | `benchmark-row-status` | `review-benchmark-change` |",
        "| Submit a generic SLURM campaign | `slurm-campaign-submit` | `artifact-provenance` |",
        "| Submit issue-791 Auxme training | `auxme-issue791-submit` | `slurm-campaign-submit` |",
        "| Stage external data | `data-staging-provenance` | `artifact-provenance` |",
        "| Synthesize evidence across issues | `evidence-synthesis` | `paper-facing-docs` |",
        "",
        "## Negative Routing",
        "",
        "- Do not use `autoresearch` for ordinary cleanup.",
        "- Do not use `paper-facing-docs` for non-claim documentation.",
        "- Do not use `gh-issue-autopilot` for ambiguous issues; route to `issue-contract-maintainer` first.",
        "- Do not use `auxme-issue791-submit` for non-issue-791 campaigns.",
        "- Do not count fallback or degraded benchmark rows as success evidence; use `benchmark-row-status`.",
        "- Do not cite local `output/` contents as durable evidence; use `artifact-provenance`.",
        "",
        "## Canonical Skill Stacks",
        "",
        "| Stack | Skills |",
        "| --- | --- |",
        "| Issue queue to PR | `gh-issue-sequencer` -> `gh-issue-autopilot` -> `implementation-verification` -> `pr-ready-check` -> `gh-pr-opener` |",
        "| Issue contract repair | `issue-contract-maintainer` -> `gh-issue-sequencer` |",
        "| PR review cleanup | `gh-pr-comment-fixer` -> `implementation-verification` -> `pr-ready-check` |",
        "| Benchmark evidence audit | `benchmark-row-status` -> `artifact-provenance` -> `evidence-synthesis` |",
        "| SLURM campaign launch | `slurm-campaign-submit` -> `artifact-provenance` |",
        "| External data staging | `data-staging-provenance` -> `artifact-provenance` -> `context-note-maintainer` |",
        "",
        "## Validation Tiers",
        "",
        "| Tier | Use for | Required proof |",
        "| --- | --- | --- |",
        "| 0 | docs-only, metadata-only | render, link, path, or registry check |",
        "| 1 | local code or CLI behavior | targeted tests plus lint where relevant |",
        "| 2 | planner, metric, benchmark, artifact behavior | targeted tests plus benchmark preflight or sample run |",
        "| 3 | campaign or paper-facing evidence | full provenance, seeds, configs, artifacts, and interpretation note |",
        "",
        "## GitHub And Project Policy",
        "",
        "- `goal-issue-implementation` owns the multi-issue loop and stop condition.",
        "- `gh-issue-sequencer` owns Project #5 queue ordering, with current maintainer direction and fresh",
        "  evidence allowed to override score order.",
        "- `gh-issue-autopilot` owns one selected issue -> branch -> validation -> ready PR.",
        "- `gh-issue-creator` owns new issue creation.",
        "- `issue-contract-maintainer` owns ambiguity, template, and decision repair.",
        "- Use Project #5 `Priority Score` as an advisory queue-ordering signal; use",
        "  `gh-issue-priority-assessor` when score inputs need review.",
        "- Batch issue cleanup separately from Project #5 metadata writes; follow",
        "  `docs/context/issue_713_batch_first_issue_workflow.md`.",
        "",
        "## Maintenance And Validation",
        "",
        "- Edit `.agents/skills/skills.yaml` when adding aliases, categories, routing metadata,",
        "  delegated skills, write scopes, or output schemas.",
        "- Run `uv run python scripts/dev/generate_skills_readme.py` after registry changes.",
        "- Run `uv run python scripts/dev/check_skills.py` after adding, renaming, or removing skills.",
        "- Keep legacy compatibility wrappers only when they reduce routing breakage.",
        "- Keep group notes under `.agents/skills/groups/`; direct children of `.agents/skills/`",
        "  must be real skill directories unless explicitly whitelisted by the checker.",
        "",
        "## Generated Skill Index",
        "",
    ]

    for category in sorted(by_category, key=lambda key: CATEGORY_TITLES.get(key, key)):
        lines.extend(
            [
                f"### {CATEGORY_TITLES.get(category, category.title())}",
                "",
                "| Skill | Kind | Phase | Writes | SLURM | Artifacts | Delegates | Use When |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name in sorted(by_category[category]):
            metadata = skills[name]
            delegates = ", ".join(f"`{delegate}`" for delegate in metadata.get("delegates_to", []))
            delegates = delegates or "none"
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{name}`",
                        metadata.get("kind", ""),
                        metadata.get("phase", ""),
                        _yes(bool(metadata.get("requires_write"))),
                        _yes(bool(metadata.get("requires_slurm"))),
                        _yes(bool(metadata.get("requires_benchmark_artifacts"))),
                        delegates,
                        _cell(metadata.get("description", "")),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Aliases",
            "",
            "| Alias | Canonical skill |",
            "| --- | --- |",
        ]
    )
    aliases: list[tuple[str, str]] = []
    for name, metadata in skills.items():
        for alias in metadata.get("aliases", []):
            aliases.append((alias, name))
    for alias, name in sorted(aliases):
        lines.append(f"| `{alias}` | `{name}` |")
    if not aliases:
        lines.append("| none | none |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Prefer the most specific skill that matches the task.",
            "- Combine skills only when they cover different phases.",
            "- This README is generated from `.agents/skills/skills.yaml`; do not hand-edit the index.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Write the generated README, or check it for drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if README is not generated")
    args = parser.parse_args()

    generated = render_readme(_load_registry())
    if args.check:
        if README.read_text(encoding="utf-8") != generated:
            print("README is stale; run `uv run python scripts/dev/generate_skills_readme.py`")
            return 1
        print(f"{README.relative_to(REPO_ROOT)} is up to date")
        return 0

    README.write_text(generated, encoding="utf-8")
    print(f"Wrote {README.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
