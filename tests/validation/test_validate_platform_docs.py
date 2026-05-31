"""Tests for the platform documentation validator."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from scripts.validation.validate_platform_docs import (
    validate_benchmark_suites,
    validate_leaderboards,
    validate_planner_zoo,
    validate_platform_docs,
    validate_policy_cards,
)


def _write(path: Path, text: str) -> None:
    """Write a dedented text fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")


def _write_registry_fixtures(root: Path) -> None:
    """Write compact registry fixtures used by multiple platform-doc tests."""
    _write(
        root / "docs/context/policy_search/candidate_registry.yaml",
        """
        version: 1
        candidates:
          demo_policy:
            status: implemented
            family: learned_policy_network
            candidate_config_path: configs/policy_search/candidates/demo_policy.yaml
            required_stages: [smoke]
        """,
    )
    _write(root / "configs/policy_search/candidates/demo_policy.yaml", "name: demo_policy\n")
    _write(
        root / "docs/context/policy_search/learned_policy_registry.md",
        """
        # Registry

        | `policy_id` | `policy_family` | `integration_status` | `reproducibility_status` | `benchmark_status` | Boundary |
        | --- | --- | --- | --- | --- | --- |
        | `demo_policy` | `learned_baseline` | `implemented` | `comparison_available` | `comparison_available` | Demo only. |
        """,
    )
    _write(
        root / "model/registry.yaml",
        """
        version: 1
        models:
          - model_id: demo_model
        """,
    )


def test_checked_in_platform_docs_validate() -> None:
    """The tracked platform docs should satisfy the public metadata contract."""
    repo_root = Path(__file__).parents[2]

    issues = validate_platform_docs(repo_root)

    assert issues == []


def test_policy_card_reports_unknown_policy_id(tmp_path: Path) -> None:
    """Policy cards should not introduce IDs absent from the registries."""
    _write_registry_fixtures(tmp_path)
    _write(
        tmp_path / "docs/policy_cards/missing.md",
        """
        # Policy Card

        ```yaml
        policy_id: missing_policy
        policy_family: learned_baseline
        card_status: current
        registry_status:
          integration_status: implemented
          reproducibility_status: comparison_available
          benchmark_status: comparison_available
        benchmark_track: demo_track
        evidence_boundary: Demo.
        not_for:
          - promotion
        ```
        """,
    )

    issues = validate_policy_cards(tmp_path)

    assert any(issue.path.endswith(".summary.policy_id") for issue in issues)


def test_planner_zoo_command_parsing_ignores_placeholders_and_split_prose(tmp_path: Path) -> None:
    """Planner-zoo command parsing should validate concrete command units only."""
    _write_registry_fixtures(tmp_path)
    _write(
        tmp_path / "docs/planner_zoo/index.md",
        """
        # Planner Zoo

        Generic example:

        ```bash
        uv run python scripts/validation/run_policy_search_candidate.py --candidate <candidate_id> --stage smoke
        ```

        Prose can mention `--candidate demo_policy` and a separate `--stage nominal_sanity`
        without creating a fake runnable command.

        | Candidate | Family | Config | Local smoke command | Notes |
        | --- | --- | --- | --- | --- |
        | `demo_policy` | Demo | [`configs/policy_search/candidates/demo_policy.yaml`](../../configs/policy_search/candidates/demo_policy.yaml) | `uv run python scripts/validation/run_policy_search_candidate.py --candidate demo_policy --stage smoke` | Smoke only. |
        """,
    )
    _write(
        tmp_path / "scripts/validation/run_policy_search_candidate.py",
        "#!/usr/bin/env python\n",
    )

    issues = validate_planner_zoo(tmp_path)

    assert issues == []


def test_policy_card_model_id_must_match_model_registry(tmp_path: Path) -> None:
    """Policy cards should keep model IDs aligned with model/registry.yaml."""
    _write_registry_fixtures(tmp_path)
    _write(
        tmp_path / "docs/policy_cards/demo.md",
        """
        # Policy Card

        ```yaml
        policy_id: demo_policy
        policy_family: learned_baseline
        card_status: current
        registry_status:
          integration_status: implemented
          reproducibility_status: comparison_available
          benchmark_status: comparison_available
        benchmark_track: demo_track
        evidence_boundary: Demo.
        not_for:
          - promotion
        ```

        | Field | Value |
        | --- | --- |
        | Model id | `missing_model` |
        """,
    )

    issues = validate_policy_cards(tmp_path)

    assert any("unknown model registry ID" in issue.message for issue in issues)


def test_leaderboard_rejects_output_evidence_uri(tmp_path: Path) -> None:
    """Leaderboard evidence URIs should be durable tracked paths, not output/."""
    _write(
        tmp_path / "docs/leaderboards/smoke.md",
        """
        # Smoke

        | planner | suite | success | collision | near_miss | low_progress | min_distance | runtime | benchmark_track | evidence_uri | status | claim_boundary |
        | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
        | `demo` | `smoke` | `1.0` | `0.0` | `0.0` | `0` | `not_recorded` | `1s` | `policy_search_smoke` | `output/demo.json` | `pass` | Smoke only. |
        """,
    )

    issues = validate_leaderboards(tmp_path)

    assert any("worktree-local artifact" in issue.message for issue in issues)


def test_leaderboard_requires_status_and_track_columns(tmp_path: Path) -> None:
    """Leaderboard tables should include status and benchmark-track fields."""
    _write(
        tmp_path / "docs/leaderboards/smoke.md",
        """
        # Smoke

        | planner | suite | success | collision | near_miss | low_progress | min_distance | runtime | evidence_uri | claim_boundary |
        | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
        | `demo` | `smoke` | `1.0` | `0.0` | `0.0` | `0` | `not_recorded` | `1s` | `docs/evidence.json` | Smoke only. |
        """,
    )

    issues = validate_leaderboards(tmp_path)

    assert any("required row contract" in issue.message for issue in issues)


def test_benchmark_suite_requires_schema_fields(tmp_path: Path) -> None:
    """Benchmark-suite pages should name benchmark track and claim boundary."""
    _write(
        tmp_path / "docs/benchmark_suites/demo.md",
        """
        # Demo Suite

        ```yaml
        suite_id: demo
        purpose: Demo.
        scenario_ids: [classic]
        seed_set: [0]
        eligible_planners: [orca]
        metrics: [success]
        expected_runtime: 1m
        canonical_command: uv run python scripts/missing.py
        ```
        """,
    )

    issues = validate_benchmark_suites(tmp_path)
    paths = {issue.path for issue in issues}

    assert "docs/benchmark_suites/demo.md.summary.benchmark_track" in paths
    assert "docs/benchmark_suites/demo.md.summary.claim_boundary" in paths
    assert any("command path does not exist" in issue.message for issue in issues)
