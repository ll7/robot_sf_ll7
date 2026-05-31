"""Tests for the policy-search candidate registry validator."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from textwrap import dedent

from scripts.validation.validate_policy_search_registry import validate_registry


def _write(path: Path, text: str) -> None:
    """Write a dedented YAML fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")


def _write_support_files(root: Path) -> None:
    """Write common registry support files."""
    _write(
        root / "configs/policy_search/funnel.yaml",
        """
        stages:
          smoke: {}
          nominal_sanity: {}
        """,
    )
    _write(
        root / "configs/policy_search/promotion_gates.yaml",
        """
        gates:
          tier_b: {}
        """,
    )
    _write(root / "configs/policy_search/candidates/demo.yaml", "name: demo\n")
    _write(root / "docs/context/policy_search/SLURM/demo.md", "# Handoff\n")
    _write(
        root / "configs/training/demo_launch.yaml",
        """
        artifact_policy:
          checkpoints_in_git: false
        execution_boundary:
          full_training_in_this_issue: false
        """,
    )


def _valid_registry_text() -> str:
    """Return a minimal valid registry fixture."""
    return """
    version: 1
    updated_at: 2026-05-20
    freshness:
      max_age_days: 90
    funnel_config: configs/policy_search/funnel.yaml
    promotion_gates: configs/policy_search/promotion_gates.yaml
    candidates:
      demo_candidate:
        status: implemented
        family: hybrid_rule_based
        training_required: false
        promotion_gate: tier_b
        candidate_config_path: configs/policy_search/candidates/demo.yaml
        hypothesis: Demo candidate.
        required_stages: [smoke]
      learned_candidate:
        status: implemented
        family: learned_policy_network
        training_required: false
        promotion_gate: tier_b
        candidate_config_path: configs/policy_search/candidates/demo.yaml
        learned_policy_registry_id: learned_candidate
        hypothesis: Demo learned candidate.
        required_stages: [smoke]
      deferred_candidate:
        status: slurm_handoff_required
        family: learned_auxiliary_cost
        training_required: true
        promotion_gate: tier_b
        slurm_handoff: docs/context/policy_search/SLURM/demo.md
        launch_packet_config_path: configs/training/demo_launch.yaml
        learned_policy_registry_id: deferred_candidate
        hypothesis: Demo deferred candidate.
    """


def test_checked_in_policy_search_registry_validates() -> None:
    """The tracked policy-search registry should satisfy the metadata contract."""
    repo_root = Path(__file__).parents[2]

    issues = validate_registry(
        repo_root / "docs/context/policy_search/candidate_registry.yaml",
        as_of=date(2026, 5, 31),
    )

    assert issues == []


def test_valid_registry_fixture_passes(tmp_path: Path) -> None:
    """A representative implemented, learned, and SLURM registry can pass."""
    _write_support_files(tmp_path)
    registry = tmp_path / "candidate_registry.yaml"
    _write(registry, _valid_registry_text())

    issues = validate_registry(registry, as_of=date(2026, 5, 31))

    assert issues == []


def test_missing_implemented_fields_are_reported(tmp_path: Path) -> None:
    """Implemented candidates need config, gate, and stage routing fields."""
    _write_support_files(tmp_path)
    registry = tmp_path / "candidate_registry.yaml"
    _write(
        registry,
        """
        version: 1
        updated_at: 2026-05-20
        funnel_config: configs/policy_search/funnel.yaml
        promotion_gates: configs/policy_search/promotion_gates.yaml
        candidates:
          bad_candidate:
            status: implemented
            family: hybrid_rule_based
            training_required: false
            hypothesis: Missing routing fields.
        """,
    )

    issues = validate_registry(registry, as_of=date(2026, 5, 31))
    paths = {issue.path for issue in issues}

    assert "candidates.bad_candidate.candidate_config_path" in paths
    assert "candidates.bad_candidate.promotion_gate" in paths
    assert "candidates.bad_candidate.required_stages" in paths


def test_stale_updated_at_requires_explanation(tmp_path: Path) -> None:
    """Stale registries should be updated or explicitly explained."""
    _write_support_files(tmp_path)
    registry = tmp_path / "candidate_registry.yaml"
    _write(registry, _valid_registry_text().replace("2026-05-20", "2026-01-01"))

    issues = validate_registry(registry, as_of=date(2026, 5, 31), max_age_days=30)

    assert any(issue.path == "updated_at" and "days old" in issue.message for issue in issues)


def test_learned_candidate_requires_registry_or_adapter_link(tmp_path: Path) -> None:
    """Learned candidates need a registry id or adapter-contract link."""
    _write_support_files(tmp_path)
    registry = tmp_path / "candidate_registry.yaml"
    _write(
        registry,
        _valid_registry_text().replace(
            "        learned_policy_registry_id: learned_candidate\n",
            "",
        ),
    )

    issues = validate_registry(registry, as_of=date(2026, 5, 31))

    assert any(
        issue.path == "candidates.learned_candidate"
        and "learned_policy_registry_id" in issue.message
        for issue in issues
    )
