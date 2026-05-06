"""Tests for policy-search promotion gate decisions."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import yaml

from scripts.tools import promote_policy_search_candidate

if TYPE_CHECKING:
    from pathlib import Path


def test_stress_slice_summary_cannot_promote_without_stage_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Promotion decisions must fail closed for non-passing local stages."""
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "candidate": "candidate_a",
                "stage": "stress_slice",
                "decision": "tracked",
                "summary": {
                    "success_rate": 0.40,
                    "collision_rate": 0.0,
                    "scenario_family": {
                        "classic": {"collision_rate": 0.0},
                        "francis2023": {"collision_rate": 0.0},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "candidates": {
                    "candidate_a": {
                        "promotion_gate": "tier_b",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    gates = tmp_path / "gates.yaml"
    gates.write_text(
        yaml.safe_dump(
            {
                "gates": {
                    "tier_b": {
                        "min_success_rate": 0.264,
                        "max_collision_rate": 0.055,
                    },
                    "scenario_stratified": {
                        "classic_collision_rate_max": 0.07,
                        "francis_collision_rate_max": 0.05,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "promotion"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_policy_search_candidate.py",
            str(summary_json),
            "--candidate-registry",
            str(registry),
            "--promotion-gates",
            str(gates),
            "--output",
            str(output_dir),
        ],
    )

    assert promote_policy_search_candidate.main() == 0

    decision = json.loads((output_dir / "candidate_a_stress_slice_promotion.json").read_text())
    assert decision["decision"] == "revise"
    assert decision["checks"]["stage_decision_passed"] is False


def test_unknown_promotion_gate_fails_closed(tmp_path: Path, monkeypatch) -> None:
    """Candidates must not promote when their configured gate is missing."""
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "candidate": "candidate_a",
                "stage": "stress_slice",
                "decision": "pass",
                "summary": {
                    "success_rate": 1.0,
                    "collision_rate": 0.0,
                    "scenario_family": {
                        "classic": {"collision_rate": 0.0},
                        "francis2023": {"collision_rate": 0.0},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "candidates": {
                    "candidate_a": {
                        "promotion_gate": "missing_gate",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    gates = tmp_path / "gates.yaml"
    gates.write_text(
        yaml.safe_dump(
            {
                "gates": {
                    "tier_b": {
                        "min_success_rate": 0.264,
                        "max_collision_rate": 0.055,
                    },
                    "scenario_stratified": {
                        "classic_collision_rate_max": 0.07,
                        "francis_collision_rate_max": 0.05,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "promotion"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_policy_search_candidate.py",
            str(summary_json),
            "--candidate-registry",
            str(registry),
            "--promotion-gates",
            str(gates),
            "--output",
            str(output_dir),
        ],
    )

    assert promote_policy_search_candidate.main() == 0

    decision = json.loads((output_dir / "candidate_a_stress_slice_promotion.json").read_text())
    assert decision["decision"] == "revise"
    assert decision["checks"]["gate_configured"] is False


def test_unregistered_candidate_fails_closed(tmp_path: Path, monkeypatch) -> None:
    """Candidates must be present in the registry before they can promote."""
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "candidate": "candidate_a",
                "stage": "stress_slice",
                "decision": "pass",
                "summary": {
                    "success_rate": 1.0,
                    "collision_rate": 0.0,
                    "scenario_family": {
                        "classic": {"collision_rate": 0.0},
                        "francis2023": {"collision_rate": 0.0},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({"candidates": {}}), encoding="utf-8")
    gates = tmp_path / "gates.yaml"
    gates.write_text(
        yaml.safe_dump(
            {
                "gates": {
                    "tier_b": {
                        "min_success_rate": 0.264,
                        "max_collision_rate": 0.055,
                    },
                    "scenario_stratified": {
                        "classic_collision_rate_max": 0.07,
                        "francis_collision_rate_max": 0.05,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "promotion"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_policy_search_candidate.py",
            str(summary_json),
            "--candidate-registry",
            str(registry),
            "--promotion-gates",
            str(gates),
            "--output",
            str(output_dir),
        ],
    )

    assert promote_policy_search_candidate.main() == 0

    decision = json.loads((output_dir / "candidate_a_stress_slice_promotion.json").read_text())
    assert decision["decision"] == "revise"
    assert decision["checks"]["candidate_registered"] is False


def test_full_matrix_h500_is_promotion_scale_stage(tmp_path: Path, monkeypatch) -> None:
    """H500 promotion summaries should not require a runner decision of pass."""
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "candidate": "candidate_a",
                "stage": "full_matrix_h500",
                "decision": "tracked",
                "summary": {
                    "success_rate": 0.90,
                    "collision_rate": 0.02,
                    "scenario_family": {
                        "classic": {"collision_rate": 0.02},
                        "francis2023": {"collision_rate": 0.02},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump({"candidates": {"candidate_a": {"promotion_gate": "tier_b"}}}),
        encoding="utf-8",
    )
    gates = tmp_path / "gates.yaml"
    gates.write_text(
        yaml.safe_dump(
            {
                "gates": {
                    "tier_b": {
                        "min_success_rate": 0.264,
                        "max_collision_rate": 0.055,
                    },
                    "scenario_stratified": {
                        "classic_collision_rate_max": 0.07,
                        "francis_collision_rate_max": 0.05,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "promotion"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_policy_search_candidate.py",
            str(summary_json),
            "--candidate-registry",
            str(registry),
            "--promotion-gates",
            str(gates),
            "--output",
            str(output_dir),
        ],
    )

    assert promote_policy_search_candidate.main() == 0

    decision = json.loads((output_dir / "candidate_a_full_matrix_h500_promotion.json").read_text())
    assert decision["decision"] == "promote"
    assert decision["checks"]["stage_decision_passed"] is True


def test_gate_name_override_can_apply_strict_h500_gate(tmp_path: Path, monkeypatch) -> None:
    """A stricter explicit gate should override the registry default for diagnostics."""
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "candidate": "candidate_a",
                "stage": "full_matrix_h500",
                "decision": "tracked",
                "summary": {
                    "success_rate": 0.90,
                    "collision_rate": 0.0208,
                    "scenario_family": {
                        "classic": {"collision_rate": 0.02},
                        "francis2023": {"collision_rate": 0.02},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump({"candidates": {"candidate_a": {"promotion_gate": "tier_b"}}}),
        encoding="utf-8",
    )
    gates = tmp_path / "gates.yaml"
    gates.write_text(
        yaml.safe_dump(
            {
                "gates": {
                    "tier_b": {
                        "min_success_rate": 0.264,
                        "max_collision_rate": 0.055,
                    },
                    "nominal_sanity": {
                        "min_success_rate": 0.80,
                        "max_collision_rate": 0.0200,
                    },
                    "scenario_stratified": {
                        "classic_collision_rate_max": 0.07,
                        "francis_collision_rate_max": 0.05,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "promotion"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_policy_search_candidate.py",
            str(summary_json),
            "--candidate-registry",
            str(registry),
            "--promotion-gates",
            str(gates),
            "--gate-name",
            "nominal_sanity",
            "--output",
            str(output_dir),
        ],
    )

    assert promote_policy_search_candidate.main() == 0

    decision = json.loads((output_dir / "candidate_a_full_matrix_h500_promotion.json").read_text())
    assert decision["gate"] == "nominal_sanity"
    assert decision["decision"] == "revise"
    assert decision["checks"]["collision_rate"] is False
