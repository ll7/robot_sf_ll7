"""Tests for run-local ORCA-residual candidate materialization."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.tools.materialize_orca_residual_candidate import materialize_candidate

REPO_ROOT = Path(__file__).parents[2]


def test_materialize_candidate_pins_checkpoint_path(tmp_path: Path) -> None:
    """Materialization should avoid mutating the checked-in candidate registry."""
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_text("checkpoint", encoding="utf-8")

    manifest = materialize_candidate(
        policy_model_id="issue_1428_orca_residual_bc_policy_smoke",
        policy_model_path=checkpoint,
        output_dir=tmp_path / "runtime",
        registry_path=REPO_ROOT / "docs/context/policy_search/candidate_registry.yaml",
        candidate_name="orca_residual_guarded_ppo_v0",
    )

    runtime_registry = Path(manifest["runtime_registry"])
    runtime_candidate = Path(manifest["runtime_candidate_config"])
    registry_payload = yaml.safe_load(runtime_registry.read_text(encoding="utf-8"))
    candidate_payload = yaml.safe_load(runtime_candidate.read_text(encoding="utf-8"))

    entry = registry_payload["candidates"]["orca_residual_guarded_ppo_v0"]
    assert entry["candidate_config_path"] == str(runtime_candidate)
    assert entry["training_required"] is False
    assert candidate_payload["params"]["model_id"] is None
    assert candidate_payload["params"]["model_path"] == str(checkpoint.resolve())
