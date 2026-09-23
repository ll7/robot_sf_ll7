"""Contract tests for the issue #9609 actor/observation diagnostic."""

# evidence-writer-exempt: writer-failure tests create synthetic prior artifact/sidecar bytes only
# under pytest tmp_path so rollback behavior can be asserted; they never write repository evidence.

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.analysis import narrow_doorway_actor_response_issue_9609 as diagnostic
from scripts.analysis.narrow_doorway_actor_response_issue_9609 import (
    OFFSETS,
    SEEDS,
    _nearest_forward_obstacle,
    _without_static_geometry,
    classify,
)


def _model_obs() -> dict[str, np.ndarray]:
    grid = np.zeros((3, 10, 10), dtype=np.float32)
    grid[0, 5, 7] = 1.0
    grid[1, 4, 6] = 1.0
    grid[2] = np.maximum(grid[0], grid[1])
    return {
        "occupancy_grid": grid,
        "occupancy_grid_meta_channel_indices": np.asarray([0, 1, -1, 2], dtype=np.int32),
        "occupancy_grid_meta_origin": np.asarray([-1.0, -1.0], dtype=np.float32),
        "occupancy_grid_meta_resolution": np.asarray([0.2], dtype=np.float32),
        "goal_current": np.asarray([2.0, 2.0], dtype=np.float32),
    }


def test_static_geometry_ablation_preserves_non_geometry_fields() -> None:
    source = _model_obs()
    ablated = _without_static_geometry(source)

    assert ablated["occupancy_grid"][0].sum() == 0.0
    np.testing.assert_array_equal(ablated["occupancy_grid"][1], source["occupancy_grid"][1])
    np.testing.assert_array_equal(ablated["occupancy_grid"][2], source["occupancy_grid"][1])
    np.testing.assert_array_equal(ablated["goal_current"], source["goal_current"])
    assert source["occupancy_grid"][0].sum() == 1.0


def test_nearest_forward_obstacle_uses_ego_grid_metadata() -> None:
    distance, count = _nearest_forward_obstacle(_model_obs())

    assert count == 1
    np.testing.assert_allclose(distance, np.hypot(0.5, 0.1), atol=1e-6)


def _supported_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for offset in OFFSETS:
            rows.append(
                {
                    "seed": seed,
                    "contact_step": 60,
                    "offset_before_contact": offset,
                    "static_geometry_present": True,
                    "nearest_forward_obstacle_m": float(offset) / 5.0,
                    "canonical_model_predict_v": 2.0,
                    "canonical_critic_value": -20.0 + float(offset) / 2.0,
                    "adapter_command_v": 2.0,
                    "executed_linear_speed_after": 2.0,
                    "adapter_forward_intent_preserved": True,
                    "actor_mean_v_delta_geometry_zero_minus_canonical": -0.5,
                    "critic_delta_geometry_zero_minus_canonical": 20.0,
                    "fallback_or_degraded": False,
                }
            )
    return rows


def test_classify_supports_only_complete_matched_signal() -> None:
    summary = classify(_supported_rows())

    assert summary["row_count"] == 12
    assert summary["result_classification"] == "actor_value_response_mismatch_supported"
    assert summary["checks"]["fallback_or_degraded_rows"] == 0


def test_classify_fails_closed_when_geometry_is_missing() -> None:
    rows = _supported_rows()
    rows[0]["static_geometry_present"] = False

    assert classify(rows)["result_classification"] == "not_identifiable"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("actor_mean_v_delta_geometry_zero_minus_canonical", None),
        ("critic_delta_geometry_zero_minus_canonical", float("nan")),
        ("actor_mean_v_delta_geometry_zero_minus_canonical", 0.5),
        ("critic_delta_geometry_zero_minus_canonical", -0.5),
    ],
)
def test_classify_fails_closed_on_missing_nonfinite_or_wrong_direction_ablation(
    field: str, replacement: float | None
) -> None:
    rows = _supported_rows()
    if replacement is None:
        del rows[0][field]
    else:
        rows[0][field] = replacement

    summary = classify(rows)

    assert summary["result_classification"] == "not_identifiable"


def test_parent_binding_write_failure_preserves_published_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The inherited builder's side-effect write must stay in private staging."""
    output_dir = tmp_path / "evidence"
    output_dir.mkdir()
    binding_path = output_dir / "binding.json"
    sidecar_path = output_dir / "binding.json.review.json"
    previous_binding = b'{"review_marker":"previous binding"}\n'
    previous_sidecar = b'{"artifact_sha256":"previous digest"}\n'
    binding_path.write_bytes(previous_binding)
    sidecar_path.write_bytes(previous_sidecar)

    monkeypatch.setattr(diagnostic, "SEEDS", (225,))
    monkeypatch.setattr(
        diagnostic,
        "_trace_seed",
        lambda _seed: ([{} for _ in diagnostic.OFFSETS], 21),
    )
    builder_paths: list[Path] = []

    def fail_after_inherited_write(output_path: Path, _gamma: float) -> dict[str, object]:
        output_path = Path(output_path)
        builder_paths.append(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        diagnostic.write_json(output_path / "binding.json", {"intermediate": True})
        raise RuntimeError("injected failure after inherited binding write")

    monkeypatch.setattr(diagnostic.parent, "build_binding", fail_after_inherited_write)

    with pytest.raises(RuntimeError, match="injected failure"):
        diagnostic.run(output_dir)

    assert builder_paths
    assert builder_paths[0].resolve() != output_dir.resolve()
    assert binding_path.read_bytes() == previous_binding
    assert sidecar_path.read_bytes() == previous_sidecar


def test_json_sidecar_publish_failure_rolls_back_previous_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "evidence"
    output_dir.mkdir()
    binding_path = output_dir / "binding.json"
    sidecar_path = output_dir / "binding.json.review.json"
    previous_binding = b'{"review_marker":"previous binding"}\n'
    previous_sidecar = b'{"artifact_sha256":"previous digest"}\n'
    binding_path.write_bytes(previous_binding)
    sidecar_path.write_bytes(previous_sidecar)
    original_replace = diagnostic.os.replace
    sidecar_replace_failed = False

    def fail_sidecar_replace_once(source: Path, target: Path) -> None:
        nonlocal sidecar_replace_failed
        if Path(target) == sidecar_path and not sidecar_replace_failed:
            sidecar_replace_failed = True
            raise OSError("injected sidecar publication failure")
        original_replace(source, target)

    monkeypatch.setattr(diagnostic.os, "replace", fail_sidecar_replace_once)

    with pytest.raises(OSError, match="injected sidecar publication failure"):
        diagnostic._write_json_with_review_sidecar(binding_path, {"new": "binding"})

    assert sidecar_replace_failed
    assert binding_path.read_bytes() == previous_binding
    assert sidecar_path.read_bytes() == previous_sidecar


def test_cli_runs_after_rollout_foresight_load_and_writes_evidence(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise main -> run -> trace so a fresh-planner pre-load assertion cannot recur."""

    def observation(step: int) -> dict[str, np.ndarray]:
        obs = _model_obs()
        obs["diagnostic_step"] = np.asarray([step], dtype=np.float32)
        return obs

    class FakeEnv:
        def __init__(self) -> None:
            self.step_index = 0
            self.env_config = SimpleNamespace()
            self.simulator = SimpleNamespace(
                robots=[SimpleNamespace(current_speed=np.asarray([2.0, 0.0], dtype=float))]
            )

        def reset(self, *, seed: int):
            self.step_index = 0
            return observation(0), {"seed": seed}

        def step(self, _action: np.ndarray):
            self.step_index += 1
            collision = self.step_index >= 21
            return (
                observation(self.step_index),
                0.0,
                collision,
                False,
                {"meta": {"is_obstacle_collision": collision}},
            )

        def close(self) -> None:
            return None

    class FakePlanner:
        def __init__(self) -> None:
            self.loaded = False
            self.closed = False

        def _build_model_obs_dict(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            self.loaded = True
            return obs

        def foresight_diagnostics(self) -> dict[str, object]:
            return {
                "foresight_prediction": {
                    "load_status": "loaded" if self.loaded else "not_attempted",
                    "effective_prediction_mode": (
                        "predictive_foresight" if self.loaded else "unavailable"
                    ),
                    "fallback_used": False,
                }
            }

        def _action_vec_to_dict_from_array(self, action: np.ndarray) -> dict[str, float]:
            return {"v": float(action[0]), "omega": float(action[1])}

        def close(self) -> None:
            self.closed = True

    planners: list[FakePlanner] = []

    def make_planner() -> FakePlanner:
        planner = FakePlanner()
        planners.append(planner)
        return planner

    def policy_outputs(
        _planner: FakePlanner, model_obs: dict[str, np.ndarray]
    ) -> dict[str, object]:
        step = float(model_obs["diagnostic_step"][0])
        geometry_present = bool(np.asarray(model_obs["occupancy_grid"])[0].sum())
        return {
            "actor_mean": np.asarray([2.5 if geometry_present else 1.5, 0.0]),
            "model_predict": np.asarray([2.0 if geometry_present else 1.0, 0.0]),
            "critic_value": (-10.0 - step) if geometry_present else (10.0 - step),
        }

    monkeypatch.setattr(diagnostic, "SEEDS", (225,))
    monkeypatch.setattr(
        diagnostic.parent,
        "_build_diagnostic_env",
        lambda seed: (FakeEnv(), SimpleNamespace(seed=seed), SimpleNamespace()),
    )
    monkeypatch.setattr(diagnostic.parent, "_make_planner", make_planner)
    monkeypatch.setattr(diagnostic.parent, "_normalize_runner_obs", lambda obs: obs)
    monkeypatch.setattr(
        diagnostic.parent, "_min_obstacle_clearance", lambda env: 4.0 - env.step_index / 10.0
    )
    monkeypatch.setattr(
        diagnostic.parent,
        "build_binding",
        lambda _output_dir, _gamma: {"generated_at_utc": "ignored", "git_head": "ignored"},
    )
    monkeypatch.setattr(diagnostic, "_policy_outputs", policy_outputs)
    monkeypatch.setattr(
        diagnostic,
        "policy_command_to_env_action",
        lambda **_kwargs: np.asarray([0.0, 0.0]),
    )
    monkeypatch.setattr(
        sys, "argv", ["narrow_doorway_actor_response_issue_9609.py", "--output-dir", str(tmp_path)]
    )

    assert diagnostic.main() == 0
    assert planners[0].loaded is True
    assert all(planner.closed for planner in planners)
    summary = json.loads((tmp_path / "mechanism_summary.json").read_text(encoding="utf-8"))
    assert summary["result_classification"] == "actor_value_response_mismatch_supported"
    assert summary["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    csv_path = tmp_path / "matched_state_rows.csv"
    csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == "# AI-GENERATED NEEDS-REVIEW"
    assert csv_lines[1] == "# distance_convention: surface_clearance"
    assert "ground_truth_clearance_before_m" in csv_lines[2]
    binding = json.loads((tmp_path / "binding.json").read_text(encoding="utf-8"))
    assert binding["distance_conventions_by_column"] == {
        "ground_truth_clearance_before_m": {
            "distance_convention": "surface_clearance",
            "definition": (
                "Shortest robot-center to static-wall-segment distance minus the robot radius; "
                "positive values mean separation between the robot footprint and wall."
            ),
            "producer": "parent._min_obstacle_clearance",
        },
        "nearest_forward_obstacle_m": {
            "distance_convention": "center_center",
            "definition": (
                "Euclidean distance in the ego occupancy grid from its robot-center origin to "
                "the center of the nearest occupied static cell in the 3 m forward corridor."
            ),
            "producer": "_nearest_forward_obstacle",
        },
    }
    binding_path = tmp_path / "binding.json"
    binding_sidecar = json.loads(
        (tmp_path / "binding.json.review.json").read_text(encoding="utf-8")
    )
    assert binding_sidecar["artifact_path"] == binding_path.name
    assert (
        binding_sidecar["artifact_sha256"] == hashlib.sha256(binding_path.read_bytes()).hexdigest()
    )
    assert binding_sidecar["domain_approval_status"] == "not_granted"
    assert binding_sidecar["dissertation_admission_status"] == "not_admitted"
    sidecar_path = tmp_path / "matched_state_rows.csv.review.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == "evidence-review-marker.v1"
    assert sidecar["artifact_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert sidecar["claim_boundary"] == diagnostic.CLAIM_BOUNDARY

    first_generation = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    assert diagnostic.main() == 0
    second_generation = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    assert first_generation == second_generation


def test_tracked_summary_has_diagnostic_claim_boundary() -> None:
    evidence = (
        Path(__file__).resolve().parents[2]
        / "docs/context/evidence/issue_9609_narrow_doorway_actor_response/mechanism_summary.json"
    )
    if not evidence.exists():
        return
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["result_classification"] in {
        "actor_value_response_mismatch_supported",
        "not_identifiable",
    }
    assert payload["evidence_tier"] == "diagnostic-only"
    assert "not benchmark" in payload["claim_boundary"]
