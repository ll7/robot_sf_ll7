"""Tests for the issue #4010 diffusion-policy planner slice."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robot_sf.planner.diffusion_policy import (  # noqa: E402
    CLAIM_BOUNDARY,
    DiffusionGuidanceSelector,
    DiffusionPolicyAdapter,
    RobotPedestrianGraphEncoder,
    build_diffusion_policy_config,
)


def _observation() -> dict[str, object]:
    return {
        "dt": 0.1,
        "robot": {
            "position": [0.0, 0.0],
            "velocity": [0.1, 0.0],
            "goal": [2.0, 0.0],
            "heading": 0.0,
            "radius": 0.3,
        },
        "agents": [
            {"position": [0.8, 0.2], "velocity": [-0.1, 0.0], "radius": 0.25},
            {"position": [1.2, -0.3], "velocity": [0.0, 0.1], "radius": 0.25},
        ],
        "obstacles": [],
    }


def test_config_builder_rejects_invalid_contract_values() -> None:
    """The first implementation slice accepts only one-step ``(v, omega)`` actions."""
    with pytest.raises(ValueError, match="denoising_steps"):
        build_diffusion_policy_config({"allow_untrained_smoke": True, "denoising_steps": 0})
    with pytest.raises(ValueError, match="action_horizon"):
        build_diffusion_policy_config({"allow_untrained_smoke": True, "action_horizon": 3})
    with pytest.raises(ValueError, match="Unsupported"):
        build_diffusion_policy_config({"allow_untrained_smoke": True, "unknown": 1})


def test_missing_checkpoint_fails_closed_without_smoke_flag() -> None:
    """Untrained inference must be explicit diagnostic smoke, not silent success evidence."""
    with pytest.raises(RuntimeError, match="allow_untrained_smoke=true"):
        DiffusionPolicyAdapter({})
    with pytest.raises(FileNotFoundError):
        DiffusionPolicyAdapter({"checkpoint_path": "/tmp/missing-issue-4010.pt"})


def test_checkpoint_path_must_be_file(tmp_path) -> None:
    """Directory-valued checkpoint paths fail closed."""
    with pytest.raises(FileNotFoundError):
        DiffusionPolicyAdapter({"checkpoint_path": str(tmp_path)})


def test_encoder_returns_finite_masked_tensors() -> None:
    """The graph encoder pads visible pedestrians and marks valid nodes."""
    adapter = DiffusionPolicyAdapter(
        {"allow_untrained_smoke": True, "seed": 1, "max_pedestrians": 3}
    )
    node_features, mask = adapter.encoder.encode_observation(_observation())
    assert node_features.shape == (4, 8)
    assert mask.tolist() == [True, True, True, False]
    assert torch.isfinite(node_features).all()


def test_encoder_forward_supports_unbatched_and_batched_inputs() -> None:
    """The graph encoder supports individual and batched observations."""
    encoder = RobotPedestrianGraphEncoder(max_pedestrians=3)
    unbatched_features = torch.randn(4, 8)
    unbatched_mask = torch.tensor([True, True, True, False])

    unbatched = encoder(unbatched_features, unbatched_mask)
    batched = encoder(unbatched_features.unsqueeze(0), unbatched_mask.unsqueeze(0))

    assert unbatched.shape == (encoder.output_dim,)
    assert batched.shape == (1, encoder.output_dim)


def test_non_finite_observation_values_fall_back_to_defaults() -> None:
    """Non-finite observation fields cannot propagate NaN or Inf into encoder features."""
    adapter = DiffusionPolicyAdapter(
        {"allow_untrained_smoke": True, "seed": 1, "max_pedestrians": 2}
    )
    observation = _observation()
    observation["robot"] = {
        "position": [float("nan"), 0.0],
        "velocity": [0.0, float("inf")],
        "goal": [2.0, 0.0],
        "heading": float("nan"),
        "radius": float("inf"),
    }
    observation["agents"] = [
        {"position": [1.0, float("inf")], "velocity": [float("nan"), 0.0], "radius": 0.25}
    ]

    node_features, _mask = adapter.encoder.encode_observation(observation)

    assert torch.isfinite(node_features).all()


def test_sampler_returns_finite_bounded_action_and_reproducible_reset() -> None:
    """Resetting the seed makes diagnostic sampling reproducible."""
    cfg = {
        "allow_untrained_smoke": True,
        "seed": 4010,
        "max_linear_speed": 0.8,
        "max_angular_speed": 0.7,
        "num_action_samples": 5,
    }
    adapter = DiffusionPolicyAdapter(cfg)
    first = adapter.plan(_observation())
    adapter.reset(seed=4010)
    second = adapter.plan(_observation())
    assert first == pytest.approx(second)
    assert 0.0 <= first[0] <= 0.8
    assert -0.7 <= first[1] <= 0.7


def test_guidance_changes_selected_candidate() -> None:
    """The candidate guidance hook is behaviorally active."""
    candidates = np.array([[0.2, 0.0], [0.8, 0.6]], dtype=float)
    selector = DiffusionGuidanceSelector({"enabled": True, "goal_progress_weight": 2.0})
    selected, diagnostics = selector.select(candidates, _observation(), (0.2, 0.0))
    assert selected == pytest.approx((0.8, 0.6))
    assert diagnostics["selected_index"] == 1

    disabled = DiffusionGuidanceSelector({"enabled": False})
    selected_disabled, disabled_diagnostics = disabled.select(
        candidates, _observation(), (0.2, 0.0)
    )
    assert selected_disabled == pytest.approx((0.2, 0.0))
    assert disabled_diagnostics["status"] == "disabled"


def test_diagnostics_exposes_diagnostic_claim_boundary() -> None:
    """Diagnostics must not look like benchmark or paper evidence."""
    adapter = DiffusionPolicyAdapter({"allow_untrained_smoke": True, "deterministic": True})
    adapter.plan(_observation())
    payload = adapter.diagnostics()["diffusion_policy"]
    assert payload["evidence_tier"] == "diagnostic-only"
    assert payload["allow_untrained_smoke"] is True
    assert payload["claim_boundary"] == CLAIM_BOUNDARY
    assert "raw_samples" not in payload
