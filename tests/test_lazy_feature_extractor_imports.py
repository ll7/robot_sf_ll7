"""Regression tests for optional ML dependency import boundaries."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    (
        "robot_sf.baselines.distributional_rl",
        "robot_sf.benchmark.predictive_planner_config",
        "robot_sf.feature_extractor",
        "robot_sf.feature_extractors",
        "robot_sf.feature_extractors.attention_extractor",
        "robot_sf.feature_extractors.config",
        "robot_sf.feature_extractors.grid_socnav_extractor",
        "robot_sf.feature_extractors.lightweight_cnn_extractor",
        "robot_sf.feature_extractors.lstm_extractor",
        "robot_sf.feature_extractors.mamba_extractor",
        "robot_sf.feature_extractors.mlp_extractor",
        "robot_sf.planner.crowdnav_height",
        "robot_sf.planner.crowdnav_pred_attng",
        "robot_sf.planner.learned_short_horizon_trainer",
        "robot_sf.planner.predictive_model",
        "robot_sf.planner.sonic_crowdnav",
        "robot_sf.tb_logging",
        "robot_sf.training.distributional_rl",
        "robot_sf.training.oracle_imitation_bc_smoke",
        "robot_sf.training.ppo_diagnostics",
        "robot_sf.training.ppo_policy",
        "robot_sf.training.risk_objectives",
        "robot_sf.training.threaded_vec_env",
    ),
)
def test_deferred_modules_do_not_import_ml_dependencies(module_name: str):
    """Importing every deferred module must leave optional ML frameworks unloaded."""
    code = f"""\
import importlib
import sys

importlib.import_module({module_name!r})

assert "torch" not in sys.modules
assert "stable_baselines3" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, check=False, text=True
    )

    assert result.returncode == 0, result.stderr


def test_lazy_legacy_export_preserves_pickle_compatibility():
    """The deferred legacy class remains a module-level pickle target."""
    code = """\
import pickle
import sys
import types

torch = types.ModuleType("torch")
torch.__path__ = []
torch_nn = types.ModuleType("torch.nn")
torch.nn = torch_nn

stable_baselines3 = types.ModuleType("stable_baselines3")
stable_baselines3.__path__ = []
common = types.ModuleType("stable_baselines3.common")
common.__path__ = []
torch_layers = types.ModuleType("stable_baselines3.common.torch_layers")
torch_layers.BaseFeaturesExtractor = type("BaseFeaturesExtractor", (), {})

sys.modules.update(
    {
        "torch": torch,
        "torch.nn": torch_nn,
        "stable_baselines3": stable_baselines3,
        "stable_baselines3.common": common,
        "stable_baselines3.common.torch_layers": torch_layers,
    }
)

from robot_sf.feature_extractor import DynamicsExtractor

assert DynamicsExtractor.__qualname__ == "DynamicsExtractor"
assert pickle.loads(pickle.dumps(DynamicsExtractor)) is DynamicsExtractor
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, check=False, text=True
    )

    assert result.returncode == 0, result.stderr


def test_lazy_factory_publishes_all_nested_pickle_targets():
    """One lazy export publishes every class it constructs for checkpoint compatibility."""
    code = """\
import pickle
import sys
import types

torch = types.ModuleType("torch")
torch.__path__ = []
torch_nn = types.ModuleType("torch.nn")
torch_nn.Module = type("Module", (), {})
torch.nn = torch_nn

sys.modules.update({"torch": torch, "torch.nn": torch_nn})

from robot_sf.planner.predictive_model import _MessageBlock, PredictiveTrajectoryModel

for exported in (_MessageBlock, PredictiveTrajectoryModel):
    assert ".<locals>." not in exported.__qualname__
    assert pickle.loads(pickle.dumps(exported)) is exported
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, check=False, text=True
    )

    assert result.returncode == 0, result.stderr


def test_lazy_callback_alias_keeps_canonical_pickle_name():
    """Keep the old callback spelling without changing the serialized class path."""
    code = """\\
import pickle
import sys
import types

stable_baselines3 = types.ModuleType("stable_baselines3")
stable_baselines3.__path__ = []
common = types.ModuleType("stable_baselines3.common")
common.__path__ = []
callbacks = types.ModuleType("stable_baselines3.common.callbacks")
callbacks.BaseCallback = type("BaseCallback", (), {})
logger = types.ModuleType("stable_baselines3.common.logger")
logger.TensorBoardOutputFormat = type("TensorBoardOutputFormat", (), {})

sys.modules.update(
    {
        "stable_baselines3": stable_baselines3,
        "stable_baselines3.common": common,
        "stable_baselines3.common.callbacks": callbacks,
        "stable_baselines3.common.logger": logger,
    }
)

from robot_sf.tb_logging import (
    AdversarialPedestrianMetricsCallback,
    AdversialPedestrianMetricsCallback,
)

assert AdversialPedestrianMetricsCallback is AdversarialPedestrianMetricsCallback
assert AdversarialPedestrianMetricsCallback.__qualname__ == "AdversarialPedestrianMetricsCallback"
assert (
    pickle.loads(pickle.dumps(AdversarialPedestrianMetricsCallback))
    is AdversarialPedestrianMetricsCallback
)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, check=False, text=True
    )

    assert result.returncode == 0, result.stderr
