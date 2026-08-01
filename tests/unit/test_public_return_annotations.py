"""Regression tests for public ``robot_sf`` return annotations."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest

import robot_sf
from robot_sf.baselines.social_force import SocialForcePlanner
from robot_sf.eval import EnvMetrics, PedEnvMetrics, PedVecEnvMetrics, VecEnvMetrics
from robot_sf.maps.import_svg_from_osm import import_svg_from_osm
from robot_sf.maps.osm_zones_yaml import OSMZonesConfig, save_zones_yaml
from robot_sf.recipes.cli import build_subparser

ANNOTATED_ANALYSIS_MODULES = (
    "robot_sf.data_analysis.extract_json_from_pickle",
    "robot_sf.data_analysis.extract_obj_from_pickle",
    "robot_sf.data_analysis.plot_dataset",
    "robot_sf.data_analysis.plot_kernel_density",
    "robot_sf.data_analysis.plot_npc_trajectory",
    "robot_sf.data_analysis.recording_analysis",
)


def test_every_public_robot_sf_function_has_a_return_annotation() -> None:
    """Keep the issue #6276 AST contract at zero missing annotations."""
    package_root = Path(robot_sf.__file__).parent
    missing: list[str] = []

    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("_")
                and node.returns is None
            ):
                relative_path = path.relative_to(package_root.parent)
                missing.append(f"{relative_path}:{node.lineno}:{node.name}")

    assert missing == []


def test_social_force_scalar_normal_return_annotation_matches_runtime() -> None:
    """Distinguish the scalar Python float from NumPy array results."""
    rng = SocialForcePlanner._RNGCompat(seed=0)

    assert inspect.signature(rng.normal).return_annotation == "float | np.ndarray"
    assert isinstance(rng.normal(), float)
    assert isinstance(rng.normal(size=2), np.ndarray)


def test_recipe_subparser_return_annotation_names_concrete_parser_type() -> None:
    """Keep the recipe builder annotation concrete while exercising its definition."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    assert inspect.signature(build_subparser).return_annotation == "argparse.ArgumentParser"
    assert isinstance(build_subparser(subparsers), argparse.ArgumentParser)


def test_public_procedure_annotations_are_runtime_visible(tmp_path: Path) -> None:
    """Exercise definition lines that otherwise sit outside readiness test lanes."""
    procedures = [
        EnvMetrics.update,
        VecEnvMetrics.update,
        PedEnvMetrics.update,
        PedVecEnvMetrics.update,
        import_svg_from_osm,
    ]

    assert all(inspect.signature(procedure).return_annotation is None for procedure in procedures)
    save_zones_yaml(OSMZonesConfig(), str(tmp_path / "zones.yaml"))


def test_annotation_only_analysis_modules_import_cleanly() -> None:
    """Import annotation-only analysis modules when their optional dependency is available."""
    pytest.importorskip("sklearn")
    imported = [importlib.import_module(module_name) for module_name in ANNOTATED_ANALYSIS_MODULES]

    assert [module.__name__ for module in imported] == list(ANNOTATED_ANALYSIS_MODULES)
