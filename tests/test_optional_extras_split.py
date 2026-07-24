"""Tests for pyproject.toml aggressive extras split and optional import guards (issue #5799)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from robot_sf.common.optional_import import require_extra

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def test_pyproject_extras_split_contract() -> None:
    """Verify pyproject.toml defines required optional extras and a slim core."""
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    project = data["project"]
    dependencies = project.get("dependencies", [])
    optional_deps = project.get("optional-dependencies", {})

    # Check required optional extra groups exist
    for required_extra in ("viz", "maps", "benchmark", "training", "all"):
        assert required_extra in optional_deps, f"Missing required extra [{required_extra}]"

    # Core dependencies must NOT contain heavy optional packages
    heavy_packages = [
        "pygame",
        "matplotlib",
        "moviepy",
        "geopandas",
        "osmnx",
        "pyproj",
        "pandas",
        "scipy",
        "stable-baselines3",
        "torch",
        "wandb",
        "tensorboard",
    ]
    core_dep_str = " ".join(dependencies)
    for pkg in heavy_packages:
        assert pkg not in core_dep_str, f"Heavy package '{pkg}' found in core dependencies"

    # Check extras package allocations
    viz_deps = " ".join(optional_deps["viz"])
    for pkg in ("pygame", "matplotlib", "pillow", "moviepy", "seaborn"):
        assert pkg in viz_deps, f"Package '{pkg}' missing from [viz] extra"

    maps_deps = " ".join(optional_deps["maps"])
    for pkg in ("osmnx", "geopandas", "pyproj"):
        assert pkg in maps_deps, f"Package '{pkg}' missing from [maps] extra"

    bench_deps = " ".join(optional_deps["benchmark"])
    for pkg in ("pandas", "scipy"):
        assert pkg in bench_deps, f"Package '{pkg}' missing from [benchmark] extra"

    all_deps = " ".join(optional_deps["all"])
    assert "robot_sf[viz,maps,benchmark,training" in all_deps


def test_require_extra_remedy_message() -> None:
    """require_extra raises ImportError with clear remedy instruction."""
    missing_module = "definitely_not_a_module_xyz_5799"
    with pytest.raises(ImportError, match=r"robot_sf\[viz\]"):
        require_extra("viz", missing_module, feature_name="Visualization")
