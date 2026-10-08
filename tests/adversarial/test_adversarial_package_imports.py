"""Tests adversarial package import boundaries."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_adversarial_package_import_does_not_eagerly_load_search() -> None:
    """Package import stays lightweight so optional planner dependencies remain optional."""

    code = (
        "import sys\n"
        "import robot_sf.adversarial\n"
        "assert 'robot_sf.adversarial.search' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_objectives_import_resolves_v2_without_loading_deferred_api() -> None:
    """Direct objective lookup registers v2 without loading the deferred package API."""

    code = (
        "import sys\n"
        "from robot_sf.adversarial.objectives import get_objective\n"
        "objective = get_objective('constraints_first_lexicographic_v2')\n"
        "assert callable(objective)\n"
        "assert 'robot_sf.adversarial._api' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "first_module", ["robot_sf.adversarial.objectives", "robot_sf.adversarial.objectives_v2"]
)
def test_fresh_objective_registry_enumerates_builtins(first_module: str) -> None:
    """Enumeration includes v2 before any lookup, regardless of scorer import order."""
    code = (
        "import importlib, sys\n"
        f"importlib.import_module({first_module!r})\n"
        "from robot_sf.adversarial.objectives import list_objectives\n"
        "expected = (\n"
        "    'constraints_first_lexicographic_v1',\n"
        "    'constraints_first_lexicographic_v2',\n"
        "    'minimize_episode_min_robot_distance',\n"
        "    'temporal_robustness',\n"
        "    'worst_case_snqi',\n"
        ")\n"
        "assert list_objectives() == expected, list_objectives()\n"
        "assert list_objectives() == expected\n"
        "assert 'robot_sf.adversarial._api' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_adversarial_search_reexport_is_lazy() -> None:
    """The package-level search re-export resolves through module __getattr__."""

    from robot_sf import adversarial
    from robot_sf.adversarial.search import (
        production_candidate_evaluator,
        run_adversarial_search,
    )

    assert adversarial.run_adversarial_search is run_adversarial_search
    assert adversarial.production_candidate_evaluator is production_candidate_evaluator
    missing_name = "definitely_missing"
    with pytest.raises(AttributeError, match="definitely_missing"):
        getattr(adversarial, missing_name)


def test_public_adversarial_reexports_resolve_on_demand() -> None:
    """Existing package-level helper names remain available through lazy exports."""

    from robot_sf.adversarial import CandidateSpec, RandomCandidateSampler

    assert CandidateSpec.__name__ == "CandidateSpec"
    assert RandomCandidateSampler.__name__ == "RandomCandidateSampler"


def test_adversarial_search_import_does_not_require_torch() -> None:
    """Search import stays available without optional CrowdNav HEIGHT torch dependency."""

    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from robot_sf.adversarial import search\n"
        "assert search is not None\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
