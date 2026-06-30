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


def test_adversarial_search_reexport_is_lazy() -> None:
    """The package-level search re-export resolves through module __getattr__."""

    from robot_sf import adversarial
    from robot_sf.adversarial.search import run_adversarial_search

    assert adversarial.run_adversarial_search is run_adversarial_search
    missing_name = "definitely_missing"
    with pytest.raises(AttributeError, match="definitely_missing"):
        getattr(adversarial, missing_name)


def test_adversarial_search_import_does_not_require_torch() -> None:
    """Search import stays available without optional CrowdNav HEIGHT torch dependency."""

    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from robot_sf.adversarial import search\n"
        "assert search is not None\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
