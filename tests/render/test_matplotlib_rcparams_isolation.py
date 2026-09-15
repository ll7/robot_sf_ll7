"""Regression test: matplotlib rcParams must not leak between tests (issue #9411).

Production render paths mutate global rcParams (notably savefig.bbox via the
latex style helper). The autouse isolation fixture in tests/conftest.py must
restore defaults after every test, so fixed-canvas assertions elsewhere never
depend on xdist worker execution order. These two tests run in definition
order: without the fixture, the second one fails.
"""

import matplotlib as mpl


def test_polluting_test_mutates_global_rcparams() -> None:
    """Simulate a render path that mutates global savefig behavior."""
    mpl.rcParams["savefig.bbox"] = "tight"

    assert mpl.rcParams["savefig.bbox"] == "tight"


def test_rcparams_restored_after_previous_test() -> None:
    """Defaults must hold regardless of what earlier tests mutated."""
    assert mpl.rcParams["savefig.bbox"] is None
