"""Task-only interlock: this benchmark audit executes no simulator episodes.

The positive-control episodes ran separately on the explicitly pinned dev seed.
Keep repository seed enforcement installed and stop real physics entry before
RNG or state mutation, including subprocesses. Static/mocked tests still run.
"""
import os
import sys

if os.environ.get("HARDCASE_NO_TEST_SIMULATION") == "1":
    try:
        import seedguard_boundaries as guard
        sys.modules["tests.support.seedguard_boundaries"] = guard
        original = guard._check_boundary_seeds

        def reject_simulation(module_name, qualified, values, seed_arg, config_arg, fields, boundary):
            if module_name not in {"random", "numpy.random"}:
                import pytest
                pytest.skip("hardcase task interlock: simulator execution excluded from benchmark audit")
            return original(module_name, qualified, values, seed_arg, config_arg, fields, boundary)

        guard._check_boundary_seeds = reject_simulation
        guard.install()
    except BaseException:
        os.write(2, b"hardcase test interlock installation failed; execution refused\n")
        os._exit(86)
