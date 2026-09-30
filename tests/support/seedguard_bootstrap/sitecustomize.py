"""Bootstrap only Python children explicitly launched by guarded pytest."""
import os
import sys

if os.environ.get("ROBOT_SF_PYTEST_SEED_GUARD") == "1":
    import seedguard_boundaries

    # Workers subsequently load the pytest plugin by its package name. Reuse the
    # same module so wrappers and pytest.raises share one exception class.
    sys.modules["tests.support.seedguard_boundaries"] = seedguard_boundaries
    seedguard_boundaries.install()
