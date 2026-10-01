"""Bootstrap only Python children explicitly launched by guarded pytest."""

import os
import sys

if os.environ.get("ROBOT_SF_PYTEST_SEED_GUARD") == "1":
    try:
        import seedguard_boundaries
    except BaseException:
        # Python otherwise prints sitecustomize errors and runs unprotected code.
        os.write(2, b"SEEDGUARD: child bootstrap failed; execution refused\n")
        os._exit(86)

    # Workers subsequently load the pytest plugin by its package name. Reuse the
    # same module so wrappers and pytest.raises share one exception class.
    sys.modules["tests.support.seedguard_boundaries"] = seedguard_boundaries
    try:
        seedguard_boundaries.install()
    except BaseException:
        os.write(2, b"SEEDGUARD: child boundary installation failed; execution refused\n")
        os._exit(86)
