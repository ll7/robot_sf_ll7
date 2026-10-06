"""Bootstrap only Python children explicitly launched by guarded pytest."""

import os
import sys
from importlib.machinery import PathFinder
from importlib.util import module_from_spec

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

    # The safety bootstrap precedes application PYTHONPATH entries. Preserve the
    # startup hook Python would otherwise have loaded (for example offline I/O
    # isolation), after installing the seed guard and before running user code.
    bootstrap_dir = os.path.realpath(os.path.dirname(__file__))
    application_paths = [p for p in sys.path if os.path.realpath(p) != bootstrap_dir]
    application_hook = PathFinder.find_spec("sitecustomize", application_paths)
    if application_hook is not None and application_hook.loader is not None:
        application_module = module_from_spec(application_hook)
        sys.modules["sitecustomize"] = application_module
        application_hook.loader.exec_module(application_module)
