"""Load safety checks for all pytest roots, including vendored simulator tests."""

pytest_plugins = ["tests.support.heldout_seed_guard"]
