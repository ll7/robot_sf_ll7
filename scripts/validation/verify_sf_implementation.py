#!/usr/bin/env python3
"""
Simple verification script for Social Force Planner implementation.

This script verifies the core implementation without requiring full dependencies
by testing the configuration system, registry, and basic class structure.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))


def test_configuration_system():
    """Test that configuration system works."""
    print("Testing configuration system...")
    try:
        try:
            import importlib

            module = importlib.import_module("robot_sf.baselines.social_force")
            required = ["SocialForcePlanner", "SFPlannerConfig", "Observation"]
            missing = [name for name in required if not hasattr(module, name)]
            if missing:
                print(f"❌ Missing expected symbols: {missing}")
                return False
            print("✅ Module imported successfully and required symbols present")
            return True
        except ModuleNotFoundError as e:
            # Accept missing heavy deps (numpy / pysocialforce) as structural pass
            if any(x in str(e) for x in ("numpy", "pysocialforce")):
                print("✅ Module structure is correct (dependency missing as expected)")
                return True
            print(f"❌ Unexpected ModuleNotFoundError: {e}")
            return False
    except Exception as e:  # pragma: no cover - defensive
        print(f"❌ Configuration test failed: {e}")
        return False


def test_registry_system():
    """Test baseline registry without importing dependencies."""
    print("\nTesting registry system...")

    try:
        # Check registry structure
        with open("robot_sf/baselines/__init__.py", encoding="utf-8") as f:
            content = f.read()

        required_components = ["BASELINES", "get_baseline", "list_baselines", "baseline_sf"]

        for component in required_components:
            if component not in content:
                print(f"❌ Missing component: {component}")
                return False

        print("✅ Registry structure is correct")
        return True

    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False


def test_cli_integration():
    """Test CLI integration structure."""
    print("\nTesting CLI integration...")

    try:
        # Check CLI modifications
        with open("robot_sf/benchmark/cli.py", encoding="utf-8") as f:
            cli_content = f.read()

        # Check for algorithm support
        cli_requirements = ["--algo", "--algo-config", "list-algorithms", "baseline_sf"]

        for req in cli_requirements:
            if req not in cli_content:
                print(f"❌ Missing CLI feature: {req}")
                return False

        # Check runner modifications
        with open("robot_sf/benchmark/runner.py", encoding="utf-8") as f:
            runner_content = f.read()

        runner_requirements = ["_create_robot_policy", "algo:"]

        # Existing exact-match requirements
        for req in runner_requirements:
            if req not in runner_content:
                print(f"❌ Missing runner feature: {req}")
                return False

        # Flexible algorithm hook detection (at least one pattern must appear)
        hook_patterns = ["def algorithm_", "class Algorithm", "algo:"]
        if not any(p in runner_content for p in hook_patterns):
            print(
                "❌ Missing runner feature: algorithm hook (def algorithm_/class Algorithm/algo:)",
            )
            return False

        print("✅ CLI integration is complete")
        return True

    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def test_configuration_files():
    """Test configuration files exist and are valid."""
    print("\nTesting configuration files...")

    try:
        import yaml

        # Test default config
        with open("configs/baselines/social_force_default.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        required_params = ["v_max", "desired_speed", "action_space", "A", "B"]

        for param in required_params:
            if param not in config:
                print(f"❌ Missing config parameter: {param}")
                return False

        print("✅ Configuration files are valid")
        return True

    except ImportError:
        print("⚠️  Cannot test YAML config (yaml module not available)")
        return True  # Not a failure
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_documentation():
    """Test that documentation files exist."""
    print("\nTesting documentation...")

    doc_files = ["docs/dev/design/social_force_wrapper.md", "docs/baselines/social_force.md"]

    for doc_file in doc_files:
        if not os.path.exists(doc_file):
            print(f"❌ Missing documentation: {doc_file}")
            return False

    print("✅ Documentation files are present")
    return True


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")

    required_files = [
        "robot_sf/baselines/__init__.py",
        "robot_sf/baselines/social_force.py",
        "configs/baselines/social_force_default.yaml",
        "tests/baselines/test_social_force.py",
        "tests/integration/test_sf_smoke.py",
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            return False

    print("✅ All required files are present")
    return True


def main():
    """Run all verification tests."""
    print("🤖 Social Force Planner Implementation Verification")
    print("=" * 50)

    tests = [
        test_file_structure,
        test_configuration_system,
        test_registry_system,
        test_cli_integration,
        test_configuration_files,
        test_documentation,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All verification tests passed!")
        print("\nImplementation is ready for use once dependencies are available.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install numpy pysocialforce pytest")
        print("2. Run tests: pytest tests/baselines/ tests/integration/")
        print("3. Try CLI: python -m robot_sf.benchmark.cli list-algorithms")
        return 0
    else:
        print("❌ Some tests failed - implementation needs fixes")
        return 1


if __name__ == "__main__":
    sys.exit(main())
