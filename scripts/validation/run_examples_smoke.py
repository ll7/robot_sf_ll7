"""Execute the manifest-driven example smoke test via pytest.

The script coordinates ``tests/examples/test_examples_run.py`` and exposes a
``--dry-run`` flag to list the targeted examples without executing them.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import pytest

from robot_sf.examples.manifest_loader import load_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the manifest-driven examples smoke test (pytest harness).",
        epilog=(
            "Additional pytest arguments can be supplied after --pytest-args, e.g.\n"
            "  uv run python scripts/validation/run_examples_smoke.py --pytest-args -k quickstart\n"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List CI-enabled examples without invoking pytest.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=(),
        metavar="PYTEST_ARGS",
        help="Additional arguments forwarded directly to pytest.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(validate_paths=True)
    ci_examples = tuple(manifest.iter_ci_enabled_examples())

    if args.dry_run:
        if not ci_examples:
            print("No CI-enabled examples declared in manifest.")
        else:
            print("CI-enabled examples (dry run):")
            for example in ci_examples:
                tag_suffix = f" [{', '.join(example.tags)}]" if example.tags else ""
                print(f" - {example.path.as_posix()}{tag_suffix}")
            print(f"Total: {len(ci_examples)} example(s)")
        return 0

    pytest_args = ["tests/examples/test_examples_run.py"]
    if args.pytest_args:
        pytest_args.extend(args.pytest_args)

    return int(pytest.main(pytest_args))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
