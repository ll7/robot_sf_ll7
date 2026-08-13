"""Build the issue #7032 negative-result mechanism-boundary atlas."""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.mechanism_boundary_atlas import main as atlas_main

DEFAULT_INPUT = Path(
    "docs/context/evidence/issue_7032_mechanism_boundary_atlas/atlas_input.v1.json"
)
DEFAULT_OUTPUT = Path("docs/context/evidence/issue_7032_mechanism_boundary_atlas/atlas.v1.json")


def main() -> int:
    """Run the deterministic atlas builder with repository defaults."""

    return atlas_main(["--input", str(DEFAULT_INPUT), "--output", str(DEFAULT_OUTPUT)])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
