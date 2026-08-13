"""Build the issue #7032 negative-result mechanism-boundary atlas."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.mechanism_boundary_atlas import main as atlas_main

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_INPUT = Path(
    "docs/context/evidence/issue_7032_mechanism_boundary_atlas/atlas_input.v1.json"
)
DEFAULT_OUTPUT = Path("docs/context/evidence/issue_7032_mechanism_boundary_atlas/atlas.v1.json")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic atlas builder with optional CLI overrides."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["--input", str(DEFAULT_INPUT), "--output", str(DEFAULT_OUTPUT)]
    return atlas_main(arguments)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
