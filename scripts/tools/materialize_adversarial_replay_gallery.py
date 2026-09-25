"""Materialize a search-manifest gallery or reconcile a compact multi-run evidence packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.adversarial.replay_gallery import build_replay_gallery

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the replay-gallery command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        help="adversarial-search-manifest.v1 JSON or a compact #9645 payload directory",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="new output directory for the gallery"
    )
    parser.add_argument("--top-k", default=5, type=int, help="maximum cases to select (1–20)")
    parser.add_argument(
        "--tolerance",
        default=1e-6,
        type=float,
        help="absolute tolerance for canonical objective and outcome comparisons",
    )
    parser.add_argument("--no-video", action="store_true", help="skip optional synthetic video")
    parser.add_argument("--no-render", action="store_true", help="skip existing figure rendering")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build one gallery and print its compact result summary."""
    args = parse_args(argv)
    result = build_replay_gallery(
        args.manifest,
        args.out,
        top_k=args.top_k,
        tolerance=args.tolerance,
        video=not args.no_video,
        render=not args.no_render,
    )
    print(
        json.dumps(
            {
                "output": args.out.as_posix(),
                "summary": result["summary"],
                "schema_version": result["schema_version"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
