"""Admit an observed critical window through the existing persistence replay gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import ValidationError

from robot_sf.adversarial.counterexample_corpus import admit_window, export_replay_slice
from robot_sf.benchmark.scenario_generation.persistence_gate import (
    ScenarioPersistenceValidationError,
)


def main() -> int:
    """Validate actual artifacts and commit or idempotently retain one case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--window-receipt", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--export-slice", type=Path)
    args = parser.parse_args()
    try:
        result = admit_window(
            args.candidate, args.window_receipt, corpus_dir=args.corpus, repo_root=args.repo_root
        )
        if args.export_slice is not None:
            args.export_slice.write_text(
                json.dumps(export_replay_slice(args.corpus), indent=2, sort_keys=True) + "\n"
            )
    except (
        OSError,
        ValueError,
        KeyError,
        ValidationError,
        ScenarioPersistenceValidationError,
    ) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
