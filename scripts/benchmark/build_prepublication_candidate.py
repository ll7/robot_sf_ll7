#!/usr/bin/env python3
"""Create a DOI-free, diagnostic release candidate from one clean checkout (#9863)."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.release_candidate import create_prepublication_candidate


def main() -> int:
    """Create and print the exact candidate identity.

    Returns:
        Zero after validation, nonzero on an input or custody error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-config", type=Path, required=True)
    parser.add_argument("--suite-policy", type=Path, required=True)
    parser.add_argument("--route-certification", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = create_prepublication_candidate(
        campaign_config=args.campaign_config,
        suite_policy=args.suite_policy,
        route_certification=args.route_certification,
        candidate_id=args.candidate_id,
        output=args.output,
    )
    print(f"candidate={candidate.path}")
    print(f"candidate_sha256={sha256_file(candidate.path)}")
    print(f"source_commit={candidate.source_sha}")
    print(f"pinned_file_count={len(candidate.pinned_files)}")
    print("evidence_class=preflight_diagnostic_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
