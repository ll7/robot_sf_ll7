"""Restore the two original native JSONL byte streams without simulation.

The archived rows only normalized the producer checkout prefix. Reversing that
literal substitution reconstructs the native manifest's original SHA-256, which
is checked before writing. All trace payloads and seed identities are preserved.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path

PRODUCER_ROOT = b"/home/luttkule/lanes/hardcase/repo"
DIGESTS = (
    "1a2f954113d3a240d96dadc7c293da13008d128066217dae7bf4b9b3b4fffa50",
    "746b454f84736b729dea74ad419d2384bc3e41cac34b78ee5e7e75a170f99286",
)


def main() -> None:
    """Regenerate native rows from committed gzip inputs and verify custody."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    destination = parser.parse_args().output
    destination.mkdir(parents=True, exist_ok=True)
    for number, expected in enumerate(DIGESTS, 1):
        name = f"window_replay_{number}.jsonl"
        data = gzip.decompress((Path(__file__).parent / "packet" / (name + ".gz")).read_bytes())
        data = data.replace(b"<repository_root>", PRODUCER_ROOT)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"original native row digest mismatch: {name}")
        output = destination / name
        if output.exists() and output.read_bytes() != data:
            raise ValueError(f"refusing to overwrite different custody bytes: {name}")
        output.write_bytes(data)
        print(f"PASS {name}: {expected}, {len(data)} bytes")


if __name__ == "__main__":
    main()
