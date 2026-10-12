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

# allow-abs-path: byte-exact original producer prefix, never a runtime input path.
PRODUCER_ROOT = b"/home/luttkule/lanes/hardcase/repo"  # allow-abs-path: original bytes
DIGESTS = (
    "2b646f046301c2b14f957160f5f5445abf3a362047106d19c24521f03d372508",
    "fc90bb1685027be6e79f92fe203d9f299969253ae72dbc1f58df7b2a504fcfd0",
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
