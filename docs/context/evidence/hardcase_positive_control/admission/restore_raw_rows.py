"""Reconstruct the v2 native rows exactly from committed gzip; no simulation."""
# evidence-writer-exempt: byte-exact native custody rows cannot embed review markers;
# the committed packet and regenerated custody targets carry separate review sidecars.

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from robot_sf.evidence.writers import write_review_sidecar


def main() -> None:
    """Restore immutable native bytes into a caller-selected custody directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    packet = Path(__file__).parent / "packet"
    provenance = json.loads((packet / "provenance.json").read_text())
    for number in (1, 2):
        name = f"window_replay_{number}.jsonl"
        raw = next(a for a in provenance["raw_artifacts"] if a["path"].endswith("/" + name))
        data = gzip.decompress((packet / (name + ".gz")).read_bytes())
        data = data.replace(b"<repository_root>", provenance["producer_repository"].encode())
        if hashlib.sha256(data).hexdigest() != raw["sha256"] or len(data) != raw["size_bytes"]:
            raise ValueError("native byte reconstruction mismatch")
        target = output / name
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError("different existing custody bytes")
        else:
            target.write_bytes(data)
            write_review_sidecar(target, repo_root=output)
        print(f"PASS {name}: {len(data)} bytes")


if __name__ == "__main__":
    main()
