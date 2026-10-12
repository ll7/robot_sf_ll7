"""Verify the retained continuation packet without running/resetting a simulator."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

from robot_sf.benchmark.scenario_generation.continuation import read_window_checkpoint
from robot_sf.benchmark.step_trace_comparator import canonical_step_trace_digest
from robot_sf.benchmark.typed_snapshot import read_typed_snapshot


def main() -> None:
    """Check companion custody and independent source/window/event equality."""
    folder = Path(__file__).resolve().parent
    packet = folder / "packet"
    source = json.loads(gzip.decompress((folder.parent / "source.jsonl.gz").read_bytes()))
    expected = source["algorithm_metadata"]["simulation_step_trace"]["steps"][370:390]
    checkpoint = packet / "checkpoint.json"
    record = read_window_checkpoint(
        checkpoint, expected_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    read_typed_snapshot(packet / record["simulator_file"])
    assert record["start_step"] == 370 and record["original_budget_steps"] == 600
    for index in (1, 2):
        row = json.loads(gzip.decompress((packet / f"window_replay_{index}.jsonl.gz").read_bytes()))
        actual = row["algorithm_metadata"]["simulation_step_trace"]["steps"]
        assert actual == expected
        assert canonical_step_trace_digest(actual) == (
            "37488c586d0a7fd203b9385377434516f108edb8bef011a6003a3236e135fed3"
        )
        near = [
            frame["step"]
            for frame in actual
            if 0
            <= min(
                math.dist(frame["robot"]["position"], p["position"]) for p in frame["pedestrians"]
            )
            - 1.4
            < 0.5
        ]
        assert len(near) == 11 and near[0] == 374
        assert all(not s["rl"]["terminated"] and not s["rl"]["truncated"] for s in actual)
        assert row["outcome"] == source["outcome"]
    entry = json.loads((folder.parent / "entry.json").read_text())
    gate_path = packet / "persistence" / f"{entry['scenario_id']}.persistence.json"
    gate = json.loads(gate_path.read_text())
    assert gate["promotion"]["verdict"] == "reject"
    assert gate["exact_replay"]["status"] == "fail"
    assert len(gate["perturbation_persistence"]["cells"]) == 9
    assert all(c["verdict"] == "unknown" for c in gate["perturbation_persistence"]["cells"])
    print("PASS: packet hashes, typed snapshot, both exact traces, events and gate rejection")


if __name__ == "__main__":
    main()
