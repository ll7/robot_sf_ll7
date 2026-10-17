"""Regenerate the wrapper-enabled BIKEFIX diagnostic and its provenance sidecars."""

import argparse
import collections
import json
from pathlib import Path

from summarize_bicycle_probe import emit_csv, load_rows, load_trace, raw_identity, stats

from robot_sf.evidence.writers import sha256_file, write_json


def main():
    """Validate paired dev cells and publish deterministic episode and pooled evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    rows = load_rows(args.input)
    manifest = json.loads((args.input / "manifest.json").read_text())
    assert manifest["safety_wrapper_enabled"] and manifest["wrapper_deadlock_recovery_enabled"]
    assert not manifest["cbf_safety_filter_enabled"]
    assert len(rows) == 180 and {r["planner"] for r in rows} == {"goal"}
    arms = ["DD", "T60-45", "T60-45-on"]
    scenarios = manifest["roster"]  # Planner roster is source-bound, not an execution axis.
    assert any(p["key"] == "goal" for p in scenarios)
    by = {(r["arm"], r["scenario"], r["seed"]): r for r in rows}
    assert len(by) == 180
    assert {r["seed"] for r in rows} == set(range(1001, 1011))
    assert len({r["scenario"] for r in rows}) == 6
    evidence = []
    for r in rows:
        assert r["probe"] == 2 and r["arm"] in arms
        assert r["reset"] == by[("DD", r["scenario"], r["seed"])]["reset"]
        trace = load_trace(args.input, r)
        labels = [t["safety_interventions"] for t in trace]
        safety_creep_steps = sum(bool(t["safety_interventions"]) and t["creeping"] for t in trace)
        terminal = labels[-1] if labels else []
        evidence.append(
            {
                "cell": r["cell"],
                "planner": r["planner"],
                "arm": r["arm"],
                "scenario": r["scenario"],
                "seed": r["seed"],
                "success": r["success"],
                "collision": r["collision"],
                "timeout": r["timeout"],
                "terminal_safety_interventions": "|".join(terminal),
                "safety_steps": sum(bool(t) for t in labels),
                "safety_creep_steps": safety_creep_steps,
                "creep_steps": sum(t["creeping"] for t in trace),
                "zero_speed_yaw_veto_steps": sum(
                    bool(t["safety_interventions"])
                    and t["final_command"][0] == 0.0
                    and abs(t["final_command"][1]) >= 0.017
                    for t in trace
                ),
            }
        )
    output = args.output_root / "docs/validation"
    output.mkdir(parents=True, exist_ok=True)
    emit_csv(output, "bicycle_wrapper_probe_episodes.csv", list(evidence[0]), evidence)
    summary = []
    for arm in arms:
        selected = [r for r in rows if r["arm"] == arm]
        cases = [r for r in evidence if r["arm"] == arm]
        summary.append(
            {
                "arm": arm,
                **stats(selected),
                "terminal_safety_interventions": sum(
                    bool(r["terminal_safety_interventions"]) for r in cases
                ),
                "terminal_intervention_labels": dict(
                    collections.Counter(
                        label
                        for r in cases
                        for label in r["terminal_safety_interventions"].split("|")
                        if label
                    )
                ),
                **{
                    field: sum(r[field] for r in cases)
                    for field in [
                        "safety_steps",
                        "safety_creep_steps",
                        "creep_steps",
                        "zero_speed_yaw_veto_steps",
                    ]
                },
            }
        )
    result = {
        "schema_version": "bicycle-wrapper-probe.v1",
        "producer_sha": manifest["source_sha"],
        "planner": "goal",
        "seeds": list(range(1001, 1011)),
        "rows": 180,
        "safety_wrapper_enabled": True,
        "deadlock_recovery_enabled": True,
        "cbf_safety_filter_enabled": False,
        "paired_reset_mismatches": [],
        "summary": summary,
        "terminal_safety_interventions_definition": "episodes whose final executed step has at least one wrapper intervention; labels count separately",
    }
    write_json(output / "bicycle_wrapper_probe_summary.json", result)
    write_json(
        output / "bicycle_wrapper_probe_provenance.json",
        {
            "schema_version": "bicycle-wrapper-probe-provenance.v1",
            "producer_sha": manifest["source_sha"],
            "source_files_sha256": manifest["source_files_sha256"],
            "raw": raw_identity(args.input),
            "summarizer_sha256": sha256_file(Path(__file__)),
            "shared_summarizer_sha256": sha256_file(
                Path(__file__).with_name("summarize_bicycle_probe.py")
            ),
            "public_sha256": {
                name: sha256_file(output / name)
                for name in [
                    "bicycle_wrapper_probe_episodes.csv",
                    "bicycle_wrapper_probe_summary.json",
                ]
            },
        },
    )


if __name__ == "__main__":
    main()
