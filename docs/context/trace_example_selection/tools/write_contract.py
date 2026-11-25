"""Write a claim's contract.yaml.txt, copying the protocol's claim, rules and spec text verbatim.

    python write_contract.py --protocol <trace_example_selection_protocol.yaml> \
        --protocol-commit <diss sha> --claim-id <id> --out <claim dir>/contract.yaml.txt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exsel

EXEC_COMMIT = "373dbfde4f39667cf9e8732dabe7df5118bdeab1"


def _literal(dumper: yaml.SafeDumper, value: str) -> yaml.Node:
    """Dump multi-line strings as literal blocks so verbatim text stays readable.

    Returns:
        The scalar node.
    """
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


yaml.SafeDumper.add_representer(str, _literal)


def span(lines: list[str], start: str, stop_prefixes: tuple[str, ...]) -> str:
    """Exact text from the line equal to ``start`` up to the next stop line (exclusive).

    Returns:
        The span with trailing blank lines removed.
    """
    i = lines.index(start)
    j = i + 1
    while j < len(lines) and not lines[j].startswith(stop_prefixes):
        j += 1
    out = lines[i:j]
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


def main() -> int:
    """Write the contract.

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--protocol-commit", required=True)
    ap.add_argument("--claim-id", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    raw = a.protocol.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    claim_text = span(lines, f"  - id: {a.claim_id}", ("  - id: ", "acquisition_spec_markdown:"))
    rules_text = span(lines, "rules:", ("sources:",))
    spec_lines = yaml.safe_load(raw)["acquisition_spec_markdown"].splitlines()
    common = span(spec_lines, "## Common steps", ("## EXSEL-",))
    own = span(spec_lines, f"## {a.claim_id}", ("## EXSEL-",))
    claim = yaml.safe_load(claim_text)[0]
    scr = claim["screening"]
    order = exsel.screening_order(scr["seed_band"], scr["selection_rng_seed"], scr["budget"])
    arms = [arm["arm"] for arm in claim["arms"]]
    contract = {
        "schema": "exsel.screening_contract.v1",
        "claim_id": a.claim_id,
        "evidence_tier": (
            "development-seed diagnostic on the 0.0.8 release code; new current-version "
            "execution, never release evidence; admits nothing into the thesis"
        ),
        "protocol": {
            "repository": "ll7/diss",
            "path": "docs/context/trace_example_selection_protocol.yaml",
            "read_at_commit": a.protocol_commit,
            "landed_in_commit": "b95d41b524f1a12655ecf5722ee980d745861350",
            "decision_id": "EXSEL-TRACE-SELECTION-PROTOCOL-2026-10-28",
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "claim_keyed_report": (
                "uv run scripts/dev/diss_worked_example_nomination.py --claims-protocol "
                "docs/context/trace_example_selection_protocol.yaml --robot-sf-repo "
                "<robot_sf_ll7 clone> (DISS_RELEASE_BUNDLE_DIR=<0.0.8 publication bundle>); "
                "exit 0, 'Frozen record: MATCH' for every claim"
            ),
        },
        "protocol_claim_verbatim": claim_text,
        "protocol_rules_verbatim": rules_text,
        "acquisition_spec_verbatim": common + "\n" + own,
        "execution": {
            "repository": "ll7/robot_sf_ll7",
            "commit": EXEC_COMMIT,
            "branch": "exscr1-exec-373dbfde (points at the release commit; no code change)",
            "code_change": "none",
            "release_campaign_config": (
                "configs/benchmarks/"
                "paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml"
            ),
            "release_campaign_config_sha256": (
                "de627db9f2999b17ed56378b9d793ee13a4b1bc15d1eb442513360d45f6d76d5"
            ),
            "scenario_matrix": (
                "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
            ),
            "scenario_matrix_sha256": (
                "af4e9a7a22f9d5f373eb4a3323984e0de6d79edd4263f1c023e9efddff2fd25f"
            ),
            "horizon_schedule_sha256": (
                "032bbf8ad354ea9394492659aa50188e2bc40078930975ef30cd3ba77208675e"
            ),
            "cell": claim["frozen"]["cell"],
            "arms": arms,
            "planner_entries": "copied verbatim from the release campaign config planners list",
            "runner": "scripts/tools/run_camera_ready_benchmark.py --skip-publication-bundle",
            "derived_config_changes": {
                "seed_policy": "fixed-list with the one screened seed",
                "scenario_candidates": "[frozen cell]",
                "planners": "the claim's arms only",
                "export_publication_bundle": False,
                "workers": 1,
                "snqi_v2_spec": (
                    "null: the release bound SNQI-v2 anchors for its 48-cell schedule and the "
                    "loader refuses a one-cell schedule; SNQI enters no class predicate or "
                    "banded measure"
                ),
                "record_simulation_step_trace": "true in the trace-capable rerun only",
            },
            "dt_s": 0.1,
            "environment": {
                "python": "3.13.14",
                "uv_lock_sha256": "def82098b23281e7c49f1f05e052e412c2de54ae2da53f1708bc0ddc2a30d023",
            },
        },
        "screening": {
            "seed_band": scr["seed_band"],
            "budget": scr["budget"],
            "selection_rng_seed": scr["selection_rng_seed"],
            "order": order,
            "procedure": (
                "one campaign per seed with all arms, seeds in this order; stop at the first "
                "accepted seed or after the budget; a run that exits non-zero halts the claim "
                "and is reported, it is not a rejection"
            ),
            "excluded_runs": (
                "the runner smoke (development seed 1001) and the qualification parity runs "
                "(development seeds 1004 and 1005, goal, classic_doorway_medium) are never "
                "screening rows"
            ),
        },
        "measures": {
            "outcome": "S if metrics.success; else C if outcome.collision_event; else T if "
            "outcome.timeout_event; else ?",
            "collision_kind": "pedestrian if C and metrics.ped_collision_count > 0; obstacle if C, "
            "no pedestrian collision and (metrics.obstacle_collision_count > 0 "
            "or metrics.wall_collisions > 0)",
            "min_surface_clearance_m": "safety_predicates.occlusion_near_miss_predicate.fields."
            "actual_minimum_separation_m - 1.4; absent when missing "
            "or >= 1e6",
            "exposure_steps": "metrics.near_misses",
            "time_to_goal_s": "round(steps * scenario_params.run_dt, 6) for outcome S only",
            "band_check": "inclusive; unrounded value against the frozen (3-decimal) bounds; an "
            "absent value lies outside; a null band is not checked",
        },
        "trace_rerun": {
            "match_fields": [
                "outcome",
                "collision_kind",
                "termination_reason",
                "steps",
                "ped_collision_count",
                "obstacle_collision_count",
                "wall_collisions",
                "near_misses",
                "min_ped_separation_m",
            ],
            "rule": "accept the trace only if every arm's rerun row equals its screening row on "
            "every match field (exact equality); otherwise report the rerun as it is "
            "(rules.never_replace)",
        },
        "mode": {
            "column": "mode in the screening ledger, per (seed, arm), from the raw row",
            "rule": "fallback if algorithm_metadata.status or pedestrian_model."
            "fallback_degraded_status reports fallback or any numeric algorithm_metadata "
            "counter whose key contains 'fallback' is > 0; degraded if integrity."
            "effective_view.degraded or the pedestrian model is not native; else native",
            "use": "a non-native row is reported and counted; it is not silently screened as a "
            "rejection or acceptance (the verdict is still recorded as the rule gives it)",
        },
        "outcome_rule": {
            "accepted": (
                "the claim has a trace example: the accepted seed, the screening index and the "
                "trace files; captions quote the frozen base rate pattern of the chosen cell, "
                "never a rate read off the trace; whether a figure is made is decided on the "
                "dissertation side"
            ),
            "no_qualifying_example": (
                "rules.no_match: the claim gets no trace example and the text uses the "
                "aggregate evidence alone"
            ),
            "trace_mismatch": (
                "rules.never_replace: the accepted seed stays the example; the rerun is "
                "reported as it is and the trace is not accepted as a match"
            ),
            "run_failure": "BLOCKED_EXTERNAL for the claim, with the failing seed and log",
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(contract, sort_keys=False, width=100, allow_unicode=True)
    a.out.write_text(text, encoding="utf-8")
    # The verbatim claim must round-trip and reparse to the protocol's own claim.
    back = yaml.safe_load(a.out.read_text(encoding="utf-8"))
    assert back["protocol_claim_verbatim"] == claim_text
    full = yaml.safe_load(raw)
    assert next(c for c in full["claims"] if c["id"] == a.claim_id) == exsel.load_claim(back)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
