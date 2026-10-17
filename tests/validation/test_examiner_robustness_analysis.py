"""Independent outcome and missingness controls for the examiner analysis."""

from __future__ import annotations

import gzip
import json

import pytest

from scripts.validation import analyze_examiner_robustness_20261012 as analysis


def _fixture(tmp_path, monkeypatch, defect=None):
    """Create a hand-calculated two-planner ranking reversal on dev identities."""
    config, evidence = tmp_path / "configs", tmp_path / "evidence"
    config.mkdir()
    (evidence / "episodes").mkdir(parents=True)
    monkeypatch.setattr(analysis, "CONFIG", config)
    monkeypatch.setattr(analysis, "EVIDENCE", evidence)
    families = {f"f{i}": f"scenario{i}" for i in range(10)}
    seeds = [1061, 1062, 1063]
    (config / "study.json").write_text(json.dumps({"families": families, "seeds": seeds}))
    jobs = [
        {"treatment": arm, "planner": p, "requested": 30}
        for arm in ("legacy", "changed")
        for p in ("a", "b")
    ]
    (evidence / "jobs.json").write_text(json.dumps({"jobs": jobs, "seed_count": 3}))
    for job in jobs:
        rows = []
        for scenario in families.values():
            for seed in seeds:
                success = (job["planner"] == "a") == (job["treatment"] == "legacy")
                row = {
                    "seed": seed,
                    "scenario_params": {"name": scenario},
                    "outcome": {"route_complete": success, "collision_event": not success},
                    "algorithm_metadata": {"status": "ok"},
                }
                target = job == jobs[-1] and scenario == "scenario0" and seed == 1061
                if target and defect == "missing":
                    continue
                if target and defect == "fallback":
                    row["algorithm_metadata"]["nested"] = {"fallback": True}
                # Optional proxy unavailability does not degrade executed outcomes.
                row["metrics"] = {"social_compliance": {"status": "unavailable"}}
                rows.append(row)
        with gzip.open(
            evidence / "episodes" / f"{job['treatment']}_{job['planner']}.jsonl.gz", "wt"
        ) as handle:
            handle.writelines(json.dumps(row) + "\n" for row in rows)
    return evidence


def test_complete_matched_reversal(tmp_path, monkeypatch):
    """Four complete groups must report exactly reversed success and collision ranks."""
    evidence = _fixture(tmp_path, monkeypatch)
    analysis.analyze()
    summary = json.loads((evidence / "summary.json").read_text())
    changed = summary["rankings"][1]
    assert summary["eligible"] == 120
    assert changed["success_tau_b"] == -1.0
    assert changed["collision_tau_b"] == -1.0
    assert changed["success_tau_ci95"] == [-1.0, -1.0]


@pytest.mark.parametrize("defect", ["missing", "fallback"])
def test_incomplete_pair_withholds_ranking(tmp_path, monkeypatch, defect):
    """One missing or fallback row invalidates the joint comparison, not the denominator."""
    evidence = _fixture(tmp_path, monkeypatch, defect)
    analysis.analyze()
    summary = json.loads((evidence / "summary.json").read_text())
    changed = summary["rankings"][1]
    assert summary["eligible"] == 119
    assert changed["complete_matched_roster"] is False
    assert changed["success_tau_b"] is None
    assert changed["collision_tau_b"] is None
    assert summary["missing"] == (1 if defect == "missing" else 0)
    assert len(summary["excluded"]) == (1 if defect == "fallback" else 0)


def test_ties_are_retained_and_constant_ranks_undefined():
    """Three hand-counted pairs give tau-b 1/2; a constant order has no tau."""
    # One concordance, zero discordance, one x tie, one y tie: 1/sqrt(2*2).
    assert analysis.tau([1, 1, 2], [1, 2, 2]) == pytest.approx(0.5)
    assert analysis.tau([1, 1, 1], [1, 2, 3]) is None


def test_equal_episode_counts_do_not_become_floating_point_ranks(tmp_path, monkeypatch):
    """Three successes each out of thirty must tie despite different seed partitions."""
    evidence = _fixture(tmp_path, monkeypatch)
    for planner, per_seed in (("a", (0, 0, 3)), ("b", (0, 1, 2))):
        path = evidence / "episodes" / f"changed_{planner}.jsonl.gz"
        rows = [json.loads(line) for line in gzip.open(path, "rt")]
        for row in rows:
            seed_index = row["seed"] - 1061
            scenario_index = int(row["scenario_params"]["name"].removeprefix("scenario"))
            success = scenario_index < per_seed[seed_index]
            row["outcome"] = {"route_complete": success, "collision_event": not success}
        with gzip.open(path, "wt") as handle:
            handle.writelines(json.dumps(row) + "\n" for row in rows)
    analysis.analyze()
    changed = json.loads((evidence / "summary.json").read_text())["rankings"][1]
    assert changed["success_tau_b"] is None
    assert [r["success"] for r in changed["planner_outcomes"]] == [0.1, 0.1]
