"""Fixture-only contract tests for the issue #9653 round coordinator."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.adversarial.coevolution import (
    CoevolutionAdapters,
    CoevolutionError,
    load_coevolution_config,
    run_coevolution,
)


def _write_config(
    tmp_path: Path,
    *,
    candidate_budget: int = 2,
    minimum_rounds: int = 2,
    maximum_rounds: int = 2,
    stop_on_no_discovery: bool = True,
    plateau_rounds: int | None = None,
) -> Path:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(exist_ok=True)
    (input_dir / "optimizer.yaml").write_text("schema: optimizer-fixture.v1\n", encoding="utf-8")
    (input_dir / "template.yaml").write_text("scenarios: []\n", encoding="utf-8")
    (input_dir / "space.yaml").write_text("schema: search-space-fixture.v1\n", encoding="utf-8")
    payload: dict[str, Any] = {
        "schema": "adversarial_coevolution_config.v1",
        "run_id": "fixture-run",
        "output_dir": "run-output",
        "inputs": {"optimizer_config": "inputs/optimizer.yaml"},
        "falsification": {
            "scenario_template": "inputs/template.yaml",
            "search_space": "inputs/space.yaml",
            "policy": "fixture_planner",
            "objective": "worst_case_snqi",
            "horizon": 12,
            "dt": 0.1,
            "workers": 1,
            "benchmark_profile": "experimental",
            "record_forces": True,
            "require_certification": False,
        },
        "rounds": {"minimum": minimum_rounds, "maximum": maximum_rounds},
        "budgets": {
            "optimizer_trials_per_method": 1,
            "falsification_candidates_per_round": candidate_budget,
        },
        "seeds": {
            "optimizer_random_base": 31,
            "optimizer_tpe_base": 32,
            "falsification_base": 33,
        },
        "heldout_case_ids": ["heldout-1"],
        "initial_regression_case_ids": [],
        "stop": {
            "stop_after_minimum_without_new_admission": stop_on_no_discovery,
            "optimization_plateau_rounds": plateau_rounds,
        },
    }
    config_path = tmp_path / "loop.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _candidate(
    candidate_id: str,
    *,
    search_status: str = "evaluated",
    execution_status: str = "ok",
    planner_outcome: str = "confirmed_failure",
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "search_status": search_status,
        "execution_status": execution_status,
        "planner_outcome": planner_outcome,
        "scenario": {"name": f"scenario-{candidate_id}", "seed": 101},
        "objective_value": 1.0,
    }


class FixtureAdapters:
    """Small deterministic owner adapters that never start a simulator."""

    def __init__(
        self,
        *,
        candidates_by_round: dict[int, list[dict[str, Any]]] | None = None,
        gates: dict[str, dict[str, str]] | None = None,
        scores_by_round: dict[int, list[float | None]] | None = None,
        fail_phase: str | None = None,
        admit: bool = True,
    ) -> None:
        """Configure scripted phase results and collect which adapters ran."""
        self.candidates_by_round = candidates_by_round or {}
        self.gates = gates or {}
        self.scores_by_round = scores_by_round or {}
        self.fail_phase = fail_phase
        self.admit = admit
        self.calls: Counter[str] = Counter()
        self.round_regression_ids: dict[int, tuple[str, ...]] = {}
        self.admitted_case_ids: list[str] = []

    def optimize(self, request, output_dir):
        self.calls["optimize"] += 1
        if self.fail_phase == "optimization":
            raise OSError("fixture optimizer unavailable")
        output_dir.mkdir(parents=True, exist_ok=True)
        planner_path = output_dir / f"planner-r{request.round_number}.yaml"
        planner_payload = f"planner_id: fixture-planner-r{request.round_number}\n"
        planner_path.write_text(planner_payload, encoding="utf-8")
        planner_digest = hashlib.sha256(planner_payload.encode("utf-8")).hexdigest()
        selection = self.scores_by_round.get(request.round_number, [0.8, 0.5, None])
        return {
            "status": "complete",
            "selected_planner": {
                "planner_id": f"fixture-planner-r{request.round_number}",
                "config_sha256": planner_digest,
                "artifact_path": planner_path.name,
            },
            "selection_tuple": selection,
            "improves_over_baseline": request.round_number > 1,
            "budget": {
                "trials_per_method": request.optimizer_trials_per_method,
                "random_seed": request.optimizer_random_seed,
                "tpe_seed": request.optimizer_tpe_seed,
            },
            "train_evaluations": [{"trial": 0, "status": "complete", "selection_tuple": selection}],
        }

    def evaluate_challenges(self, request, _planner, _output_dir):
        self.calls["evaluate"] += 1
        self.round_regression_ids[request.round_number] = request.regression_case_ids
        return {
            "status": "complete",
            "heldout_rows": [
                {"case_id": case_id, "status": "solved", "metrics": {"success": True}}
                for case_id in request.heldout_case_ids
            ],
            "regression_rows": [
                {"case_id": case.case_id, "status": "unsolved", "metrics": {"success": False}}
                for case in request.regression_cases
            ],
        }

    def falsify(self, request, _planner, _output_dir):
        self.calls["falsify"] += 1
        candidates = self.candidates_by_round.get(request.round_number)
        if candidates is None:
            candidates = [
                _candidate("new-case")
                if request.round_number == 1
                else _candidate("no-new-case", planner_outcome="no_failure")
            ]
            candidates.extend(
                _candidate(f"routine-{index}", planner_outcome="no_failure")
                for index in range(len(candidates), request.falsification_candidates)
            )
        return {
            "status": "complete",
            "candidate_budget": request.falsification_candidates,
            "seed": request.falsification_seed,
            "candidates": candidates,
            "invalid_count": sum(row["search_status"] == "invalid" for row in candidates),
            "failed_count": sum(row["search_status"] == "failed" for row in candidates),
        }

    def verify_discovery(self, _request, _planner, candidate, _output_dir):
        self.calls["verify"] += 1
        defaults = {
            "replay_status": "exact_match",
            "feasibility_status": "empirically_feasible",
            "admissibility_status": "admissible",
            "planner_outcome": candidate["planner_outcome"],
        }
        return {**defaults, **self.gates.get(candidate["candidate_id"], {})}

    def admit_case(self, _request, _planner, case, _output_dir):
        self.calls["admit"] += 1
        self.admitted_case_ids.append(case["case_id"])
        return {
            "status": "admitted" if self.admit else "rejected",
            "admitted": self.admit,
            "case_id": case["case_id"],
            "stable_id": case["case_id"],
        }

    def bundle(self) -> CoevolutionAdapters:
        return CoevolutionAdapters(
            optimize=self.optimize,
            evaluate_challenges=self.evaluate_challenges,
            falsify=self.falsify,
            verify_discovery=self.verify_discovery,
            admit_case=self.admit_case,
        )


def test_round_one_admission_is_evaluated_as_round_two_regression(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    adapters = FixtureAdapters()

    result = run_coevolution(config, adapters.bundle())

    assert result["status"] == "complete"
    assert result["stop"]["reason"] == "no_new_admissible_counterexample_under_budget"
    assert len(result["rounds"]) == 2
    case_id = adapters.admitted_case_ids[0]
    assert adapters.round_regression_ids == {1: (), 2: (case_id,)}
    eval_path = Path(
        config.parent / "run-output" / "round_002" / "phases" / "challenge_evaluation.json"
    )
    persisted = json.loads(eval_path.read_text(encoding="utf-8"))
    assert [row["case_id"] for row in persisted["output"]["regression_rows"]] == [case_id]
    assert result["rounds"][1]["summary"]["regression_case_count"] == 1


def test_no_discovery_still_completes_configured_minimum_two_rounds(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    adapters = FixtureAdapters(
        candidates_by_round={
            round_number: [
                _candidate(f"no-failure-{round_number}-a", planner_outcome="no_failure"),
                _candidate(f"no-failure-{round_number}-b", planner_outcome="no_failure"),
            ]
            for round_number in (1, 2)
        }
    )

    result = run_coevolution(config, adapters.bundle())

    assert result["stop"]["after_round"] == 2
    assert result["stop"]["reason"] == "no_new_admissible_counterexample_under_budget"
    assert adapters.calls["optimize"] == 2
    assert adapters.calls["falsify"] == 2
    assert adapters.calls["admit"] == 0


def test_unknown_mismatch_invalid_fallback_degraded_and_failed_remain_distinct(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path, candidate_budget=7)
    candidates = [
        _candidate("unknown-feasibility"),
        _candidate("replay-mismatch"),
        _candidate(
            "invalid",
            search_status="invalid",
            execution_status="not_run",
            planner_outcome="not_assessed",
        ),
        _candidate("fallback", execution_status="fallback"),
        _candidate("degraded", execution_status="degraded"),
        _candidate("infeasible"),
        _candidate(
            "search-failed",
            search_status="failed",
            execution_status="failed",
            planner_outcome="unknown",
        ),
    ]
    adapters = FixtureAdapters(
        candidates_by_round={1: candidates, 2: candidates},
        gates={
            "unknown-feasibility": {
                "feasibility_status": "unknown",
                "admissibility_status": "unknown",
            },
            "replay-mismatch": {"replay_status": "mismatch"},
            "infeasible": {
                "feasibility_status": "infeasible",
                "admissibility_status": "inadmissible",
            },
        },
    )

    result = run_coevolution(config, adapters.bundle())

    rows = json.loads(
        (
            Path(config.parent / "run-output" / "round_001" / "phases" / "discovery_admission.json")
        ).read_text(encoding="utf-8")
    )["output"]["rows"]
    indexed = {row["candidate_id"]: row for row in rows}
    assert indexed["unknown-feasibility"]["gate"]["feasibility_status"] == "unknown"
    assert indexed["unknown-feasibility"]["admission"]["status"] == "not_eligible"
    assert indexed["replay-mismatch"]["gate"]["replay_status"] == "mismatch"
    assert indexed["replay-mismatch"]["admission"]["status"] == "not_eligible"
    assert indexed["infeasible"]["gate"]["feasibility_status"] == "infeasible"
    assert indexed["infeasible"]["gate"]["admissibility_status"] == "inadmissible"
    assert indexed["infeasible"]["admission"]["status"] == "not_eligible"
    assert indexed["invalid"]["search_status"] == "invalid"
    assert indexed["invalid"]["gate"]["admissibility_status"] == "invalid"
    assert indexed["fallback"]["execution_status"] == "fallback"
    assert indexed["fallback"]["gate"]["replay_status"] == "not_run"
    assert indexed["degraded"]["execution_status"] == "degraded"
    assert indexed["degraded"]["gate"]["replay_status"] == "not_run"
    assert indexed["search-failed"]["search_status"] == "failed"
    assert indexed["search-failed"]["raw_candidate"]["execution_status"] == "failed"
    assert adapters.calls["admit"] == 0
    assert result["stop"]["reason"] == "no_new_admissible_counterexample_under_budget"


def test_optimization_improvement_is_recorded_and_maximum_round_budget_is_explicit(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path, stop_on_no_discovery=False)
    adapters = FixtureAdapters(
        scores_by_round={1: [0.7, 0.5, None], 2: [0.8, 0.5, None]},
        candidates_by_round={
            round_number: [
                _candidate(f"none-{round_number}-a", planner_outcome="no_failure"),
                _candidate(f"none-{round_number}-b", planner_outcome="no_failure"),
            ]
            for round_number in (1, 2)
        },
    )

    result = run_coevolution(config, adapters.bundle())

    assert result["status"] == "budget_exhausted"
    assert result["stop"]["reason"] == "maximum_round_budget_exhausted"
    assert result["rounds"][1]["summary"]["selection_improves_over_prior_best"] is True
    assert result["rounds"][1]["summary"]["optimizer_improves_over_baseline"] is True


def test_optimization_plateau_is_a_separate_configured_stop(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        maximum_rounds=4,
        stop_on_no_discovery=False,
        plateau_rounds=1,
    )
    adapters = FixtureAdapters(
        scores_by_round={1: [0.7, 0.5, None], 2: [0.7, 0.5, None]},
        candidates_by_round={
            round_number: [
                _candidate(f"plateau-{round_number}-a", planner_outcome="no_failure"),
                _candidate(f"plateau-{round_number}-b", planner_outcome="no_failure"),
            ]
            for round_number in (1, 2)
        },
    )

    result = run_coevolution(config, adapters.bundle())

    assert result["status"] == "complete"
    assert result["stop"]["reason"] == "planner_optimization_plateau"
    assert result["stop"]["consecutive_rounds_without_new_best"] == 1
    assert len(result["rounds"]) == 2


def test_infrastructure_failure_is_persisted_without_running_later_phases(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    adapters = FixtureAdapters(fail_phase="optimization")

    result = run_coevolution(config, adapters.bundle())

    assert result["status"] == "diagnostic"
    assert result["stop"]["reason"] == "infrastructure_failure"
    assert result["rounds"][0]["failure"]["phase"] == "optimization"
    failure = json.loads(
        (
            Path(config.parent / "run-output" / "round_001" / "phases" / "optimization.json")
        ).read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert failure["output"]["error_type"] == "OSError"
    assert adapters.calls == Counter({"optimize": 1})


def test_resume_reuses_completed_phases_and_rejects_changed_phase_digest(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_coevolution_config(config_path)
    adapters = FixtureAdapters()
    first = run_coevolution(config, adapters.bundle())
    calls_after_first_run = adapters.calls.copy()

    resumed = run_coevolution(config, adapters.bundle(), resume=True)

    assert resumed == first
    assert adapters.calls == calls_after_first_run

    optimizer_path = config.output_dir / "round_001" / "phases" / "optimization.json"
    optimizer_phase = json.loads(optimizer_path.read_text(encoding="utf-8"))
    planner_config = Path(optimizer_phase["output"]["selected_planner"]["config_path"])
    original_planner_bytes = planner_config.read_bytes()
    planner_config.write_text("planner_id: changed\n", encoding="utf-8")
    with pytest.raises(CoevolutionError, match="selected planner config digest mismatch"):
        run_coevolution(config, adapters.bundle(), resume=True)
    planner_config.write_bytes(original_planner_bytes)

    phase_path = config.output_dir / "round_001" / "phases" / "optimization.json"
    payload = json.loads(phase_path.read_text(encoding="utf-8"))
    payload["output"]["selection_tuple"][0] = 0.123
    phase_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(CoevolutionError, match="phase artifact digest changed"):
        run_coevolution(config, adapters.bundle(), resume=True)
    assert adapters.calls == calls_after_first_run


def test_resume_after_interruption_reuses_prior_rounds_and_does_not_retry_running_phase(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path, stop_on_no_discovery=False)

    class InterruptSecondRound(FixtureAdapters):
        """Simulate a process interruption after round one has committed."""

        def optimize(self, request, output_dir):
            if request.round_number == 2:
                self.calls["optimize"] += 1
                raise KeyboardInterrupt("simulated process interruption")
            return super().optimize(request, output_dir)

    adapters = InterruptSecondRound()
    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        run_coevolution(config, adapters.bundle())
    calls_after_interruption = adapters.calls.copy()
    assert calls_after_interruption["optimize"] == 2

    result = run_coevolution(config, adapters.bundle(), resume=True)

    assert result["status"] == "diagnostic"
    assert result["stop"]["reason"] == "infrastructure_failure"
    assert adapters.calls == calls_after_interruption
    assert all(phase["status"] == "complete" for phase in result["rounds"][0]["phases"].values())


def test_round_contract_rejects_a_minimum_of_one(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, minimum_rounds=1)

    with pytest.raises(ValueError, match="rounds.minimum must be an integer >= 2"):
        load_coevolution_config(config_path)
