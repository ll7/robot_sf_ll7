"""Fixture-only contract tests for the issue #9653 round coordinator."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.adversarial import coevolution as coevolution_module
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
            "sampler": "random",
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
        "scenario": {
            "name": f"scenario-{candidate_id}",
            "seed": 101,
            "geometry": {"points": [[0.0, 1.0], [2.0, 3.0]]},
        },
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
        self.previous_planners: dict[int, dict[str, Any] | None] = {}
        self.admitted_case_ids: list[str] = []

    def optimize(self, request, output_dir):
        self.calls["optimize"] += 1
        self.previous_planners[request.round_number] = request.to_json()[
            "previous_selected_planner"
        ]
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
            "baseline_planner": (
                {
                    "planner_id": request.previous_selected_planner["planner_id"],
                    "config_sha256": request.previous_selected_planner["config_sha256"],
                }
                if request.previous_selected_planner is not None
                else None
            ),
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
            "sampler": request.falsification_sampler,
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
    assert adapters.previous_planners[1] is None
    assert adapters.previous_planners[2]["planner_id"] == "fixture-planner-r1"
    assert (
        adapters.previous_planners[2]["config_sha256"]
        == hashlib.sha256(b"planner_id: fixture-planner-r1\n").hexdigest()
    )
    eval_path = Path(
        config.parent / "run-output" / "round_002" / "phases" / "challenge_evaluation.json"
    )
    persisted = json.loads(eval_path.read_text(encoding="utf-8"))
    assert [row["case_id"] for row in persisted["output"]["regression_rows"]] == [case_id]
    assert result["rounds"][1]["summary"]["regression_case_count"] == 1
    round_two_input = json.loads(
        (config.parent / "run-output" / "round_002" / "round_input.json").read_text(
            encoding="utf-8"
        )
    )
    first_selected = json.loads(
        (config.parent / "run-output" / "round_001" / "phases" / "optimization.json").read_text(
            encoding="utf-8"
        )
    )["output"]["selected_planner"]
    assert round_two_input["previous_selected_planner"] == first_selected
    second_optimizer = json.loads(
        (config.parent / "run-output" / "round_002" / "phases" / "optimization.json").read_text(
            encoding="utf-8"
        )
    )["output"]
    assert second_optimizer["baseline_planner"] == {
        "planner_id": first_selected["planner_id"],
        "config_sha256": first_selected["config_sha256"],
    }


def test_adapter_mutation_cannot_change_selected_planner_or_round_carry(tmp_path: Path) -> None:
    config = _write_config(tmp_path)

    class MutatingAdapters(FixtureAdapters):
        def __init__(self):
            super().__init__()
            self.seen_planner_ids: list[tuple[str, int, str]] = []

        def _mutate(self, phase: str, request, planner):
            self.seen_planner_ids.append((phase, request.round_number, planner["planner_id"]))
            planner["planner_id"] = f"mutated-by-{phase}"

        def evaluate_challenges(self, request, planner, output_dir):
            self._mutate("evaluator", request, planner)
            return super().evaluate_challenges(request, planner, output_dir)

        def falsify(self, request, planner, output_dir):
            self._mutate("falsifier", request, planner)
            return super().falsify(request, planner, output_dir)

        def verify_discovery(self, request, planner, candidate, output_dir):
            self._mutate("verifier", request, planner)
            return super().verify_discovery(request, planner, candidate, output_dir)

        def admit_case(self, request, planner, case, output_dir):
            self._mutate("admitter", request, planner)
            return super().admit_case(request, planner, case, output_dir)

    adapters = MutatingAdapters()
    result = run_coevolution(config, adapters.bundle())

    assert result["status"] == "complete"
    assert adapters.seen_planner_ids == [
        ("evaluator", 1, "fixture-planner-r1"),
        ("falsifier", 1, "fixture-planner-r1"),
        ("verifier", 1, "fixture-planner-r1"),
        ("admitter", 1, "fixture-planner-r1"),
        ("evaluator", 2, "fixture-planner-r2"),
        ("falsifier", 2, "fixture-planner-r2"),
    ]
    output_dir = config.parent / "run-output"
    first_optimizer = json.loads(
        (output_dir / "round_001" / "phases" / "optimization.json").read_text(encoding="utf-8")
    )["output"]
    second_input = json.loads(
        (output_dir / "round_002" / "round_input.json").read_text(encoding="utf-8")
    )
    second_optimizer = json.loads(
        (output_dir / "round_002" / "phases" / "optimization.json").read_text(encoding="utf-8")
    )["output"]
    expected_prior = first_optimizer["selected_planner"]
    assert expected_prior["planner_id"] == "fixture-planner-r1"
    assert second_input["previous_selected_planner"] == expected_prior
    assert second_optimizer["baseline_planner"] == {
        "planner_id": expected_prior["planner_id"],
        "config_sha256": expected_prior["config_sha256"],
    }
    first_discovery = json.loads(
        (output_dir / "round_001" / "phases" / "discovery_admission.json").read_text(
            encoding="utf-8"
        )
    )["output"]
    assert first_discovery["rows"][0]["raw_candidate"]["target_planner_id"] == (
        "fixture-planner-r1"
    )
    assert (
        first_discovery["newly_admitted_cases"][0]["source_evidence"]["target_planner_id"]
        == "fixture-planner-r1"
    )


def test_round_two_rejects_an_optimizer_that_resets_its_baseline(tmp_path: Path) -> None:
    config = _write_config(tmp_path)

    class ResetBaseline(FixtureAdapters):
        def optimize(self, request, output_dir):
            result = super().optimize(request, output_dir)
            if request.round_number == 2:
                result["baseline_planner"]["planner_id"] = "static-config-baseline"
            return result

    adapters = ResetBaseline()
    result = run_coevolution(config, adapters.bundle())

    assert result["status"] == "diagnostic"
    assert result["rounds"][1]["failure"]["phase"] == "optimization"
    assert (
        "does not match the prior round's selected planner"
        in result["rounds"][1]["failure"]["error"]
    )
    assert adapters.previous_planners[2]["planner_id"] == "fixture-planner-r1"
    assert adapters.calls == Counter(
        {"optimize": 2, "evaluate": 1, "falsify": 1, "verify": 1, "admit": 1}
    )


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


def test_falsification_sampler_is_required_and_must_match_frozen_round_input(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["falsification"].pop("sampler")
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="falsification.sampler must be a non-empty string"):
        load_coevolution_config(config_path)

    mismatch_dir = tmp_path / "mismatch"
    mismatch_dir.mkdir()
    config_path = _write_config(mismatch_dir)

    class WrongSampler(FixtureAdapters):
        def falsify(self, request, selected_planner, output_dir):
            result = super().falsify(request, selected_planner, output_dir)
            result["sampler"] = "coordinate"
            return result

    adapters = WrongSampler()
    result = run_coevolution(config_path, adapters.bundle())

    assert result["status"] == "diagnostic"
    assert result["rounds"][0]["failure"]["phase"] == "falsification"
    assert "sampler differs from the frozen round input" in result["rounds"][0]["failure"]["error"]
    assert adapters.calls == Counter({"optimize": 1, "evaluate": 1, "falsify": 1})
    round_input = json.loads(
        (mismatch_dir / "run-output" / "round_001" / "round_input.json").read_text(encoding="utf-8")
    )
    assert round_input["search_execution"]["sampler"] == "random"
    phase = json.loads(
        (mismatch_dir / "run-output" / "round_001" / "phases" / "falsification.json").read_text(
            encoding="utf-8"
        )
    )
    assert phase["output"]["raw_output"]["sampler"] == "coordinate"


def test_nested_regression_case_payload_is_detached_and_deeply_immutable(tmp_path: Path) -> None:
    config = load_coevolution_config(_write_config(tmp_path))

    class MutatingEvaluator(FixtureAdapters):
        def evaluate_challenges(self, request, planner, output_dir):
            if request.round_number == 2:
                case = request.regression_cases[0]
                before = request.to_json()
                assert case.scenario is not None
                with pytest.raises(TypeError):
                    case.scenario["geometry"]["points"][0][0] = -99.0
                assert request.previous_selected_planner is not None
                with pytest.raises(TypeError):
                    request.previous_selected_planner["planner_id"] = "mutated"
                detached = case.to_json()
                detached["scenario"]["geometry"]["points"][0][0] = -99.0
                assert request.to_json() == before
            return super().evaluate_challenges(request, planner, output_dir)

    result = run_coevolution(config, MutatingEvaluator().bundle())

    assert result["status"] == "complete"
    round_input = json.loads(
        (config.output_dir / "round_002" / "round_input.json").read_text(encoding="utf-8")
    )
    points = round_input["regression_cases"][0]["scenario"]["geometry"]["points"]
    assert points == [[0.0, 1.0], [2.0, 3.0]]


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


def test_resume_recovers_manifest_pair_after_crash_between_atomic_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_coevolution_config(_write_config(tmp_path))
    adapters = FixtureAdapters()
    atomic_write = coevolution_module._atomic_write_json
    crashed = False
    paired_run_manifest_writes = 0

    def crash_before_run_manifest(path: Path, payload: Any) -> str:
        nonlocal crashed, paired_run_manifest_writes
        if (
            path == config.output_dir / "run_manifest.json"
            and (config.output_dir / ".manifest_transaction.json").is_file()
        ):
            paired_run_manifest_writes += 1
            if paired_run_manifest_writes == 2:
                crashed = True
                raise KeyboardInterrupt("simulated crash between manifest replacements")
        return atomic_write(path, payload)

    with monkeypatch.context() as patcher:
        patcher.setattr(coevolution_module, "_atomic_write_json", crash_before_run_manifest)
        with pytest.raises(KeyboardInterrupt, match="between manifest replacements"):
            run_coevolution(config, adapters.bundle())

    assert crashed
    assert paired_run_manifest_writes == 2
    assert adapters.calls == Counter()
    transaction_path = config.output_dir / ".manifest_transaction.json"
    assert transaction_path.is_file()
    round_before_recovery = json.loads(
        (config.output_dir / "round_001" / "round_manifest.json").read_text(encoding="utf-8")
    )
    run_before_recovery = json.loads(
        (config.output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert round_before_recovery["phases"]["optimization"]["status"] == "running"
    assert run_before_recovery["rounds"][0]["phases"] == {}
    assert run_before_recovery["rounds"][0] != round_before_recovery

    result = run_coevolution(config, adapters.bundle(), resume=True)

    assert result["status"] == "diagnostic"
    assert result["stop"]["reason"] == "infrastructure_failure"
    assert adapters.calls == Counter()
    assert not transaction_path.exists()
    round_after_recovery = json.loads(
        (config.output_dir / "round_001" / "round_manifest.json").read_text(encoding="utf-8")
    )
    run_after_recovery = json.loads(
        (config.output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_after_recovery["rounds"] == [round_after_recovery]


def test_round_contract_rejects_a_minimum_of_one(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, minimum_rounds=1)

    with pytest.raises(ValueError, match="rounds.minimum must be an integer >= 2"):
        load_coevolution_config(config_path)
