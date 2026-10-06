"""Production reference and learned-family guards; no environment reset or step."""

import hashlib
import importlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from robot_sf.benchmark.result_provenance import build_execution_context_provenance
from robot_sf.benchmark.snqi import execution_context as gate
from robot_sf.evidence.writers import write_json
from tests.benchmark.test_snqi_execution_context import EVIDENCE
from tests.unit.benchmark.test_snqi_v2 import spec_files as _spec_files

spec_files = _spec_files


@pytest.mark.parametrize("missing", ["torch_version", "stable_baselines3_version", "both"])
def test_bound_reference_requires_recorded_learned_versions(tmp_path, missing):
    """An otherwise valid, digest-bound reference cannot omit learned-stack observations."""
    anchors = tmp_path / "anchors.v2.0.acquired.json"
    write_json(anchors, json.loads((EVIDENCE / anchors.name).read_bytes()))
    receipt = json.loads((EVIDENCE / "determinism-receipt.json").read_bytes())
    reference = build_execution_context_provenance()
    reference.update(torch_version="fixture-torch", stable_baselines3_version="fixture-sb3")
    receipt["execution_contexts"] = {"original": reference, "repeat": deepcopy(reference)}
    receipt_path = tmp_path / "determinism-receipt.json"
    proof = json.loads((EVIDENCE / "acquisition-proof.json").read_bytes())
    proof["anchors_sha256"] = hashlib.sha256(anchors.read_bytes()).hexdigest()

    def deliver():
        write_json(receipt_path, receipt)
        digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        proof["determinism_receipt"] = {"path": receipt_path.name, "sha256": digest}
        write_json(tmp_path / "acquisition-proof.json", proof)
        return {"determinism_receipt_path": receipt_path, "determinism_receipt_sha256": digest}

    assert gate.load_calibration_context(anchors, deliver()) == reference
    for field in ("torch_version", "stable_baselines3_version"):
        if missing in {field, "both"}:
            reference.pop(field)
    receipt["execution_contexts"]["repeat"] = deepcopy(reference)
    field = "torch_version" if missing == "both" else missing
    with pytest.raises(ValueError, match="calibration execution context missing " + field):
        gate.load_calibration_context(anchors, deliver())


# Independent catalog examples; none was guarded by ca9aeece's literal roster.
NEW_LEARNED_NAMES = (
    "sampling",
    "drl_vo",
    "drlvo",
    "drl-vo",
    "sonic_crowdnav",
    "sonic_gst",
    "sac",
    "distributional_rl",
    "qr_dqn",
    "crowdnav_height",
    "dr_mpc",
    "drmpc",
    "learned_prediction_mpc",
    "learned_short_horizon_mpc",
    "model_based_local_planner",
    "learned_prediction_planner",
    "hybrid_global_rl",
    "global_rl_local",
    "route_conditioned_rl",
    "hybrid_route_rl",
    "gensafenav_ours_gst",
    "gensafe_ours_gst",
    "ours_gst",
    "gensafenav_ours_gst_guarded",
    "gensafenav_gst_predictor_rand",
    "gensafenav_gst_predictor_rand_guarded",
    "hybrid_portfolio",
    "gap_prediction",
)


@pytest.mark.parametrize("algo", NEW_LEARNED_NAMES)
def test_registered_learned_family_cannot_bypass_worker_or_roster(monkeypatch, tmp_path, algo):
    reference = build_execution_context_provenance()
    monkeypatch.setenv(gate.CONTEXT_ENV, json.dumps(reference))
    observed = gate.admit_episode_context(" " + algo.upper() + " ")
    assert observed is not None, "registered learned alias bypassed admission: " + algo
    assert observed["cpu_model"] == reference["cpu_model"]
    reference["cpu_model"] = "mismatched fixture CPU"
    monkeypatch.setenv(gate.CONTEXT_ENV, json.dumps(reference))
    with pytest.raises(ValueError, match="execution context mismatch: cpu_model"):
        gate.admit_episode_context(" " + algo.upper() + " ")
    with pytest.raises(ValueError, match="census missing arms: " + algo):
        gate.verify_episode_contexts(tmp_path, reference, (algo,))


def test_readiness_alias_registration_flows_to_context_roster(monkeypatch):
    from robot_sf.benchmark import algorithm_readiness

    try:
        with monkeypatch.context() as registry:
            ppo = algorithm_readiness.get_algorithm_readiness("ppo")
            registry.setitem(
                algorithm_readiness._ALIAS_INDEX,
                "ppo",
                replace(ppo, aliases=(*ppo.aliases, "fixture-new-learned-alias")),
            )
            importlib.reload(gate)
            assert "fixture-new-learned-alias" in gate.LEARNED_ALGORITHMS
    finally:
        importlib.reload(gate)


@pytest.mark.parametrize("algo", ["planner_selector_v2", "planner_selector_v2_diagnostic"])
def test_selector_refuses_before_effective_context_and_children(monkeypatch, tmp_path, algo):
    from robot_sf.benchmark.map_runner_policies import map_runner_policy_resolution as resolution

    monkeypatch.setenv(gate.CONTEXT_ENV, json.dumps(build_execution_context_provenance()))
    monkeypatch.setattr(
        resolution,
        "_resolve_policy_search_candidate_runtime",
        lambda **_kw: (" " + algo.upper() + " ", {"candidate": "ppo"}),
    )
    reached = []
    monkeypatch.setattr(
        resolution,
        "_apply_planner_selector_v2_context",
        lambda *_a, **_kw: reached.append("selector context") or {},
    )
    with pytest.raises(ValueError, match="calibrated execution refuses planner_selector_v2"):
        resolution.resolve_episode_policy_runtime(
            default_algo="goal",
            algo_config_path=None,
            scenario={"name": "dev-fixture"},
            seed=1004,
        )
    assert reached == []

    with pytest.raises(ValueError, match="calibrated execution refuses planner_selector_v2"):
        gate.admit_episode_context(algo)
    monkeypatch.delenv(gate.CONTEXT_ENV)
    with pytest.raises(ValueError, match="calibrated execution refuses planner_selector_v2"):
        gate.verify_episode_contexts(tmp_path, build_execution_context_provenance(), (algo,))
    rows = tmp_path / "runs/goal/episodes.jsonl"
    rows.parent.mkdir(parents=True)
    write_json(rows, {"algo": algo, "seed": 1004}, indent=None)
    with pytest.raises(ValueError, match="calibrated execution refuses planner_selector_v2"):
        gate.verify_episode_contexts(tmp_path, build_execution_context_provenance(), ("goal",))
    # Diagnostic resolution outside calibrated production remains available.
    resolution.resolve_episode_policy_runtime(
        default_algo="goal",
        algo_config_path=None,
        scenario={"name": "dev-fixture"},
        seed=1004,
    )
    assert reached == ["selector context"]
    assert gate.admit_episode_context(algo) is None


@pytest.mark.parametrize("algo", ["planner_selector_v2", "planner_selector_v2_diagnostic"])
def test_production_selector_refuses_before_preflight(
    tmp_path, monkeypatch, capsys, spec_files, algo
):
    from scripts.tools import run_benchmark_release as runner
    from tests.benchmark import test_snqi_context_fix_round as support
    from tests.tools import test_run_benchmark_release as release_support

    original_manifest = release_support._manifest_fixture

    def manifest():
        result = original_manifest()
        result.planner_keys = (algo,)
        return result

    monkeypatch.setattr(release_support, "_manifest_fixture", manifest)
    original_main = runner.main

    class SelectorRefused(Exception):
        pass

    def refusing_main(args):
        assert original_main(args) == 2, "production selector reached preflight"
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == "snqi_v2_execution_context_refused"
        assert "calibrated execution refuses planner_selector_v2" in result["status_reason"]
        assert result["campaign_execution_status"] == "not_started"
        raise SelectorRefused

    monkeypatch.setattr(runner, "main", refusing_main)
    with pytest.raises(SelectorRefused):
        support.test_post_freeze_external_receipt_admits_same_source_through_release_cli(
            tmp_path,
            monkeypatch,
            capsys,
            spec_files,
        )
