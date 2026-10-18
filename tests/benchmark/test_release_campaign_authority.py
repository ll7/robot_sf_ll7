"""D-083 release-template admission; real input bytes, no reset or step."""
# seed-holdout: synthetic-fixture begin

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark import release_protocol as protocol
from robot_sf.benchmark import spawn_preflight as spawn
from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config
from robot_sf.benchmark.seed_bands import EVAL_SEEDS_0_0_8, RETIRED_EVAL_SEEDS_0_0_7
from tests.benchmark.test_sealed_source_pins import (
    EnvironmentAttempt,
    materialize,
    worker_stub,
)
from tests.benchmark.test_sealed_source_pins import (
    sealed_repository as _sealed_repository,
)

sealed_repository = _sealed_repository
ROOT = Path(__file__).resolve().parents[2]
AUTHORED = (
    "configs/benchmarks/"
    "paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml"
)
IDENTITY_TEMPLATE = "configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml"


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    """A simulation attempt escapes the guard's ordinary exception handling."""
    from robot_sf.gym_env import environment_factory
    from robot_sf.gym_env.robot_env import RobotEnv

    def abort(*_args, **_kwargs):
        raise EnvironmentAttempt("D-083 witness attempted environment/reset/step")

    for name in vars(environment_factory):
        if name.startswith("make_") and callable(getattr(environment_factory, name)):
            monkeypatch.setattr(environment_factory, name, abort)
    for name in ("__init__", "reset", "step"):
        monkeypatch.setattr(RobotEnv, name, abort)
    monkeypatch.setattr(spawn, "make_robot_env", abort)


def test_d083_authored_template_resolves_exact_sealed_seed_file():
    """The real selected campaign transports the sealed tuple, without executing it."""
    cfg = load_campaign_config(ROOT / AUTHORED, repository_root=ROOT)
    scenarios = _load_campaign_scenarios(cfg, repository_root=ROOT)
    assert tuple(protocol._resolved_seed_inventory(scenarios)) == EVAL_SEEDS_0_0_8
    assert cfg.seed_policy.seed_set == "release_eval_0_0_8"
    assert cfg.seed_policy.seed_sets_path == ROOT / "configs/benchmarks/seed_sets_0_0_8.yaml"
    assert cfg.seed_policy.seeds == ()
    assert len(scenarios) == 48
    assert cfg.protocol_version == "0.0.8"
    assert cfg.horizon is None
    payload = yaml.safe_load((ROOT / IDENTITY_TEMPLATE).read_text())
    selected = ((ROOT / IDENTITY_TEMPLATE).parent / payload["canonical_campaign_config"]).resolve()
    assert selected == ROOT / AUTHORED


def test_d083_materialized_selected_template_passes_guard_without_execution(
    sealed_repository, monkeypatch
):
    """Real source-bound admission reaches recording workers; retired input cannot."""
    repo = sealed_repository
    template = repo / IDENTITY_TEMPLATE
    payload = yaml.safe_load(template.read_text())
    assert (template.parent / payload["canonical_campaign_config"]).resolve() == repo / AUTHORED
    manifest = materialize(repo, "main")
    assert manifest.canonical_campaign_config_path == repo / AUTHORED
    assert manifest.resolved_seeds == EVAL_SEEDS_0_0_8
    assert (
        protocol.sealed_seed_execution_problem(
            manifest, EVAL_SEEDS_0_0_8, source_commit=manifest.source_sha, repository_root=repo
        )
        is None
    )
    reached = worker_stub(monkeypatch)
    report = spawn.run_manifest_preflight(manifest, workers=1, source_commit=manifest.source_sha)
    assert report["status"] == "valid", report
    assert len(reached) == 48
    assert all(tuple(seeds) == EVAL_SEEDS_0_0_8 for seeds in reached)

    # Preserve the valid freeze; only replace the synthetic refusal's seed payload.
    retired = replace(
        manifest,
        resolved_seeds=RETIRED_EVAL_SEEDS_0_0_7,
        seed_policy={"mode": "fixed-list", "seeds": list(RETIRED_EVAL_SEEDS_0_0_7)},
    )
    reached.clear()
    with pytest.raises(ValueError, match="retired evaluation seeds are forbidden for execution"):
        spawn.guard_manifest_execution(retired, source_commit=manifest.source_sha)
    refusal = spawn.run_manifest_preflight(retired, source_commit=manifest.source_sha)
    assert refusal["status"] == "invalid"
    assert "retired evaluation seeds are forbidden for execution" in refusal["input_error"]
    assert reached == []

    for old_name in (
        "paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate.yaml",
        "paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml",
    ):
        old = replace(
            manifest, canonical_campaign_config_path=repo / "configs/benchmarks" / old_name
        )
        assert "canonical repository paths" in protocol.sealed_seed_execution_problem(
            old, EVAL_SEEDS_0_0_8, repository_root=repo
        )


# seed-holdout: synthetic-fixture end
