"""Tests for the empty-world sweep seed guard and pedestrian removal (#9978)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Compile the supplied image so main exercises the production capture contract.
_SWEEP_PATH = Path(__file__).resolve().parents[2] / "scripts/validation/run_empty_world_sweep.py"
_SWEEP_SPEC = importlib.util.spec_from_file_location("captured_empty_world_sweep", _SWEEP_PATH)
sweep = importlib.util.module_from_spec(_SWEEP_SPEC)
sweep._SWEEP_SOURCE_BYTES = _SWEEP_PATH.read_bytes()
exec(compile(sweep._SWEEP_SOURCE_BYTES, str(_SWEEP_PATH), "exec"), sweep.__dict__)  # noqa: S102

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("seeds", [[1001], [1001, 1002], [1030], [1001, 1030]])
def test_seed_guard_accepts_dev_seeds(seeds: list[int]) -> None:
    """Development seeds 1001-1030 pass unchanged."""
    assert sweep.assert_dev_seeds(seeds) == seeds


@pytest.mark.parametrize(
    "seeds",
    [[111], [140], [125], [1000], [1031], [1001, 111], [], [True], ["x"], [1001.5]],
)
def test_seed_guard_rejects_everything_else(seeds: list[object]) -> None:
    """Holdout 111-140, out-of-range, empty and non-integer seeds abort."""
    with pytest.raises(sweep.SeedGuardError):
        sweep.assert_dev_seeds(seeds)


def test_main_aborts_on_holdout_seed_before_touching_anything(tmp_path: Path) -> None:
    """The CLI rejects a holdout seed before verifying the head or running a campaign."""
    with pytest.raises(sweep.SeedGuardError):
        sweep.main(["--head-sha", "HEAD", "--seeds", "111", "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_remove_pedestrians_clears_every_source_and_sets_dev_seeds() -> None:
    """Density, single pedestrians and groups are removed; the rest is kept."""
    scenario = {
        "name": "fixture",
        "map_id": "classic_cross_trap",
        "simulation_config": {
            "max_episode_steps": 600,
            "ped_density": 0.08,
            "population_size": 4,
            "goal_completion_policy": "goal_zone_entry_v1",
        },
        "single_pedestrians": [{"id": "p1"}],
        "social_groups": [{"id": "g1"}],
        "seeds": [131, 132],  # seed-holdout: synthetic-fixture
    }
    out = sweep.remove_pedestrians(scenario, [1001, 1002])
    assert sweep.pedestrian_residue(out) == []
    assert out["simulation_config"]["ped_density"] == 0.0
    assert out["simulation_config"]["max_episode_steps"] == 600
    assert out["simulation_config"]["goal_completion_policy"] == "goal_zone_entry_v1"
    assert out["single_pedestrians"] == []
    assert out["social_groups"] == []
    assert out["seeds"] == [1001, 1002]
    assert scenario["single_pedestrians"] == [{"id": "p1"}]  # input untouched
    assert scenario["seeds"] == [131, 132]  # seed-holdout: synthetic-fixture


def test_remove_pedestrians_refuses_holdout_seeds() -> None:
    """Pedestrian removal cannot smuggle in evaluation seeds."""
    with pytest.raises(sweep.SeedGuardError):
        sweep.remove_pedestrians({"name": "x", "simulation_config": {}}, [111, 112])


def test_residue_detects_a_scenario_that_still_has_pedestrians() -> None:
    """A scenario that was not passed through the removal is reported."""
    residue = sweep.pedestrian_residue(
        {"simulation_config": {"ped_density": 0.05}, "single_pedestrians": [{"id": "a"}]}
    )
    assert "ped_density != 0" in residue
    assert "single_pedestrians present" in residue
    assert "runtime map-pedestrian removal flag missing" in residue


def test_classify_outcome_prefers_flags_over_ambiguous_reason() -> None:
    """A horizon run with reason 'terminated' is a timeout, not a generic termination."""
    row = {"termination_reason": "terminated", "outcome": {"timeout_event": True}, "metrics": {}}
    assert sweep.classify_outcome(row) == "timeout"
    assert sweep.classify_outcome({"metrics": {"success": True}}) == "success"
    assert sweep.classify_outcome({"outcome": {"collision_event": True}}) == "collision"


def test_width_diagnostic_does_not_inherit_historical_snqi_anchors(tmp_path):
    """Actor-free v2 diagnostics must reach metrics without incompatible v1 anchors."""
    import yaml

    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

    cfg_path, _ = sweep.build_derived_inputs(
        "width",
        seeds=[1001],
        arms=["goal"],
        scenarios_filter=None,
        workers=1,
        out_dir=tmp_path,
        step_trace=True,
    )
    payload = yaml.safe_load(cfg_path.read_text())
    assert payload["snqi_weights"] is None
    assert payload["snqi_baseline"] is None
    cfg = load_campaign_config(cfg_path)
    assert cfg.snqi_weights_path is None
    assert cfg.snqi_baseline_path is None
    source = yaml.safe_load((REPO_ROOT / sweep.SUITES["width"]).read_text())
    assert source["snqi_weights"] is not None
    assert source["snqi_baseline"] is not None


@pytest.mark.parametrize(
    ("arm_key", "profile"),
    [
        ("predictive_mppi", "configs/algos/predictive_mppi_release_v0_0_8.yaml"),
        ("prediction_planner", "configs/algos/prediction_planner_release_v0_0_8.yaml"),
    ],
)
def test_generated_width_predictive_arms_use_published_profiles(tmp_path, arm_key, profile):
    """Generated predictive arms select the published eight-step release profiles."""
    import yaml

    cfg_path, _ = sweep.build_derived_inputs(
        "width",
        seeds=[1001],
        arms=[arm_key],
        scenarios_filter=None,
        workers=1,
        out_dir=tmp_path,
        step_trace=False,
    )
    arm = yaml.safe_load(cfg_path.read_text())["planners"][0]
    assert arm["algo_config"] == profile
    settings = yaml.safe_load((REPO_ROOT / arm["algo_config"]).read_text())
    assert settings["predictive_horizon_steps"] == 8
    assert settings["predictive_rollout_dt"] == 0.1
    assert settings["predictive_adaptive_horizon_enabled"] is False
    assert settings["predictive_horizon_boost_steps"] == 0
    if arm_key == "predictive_mppi":
        assert settings["horizon_steps"] == 8
        assert settings["rollout_dt"] == 0.1


def _provenance_sweep(tmp_path, monkeypatch):
    """Load a private source image and isolate campaign I/O from provenance checks."""
    import importlib.util

    from robot_sf.benchmark import camera_ready_campaign

    script = tmp_path / "run_empty_world_sweep.py"
    loaded = Path(sweep.__file__).read_bytes()
    script.write_bytes(loaded)
    spec = importlib.util.spec_from_file_location("provenance_sweep", script)
    module = importlib.util.module_from_spec(spec)
    module._SWEEP_SOURCE_BYTES = loaded
    exec(compile(loaded, str(script), "exec"), module.__dict__)  # noqa: S102
    monkeypatch.setattr(module, "verify_head", lambda _: "a" * 40)
    monkeypatch.setattr(module, "build_derived_inputs", lambda *_, **__: (tmp_path / "cfg", []))
    monkeypatch.setattr(module, "_preflight_receipt_inputs", lambda _: None)
    monkeypatch.setattr(camera_ready_campaign, "load_campaign_config", lambda _: object())
    monkeypatch.setattr(module, "load_rows", lambda _: [])
    monkeypatch.setattr(module, "reconcile_execution", lambda *_, **__: ([], {"complete": True}))
    return module, script, camera_ready_campaign


@pytest.mark.parametrize("replace_in_suite", ["main", "width"])
def test_loaded_harness_replaced_during_run_is_not_certified(
    tmp_path, monkeypatch, replace_in_suite
):
    """Loaded A cannot certify replacement B, even when B matches the manifest."""
    import hashlib
    import json

    module, script, campaign = _provenance_sweep(tmp_path, monkeypatch)
    loaded = script.read_bytes()
    replacement = loaded + b"\n# Replacement execution image.\n"
    out = tmp_path / "out"
    out.mkdir()
    (out / "harness.json").write_text(
        json.dumps(
            {"runtime_source_sha": "a" * 40, "harness_sha256": hashlib.sha256(loaded).hexdigest()}
        )
    )
    called = []

    def run(_cfg, **kwargs):
        suite = kwargs["label"].removeprefix("empty_world_")
        called.append(suite)
        if suite == replace_in_suite:
            script.write_bytes(replacement)
            # Match the approved on-disk replacement, as in the review probe.
            (out / "harness.json").write_text(
                json.dumps(
                    {
                        "runtime_source_sha": "a" * 40,
                        "harness_sha256": hashlib.sha256(replacement).hexdigest(),
                    }
                )
            )
        return {}

    monkeypatch.setattr(campaign, "run_campaign", run)
    with pytest.raises(RuntimeError, match="sweep harness.*changed"):
        module.main(["--head-sha", "HEAD", "--output-dir", str(out)])
    assert called == (["main"] if replace_in_suite == "main" else ["main", "width"])
    assert not (out / f"execution_{replace_in_suite}.json").exists()
    if replace_in_suite == "width":
        assert (
            json.loads((out / "execution_main.json").read_text())["sweep_harness_sha256"]
            == hashlib.sha256(loaded).hexdigest()
        )


@pytest.mark.parametrize("fault", [None, "missing", "digest", "head", "suite", "seeds", "manifest"])
def test_reanalysis_preserves_or_refuses_original_provenance(tmp_path, monkeypatch, fault):
    """Reanalysis cannot manufacture or overwrite historical execution identity."""
    import hashlib
    import json

    module, script, campaign = _provenance_sweep(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    meta = {
        "suite": "main",
        "head_sha": "a" * 40,
        "sweep_harness_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "seeds": list(module.DEFAULT_SEEDS),
        "workers": 3,
        "runtime_s": 12.5,
        "episodes": 0,
        "complete": True,
    }
    if fault in {"digest", "head", "suite", "seeds"}:
        meta[
            {
                "digest": "sweep_harness_sha256",
                "head": "head_sha",
                "suite": "suite",
                "seeds": "seeds",
            }[fault]
        ] = None
    receipt = out / "execution_main.json"
    if fault != "missing":
        receipt.write_text(json.dumps(meta))
    before = receipt.read_bytes() if receipt.exists() else None
    if fault == "manifest":
        (out / "harness.json").write_text(
            json.dumps({"runtime_source_sha": "a" * 40, "harness_sha256": "b" * 64})
        )
    monkeypatch.setattr(
        campaign, "run_campaign", lambda *_, **__: pytest.fail("reanalysis collected")
    )
    args = ["--head-sha", "HEAD", "--suite", "main", "--summarize-only", "--output-dir", str(out)]
    if fault:
        with pytest.raises(ValueError, match="original execution provenance"):
            module.main(args)
        assert (receipt.read_bytes() if receipt.exists() else None) == before
    else:
        assert module.main(args) == 0
        assert json.loads(receipt.read_text()) == meta


@pytest.mark.parametrize("summarize_only", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_import_entry_cannot_certify_unexecuted_source(
    tmp_path, monkeypatch, summarize_only, captured
):
    """Importing A while restoring selected B must refuse collection and reanalysis."""
    import hashlib
    import importlib.util
    import json
    import sys

    from robot_sf.benchmark import camera_ready_campaign

    script = tmp_path / "run_empty_world_sweep.py"
    selected = Path(sweep.__file__).read_bytes()
    loaded = selected + b'\nIMPORT_IMAGE_MARKER = "loaded-image"\n'
    script.write_bytes(loaded)
    spec = importlib.util.spec_from_file_location("import_entry_sweep", script)
    module = importlib.util.module_from_spec(spec)
    previous_profile = sys.getprofile()
    switched = []

    def restore_selected(frame, event, _arg):
        if event == "call" and frame.f_code.co_filename == str(script):
            script.write_bytes(selected)
            switched.append(True)
            sys.setprofile(previous_profile)

    sys.setprofile(restore_selected)
    try:
        if captured:
            module._SWEEP_SOURCE_BYTES = loaded
            exec(compile(loaded, str(script), "exec"), module.__dict__)  # noqa: S102
        else:
            spec.loader.exec_module(module)
    finally:
        sys.setprofile(previous_profile)
    assert switched == [True]
    assert module.IMPORT_IMAGE_MARKER == "loaded-image"
    assert script.read_bytes() == selected
    assert loaded != selected

    # Retain selected-B provenance, as in the normal-path review witness.
    out = tmp_path / "out"
    out.mkdir()
    selected_digest = hashlib.sha256(selected).hexdigest()
    (out / "harness.json").write_text(
        json.dumps({"runtime_source_sha": "a" * 40, "harness_sha256": selected_digest})
    )
    (out / "execution_main.json").write_text(
        json.dumps(
            {
                "suite": "main",
                "head_sha": "a" * 40,
                "sweep_harness_sha256": selected_digest,
                "seeds": list(module.DEFAULT_SEEDS),
            }
        )
    )
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    monkeypatch.setattr(module, "verify_head", lambda _: "a" * 40)
    monkeypatch.setattr(module, "build_derived_inputs", lambda *_, **__: (tmp_path / "cfg", []))
    monkeypatch.setattr(module, "_preflight_receipt_inputs", lambda _: None)
    monkeypatch.setattr(camera_ready_campaign, "load_campaign_config", lambda _: object())
    monkeypatch.setattr(camera_ready_campaign, "run_campaign", lambda *_, **__: {})
    monkeypatch.setattr(module, "load_rows", lambda _: [])
    monkeypatch.setattr(module, "reconcile_execution", lambda *_, **__: ([], {"complete": True}))
    args = ["--head-sha", "HEAD", "--suite", "main", "--output-dir", str(out)]
    if summarize_only:
        args.append("--summarize-only")
    # Captured A must reject B; an ordinary import must refuse before any I/O.
    error = "sweep harness.*changed" if captured else "requires a captured source image"
    # Reanalysis translates captured-image drift into a provenance conflict.
    error_type = ValueError if captured and summarize_only else RuntimeError
    if captured and summarize_only:
        error = "original execution provenance"
    with pytest.raises(error_type, match=error):
        module.main(args)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_real_campaign_replacement_cannot_issue_execution_metadata(tmp_path, monkeypatch):
    """The real goal campaign runs, but replacement bytes cannot become its identity."""
    import importlib.util
    import sys

    from robot_sf.benchmark import camera_ready_campaign

    script = tmp_path / "run_empty_world_sweep.py"
    loaded = Path(sweep.__file__).read_bytes()
    script.write_bytes(loaded)
    spec = importlib.util.spec_from_file_location("real_provenance_sweep", script)
    module = importlib.util.module_from_spec(spec)
    module._SWEEP_SOURCE_BYTES = loaded
    exec(compile(loaded, str(script), "exec"), module.__dict__)  # noqa: S102
    monkeypatch.setattr(module, "REPO_ROOT", REPO_ROOT)
    out = tmp_path / "out"
    previous_profile = sys.getprofile()
    switched = []

    def replace(frame, event, _arg):
        if event == "call" and frame.f_code is camera_ready_campaign.run_campaign.__code__:
            script.write_bytes(loaded + b"\n# Replaced while the campaign is executing.\n")
            switched.append(True)
            sys.setprofile(previous_profile)

    sys.setprofile(replace)
    try:
        with pytest.raises(RuntimeError, match="sweep harness.*changed"):
            module.main(
                [
                    "--head-sha",
                    "HEAD",
                    "--suite",
                    "main",
                    "--arms",
                    "goal",
                    "--scenarios",
                    "classic_bottleneck_low",
                    "--seeds",
                    str(module.DEV_SEED_MIN),
                    "--workers",
                    "1",
                    "--output-dir",
                    str(out),
                ]
            )
    finally:
        sys.setprofile(previous_profile)
    assert switched == [True]
    assert len(module.load_rows(out / "campaigns" / "empty_world_main")) == 1
    assert not (out / "execution_main.json").exists()


def test_direct_cli_hashes_the_buffer_it_executes(tmp_path):
    """An edit during initial imports cannot relabel the direct CLI's executed code."""
    import hashlib
    import os
    import subprocess
    import sys

    script = tmp_path / "run_empty_world_sweep.py"
    loaded = Path(sweep.__file__).read_bytes() + (
        b"\nif __name__ == '__main__':\n    print('EXECUTED_MARKER=loaded')\n"
    )
    script.write_bytes(loaded)
    replacement = (
        loaded.replace(b"EXECUTED_MARKER=loaded", b"EXECUTED_MARKER=replacement")
        + b"\n# Replacement before the direct CLI starts collection.\n"
    )
    replacement_file = tmp_path / "replacement.py"
    replacement_file.write_bytes(replacement)
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import os, sys; from pathlib import Path\n"
        "def replace(frame, event, arg):\n"
        "    if event == 'call' and frame.f_code.co_filename == os.environ['PROBE_SOURCE']:\n"
        "        sys.setprofile(None)\n"
        "        Path(os.environ['PROBE_SOURCE']).write_bytes(Path(os.environ['PROBE_REPLACEMENT']).read_bytes())\n"
        "sys.setprofile(replace)\n"
    )
    # Stop after loading, before main, to inspect the image whose functions exist.
    # The production main entrypoint is replaced only in this private test image.
    for target in (script, replacement_file):
        target.write_bytes(
            target.read_bytes().replace(
                b"raise SystemExit(main())", b"print('LOADED_IMAGE=' + SWEEP_HARNESS_SHA256)"
            )
        )
    loaded = script.read_bytes()
    replacement = replacement_file.read_bytes()
    env = dict(
        os.environ,
        PYTHONPATH=str(guard),
        PROBE_SOURCE=str(script),
        PROBE_REPLACEMENT=str(replacement_file),
    )
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, check=True
    )
    images = [
        line.split("=", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("LOADED_IMAGE=")
    ]
    assert images and set(images) == {hashlib.sha256(replacement).hexdigest()}
    assert "EXECUTED_MARKER=replacement" in result.stdout
    assert "EXECUTED_MARKER=loaded" not in result.stdout
    # Compare an executed marker, not just a second read of a pathname.
    assert loaded != replacement
