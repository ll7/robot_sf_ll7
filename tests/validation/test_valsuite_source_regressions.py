"""Exercise actual campaign rows on synthetic traces; also runnable on pre-fix runner.

The old runner produces numerical values with different definitions or null onset.
The replacements are independent known answers, not import-error witnesses.
"""

import numpy as np
import pytest

from scripts.validation import pedestrian_validation_10074 as suite


def _fake_trace(case):
    def trace(state, segments, config, steps, **kwargs):
        if case == "V1":
            t = np.arange(201) * 0.1
            v = 1.3 * (1 - np.exp(-t / 0.54))
            v[t > 6] = 1.6
            p = np.zeros((201, 1, 2))
            p[1:, 0, 0] = np.cumsum(v[1:]) * 0.1
            speeds = v[1:, None]
        elif case == "V2":
            t = np.arange(51) * 0.1
            p = np.zeros((51, 1, 2))
            p[:, 0, 0] = 4 + t - 0.04 * t * t
            speeds = (1 - 0.08 * t[1:])[:, None]
        elif case == "V4":
            t = np.arange(201) * 0.1
            x = t[:, None] - 0.5 - 0.02 * np.arange(len(state))[None, :]
            p = np.stack([x, np.zeros_like(x)], axis=-1)
            speeds = np.ones((200, len(state)))
        elif case == "V5":
            t = np.arange(161) * 0.1
            p = np.zeros((161, 1, 2))
            p[:, 0, 0] = t
            p[:, 0, 1] = 0.75
            speeds = np.ones((160, 1))
        else:
            speed = 1.15
            t = np.arange(106) * 0.1
            x = t * speed
            p = np.zeros((len(t), len(state), 2))
            p[:, 0, 0] = x
            if len(state) > 1:
                p[:, 0, 1] = np.clip(x - 4, 0, 1) * 0.4
                p[:, 1, 0] = 12 - x
            speeds = np.ones((len(t) - 1, len(state))) * speed
        return p, speeds

    return trace


@pytest.mark.parametrize(
    "case,variant,key,expected",
    [
        ("V1", "native", "fitted_desired_speed_m_s", 1.3),
        ("V2", "0.9", "speed_drop_m_s", 0.32),
        ("V4", "2.4", "all_data_specific_flow_persons_m_s", 350 / (349 * 0.02 * 2.4)),
        ("V5", "diagnostic", "lateral_cm_to_edge_m", 0.5),
        ("V6", "1.15", "onset_m", 2.645),
    ],
)
def test_source_case_known_answer_replaces_wrong_or_missing_estimator(
    monkeypatch, case, variant, key, expected
):
    fake = _fake_trace(case)
    monkeypatch.setattr(suite.reused.harness, "simulate", fake)

    def source_trace(*args, **kwargs):
        p, v = fake(*args, **kwargs)
        return p, v, np.ones(p.shape[1])

    monkeypatch.setattr(suite, "protocol_simulate", source_trace, raising=False)
    row = suite.run_task((case, 1001, variant, 0.4, "baseline"))
    # Fallback fields make the pre-fix failure a concrete incorrect numeric answer.
    fallback = (
        row.get("speed_m_s")
        if case == "V1"
        else row.get("flow_persons_s")
        if case == "V4"
        else None
    )
    if case == "V6":
        # Known .05 rad/s onset: 2.645 m ahead of the x=6 m PoMD.
        assert row["onset_threshold_rad_s"] == 0.05
    assert row.get(key, fallback) == pytest.approx(expected, abs=1e-8), (
        f"{case} source quantity {key}"
    )


def test_all_cases_carry_pair_step_measure_not_initial_overlap_only(monkeypatch):
    fake = _fake_trace("V4")
    monkeypatch.setattr(suite.reused.harness, "simulate", fake)

    def source_trace(*args, **kwargs):
        p, v = fake(*args, **kwargs)
        return p, v, np.ones(p.shape[1])

    monkeypatch.setattr(suite, "protocol_simulate", source_trace, raising=False)
    row = suite.run_task(("V3", 1001, "1.0", 0.4, "baseline"))
    pair = row.get("pair_overlap", {}).get("non_group", {})
    assert pair.get("pair_steps") == 200 * 60 * 59 // 2
    assert pair["minimum_centre_distance_m"] == pytest.approx(0.02)


def test_gate_reports_censoring_contact_and_author_tolerances():
    row = {
        "case": "V1",
        "variant": "native",
        "seed": 1001,
        "fitted_desired_speed_m_s": 1.3,
        "fitted_tau_s": 0.54,
        "pair_overlap": {"all": {"below_2r_count": 0}},
    }
    assert suite.acceptance_gate([row])["exit_code"] == 0
    row["pair_overlap"]["all"]["below_2r_count"] = 1
    assert suite.acceptance_gate([row])["exit_code"] == 3
    row["fitted_desired_speed_m_s"] = None
    result = suite.acceptance_gate([row])
    assert result["exit_code"] == 3
    assert len(result["physical_violations"]) == 1
    assert len(result["measurement_missing"]) == 1


def test_saved_acquisition_requires_unique_grid_and_exact_raw_bytes(tmp_path):
    import hashlib
    import json

    from robot_sf.evidence.writers import write_json, write_text

    config = suite.load_config(suite.DEFAULT_CONFIG)
    config["seeds"] = [1001]
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    out = tmp_path / "data"
    out.mkdir()
    grid = suite.protocol_tasks(config, 0.4, "baseline", {})
    write_json(
        out / "identity.json",
        {
            "protocol": "source",
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "mode": "baseline",
            "radius_m": 0.4,
            "episode_n": len(grid),
            "seeds": [1001],
        },
    )
    for index, task in enumerate(grid):
        trace = out / f"trajectory_{index:04}.npz"
        np.savez_compressed(trace, positions=np.zeros((3, 1, 2)))
        write_json(
            out / f"case_{index:04}.json",
            {
                "case": task[0],
                "seed": task[1],
                "variant": task[2],
                "raw_trajectory": trace.name,
                "raw_trajectory_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            },
        )
    for name in ["table.json", "gate.json"]:
        write_json(out / name, {})
    write_text(out / "table.md", "Synthetic acquisition", issue_ref="#10074")

    def manifest():
        write_text(
            out / "SHA256SUMS",
            "".join(
                f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
                for p in sorted(out.iterdir())
                if p.name != "SHA256SUMS"
            ),
            issue_ref="#10074",
        )

    manifest()
    assert len(suite.verify_acquisition(out, config, config_path)) == 25
    path = out / "case_0001.json"
    original = path.read_bytes()
    row = json.loads((out / "case_0000.json").read_text())
    write_json(path, row)
    manifest()  # A valid manifest cannot disguise a duplicate/missing case.
    with pytest.raises(ValueError, match="duplicate"):
        suite.verify_acquisition(out, config, config_path)
    write_json(path, json.loads(original))
    manifest()
    trace = out / "trajectory_0000.npz"
    trace.write_bytes(trace.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        suite.verify_acquisition(out, config, config_path)
    assert suite.source_main(["--out", str(out), "--config", str(config_path), "--gate-only"]) == 4


@pytest.mark.parametrize(
    "extra",
    [
        [],
        [
            "--mode",
            "radius",
            "--radius",
            ".28",
            "--speed-tier",
            "literature",
            "--wall-profile",
            "gradient_v3",
        ],
    ],
)
def test_acquisition_cli_writes_replayable_known_answer_raw_trace(tmp_path, monkeypatch, extra):
    import json
    from concurrent.futures import ThreadPoolExecutor

    from robot_sf.evidence.writers import write_json

    config = suite.load_config(suite.DEFAULT_CONFIG)
    config["seeds"] = [1001]
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    monkeypatch.setattr(suite, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(
        suite,
        "protocol_tasks",
        lambda config, radius, mode, options: [("V1", 1001, "native", radius, mode, options)],
    )
    fake = _fake_trace("V1")

    def trace(*args, **kwargs):
        p, v = fake(*args, **kwargs)
        return p, v, np.array([1.3])

    monkeypatch.setattr(suite, "protocol_simulate", trace)
    out = tmp_path / "data"
    assert (
        suite.source_main(
            ["--out", str(out), "--config", str(config_path), "--workers", "1", *extra]
        )
        == 0
    )
    row = json.loads((out / "case_0000.json").read_text())
    assert row["fitted_desired_speed_m_s"] == pytest.approx(1.3)
    raw = np.load(out / row["raw_trajectory"])
    assert np.array_equal(raw["positions"], fake(None, None, None, None)[0])
    assert suite.source_main(["--out", str(out), "--config", str(config_path), "--gate-only"]) == 0
    trace_path = out / row["raw_trajectory"]
    trace_path.write_bytes(trace_path.read_bytes() + b"corrupt")
    assert suite.source_main(["--out", str(out), "--config", str(config_path), "--gate-only"]) == 4


def test_gate_rejects_incomplete_all_data_even_with_a_steady_window():
    row = {
        "case": "V4",
        "variant": "2.4",
        "seed": 1001,
        "all_data_specific_flow_persons_m_s": None,
        "steady_specific_flow_persons_m_s": 2.5,
        "pair_overlap": {"all": {"below_2r_count": 0}},
    }
    assert suite.acceptance_gate([row])["exit_code"] == 2
    row["all_data_specific_flow_persons_m_s"] = float("nan")
    assert suite.acceptance_gate([row])["exit_code"] == 2
    assert suite.acceptance_gate([])["exit_code"] == 4


def test_literature_cap_preserves_normal_simulator_step_and_desired_force(monkeypatch):
    from types import SimpleNamespace

    observed = {"force_desired": [], "integration_caps": [], "callbacks": []}

    class Peds:
        def __init__(self, state):
            self.state = state.copy()
            self.max_speeds = np.array([1.3])

        def pos(self):
            return self.state[:, :2]

        def vel(self):
            return self.state[:, 2:4]

        def size(self):
            return len(self.state)

        def step(self, force):
            cap = float(self.max_speeds[0])
            observed["integration_caps"].append(cap)
            self.state[:, 0] += 0.1 * cap
            self.state[:, 2] = cap

    class Sim:
        def __init__(self, state, **kwargs):
            self.peds = Peds(state)
            self.t = 0
            self.forces = [self.desired_force]

        def desired_force(self):
            observed["force_desired"].append(float(self.peds.max_speeds[0]))
            return np.zeros((1, 2))

        def step(self):
            self.peds.step(self.desired_force())
            observed["callbacks"].append(self.t)
            self.t += 1

    monkeypatch.setattr(suite.pysocialforce, "Simulator", Sim)
    config = SimpleNamespace(scene_config=SimpleNamespace(dt_secs=0.1, agent_radius=0.25))
    p, _, desired = suite.protocol_simulate(np.zeros((1, 7)), [], config, 2, speed_cap_m_s=3)
    assert observed["callbacks"] == [0, 1], "literature tier bypasses normal simulator stepping"
    assert observed["force_desired"] == [1.3, 1.3]
    assert observed["integration_caps"] == [3.0, 3.0]
    assert p[:, 0, 0] == pytest.approx([0.0, 0.3, 0.6])
    assert desired.tolist() == [1.3]
