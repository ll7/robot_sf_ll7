"""Tests for the offline behavior-token diagnostics prototype (issue #4627).

These tests use synthetic in-memory traces only — they never run a simulation
campaign. They exercise the documented contracts of the ``experiments.behavior_tokens``
namespace: window extraction, missing-feature discipline, no-trace skipping, quantizer
determinism, one-token-per-valid-window, bounded example selection, and the
experimental claim boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The ``experiments`` namespace resolves to whichever checkout is first on sys.path.
# From a linked worktree the editable install points at the main checkout, so pin the
# worktree root ahead of it to load the code under test in this worktree.
_WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from experiments.behavior_tokens import (  # noqa: E402
    extract_windows,
    inspect_token_motifs,
    quantize_trace_windows,
    schemas,
)

_BEHAVIOR_TOKENS_DIR = _WORKTREE_ROOT / "experiments" / "behavior_tokens"


def _step(step_idx: int, robot_pos, ped_pos, *, ped_vel=None, linear=0.5, angular=0.0):
    """Build one synthetic simulation-step-trace step dict."""
    ped: dict = {"id": 1, "position": [float(ped_pos[0]), float(ped_pos[1])]}
    if ped_vel is not None:
        ped["velocity"] = [float(ped_vel[0]), float(ped_vel[1])]
    return {
        "step": step_idx,
        "time_s": round((step_idx + 1) * 0.1, 3),
        "robot": {
            "position": [float(robot_pos[0]), float(robot_pos[1])],
            "heading": 0.0,
            "velocity": [float(linear), 0.0],
        },
        "pedestrians": [ped],
        "planner": {"selected_action": {"linear_velocity": linear, "angular_velocity": angular}},
    }


def _episode_row(
    n_steps=20,
    *,
    ped_vel=None,
    scenario_id="s_synth",
    planner="goal",
    outcome="success",
    seed=7,
    angular_pattern=None,
):
    """Build a synthetic episode JSONL row carrying a simulation step trace."""
    steps = []
    for i in range(n_steps):
        robot_pos = (float(i) * 0.2, 0.0)
        ped_pos = (5.0 - i * 0.15, 0.3)  # pedestrian slowly approaching
        angular = 0.0 if angular_pattern is None else angular_pattern[i % len(angular_pattern)]
        steps.append(_step(i, robot_pos, ped_pos, ped_vel=ped_vel, angular=angular))
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "status": outcome,
        "outcome": outcome,
        "scenario_params": {"algo": planner},
        "algorithm_metadata": {
            "algorithm": planner,
            "simulation_step_trace": {"schema_version": "simulation-step-trace.v1", "steps": steps},
        },
    }


def test_extract_windows_expected_count():
    """A 20-step trace with window=10 stride=5 yields exactly 3 windows."""
    row = _episode_row(n_steps=20)
    records, reason = extract_windows.extract_windows_from_row(
        "output/run/episodes.jsonl", 0, row, window_steps=10, stride_steps=5
    )
    assert reason is None
    assert len(records) == 3
    # Metadata is carried through onto every window.
    for record in records:
        assert record.scenario_id == "s_synth"
        assert record.planner_key == "goal"
        assert record.seed == 7
        assert record.n_steps == 10
        assert record.feature_schema_version == schemas.FEATURE_SCHEMA_VERSION
        assert set(record.features) == set(schemas.FEATURE_NAMES)


def test_missing_optional_fields_are_null_not_zero():
    """Pedestrian-velocity features must be null (not fabricated zeros) when absent."""
    row = _episode_row(n_steps=12, ped_vel=None)
    records, reason = extract_windows.extract_windows_from_row(
        "output/run/episodes.jsonl", 0, row, window_steps=10, stride_steps=5
    )
    assert reason is None
    record = records[0]
    # Features that need pedestrian velocities cannot be derived -> null + missing.
    assert record.features["rel_speed_to_nearest_mean_m_s"] is None
    assert record.features["ped_speed_change_near_robot_m_s"] is None
    assert "rel_speed_to_nearest_mean_m_s" in record.missing_features
    # route progress is never present in the step trace -> always null.
    assert record.features["route_progress_delta_m"] is None
    # But robot-speed features ARE derivable and must not be null.
    assert isinstance(record.features["robot_speed_mean_m_s"], float)
    # A genuine zero (straight commands) is 0.0 and NOT treated as missing.
    assert record.features["oscillation_rate"] == 0.0
    assert "oscillation_rate" not in record.missing_features


def test_no_trace_row_skipped_with_reason():
    """Rows without a simulation step trace are skipped with an explicit reason."""
    records, reason = extract_windows.extract_windows_from_row(
        "output/run/episodes.jsonl", 0, {"scenario_id": "x"}, window_steps=10, stride_steps=5
    )
    assert records == []
    assert reason == "no_simulation_step_trace"

    # A trace shorter than the window is also skipped with a distinct reason.
    short = _episode_row(n_steps=4)
    records2, reason2 = extract_windows.extract_windows_from_row(
        "output/run/episodes.jsonl", 0, short, window_steps=10, stride_steps=5
    )
    assert records2 == []
    assert reason2.startswith("trace_too_short")


def _synthetic_windows(count=15):
    """Build a spread of window dicts (via the real extractor) for quantization tests."""
    windows = []
    for r in range(count):
        # Vary pedestrian approach speed and command angle so features differ per row.
        angular = [0.0, 0.1 * (r % 3)]
        row = _episode_row(
            n_steps=15,
            ped_vel=(-0.1 - 0.02 * r, 0.0),
            scenario_id=f"s{r % 3}",
            planner=f"p{r % 2}",
            outcome="collision" if r % 4 == 0 else "success",
            angular_pattern=angular,
        )
        records, _ = extract_windows.extract_windows_from_row(
            f"output/run/ep{r}.jsonl", r, row, window_steps=10, stride_steps=5
        )
        windows.extend(record.to_json_dict() for record in records)
    return windows


def test_quantizer_is_deterministic():
    """Repeated quantization on identical input yields identical token assignments."""
    windows = _synthetic_windows()
    first, meta_first = quantize_trace_windows.run_quantization(
        windows, num_tokens=4, seed=0, min_finite_fraction=0.9
    )
    second, meta_second = quantize_trace_windows.run_quantization(
        windows, num_tokens=4, seed=0, min_finite_fraction=0.9
    )
    assert [a["token_id"] for a in first] == [a["token_id"] for a in second]
    assert meta_first["feature_columns"] == meta_second["feature_columns"]
    assert meta_first["quantizer_schema_version"] == schemas.QUANTIZER_SCHEMA_VERSION


def test_one_token_per_valid_window():
    """Every valid window receives exactly one integer token id."""
    windows = _synthetic_windows()
    assignments, metadata = quantize_trace_windows.run_quantization(
        windows, num_tokens=4, seed=0, min_finite_fraction=0.9
    )
    assert len(assignments) == metadata["windows_valid"]
    window_ids = {w["window_id"] for w in windows}
    for item in assignments:
        assert item["window_id"] in window_ids
        assert isinstance(item["token_id"], int)
    # Each assigned window id is unique.
    assert len({a["window_id"] for a in assignments}) == len(assignments)


def test_inspection_selects_bounded_examples(tmp_path):
    """Motif inspection emits at most N representative examples per frequent token."""
    windows_list = _synthetic_windows()
    assignments, _ = quantize_trace_windows.run_quantization(
        windows_list, num_tokens=3, seed=0, min_finite_fraction=0.9
    )
    windows_by_id = {w["window_id"]: w for w in windows_list}
    assign_map = {a["window_id"]: a["token_id"] for a in assignments}
    summary = inspect_token_motifs.build_summary(windows_by_id, assign_map)
    assert summary["assignments_total"] == len(assignments)
    assert summary["windows_joined"] == len(assignments)

    manifest = inspect_token_motifs.write_inspection_outputs(
        tmp_path / "inspection",
        windows_by_id,
        assign_map,
        summary,
        min_token_count=1,
        examples_per_token=3,
    )
    assert Path(manifest["token_summary_json"]).is_file()
    assert Path(manifest["frequent_tokens_md"]).is_file()
    for example_file in manifest["example_files"]:
        lines = [line for line in Path(example_file).read_text().splitlines() if line.strip()]
        assert len(lines) <= 3
    # Every token carries a heuristic candidate motif label.
    for token in summary["tokens"]:
        assert token["candidate_motif_label"].endswith("_candidate")


def test_readme_and_metadata_carry_claim_boundary():
    """README and generated metadata must record the experimental claim boundary."""
    readme = (_BEHAVIOR_TOKENS_DIR / "README.md").read_text().lower()
    assert "experimental" in readme
    assert "not" in readme and "dissertation" in readme
    assert "no safety decision may depend" in readme

    windows = _synthetic_windows()
    _, metadata = quantize_trace_windows.run_quantization(
        windows, num_tokens=3, seed=0, min_finite_fraction=0.9
    )
    assert "claim_boundary" in metadata
    assert "diagnostic" in metadata["claim_boundary"].lower()

    summary = inspect_token_motifs.build_summary(
        {w["window_id"]: w for w in windows},
        {w["window_id"]: 0 for w in windows},
    )
    assert "claim_boundary" in summary
