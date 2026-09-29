"""Regression cases for the #9668 evaluation seed diff gate."""

# seed-holdout: synthetic-fixture

from pathlib import Path

import pytest

from scripts.validation.check_seed_holdout_diff import check_diff


def _diff(path: str, added: str, *, context: str = "") -> str:
    rows = [f"diff --git a/{path} b/{path}", f"+++ b/{path}", "@@ -0,0 +1,9 @@"]
    rows.extend(f" {line}" for line in context.splitlines())
    rows.extend(f"+{line}" for line in added.splitlines())
    return "\n".join(rows) + "\n"


@pytest.mark.parametrize(
    ("path", "added"),
    [
        ("configs/adversarial/issue_9645_pilot_space.v1.yaml", "seeds: [111, 112]"),
        ("tests/benchmark/test_pilot.py", "@pytest.mark.parametrize('seed', [113, 114])"),
        ("scripts/benchmark/run_diagnostic.py", "command = 'uv run benchmark --seed 115'"),
        ("docs/plan/diagnostic.md", "uv run benchmark --seeds 116 117"),
        ("docs/context/evidence/diagnostic/README.md", "uv run benchmark --seed 117"),
        ("tests/benchmark/test_pilot.py", "env = make_robot_env(seed=118)"),
    ],
)
def test_planner_seed_patterns_fail(tmp_path: Path, path: str, added: str) -> None:
    findings = check_diff(_diff(path, added), tmp_path)
    assert len(findings) == 1
    assert findings[0].path == path


def test_yaml_seed_list_continuation_fails(tmp_path: Path) -> None:
    diff = _diff("configs/adversarial/pilot.yaml", "  - 119", context="seeds:")
    assert len(check_diff(diff, tmp_path)) == 1


def test_named_seed_list_fails(tmp_path: Path) -> None:
    diff = _diff("configs/benchmarks/seed_list_v1.yaml", "  - 121", context="classic_interactions:")
    assert len(check_diff(diff, tmp_path)) == 1


def test_long_yaml_seed_list_continuation_fails(tmp_path: Path) -> None:
    path = "configs/adversarial/pilot.yaml"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text("seeds:\n" + "".join(f"  - {seed}\n" for seed in range(111, 140)) + "  - 140\n")
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -30,0 +31 @@\n+  - 140\n"
    assert len(check_diff(diff, tmp_path)) == 1


def test_multiline_parametrize_fails(tmp_path: Path) -> None:
    diff = _diff(
        "tests/benchmark/test_pilot.py",
        "[111, 112],",
        context='@pytest.mark.parametrize(\n    "seed",',
    )
    assert len(check_diff(diff, tmp_path)) == 1


@pytest.mark.parametrize(
    ("path", "added"),
    [
        ("configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml", "seeds: [111, 140]"),
        ("configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml", "  - 111"),
        ("tests/benchmark/test_setup.py", "seed = 111  # seed-holdout: setup-only"),
        ("tests/benchmark/test_fixture.py", "seed = 112  # seed-holdout: synthetic-fixture"),
        ("docs/plan/figure.md", "Image width: 120 px"),
        ("docs/context/evidence/diagnostic/summary.json", '"seed": 111, "metric": 0.5'),
        ("configs/adversarial/pilot.yaml", "max_steps: 120"),
        ("tests/benchmark/test_setup.py", "seed = 1001"),
        ("tests/benchmark/test_setup.py", "seed = 1110"),
    ],
)
def test_allowed_or_unrelated_values_pass(tmp_path: Path, path: str, added: str) -> None:
    assert check_diff(_diff(path, added), tmp_path) == []


def test_file_marker_passes(tmp_path: Path) -> None:
    path = "tests/benchmark/test_setup.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text("# seed-holdout: setup-only\nseed = 111\n")
    assert check_diff(_diff(path, "seed = 111"), tmp_path) == []


def test_release_allowlist_is_exact(tmp_path: Path) -> None:
    neighbor = "configs/benchmarks/releases/benchmark_data_release_s30_h600_pilot.yaml"
    assert len(check_diff(_diff(neighbor, "seeds: [111]"), tmp_path)) == 1


def test_removed_and_unchanged_lines_are_ignored(tmp_path: Path) -> None:
    path = "configs/adversarial/pilot.yaml"
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -1,2 +1,2 @@\n-seeds: [111]\n seeds: [112]\n+seeds: [1001]\n"
    assert check_diff(diff, tmp_path) == []
