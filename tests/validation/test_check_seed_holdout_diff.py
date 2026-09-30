"""Regression cases for the #9668 evaluation seed diff gate."""

# seed-holdout: synthetic-fixture begin

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


def test_issue_9918_exact_pilot_space_diff_flags_seed_range(tmp_path: Path) -> None:
    path = "configs/adversarial/issue_9645_pilot_space.v1.yaml"
    diff = f"""diff --git a/{path} b/{path}
new file mode 100644
index 000000000..09ed512ad
--- /dev/null
+++ b/{path}
@@ -0,0 +1,29 @@
+schema_version: adversarial-search-space.v1
+description: >-
+  Issue #9645 paired Random/TPE pilot derived from the crossing/TTC source bounds.
+  Only candidate-effective dimensions vary; pedestrian timing controls are omitted
+  because the crossing template has no bound pedestrian ID. The lower start/goal
+  bounds are clipped to 2.5 m after the original 1.0/2.0 m search bounds produced
+  an obstacle-inflated start. The simulator scenario seed is fixed to 123 so sampler
+  proposal seeds do not change the environment.
+variables:
+  start_x:
+    min: 2.5
+    max: 3.0
+  start_y:
+    min: 2.5
+    max: 4.0
+  goal_x:
+    min: 7.0
+    max: 9.0
+  goal_y:
+    min: 2.5
+    max: 4.0
+  pedestrian_speed_mps:
+    min: 0.8
+    max: 1.4
+  scenario_seed:
+    min: 123
+    max: 123
+constraints:
+  min_start_goal_distance_m: 2.0
"""
    findings = check_diff(diff, tmp_path)
    assert [(finding.path, finding.line) for finding in findings] == [(path, 26), (path, 27)]


@pytest.mark.parametrize(
    "added",
    [
        "scenario_seed: {min: 123, max: 123}",
        "episode_seed: {low: 123, high: 123}",
        "  low: 123",
        "  high: 123",
    ],
)
def test_yaml_seed_range_forms_fail(tmp_path: Path, added: str) -> None:
    context = "scenario_seed:" if added.lstrip().startswith(("low:", "high:")) else ""
    assert (
        len(check_diff(_diff("configs/adversarial/pilot.yaml", added, context=context), tmp_path))
        == 1
    )


def test_unrelated_yaml_range_below_seed_sibling_passes(tmp_path: Path) -> None:
    path = "configs/adversarial/pilot.yaml"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text("scenario_seed:\n  min: 1001\nstart_x:\n  min: 123\n")
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -3,0 +4 @@\n+  min: 123\n"
    assert check_diff(diff, tmp_path) == []


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
    ("context", "added"),
    [
        ('@pytest.mark.parametrize(\n    "seed",', "    list(range(111, 141)),"),
        ('@pytest.mark.parametrize(\n    "seed",', "    list(range(100, 150)),"),
        ("seeds = [", "    *range(100, 150),"),
        ("seeds = list(", "    range(120, 130)"),
    ],
)
def test_multiline_seed_range_overlapping_holdout_fails(
    tmp_path: Path, context: str, added: str
) -> None:
    path = "tests/benchmark/test_pilot.py"
    findings = check_diff(_diff(path, added, context=context), tmp_path)
    assert len(findings) == 1


@pytest.mark.parametrize(
    ("path", "added"),
    [
        ("scripts/benchmark/run_pilot.py", "for seed in range(142): run_episode(seed)"),
        ("scripts/benchmark/run_pilot.py", "for seed in range(110, 121, 5): run_episode(seed)"),
        ("scripts/benchmark/run_pilot.py", "for seed in range(140, 110, -1): run_episode(seed)"),
        ("tests/benchmark/test_pilot.py", "seeds = list(range(142))"),
        ("tests/benchmark/test_pilot.py", "seeds = list(range(140, 110, -1))"),
    ],
)
def test_one_and_three_argument_seed_ranges_fail(tmp_path: Path, path: str, added: str) -> None:
    assert len(check_diff(_diff(path, added), tmp_path)) == 1


@pytest.mark.parametrize(
    "added",
    [
        "for seed in range(0, 141, +1): run_episode(seed)",
        "for seed in range(142,): run_episode(seed)",
        "for seed in range(0, 141,): run_episode(seed)",
        "for seed in range(1_00, 1_42): run_episode(seed)",
        "for seed in range(0, 0x8e): run_episode(seed)",
        "for seed in list(range(111, 141)): run_episode(seed)",
        "for seed in tuple(range(111, 141)): run_episode(seed)",
        "list(map(run_episode, range(111, 141)))",
    ],
)
def test_literal_range_spellings_and_wrappers_fail(tmp_path: Path, added: str) -> None:
    assert len(check_diff(_diff("scripts/benchmark/run_pilot.py", added), tmp_path)) == 1


def test_multiline_episode_range_fails(tmp_path: Path) -> None:
    added = "for seed in range(111, 141):\n    run_episode(seed)"
    findings = check_diff(_diff("scripts/benchmark/run_pilot.py", added), tmp_path)
    assert [(finding.line, finding.text) for finding in findings] == [(2, "run_episode(seed)")]


@pytest.mark.parametrize(
    "added",
    [
        "for seed in range(111): run_episode(seed)",
        "for seed in range(100, 111): run_episode(seed)",
        "for seed in range(141, 110, -50): run_episode(seed)",
    ],
)
def test_seed_ranges_outside_holdout_pass(tmp_path: Path, added: str) -> None:
    assert check_diff(_diff("scripts/benchmark/run_pilot.py", added), tmp_path) == []


@pytest.mark.parametrize("bounds", ["range(100, 111)", "range(141, 150)"])
def test_multiline_seed_range_outside_holdout_passes(tmp_path: Path, bounds: str) -> None:
    diff = _diff("tests/benchmark/test_pilot.py", bounds, context="seeds = [")
    assert check_diff(diff, tmp_path) == []


def test_unrelated_range_overlap_passes(tmp_path: Path) -> None:
    diff = _diff("tests/benchmark/test_pilot.py", "    list(range(100, 150)),")
    assert check_diff(diff, tmp_path) == []


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


def test_bounded_setup_marker_passes(tmp_path: Path) -> None:
    path = "tests/benchmark/test_setup.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text(
        "# seed-holdout: setup-only begin\nseed = 111\n# seed-holdout: setup-only end\n"
    )
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -1,0 +2 @@\n+seed = 111\n"
    assert check_diff(diff, tmp_path) == []


@pytest.mark.parametrize("kind", ["setup-only", "synthetic-fixture"])
def test_marker_block_does_not_exempt_later_episode(tmp_path: Path, kind: str) -> None:
    path = "tests/benchmark/test_mixed.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text(
        f"# seed-holdout: {kind} begin\n"
        "seed = 111\n"
        f"# seed-holdout: {kind} end\n"
        "run_episode(seed=112)\n"
    )
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -3,0 +4 @@\n+run_episode(seed=112)\n"
    assert [(finding.line, finding.text) for finding in check_diff(diff, tmp_path)] == [
        (4, "run_episode(seed=112)")
    ]


@pytest.mark.parametrize("header", ["setup-only", "synthetic-fixture"])
def test_undelimited_header_does_not_exempt_later_line(tmp_path: Path, header: str) -> None:
    path = "tests/benchmark/test_mixed.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text(f"# seed-holdout: {header}\nrun_episode(seed=112)\n")
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -1,0 +2 @@\n+run_episode(seed=112)\n"
    assert len(check_diff(diff, tmp_path)) == 1


def test_unclosed_marker_does_not_exempt_later_line(tmp_path: Path) -> None:
    path = "tests/benchmark/test_mixed.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text("# seed-holdout: setup-only begin\nrun_episode(seed=112)\n")
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -1,0 +2 @@\n+run_episode(seed=112)\n"
    assert len(check_diff(diff, tmp_path)) == 1


def test_inline_marker_does_not_exempt_whole_file(tmp_path: Path) -> None:
    path = "tests/benchmark/test_mixed.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text("seed = 111  # seed-holdout: setup-only\nenv = make_robot_env(seed=112)\n")
    assert len(check_diff(_diff(path, "env = make_robot_env(seed=112)"), tmp_path)) == 1


def test_release_allowlist_is_exact(tmp_path: Path) -> None:
    neighbor = "configs/benchmarks/releases/benchmark_data_release_s30_h600_pilot.yaml"
    assert len(check_diff(_diff(neighbor, "seeds: [111]"), tmp_path)) == 1


def test_removed_and_unchanged_lines_are_ignored(tmp_path: Path) -> None:
    path = "configs/adversarial/pilot.yaml"
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -1,2 +1,2 @@\n-seeds: [111]\n seeds: [112]\n+seeds: [1001]\n"
    assert check_diff(diff, tmp_path) == []


def test_release_candidate_setup_manifest_is_not_episode_execution(tmp_path: Path) -> None:
    path = "tests/benchmark/test_release_candidate.py"
    diff = (
        f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -0,0 +1,5 @@\n"
        "+@pytest.fixture\n"
        "+def candidate_repo(tmp_path: Path) -> tuple[Path, Path, dict]:\n"
        '+    """Build a small committed checkout with a real 48-scenario source closure."""\n'
        "+    seed_policy = {\n"
        '+        "resolved_seeds": list(range(111, 141)),\n'
    )
    assert check_diff(diff, tmp_path) == []


def test_moved_sampler_defaults_are_not_new_seeds(tmp_path: Path) -> None:
    path = "scripts/tools/compare_adversarial_samplers.py"
    diff = (
        f"diff --git a/{path} b/{path}\n+++ b/{path}\n"
        "@@ -1216,0 +1480,2 @@\n"
        "+        config = SearchConfig.from_files(\n"
        "+            seed=(args.seed or [123])[0],\n"
        "@@ -1223,0 +1489 @@\n"
        "+    seeds = args.seed or [123]\n"
        "@@ -1302,52 +1577,12 @@\n"
        "-            seed=(args.seed or [123])[0],\n"
        "-        seeds = args.seed or [123]\n"
    )
    assert check_diff(diff, tmp_path) == []


@pytest.mark.parametrize(
    "added",
    [
        "run_episode(seed=(args.seed or [123])[0])",
        "run_episode(\n            seed=(args.seed or [123])[0],\n        )",
    ],
)
def test_sampler_file_episode_call_is_flagged(tmp_path: Path, added: str) -> None:
    path = "scripts/tools/compare_adversarial_samplers.py"
    assert len(check_diff(_diff(path, added), tmp_path)) == 1


def test_sampler_exception_requires_exact_default(tmp_path: Path) -> None:
    path = "scripts/tools/compare_adversarial_samplers.py"
    diff = _diff(
        path,
        "            seed=(args.seed or [124])[0],",
        context="        config = SearchConfig.from_files(",
    )
    assert len(check_diff(diff, tmp_path)) == 1


def test_sampler_default_in_later_environment_call_is_flagged(tmp_path: Path) -> None:
    path = "scripts/tools/compare_adversarial_samplers.py"
    diff = _diff(
        path,
        "            seed=(args.seed or [123])[0],",
        context=(
            "        config = SearchConfig.from_files(\n"
            "            seed=seeds[0],\n"
            "        )\n"
            "        env = make_robot_env("
        ),
    )
    assert [(finding.path, finding.text) for finding in check_diff(diff, tmp_path)] == [
        (path, "seed=(args.seed or [123])[0],")
    ]


def test_moved_episode_seed_line_is_flagged(tmp_path: Path) -> None:
    path = "scripts/benchmark/run_pilot.py"
    diff = (
        f"diff --git a/{path} b/{path}\n+++ b/{path}\n"
        "@@ -20 +20,0 @@\n-    scenario_seed = 111\n"
        "@@ -90,0 +90 @@\n+scenario_seed = 111\n"
    )
    assert len(check_diff(diff, tmp_path)) == 1


def test_moved_seed_argument_into_episode_call_is_flagged(tmp_path: Path) -> None:
    path = "scripts/benchmark/run_pilot.py"
    diff = (
        f"diff --git a/{path} b/{path}\n+++ b/{path}\n"
        "@@ -10 +10,0 @@\n-    seed=123,\n"
        "@@ -40,0 +40,2 @@\n+    run_episode(\n+        seed=123,\n"
    )
    assert len(check_diff(diff, tmp_path)) == 1


def test_deletion_in_another_file_does_not_hide_new_episode_seed(tmp_path: Path) -> None:
    removed_path = "scripts/benchmark/old_pilot.py"
    added_path = "scripts/benchmark/new_pilot.py"
    diff = (
        f"diff --git a/{removed_path} b/{removed_path}\n"
        "--- a/scripts/benchmark/old_pilot.py\n+++ /dev/null\n"
        "@@ -1 +0,0 @@\n-scenario_seed = 111\n"
        f"diff --git a/{added_path} b/{added_path}\n"
        f"+++ b/{added_path}\n@@ -0,0 +1 @@\n+scenario_seed = 111\n"
    )
    assert len(check_diff(diff, tmp_path)) == 1


@pytest.mark.parametrize(
    "added",
    [
        "scenario_seed = 111",
        "simulator_seed: 123",
        "environment_seed = 124",
        "world_seed = 125",
        'scenario["seeds"] = [111]',
        "for seed in range(111, 141): run_episode(seed)",
    ],
)
def test_direct_episode_seed_forms_fail(tmp_path: Path, added: str) -> None:
    findings = check_diff(_diff("scripts/benchmark/run_pilot.py", added), tmp_path)
    assert len(findings) == 1


@pytest.mark.parametrize("added", ["sampler_seed = 111", "rng_seed = 111"])
def test_sampler_rng_seeds_pass(tmp_path: Path, added: str) -> None:
    assert check_diff(_diff("scripts/tools/compare_adversarial_samplers.py", added), tmp_path) == []


def test_analysis_jsonl_episode_rows_are_stored_fixture_data(tmp_path: Path) -> None:
    path = "tests/analysis/fixtures/issue_9668_0_0_7_goal_sample.jsonl"
    row = '{"episode_id":"classic_bottleneck_low--111--example","seed":111}'
    assert check_diff(_diff(path, row), tmp_path) == []


def test_parametrized_rejection_seeds_do_not_run_episodes(tmp_path: Path) -> None:
    path = "tests/benchmark/test_issue_9748_v4_tuning_runner.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    decorator = '@pytest.mark.parametrize("seed", [101, 111, 140, 1031])'
    file.write_text(
        decorator
        + "\ndef test_release_and_non_dev_seeds_are_rejected(seed: int) -> None:\n"
        + "    with pytest.raises(ValueError):\n"
        + '        runner._choose([seed], list(range(1001, 1031)), label="seed")\n'
    )
    assert check_diff(_diff(path, decorator), tmp_path) == []


def test_parametrized_rejection_seed_that_steps_episode_fails(tmp_path: Path) -> None:
    path = "tests/benchmark/test_rejection.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    decorator = '@pytest.mark.parametrize("seed", [111])'
    file.write_text(
        decorator
        + "\ndef test_rejects_bad_seed(seed: int) -> None:\n"
        + "    run_episode(seed)\n"
        + "    with pytest.raises(ValueError):\n"
        + "        reject(seed)\n"
    )
    assert len(check_diff(_diff(path, decorator), tmp_path)) == 1


def test_seed_alias_rejection_payload_does_not_run_episodes(tmp_path: Path) -> None:
    path = "tests/validation/test_issue_9748_dev_split.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    assignment = '    payload["entries"][0]["metadata"] = {"trial": {"episode_seed": 111}}'
    file.write_text(
        "def test_tuning_log_rejects_seed_aliases_outside_entry_seeds() -> None:\n"
        "    payload = _valid_log_payload()\n"
        + assignment
        + "\n    with pytest.raises(CHECKER.ValidationError):\n"
        "        CHECKER._validate_tuning_log(path)\n"
    )
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -2,0 +3 @@\n+{assignment}\n"
    assert check_diff(diff, tmp_path) == []


# The marker text is assembled so that this file never carries the marker literally.
RELEASE_KIND = "release-evaluation"
RELEASE_PATHS = [
    "configs/benchmarks/releases/example_release.yaml",
    "configs/benchmarks/paper_experiment_matrix_example.yaml",
]


def _release_manifest(kind: str = RELEASE_KIND, *, closing: bool = True) -> str:
    lines = [f"# seed-holdout: {kind} begin", "seeds:", "  - 111", "  - 112"]
    if closing:
        lines.append(f"# seed-holdout: {kind} end")
    return "\n".join(lines) + "\n"


def _file_diff(path: str, content: str) -> str:
    added = content.splitlines()
    rows = [f"diff --git a/{path} b/{path}", f"+++ b/{path}", f"@@ -0,0 +1,{len(added)} @@"]
    rows.extend(f"+{line}" for line in added)
    return "\n".join(rows) + "\n"


def _write(root: Path, path: str, content: str) -> None:
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)


@pytest.mark.parametrize("path", RELEASE_PATHS)
def test_release_evaluation_block_passes_in_release_manifest(tmp_path: Path, path: str) -> None:
    content = _release_manifest()
    _write(tmp_path, path, content)
    assert check_diff(_file_diff(path, content), tmp_path) == []


@pytest.mark.parametrize("path", RELEASE_PATHS)
def test_release_evaluation_line_marker_passes_in_release_manifest(
    tmp_path: Path, path: str
) -> None:
    content = f"seeds: [111, 112]  # seed-holdout: {RELEASE_KIND}\n"
    _write(tmp_path, path, content)
    assert check_diff(_file_diff(path, content), tmp_path) == []


@pytest.mark.parametrize(
    "path",
    [
        "configs/adversarial/pilot.yaml",
        "configs/benchmarks/other_matrix.yaml",
        "tests/benchmark/test_pilot.yaml",
        "scripts/benchmark/run_pilot.yaml",
    ],
)
def test_release_evaluation_block_rejected_outside_release_manifest(
    tmp_path: Path, path: str
) -> None:
    content = _release_manifest()
    _write(tmp_path, path, content)
    findings = check_diff(_file_diff(path, content), tmp_path)
    assert findings
    assert {finding.path for finding in findings} == {path}
    assert any("outside release manifests" in finding.text for finding in findings)
    assert any(finding.text.startswith("- 111") for finding in findings)


def test_release_evaluation_line_marker_rejected_outside_release_manifest(
    tmp_path: Path,
) -> None:
    path = "tests/benchmark/test_pilot.py"
    added = f"seeds = [111, 112]  # seed-holdout: {RELEASE_KIND}"
    findings = check_diff(_diff(path, added), tmp_path)
    assert len(findings) == 1
    assert "outside release manifests" in findings[0].text


def test_release_evaluation_marker_rejected_even_without_seed(tmp_path: Path) -> None:
    path = "scripts/benchmark/run_pilot.py"
    findings = check_diff(_diff(path, f"# seed-holdout: {RELEASE_KIND} begin"), tmp_path)
    assert len(findings) == 1


def test_release_evaluation_marker_in_release_prefix_lookalike_rejected(tmp_path: Path) -> None:
    path = "configs/benchmarks/releases_extra/example.yaml"
    content = _release_manifest()
    _write(tmp_path, path, content)
    assert check_diff(_file_diff(path, content), tmp_path)


def test_unpaired_release_evaluation_block_rejected(tmp_path: Path) -> None:
    path = RELEASE_PATHS[0]
    content = _release_manifest(closing=False)
    _write(tmp_path, path, content)
    findings = check_diff(_file_diff(path, content), tmp_path)
    assert [finding.text for finding in findings] == ["- 111", "- 112"]


def test_mismatched_release_evaluation_block_rejected(tmp_path: Path) -> None:
    path = RELEASE_PATHS[0]
    content = (
        f"# seed-holdout: {RELEASE_KIND} begin\nseeds:\n  - 111\n# seed-holdout: setup-only end\n"
    )
    _write(tmp_path, path, content)
    assert [finding.text for finding in check_diff(_file_diff(path, content), tmp_path)] == [
        "- 111"
    ]


def test_release_evaluation_block_does_not_exempt_later_seed(tmp_path: Path) -> None:
    path = RELEASE_PATHS[0]
    content = _release_manifest() + "pilot_seeds: [113]\n"
    _write(tmp_path, path, content)
    findings = check_diff(_file_diff(path, content), tmp_path)
    assert [finding.text for finding in findings] == ["pilot_seeds: [113]"]


@pytest.mark.parametrize("kind", ["setup-only", "synthetic-fixture"])
def test_existing_markers_still_work_outside_release_manifests(tmp_path: Path, kind: str) -> None:
    path = "tests/benchmark/test_setup.py"
    content = f"# seed-holdout: {kind} begin\nseed = 111\n# seed-holdout: {kind} end\n"
    _write(tmp_path, path, content)
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -1,0 +2 @@\n+seed = 111\n"
    assert check_diff(diff, tmp_path) == []
    assert check_diff(_diff(path, f"seed = 111  # seed-holdout: {kind}"), tmp_path) == []


# seed-holdout: synthetic-fixture end

# seed-holdout: synthetic-fixture begin


@pytest.mark.parametrize(
    "reference",
    [
        "seed_set: release_eval_0_0_8",
        "seed_sets_path: configs/benchmarks/seed_sets_0_0_8.yaml",
        "from robot_sf.benchmark.seed_bands import EVAL_SEEDS_0_0_8",
    ],
)
@pytest.mark.parametrize(
    "path",
    [
        "configs/adversarial/new_pilot.yaml",
        "scripts/benchmark/new_pilot.py",
        "robot_sf/new_pilot.py",
    ],
)
def test_new_sealed_names_require_explicit_allowlist(tmp_path, reference, path):
    assert len(check_diff(_diff(path, reference), tmp_path)) == 1


@pytest.mark.parametrize(
    "added",
    [
        "seeds: [50_036]",
        "env.reset(seed=0xC374)",
        "seeds:\n  - 50_036",
        "seeds:\n  - 0xC374",
        "scenario_seed: {min: 50030, max: 50040}",
        "scenario_seed:\n  min: 50030\n  max: 50040",
        "scenario_seed:\n  low: 0xC36E\n  high: 50_040",
    ],
)
def test_literal_spellings_and_enclosing_yaml_bounds_fail(tmp_path, added):
    path = "configs/adversarial/new_pilot.yaml"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(added + "\n")
    assert check_diff(_diff(path, added), tmp_path)


def test_added_yaml_bound_uses_unchanged_sibling(tmp_path):
    path = "configs/adversarial/new_pilot.yaml"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text("scenario_seed:\n  min: 50030\n  max: 50040\n")
    diff = f"+++ b/{path}\n@@ -3 +3 @@\n-  max: 50035\n+  max: 50040\n"
    assert len(check_diff(diff, tmp_path)) == 1


@pytest.mark.parametrize(
    "path",
    [
        "robot_sf/benchmark/seed_bands.py",
        "robot_sf/benchmark/release_protocol.py",
        "docs/release/0.0.8/decisions.md",
        "configs/benchmarks/seed_sets_0_0_8.yaml",
    ],
)
def test_legitimate_sealed_names_are_allowlisted(tmp_path, path):
    assert check_diff(_diff(path, "EVAL_SEEDS_0_0_8"), tmp_path) == []


def test_yaml_bounds_do_not_cross_sibling_keys(tmp_path):
    added = "scenario_seed:\n  min: 50030\nstart_x:\n  max: 50040"
    path = "configs/adversarial/new_pilot.yaml"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(added + "\n")
    assert check_diff(_diff(path, added), tmp_path) == []


# seed-holdout: synthetic-fixture end


@pytest.mark.parametrize(
    ("path", "added", "context"),
    [
        ("configs/benchmarks/seed_sets_v1.yaml", "my_pilot: [50036, 116]", ""),
        ("configs/benchmarks/seed_sets_v1.yaml", "  - 50036", "dev:\n  - 1001"),
        ("configs/benchmarks/seed_sets_pilot.yaml", "my_pilot:\n  - 50036\n  - 116", ""),
        ("configs/benchmarks/seed_list_pilot.yaml", "my_pilot: [116]", ""),
    ],
)
def test_every_seed_file_value_is_seed_context(tmp_path, path, added, context):
    assert check_diff(_diff(path, added, context=context), tmp_path)


@pytest.mark.parametrize(
    "name", ["EVAL_SEED_SET_0_0_8", "HELD_OUT_SEEDS", "RETIRED_EVAL_SEEDS_0_0_7", "paper_eval_s30"]
)
def test_all_holdout_names_need_allowlist(tmp_path, name):
    assert check_diff(_diff("scripts/benchmark/pilot.py", f"seeds = {name}"), tmp_path)
    assert not check_diff(_diff("tests/benchmark/test_newseeds.py", f"seeds = {name}"), tmp_path)
