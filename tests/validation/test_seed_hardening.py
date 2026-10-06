"""Pure text and dispatch controls for public held-out seed hardening."""

# seed-holdout: synthetic-fixture begin
from pathlib import Path

import pytest

from scripts.validation.check_seed_holdout_diff import check_diff


def added_diff(path, added):
    return (
        f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -0,0 +1,{len(added.splitlines())} @@\n"
        + "\n".join("+" + line for line in added.splitlines())
        + "\n"
    )


@pytest.mark.parametrize(
    "path", ["tests/support/seedguard_boundaries.py", "tests/test_heldout_seed_guard.py"]
)
def test_symbolic_simulation_loop_in_guard_file_is_refused(tmp_path, path):
    text = "for seed in EVAL_SEEDS_0_0_8:\n    env.reset(seed=seed)\n"
    assert check_diff(added_diff(path, text), tmp_path)
    assert not check_diff(
        added_diff(
            path,
            "from robot_sf.benchmark.seed_bands import HELD_OUT_SEEDS  # seed-holdout: setup-only (guard policy metadata)",
        ),
        tmp_path,
    )


@pytest.mark.parametrize(
    "line", ["config.seed = 111", "config.desired_speed_seed = 111", "config.pedestrian_seed = 111"]
)
def test_dotted_seed_attributes_are_refused(tmp_path, line):
    assert check_diff(added_diff("scripts/probe.py", line), tmp_path)


def test_yaml_merge_cannot_inherit_heldout_seed(tmp_path):
    path = "configs/probe.yaml"
    target = tmp_path / path
    target.parent.mkdir()
    target.write_text("defaults: &defaults\n  seeds: [111]\nscenario:\n  <<: *defaults\n")
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -3,0 +4 @@\n+  <<: *defaults\n"
    assert check_diff(diff, tmp_path)
    target.write_text("defaults: &defaults\n  seeds: [1001]\nscenario:\n  <<: *defaults\n")
    assert not check_diff(diff, tmp_path)


def test_seedless_dispatch_requires_a_declared_inventory():
    from robot_sf.benchmark.map_runner.map_runner_identity import _select_seeds

    with pytest.raises(ValueError, match="seed inventory"):
        _select_seeds({}, suite_seeds={}, suite_key="classic_interactions")
    assert _select_seeds({}, suite_seeds={"default": [1001]}, suite_key="classic_interactions") == [
        1001
    ]


def test_refusal_probe_source_has_no_real_sealed_seed():
    from robot_sf.benchmark import seed_bands

    source = (Path(__file__).resolve().parents[1] / "test_heldout_seed_guard.py").read_text()
    assert all(str(seed) not in source for seed in seed_bands.EVAL_SEEDS_0_0_8)


@pytest.mark.parametrize(
    "anchor, alias",
    [("base: &s 111", "seed: *s"), ("eval: &p {seeds: [111]}", "seed_policy: *p")],
)
def test_scalar_and_mapping_aliases_cannot_hide_heldout_seeds(tmp_path, anchor, alias):
    path = "configs/probe.yaml"
    target = tmp_path / path
    target.parent.mkdir()
    target.write_text(f"{anchor}\nscenario:\n  {alias}\n")
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -2,0 +3 @@\n+  {alias}\n"
    assert check_diff(diff, tmp_path)
    target.write_text(f"{anchor.replace('111', '1001')}\nscenario:\n  {alias}\n")
    assert not check_diff(diff, tmp_path)


# seed-holdout: synthetic-fixture end
