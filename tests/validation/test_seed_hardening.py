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
    assert check_diff(diff, tmp_path)  # New indirection is forbidden even for dev values.


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
    assert check_diff(diff, tmp_path)  # New indirection is forbidden even for dev values.


@pytest.mark.parametrize("bounds", ["min: 110, max: 142", "low: 110, high: 142"])
@pytest.mark.parametrize("alias", ["<<: *defaults", "seed_policy: *defaults"])
def test_yaml_alias_cannot_inherit_heldout_seed_range(tmp_path, bounds, alias):
    """Adding an alias must check inherited bounds, even without a held-out literal."""
    path = "configs/probe.yaml"
    target = tmp_path / path
    target.parent.mkdir()
    source = f"defaults: &defaults\n  scenario_seed: {{{bounds}}}\nscenario:\n  {alias}\n"
    target.write_text(source)
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -3,0 +4 @@\n+  {alias}\n"
    findings = check_diff(diff, tmp_path)
    assert any(f.line == 4 and "indirection is not allowed" in f.text for f in findings)

    # The parser barrier also rejects new indirection with development bounds.
    target.write_text(source.replace("110", "1001").replace("142", "1003"))
    assert check_diff(diff, tmp_path)  # New indirection is forbidden even for dev values.


@pytest.mark.parametrize("band", ["RETIRED_EVAL_SEEDS_0_0_7", "EVAL_SEEDS_0_0_8"])
@pytest.mark.parametrize("names", [("min", "max"), ("low", "high")])
@pytest.mark.parametrize("spelling", ["'{value}'", "{value}.0", "'{value}.0'"])
@pytest.mark.parametrize("alias", ["<<: *defaults", "seed_policy: *defaults"])
def test_yaml_alias_coercible_seed_range(tmp_path, band, names, spelling, alias):
    """Consumer-coercible bounds must not hide an inherited held-out range."""
    from robot_sf.benchmark import seed_bands

    seeds = getattr(seed_bands, band)
    path = "configs/probe.yaml"
    target = tmp_path / path
    target.parent.mkdir()
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -2,0 +3 @@\n+  {alias}\n"

    def write_range(low, high):
        bounds = ", ".join(
            f"{name}: {spelling.format(value=value)}"
            for name, value in zip(names, (low, high), strict=True)
        )
        target.write_text(
            f"defaults: &defaults {{scenario_seed: {{{bounds}}}}}\nscenario:\n  {alias}\n"
        )

    write_range(min(seeds) - 1, max(seeds) + 1)
    assert any(
        f.line == 3 and "indirection is not allowed" in f.text for f in check_diff(diff, tmp_path)
    )
    write_range(1001, 1003)
    assert check_diff(diff, tmp_path)  # New indirection is forbidden even for dev values.


@pytest.mark.parametrize("names", [("min", "max"), ("low", "high")])
@pytest.mark.parametrize("bound_index", [0, 1])
@pytest.mark.parametrize(
    "invalid",
    [
        "1001.5",
        "'1001.000000000000000001'",
        "'nope'",
        "true",
        "null",
        ".nan",
        ".inf",
        "[]",
        "{}",
        None,
    ],
)
def test_yaml_alias_invalid_seed_range_fails_closed(tmp_path, names, bound_index, invalid):
    """Malformed bounds must yield a finding rather than skip or truncate the range."""
    path = "configs/probe.yaml"
    target = tmp_path / path
    target.parent.mkdir()
    values = ["1001", "1003"]
    values[bound_index] = invalid
    bounds = ", ".join(
        f"{name}: {value}" for name, value in zip(names, values, strict=True) if value is not None
    )
    target.write_text(
        f"defaults: &defaults {{scenario_seed: {{{bounds}}}}}\nscenario:\n  <<: *defaults\n"
    )
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -2,0 +3 @@\n+  <<: *defaults\n"
    assert any(
        f.line == 3 and "indirection is not allowed" in f.text for f in check_diff(diff, tmp_path)
    )


@pytest.mark.parametrize("band", ["RETIRED_EVAL_SEEDS_0_0_7", "EVAL_SEEDS_0_0_8"])
@pytest.mark.parametrize("names", [("min", "max"), ("low", "high")])
@pytest.mark.parametrize("flow", [True, False])
def test_yaml_merge_retains_enclosing_seed_range(tmp_path, band, names, flow):
    """Only the merge line changes; the seed key and anchor remain unchanged."""
    from robot_sf.benchmark import seed_bands

    seeds = getattr(seed_bands, band)
    path = "configs/probe.yaml"
    target = tmp_path / path
    target.parent.mkdir()
    alias = "<<: *bounds"
    diff = f"diff --git a/{path} b/{path}\n+++ b/{path}\n@@ -3,0 +4 @@\n+    {alias}\n"

    def write_range(low, high):
        bounds = f"{names[0]}: {low}, {names[1]}: {high}"
        opening, closing = (" {", "  }\n") if flow else ("", "")
        target.write_text(
            f"defaults: &bounds {{{bounds}}}\nscenario:\n"
            f"  scenario_seed:{opening}\n    {alias}\n{closing}"
        )

    write_range(min(seeds) - 1, max(seeds) + 1)
    assert any(
        f.line == 4 and "indirection is not allowed" in f.text for f in check_diff(diff, tmp_path)
    )
    write_range(1001, 1003)
    assert check_diff(diff, tmp_path)  # New indirection is forbidden even for dev values.


# seed-holdout: synthetic-fixture end
