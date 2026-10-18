"""Pure text and dispatch controls for public held-out seed hardening."""

# seed-holdout: synthetic-fixture begin
import ast
import base64
import difflib
import json
import os
import shlex
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import scripts.validation.check_seed_holdout_diff as guard
from robot_sf.adversarial.config import MultiPedAdversarialConfig, SearchSpaceConfig
from robot_sf.benchmark import seed_bands
from robot_sf.benchmark.cli import get_parser
from robot_sf.benchmark.runtime_seed_guard import check_simulation_seed
from robot_sf.training.scenario_loader import load_scenarios
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


# Independent reviewer probes: parse configs without stepping an environment.


def _review_diff(before, after, path="configs/probe.yaml"):
    return "".join(
        difflib.unified_diff(
            before.splitlines(True),
            after.splitlines(True),
            fromfile="a/" + path,
            tofile="b/" + path,
        )
    )


def _review_write(root, source, path="configs/probe.yaml"):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.encode("utf-8"))
    return target


def _review_b64(value):
    return base64.b64encode(str(value).encode()).decode()


def _review_space(bounds):
    return (
        "variables:\n"
        + "".join(
            f"  {name}: {{min: 0, max: 1}}\n" for name in ("start_x", "start_y", "goal_x", "goal_y")
        )
        + f"  scenario_seed: {bounds}\n"
    )


@pytest.mark.parametrize("band", ["RETIRED_EVAL_SEEDS_0_0_7", "EVAL_SEEDS_0_0_8"])
def test_review_binary_range_requires_refusal(tmp_path, band):
    seeds = getattr(seed_bands, band)
    before = _review_space(
        f"{{min: !!binary {_review_b64(1001)}, max: !!binary {_review_b64(1003)}}}"
    )
    target = _review_write(tmp_path, before)
    safe = SearchSpaceConfig.from_file(target).scenario_seed
    assert not any(safe.min <= s <= safe.max for s in seed_bands.HELD_OUT_SEEDS)
    assert check_diff(_review_diff("", before), tmp_path) == []
    after = _review_space(
        f"{{min: !!binary {_review_b64(min(seeds) - 1)}, max: !!binary {_review_b64(max(seeds) + 1)}}}"
    )
    _review_write(tmp_path, after)
    actual = SearchSpaceConfig.from_file(target).scenario_seed
    assert all(actual.min <= s <= actual.max for s in seeds)
    assert check_diff(_review_diff(before, after), tmp_path), (
        "consumer-accepted binary range bypassed guard"
    )


def _review_multi(seed):
    return (
        f"family: review\nscenario_seed: {seed}\npedestrians:\n"
        "  - id: p1\n    start: {x: 0, y: 0, theta: 0}\n"
        "    goal: {x: 3, y: 0, theta: 0}\n"
    )


@pytest.mark.parametrize("band", ["RETIRED_EVAL_SEEDS_0_0_7", "EVAL_SEEDS_0_0_8"])
@pytest.mark.parametrize("encoding", ["binary", "fractional"])
def test_review_scalar_coercion_requires_refusal(tmp_path, band, encoding):
    seed = min(getattr(seed_bands, band))

    def encode(s):
        return f"!!binary {_review_b64(s)}" if encoding == "binary" else f"{s}.5"

    before = _review_multi(encode(1001))
    target = _review_write(tmp_path, before)
    assert MultiPedAdversarialConfig.from_file(target).scenario_seed == 1001
    assert check_diff(_review_diff("", before), tmp_path) == []
    after = _review_multi(encode(seed))
    _review_write(tmp_path, after)
    assert MultiPedAdversarialConfig.from_file(target).scenario_seed == seed
    assert check_diff(_review_diff(before, after), tmp_path), (
        "consumer-accepted scalar bypassed guard"
    )


def _review_git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.mark.parametrize("node", ["map", "seq", "int"])
def test_review_changed_anchor_property_line_requires_refusal(tmp_path, node):
    body = {"map": "{min: 1001, max: 1003}", "seq": "[1001, 1003]", "int": "1001"}[node]
    before = f"value: !!{node}\n  &v {body}\ncopy: *v\nlabel: old\n"
    _review_write(tmp_path, before)
    _review_git(tmp_path, "init", "-q")
    _review_git(tmp_path, "add", "configs/probe.yaml")
    _review_git(
        tmp_path,
        "-c",
        "user.name=Review",
        "-c",
        "user.email=review@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    unchanged = before.replace("old", "new")
    _review_write(tmp_path, unchanged)
    assert check_diff(_review_diff(before, unchanged), tmp_path, base_ref="HEAD") == []
    changed = unchanged.replace("&v ", "&v  ")
    _review_write(tmp_path, changed)
    assert yaml.safe_load(changed) == yaml.safe_load(unchanged)
    assert check_diff(_review_diff(before, changed), tmp_path, base_ref="HEAD"), (
        "edited anchor line was grandfathered by preceding tag line"
    )


@pytest.mark.parametrize("band", ["RETIRED_EVAL_SEEDS_0_0_7", "EVAL_SEEDS_0_0_8"])
def test_review_preexisting_alias_binary_range_requires_refusal(tmp_path, band):
    seeds = getattr(seed_bands, band)
    before = (
        f"bounds: !!map\n  &v {{min: !!binary {_review_b64(1001)}, max: !!binary {_review_b64(1003)}}}\n"
        + _review_space("*v")
    )
    target = _review_write(tmp_path, before)
    baseline = SearchSpaceConfig.from_file(target).scenario_seed
    assert not any(baseline.min <= s <= baseline.max for s in seed_bands.HELD_OUT_SEEDS)
    _review_git(tmp_path, "init", "-q")
    _review_git(tmp_path, "add", "configs/probe.yaml")
    _review_git(
        tmp_path,
        "-c",
        "user.name=Review",
        "-c",
        "user.email=review@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    after = before.replace(_review_b64(1001), _review_b64(min(seeds) - 1)).replace(
        _review_b64(1003), _review_b64(max(seeds) + 1)
    )
    _review_write(tmp_path, after)
    actual = SearchSpaceConfig.from_file(target).scenario_seed
    assert all(actual.min <= s <= actual.max for s in seeds)
    assert check_diff(_review_diff(before, after), tmp_path, base_ref="HEAD"), (
        "inherited held-out binary range bypassed both parser barrier and semantic check"
    )


@pytest.mark.parametrize(
    "source",
    [
        "x: &a 1001\ny: *a\n",
        "x: &a [1001, 1003]\ny: *a\n",
        "x: &a {min: 1001, max: 1003}\ny: {<<: *a}\n",
        "x: &a {min: 1001, max: 1003}\ny:\n  ? <<\n  : *a\n",
        "x: &a {min: 1001, max: 1003}\ny:\n  <<: [\n    *a\n  ]\n",
        "x: &a {min: 1001, max: 1003}\ny:\n  <<:\n    - *a\n",
        "x: !!map\n  &a {min: 1001, max: 1003}\n",
        "x: &a {min: 1001, max: 1003}\ny: {!!merge '<<': *a}\n",
        "x: &a 1001 # seed-holdout: synthetic-fixture\n",
        "# seed-holdout: synthetic-fixture begin\nx: &a 1001\n# seed-holdout: synthetic-fixture end\n",
    ],
)
def test_review_new_indirection_barrier(tmp_path, source):
    _review_write(tmp_path, source)
    findings = check_diff(_review_diff("", source), tmp_path)
    assert any("indirection is not allowed" in f.text for f in findings)


@pytest.mark.parametrize(
    "shape",
    [
        "value: '{s}'\n",
        "value: {s}.0\n",
        "value: !!int '{s}'\n",
        "value: 0x{hex}\n",
        'value: "{unicode}"\n',
        "value: {{min: '{lo}', max: '{hi}', step: 1}}\n",
        "value: ['{lo}', '{hi}']\n",
        "value: {{low: '{lo}', high: '{hi}'}}\n",
        "value: {{start: {lo}.5, stop: {hi}.5}}\n",
        "value: |\n  {s}\n",
        "value: >\n  {s}\n",
    ],
)
@pytest.mark.parametrize("band", ["RETIRED_EVAL_SEEDS_0_0_7", "EVAL_SEEDS_0_0_8"])
def test_review_other_yaml_semantics(tmp_path, shape, band):
    seeds = getattr(seed_bands, band)
    s = min(seeds)
    source = shape.format(
        s=s,
        lo=s - 1,
        hi=max(seeds) + 1,
        hex=format(s, "x"),
        unicode="".join("\\u" + format(ord(c), "04x") for c in str(s)),
    )
    _review_write(tmp_path, source)
    assert check_diff(_review_diff("", source), tmp_path)


@pytest.mark.parametrize("source", ["x: [\n", "x: !include leaf.yaml\n", "x: &a [*a]\n"])
def test_review_invalid_or_recursive_yaml_refuses(tmp_path, source):
    _review_write(tmp_path, source)
    assert check_diff(_review_diff("", source), tmp_path)


@pytest.mark.parametrize("band", ["RETIRED_EVAL_SEEDS_0_0_7", "EVAL_SEEDS_0_0_8"])
@pytest.mark.parametrize(
    "form",
    [
        "binary",
        "octal11",
        "octal12",
        "hex",
        "underscores",
        "sexagesimal",
        "sexagesimal_float",
        "integral",
        "exponent",
        "quoted_binary",
        "quoted_octal12",
        "quoted_octal11",
        "zero_sexagesimal_float",
        "yaml12_leading_zero",
        "quoted_hex",
        "quoted_sexagesimal",
        "binary_bytes",
    ],
)
@pytest.mark.parametrize("container", ["scalar", "mapping", "sequence"])
def test_review_yaml_numeric_grammars(tmp_path, band, form, container):
    """All supported numeric spellings use the same held-out admission policy."""
    from robot_sf.benchmark import seed_bands

    def encode(value):
        minutes, seconds = divmod(value, 60)
        hours, minute = divmod(minutes, 60)
        text = {
            "binary": f"0b{value:b}",
            "octal11": f"0{value:o}",
            "octal12": f"0o{value:o}",
            "hex": f"0x{value:x}",
            "underscores": "_".join(str(value)),
            "sexagesimal": f"{minutes}:{seconds:02d}",
            "sexagesimal_float": f"{minutes}:{seconds:02d}.0",
            "integral": f"{value}.0",
            "exponent": f"{value}e0",
            "quoted_binary": f"'+0b{value:b}'",
            "quoted_octal12": f"'+0o{value:o}'",
            "quoted_octal11": f"'0{value:o}'",
            "zero_sexagesimal_float": f"'0:{hours}:{minute:02d}:{seconds:02d}.0'",
            "yaml12_leading_zero": f"0{value}",
            "quoted_hex": f"'+0x{value:x}'",
            "quoted_sexagesimal": f"'{minutes}:{seconds:02d}'",
            "binary_bytes": f"!!binary {_review_b64(value)}",
        }[form]
        return text

    def source(low, high):
        if container == "scalar":
            return f"value: {encode(low)}\n"
        if container == "mapping":
            return f"value: {{min: {encode(low)}, max: {encode(high)}, step: 1}}\n"
        return f"value: [{encode(low)}, {encode(high)}]\n"

    before = source(1001, 1003)
    _review_write(tmp_path, before)
    assert check_diff(_review_diff("", before), tmp_path) == []
    seeds = getattr(seed_bands, band)
    after = source(min(seeds), max(seeds))
    _review_write(tmp_path, after)
    assert check_diff(_review_diff(before, after), tmp_path)


@pytest.mark.parametrize("band", ["RETIRED_EVAL_SEEDS_0_0_7", "EVAL_SEEDS_0_0_8"])
@pytest.mark.parametrize("offset", ["-.5", ".5", "-.000000000000000001", ".000000000000000001"])
@pytest.mark.parametrize("quoted", [False, True])
def test_review_fraction_floor_or_ceiling_refuses(tmp_path, band, offset, quoted):
    from decimal import Decimal

    from robot_sf.benchmark import seed_bands

    seed = min(getattr(seed_bands, band))
    value = Decimal(seed) + Decimal(offset)
    text = f"'{value}'" if quoted else str(value)
    source = f"value: {text}\n"
    _review_write(tmp_path, source)
    assert check_diff(_review_diff("", source), tmp_path)


@pytest.mark.parametrize(
    "source",
    [
        "value: 1001.5\n",
        "value: '1001.0001'\n",
        "value: .inf\n",
        "value: .nan\n",
        "value: '0bnotnumeric'\n",
        "value: true\n",
        "value: {min: 1001.1, max: 1001.9}\n",
    ],
)
def test_review_numeric_controls(tmp_path, source):
    _review_write(tmp_path, source)
    assert check_diff(_review_diff("", source), tmp_path) == []


@pytest.mark.parametrize("band", ["RETIRED_EVAL_SEEDS_0_0_7", "EVAL_SEEDS_0_0_8"])
def test_review_unchanged_alias_binary_numeric_range(tmp_path, band):
    """Keep the property line identical so only resolved semantics can refuse."""
    from robot_sf.benchmark import seed_bands

    before = "bounds: &v {min: 1001, max: 1003}\ncopy: *v\nlabel: old\n"
    _review_write(tmp_path, before)
    _review_git(tmp_path, "init", "-q")
    _review_git(tmp_path, "add", "configs/probe.yaml")
    _review_git(
        tmp_path,
        "-c",
        "user.name=Review",
        "-c",
        "user.email=review@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    seeds = getattr(seed_bands, band)
    heldout = f"bounds: &v {{min: 0b{min(seeds) - 1:b}, max: 0b{max(seeds) + 1:b}}}\ncopy: *v\nlabel: old\n"
    # Commit the held-out target before an unrelated edit, preserving both alias tokens.
    _review_write(tmp_path, heldout)
    _review_git(tmp_path, "add", "configs/probe.yaml")
    _review_git(
        tmp_path,
        "-c",
        "user.name=Review",
        "-c",
        "user.email=review@example.invalid",
        "commit",
        "-qm",
        "existing range",
    )
    after = heldout.replace("old", "new")
    _review_write(tmp_path, after)
    assert check_diff(_review_diff(heldout, after), tmp_path, base_ref="HEAD")


# Independent residual boundary probes; tracked limitations remain explicit xfails.


def _review_residual_result(rows, name, root, source, path, consumer_heldout, before=""):
    findings = guard.check_diff(_review_diff(before, source, path), root)
    rows.append((name, bool(findings), consumer_heldout))


def _review_residual_rows(tmp_path):
    from robot_sf.benchmark import seed_bands

    rows = []
    root = tmp_path
    seeds = seed_bands.RETIRED_EVAL_SEEDS_0_0_7
    lo, hi = min(seeds) - 1, max(seeds) + 1
    source = (
        json.dumps(
            {
                "variables": {
                    **{n: {"min": 0, "max": 1} for n in ("start_x", "start_y", "goal_x", "goal_y")},
                    "scenario_seed": {"min": str(lo), "max": str(hi)},
                }
            },
            indent=2,
        )
        + "\n"
    )
    path = "configs/probe.json"
    actual = SearchSpaceConfig.from_file(_review_write(root, source, path)).scenario_seed
    _review_residual_result(
        rows,
        "json-string-bounds",
        root,
        source,
        path,
        all(actual.min <= s <= actual.max for s in seeds),
    )

    payload = {
        "family": "review",
        "scenario_seed": str(min(seeds)),
        "pedestrians": [
            {
                "id": "p1",
                "start": {"x": 0, "y": 0, "theta": 0},
                "goal": {"x": 3, "y": 0, "theta": 0},
            }
        ],
    }
    source = json.dumps(payload).replace('"111"', '"\\u0031\\u0031\\u0031"') + "\n"
    actual = MultiPedAdversarialConfig.from_file(_review_write(root, source, path)).scenario_seed
    _review_residual_result(rows, "json-unicode-scalar", root, source, path, actual in seeds)

    path = "configs/probe.cfg"
    source = _review_space(f"{{min: '{lo}', max: '{hi}'}}")
    actual = SearchSpaceConfig.from_file(_review_write(root, source, path)).scenario_seed
    _review_residual_result(
        rows,
        "yaml-content-non-yaml-extension",
        root,
        source,
        path,
        all(actual.min <= s <= actual.max for s in seeds),
    )

    path = "configs/top.yaml"
    leaf = "scenarios:\n  - name: review\n    seeds: [111]\n"
    _review_write(root, leaf, "configs/nested/leaf.yaml")
    _review_write(root, "includes: [leaf.yaml]\n", "configs/nested/middle.yaml")
    source = "includes: [nested/middle.yaml]\n"
    actual = load_scenarios(_review_write(root, source, path))
    _review_residual_result(
        rows,
        "activate-unchanged-two-level-include",
        root,
        source,
        path,
        actual[0]["seeds"][0] in seeds,
    )
    _review_residual_result(
        rows, "changed-nested-leaf-literal", root, leaf, "configs/nested/leaf.yaml", True
    )

    path = "scripts/review.sh"
    source = (
        "robot_sf_bench run --matrix configs/top.yaml --out output/review.jsonl --base-seed 111\n"
    )
    parsed = get_parser().parse_args(shlex.split(source)[1:])
    _review_residual_result(rows, "cli-base-seed", root, source, path, parsed.base_seed in seeds)
    source = "tool --seed 111\n"
    _review_residual_result(rows, "cli-literal-seed-control", root, source, path, True)

    path = "scripts/review.py"
    source = 'os.environ["ROBOT_SF_SEED"] = "\\x31\\x31\\x31"\n'
    literal = ast.parse(source).body[0].value.value
    _review_write(root, source, path)
    _review_residual_result(
        rows, "environment-escaped-literal", root, source, path, int(literal) in seeds
    )
    source = 'os.environ["ROBOT_SF_SEED"] = "".join(("1", "11"))\n'
    _review_write(root, source, path)
    _review_residual_result(
        rows, "environment-computed-string", root, source, path, int("".join(("1", "11"))) in seeds
    )

    refused = False
    with patch.dict(
        os.environ, {"ALLOW_HELD_OUT_SEEDS": "1", "ROBOT_SF_ALLOW_HELD_OUT_SEEDS": "1"}
    ):
        try:
            check_simulation_seed(min(seeds), boundary="read-only-review")
        except ValueError:
            refused = True
    rows.append(("runtime-environment-permission-control", refused, True))
    return rows


class _ReviewMissingRefusal(AssertionError):
    """Only the documented missing guard decision may be an expected failure."""


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            "json-string-bounds",
            marks=pytest.mark.xfail(
                strict=True,
                raises=_ReviewMissingRefusal,
                reason="#10059: JSON semantic admission remains outside the YAML repair",
            ),
        ),
        pytest.param(
            "json-unicode-scalar",
            marks=pytest.mark.xfail(
                strict=True,
                raises=_ReviewMissingRefusal,
                reason="#10059: JSON semantic admission remains outside the YAML repair",
            ),
        ),
        pytest.param(
            "yaml-content-non-yaml-extension",
            marks=pytest.mark.xfail(
                strict=True,
                raises=_ReviewMissingRefusal,
                reason="#10059: content sniffing remains outside the YAML repair",
            ),
        ),
        pytest.param(
            "activate-unchanged-two-level-include",
            marks=pytest.mark.xfail(
                strict=True,
                raises=_ReviewMissingRefusal,
                reason="#10059: recursive include closure remains separately owned",
            ),
        ),
        "changed-nested-leaf-literal",
        pytest.param(
            "cli-base-seed",
            marks=pytest.mark.xfail(
                strict=True,
                raises=_ReviewMissingRefusal,
                reason="#10059: CLI base-seed residual outside YAML repair",
            ),
        ),
        "cli-literal-seed-control",
        "environment-escaped-literal",
        pytest.param(
            "environment-computed-string",
            marks=pytest.mark.xfail(
                strict=True,
                raises=_ReviewMissingRefusal,
                reason="#10059: dynamic environment values require exact-head review",
            ),
        ),
        "runtime-environment-permission-control",
    ],
)
def test_review_residual_boundaries(tmp_path, name):
    rows = {row[0]: row[1:] for row in _review_residual_rows(tmp_path)}
    refused, consumer_heldout = rows[name]
    assert consumer_heldout
    if not refused:
        raise _ReviewMissingRefusal(name)


# seed-holdout: synthetic-fixture end
