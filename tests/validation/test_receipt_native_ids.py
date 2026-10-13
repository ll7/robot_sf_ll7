"""Receipt identity must be a fixed point of the imported native identity functions."""

import itertools
import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from robot_sf.benchmark.camera_ready import _config, _run_state
from robot_sf.training import scenario_loader
from scripts.ci import default_path_equivalence as eq
from scripts.ci import make_behaviour_receipt as producer
from scripts.ci import receipt_provenance as provenance
from scripts.validation import run_empty_world_sweep as native_sweep
from tests.validation.test_default_path_equivalence import (
    scheduler_endpoint,  # noqa: F401
)
from tests.validation.test_receipt_canonical_ids import (
    _relabel_sweeps,
    _sealed_bytes,
)
from tests.validation.test_receipt_canonical_ids import (
    admitted as _admitted,
)
from tests.validation.test_receipt_canonical_ids import (
    exact_source_sweeps as _exact_source_sweeps,
)
from tests.validation.test_receipt_canonical_ids import (
    identity_sweep as _identity_sweep,
)

admitted = _admitted
exact_source_sweeps = _exact_source_sweeps
identity_sweep = _identity_sweep

# Explicit examples cover the native sanitizer's transformation classes. Random
# Unicode fragments extend them; the native function decides what changes.
NATIVE_ALIASES = st.one_of(
    st.sampled_from(
        [
            "PPO",
            "_ppo",
            "ppo_",
            "ppo!",
            "!ppo",
            "__ppo__",
            "pp!o",
            " ppo ",
            "\tppo\n",
            "\u2003ppo\u2003",
            "ppoé",
            "ppo😀",
            "!!!",
            "___",
            "",
        ]
    ),
    st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=18),
)
PROPERTY = settings(
    max_examples=24,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _native_changes(value, field):
    directory = _config._sanitize_name(value)
    if directory != value:
        return True
    if field in {"arms", "arm"}:
        return (
            native_sweep._arm_key(
                {"_source_file": f"runs/{directory}__differential_drive/episodes.jsonl"}
            )
            != value
        )
    scenario = {"name": value}
    return (
        _config._scenario_name_key(scenario) != value
        or scenario_loader._scenario_runtime_identity(scenario) != value
    )


@PROPERTY
@example("PPO")
@example("_ppo")
@example("ppo!")
@example("__ppo__")
@example("ppo__shadow")
@example("pp!o")
@example("\tppo\n")
@example("ppoé")
@example("!!!")
@given(NATIVE_ALIASES)
@pytest.mark.parametrize("field", ["arms", "maps"])
def test_native_transform_classes_refuse_producer_scope(tmp_path, field, alias):
    assume(_native_changes(alias, field))
    path = tmp_path / "scope-native.json"
    scope = {"arms": ["ppo"], "maps": ["open"]}
    path.write_text(json.dumps(scope))
    assert producer._load_scope(path) == scope
    scope[field] = [alias]
    raw = json.dumps(scope).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="receipt identity refused: noncanonical"):
        producer._load_scope(path)
    assert path.read_bytes() == raw


@PROPERTY
@example("PPO")
@example("_ppo")
@example("ppo!")
@example("__ppo__")
@example("ppo__shadow")
@example("pp!o")
@example("\tppo\n")
@example("ppoé")
@example("!!!")
@given(NATIVE_ALIASES)
@pytest.mark.parametrize("field", ["arm", "scenario"])
@pytest.mark.parametrize("reader", ["rows", "paths", "statuses", "metadata"])
def test_native_transform_classes_refuse_all_readers(identity_sweep, field, reader, alias):
    assume(_native_changes(alias, field))
    paths = [
        identity_sweep / "episodes_main.jsonl",
        next(identity_sweep.glob("campaigns/**/episodes.jsonl")),
    ]
    # Every Hypothesis example restores the accepted source before its mutation.
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if path.name == "episodes_main.jsonl":
                row.update(arm="goal", scenario="open")
            else:
                row["scenario_id"] = "open"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def read():
        if reader == "metadata":
            return producer._verify_sweep_metadata(identity_sweep, "a" * 40)
        return {
            "rows": producer._load_sweep_rows,
            "paths": producer._sweep_episode_paths,
            "statuses": producer._sweep_execution_statuses,
        }[reader](identity_sweep)

    read()
    for path in paths:
        if path.name != "episodes_main.jsonl" and field == "arm":
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row[field if path.name == "episodes_main.jsonl" else "scenario_id"] = alias
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(ValueError, match="receipt identity refused: noncanonical"):
        read()
    assert {path: path.read_bytes() for path in paths} == before


@pytest.mark.parametrize(
    "variant",
    [
        "case",
        "leading_underscore",
        "trailing_punctuation",
        "leading_punctuation",
        "trailing_underscore",
        "unicode",
    ],
)
def test_resealed_native_directory_alias_refuses_full_producer(
    exact_source_sweeps, capsys, variant
):
    args, baseline, head = exact_source_sweeps
    assert producer.main(args) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 2142
    path = Path(args[args.index("--scope-path") + 1])
    scope = json.loads(path.read_text())
    first, old = scope["arms"][:2]
    alias = {
        "case": first.upper(),
        "leading_underscore": "_" + first,
        "trailing_punctuation": first + "!",
        "leading_punctuation": "!" + first,
        "trailing_underscore": first + "_",
        "unicode": first + "é",
    }[variant]
    assert _config._sanitize_name(alias) == first
    scope["arms"][1] = alias
    scope["arm_algorithms"][alias] = scope["arm_algorithms"].pop(old)
    path.write_text(json.dumps(scope))
    assert len({_config._sanitize_name(arm) for arm in scope["arms"]}) * 51 * 3 == 1989
    _relabel_sweeps((baseline, head), field="arms", old=old, new=alias)
    before = _sealed_bytes((baseline, head))
    with pytest.raises(ValueError, match="receipt identity refused: noncanonical"):
        producer.main(args)
    assert not capsys.readouterr().out
    assert _sealed_bytes((baseline, head)) == before


@pytest.mark.parametrize("field", ["arms", "maps"])
def test_scope_refuses_native_normalization_collisions(field):
    scope = {"arms": ["ppo"], "maps": ["open"]}
    first = scope[field][0]
    scope[field].append(first + "!")
    assert len({_config._sanitize_name(value) for value in scope[field]}) == 1
    with pytest.raises(ValueError, match="normalized scope collision"):
        provenance.validate_scope_ids(scope)


def test_seed_slice_refuses_normalized_collisions():
    with pytest.raises(ValueError, match="normalized seed collision"):
        producer._check_seed_range([1001, 1001, 1002])


@pytest.mark.parametrize("field", ["arm", "map", "scenario", "seed"])
def test_receipt_uses_live_native_normalizer_functions(monkeypatch, field):
    """Copying a native algorithm would miss this changed runtime binding."""
    if field == "seed":
        value = 1001
        monkeypatch.setattr(_run_state, "_coerce_identifier", lambda _value: 1002)

        def read():
            return provenance.canonical_seed(value)
    else:
        value = "ppo" if field == "arm" else "open"
        monkeypatch.setattr(_config, "_sanitize_name", lambda _value: "changed_native_id")

        def read():
            return provenance.canonical_id(value, field)

    with pytest.raises(ValueError, match="receipt identity refused: noncanonical"):
        read()


def test_real_pilot10342_canonical_ids_pass_producer_and_readers(tmp_path):
    """Public ID axes extracted from all 2142 original pilot summary rows remain admissible."""
    fixture = Path(__file__).with_name("fixtures") / "pilot10342_canonical_ids.json"
    pilot = json.loads(fixture.read_text())
    assert pilot["source_sha"] == "0b039f0bb9128e7abe5c700b0196782ca5cce4b6"
    assert len(pilot["arms"]) == 14 and len(pilot["maps"]) == 51
    assert pilot["seeds"] == [1001, 1002, 1003]
    scope = {"arms": pilot["arms"], "maps": pilot["maps"]}
    scope_file = tmp_path / "pilot-scope.json"
    scope_file.write_text(json.dumps(scope))
    assert producer._load_scope(scope_file) == scope
    eq.validate_scope_entries(scope)
    expected = set(itertools.product(pilot["arms"], pilot["maps"], pilot["seeds"]))
    assert len(expected) == pilot["episodes"] == 2142
    root = tmp_path / "pilot-identities"
    root.mkdir()
    summaries = []
    for arm in pilot["arms"]:
        rows = []
        source = f"runs/{arm}__differential_drive/episodes.jsonl"
        raw_path = root / "campaigns/empty_world_main" / source
        raw_path.parent.mkdir(parents=True)
        for scenario, seed in itertools.product(pilot["maps"], pilot["seeds"]):
            key = provenance.canonical_pair_key(arm, scenario, seed)
            assert key == (arm, scenario, seed)
            rows.append(
                {
                    "scenario_id": scenario,
                    "seed": seed,
                    "steps": 1,
                    "metrics": {"success": False, "collisions": 0, "total_collision_count": 0},
                }
            )
            summaries.append(
                {
                    "arm": arm,
                    "scenario": scenario,
                    "seed": seed,
                    "source_file": source,
                    "execution_status": "written",
                    "success": False,
                    "collisions": 0,
                    "steps": 1,
                }
            )
        raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "episodes_main.jsonl").write_text("".join(json.dumps(row) + "\n" for row in summaries))
    assert set(producer._load_sweep_rows(root)) == expected
    assert set(producer._sweep_episode_paths(root)) == expected
    assert set(producer._sweep_execution_statuses(root)) == expected


@pytest.mark.parametrize(
    "owner,function,field",
    [
        (_config, "_scenario_name_key", "scenario"),
        (scenario_loader, "_scenario_runtime_identity", "map"),
        (native_sweep, "_arm_key", "arm"),
    ],
)
def test_receipt_uses_live_scenario_and_roster_normalizers(monkeypatch, owner, function, field):
    value = "ppo" if field == "arm" else "open"
    monkeypatch.setattr(owner, function, lambda _value: "changed_native_id")
    with pytest.raises(ValueError, match="receipt identity refused: noncanonical"):
        provenance.canonical_id(value, field)
