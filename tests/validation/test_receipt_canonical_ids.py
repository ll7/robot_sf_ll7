"""Canonical pair identity is mandatory before coverage counts or key coercions."""

import copy
import json
from pathlib import Path

import pytest

from scripts.ci import behaviour_receipt as gate
from scripts.ci import default_path_equivalence as eq
from scripts.ci import make_behaviour_receipt as producer
from tests.validation.test_default_path_equivalence import (
    _episode,
    _refresh_equivalence_binding,
    _write_equivalence_sweep,
    scheduler_endpoint,  # noqa: F401
)
from tests.validation.test_default_path_equivalence import (
    admitted as _admitted,
)
from tests.validation.test_default_path_equivalence import (
    exact_source_sweeps as _exact_source_sweeps,
)

admitted = _admitted
exact_source_sweeps = _exact_source_sweeps


def _relabel_sweeps(roots, *, field, old, new):
    """Retain full rows, allocation custody and normal producer seals after relabeling."""
    for root in roots:
        path = root / "episodes_main.jsonl"
        summaries = [json.loads(line) for line in path.read_text().splitlines()]
        for row in summaries:
            label = "arm" if field == "arms" else "scenario"
            if row[label] == old:
                row[label] = new
                if field == "arms":
                    row["source_file"] = row["source_file"].replace(old + "__", new + "__")
        path.write_text("".join(json.dumps(row) + "\n" for row in summaries))
        for raw in list(root.glob("campaigns/**/episodes.jsonl")):
            if field == "arms":
                if raw.parent.name.startswith(old + "__"):
                    raw.parent.rename(
                        raw.parent.with_name(raw.parent.name.replace(old + "__", new + "__"))
                    )
            else:
                rows = [json.loads(line) for line in raw.read_text().splitlines()]
                for row in rows:
                    if row["scenario_id"] == old:
                        row["scenario_id"] = new
                raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        _refresh_equivalence_binding(root, validate_identity=False)


def _sealed_bytes(roots):
    from scripts.ci import receipt_provenance

    return {
        path: path.read_bytes()
        for root in roots
        for path in [*receipt_provenance.source_files(root), root / "execution_binding.json"]
    }


@pytest.mark.parametrize("field,variant", [("arms", "space"), ("maps", "space"), ("maps", "case")])
def test_producer_refuses_resealed_selector_alias(exact_source_sweeps, capsys, field, variant):
    """An internally coherent 2142-row matrix must not double-count selector aliases."""
    args, baseline, head = exact_source_sweeps
    assert producer.main(args) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 2142
    scope_path = Path(args[args.index("--scope-path") + 1])
    scope = json.loads(scope_path.read_text())
    original = scope[field][0]
    old = scope[field][1]
    alias = original + " " if variant == "space" else original.upper()
    scope[field][1] = alias
    if field == "arms":
        scope["arm_algorithms"][alias] = scope["arm_algorithms"].pop(old)
    scope_path.write_text(json.dumps(scope))
    roots = (baseline, head)
    _relabel_sweeps(roots, field=field, old=old, new=alias)
    # Independent cardinalities demonstrate the claimed coverage defect.
    arms = {value.strip() for value in scope["arms"]}
    maps = {value.strip().lower() for value in scope["maps"]}
    assert len(arms) * len(maps) * 3 == (1989 if field == "arms" else 2100)
    sealed_before = _sealed_bytes(roots)
    scope_before = scope_path.read_bytes()
    with pytest.raises(ValueError, match="receipt identity refused: noncanonical"):
        producer.main(args)
    assert not capsys.readouterr().out
    assert scope_path.read_bytes() == scope_before
    assert _sealed_bytes(roots) == sealed_before


@pytest.fixture
def identity_sweep(tmp_path):
    root = tmp_path / "11"
    _write_equivalence_sweep(
        root, "a" * 40, "11", "b" * 40, "c" * 64, {"arms": ["goal"], "maps": ["open"]}
    )
    metadata = root / "execution_main.json"
    values = json.loads(metadata.read_text())
    values.update(expected_slots=3, written_rows=3, total_episodes=3)
    metadata.write_text(json.dumps(values))
    _refresh_equivalence_binding(root)
    (root / "execution_binding.json").unlink()
    return root


def _change_identity(root, field, value):
    summary = root / "episodes_main.jsonl"
    rows = [json.loads(line) for line in summary.read_text().splitlines()]
    for row in rows:
        if field != "seed" or row[field] == 1001:
            row[field] = value
    if field == "arm":
        for row in rows:
            row["source_file"] = row["source_file"].replace("goal__", value + "__")
    summary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    raw = next(root.glob("campaigns/**/episodes.jsonl"))
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    if field != "arm":
        raw_field = "scenario_id" if field == "scenario" else field
        for row in rows:
            if field != "seed" or row[raw_field] == 1001:
                row[raw_field] = value
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        raw.parent.rename(raw.parent.with_name(value + "__differential_drive"))


@pytest.mark.parametrize("reader", ["rows", "paths", "statuses", "metadata"])
@pytest.mark.parametrize(
    "field,value", [("arm", "goal "), ("scenario", "open "), ("scenario", "OPEN"), ("seed", 1001.0)]
)
def test_producer_readers_refuse_noncanonical_pair_ids(identity_sweep, reader, field, value):
    def read():
        if reader == "metadata":
            return producer._verify_sweep_metadata(identity_sweep, "a" * 40)
        return {
            "rows": producer._load_sweep_rows,
            "paths": producer._sweep_episode_paths,
            "statuses": producer._sweep_execution_statuses,
        }[reader](identity_sweep)

    read()
    _change_identity(identity_sweep, field, value)
    with pytest.raises(ValueError, match="receipt identity refused: noncanonical"):
        read()


@pytest.mark.parametrize(
    "field,value",
    [
        ("arm", "goal "),
        ("map", "open "),
        ("map", "OPEN"),
        ("seed", 1001.0),
        ("seed", "1001"),
        ("seed", True),
    ],
)
def test_shared_receipt_keys_refuse_noncanonical_ids(field, value):
    row = {"arm": "goal", "map": "open", "seed": 1001}
    assert gate._key(row) == ("goal", "open", 1001)
    row[field] = value
    before = copy.deepcopy(row)
    with pytest.raises(ValueError, match="receipt identity refused: noncanonical"):
        gate._key(row)
    assert row == before


def test_comparison_refuses_float_seed_keys_before_set_coverage():
    scope = {"arms": ["goal"], "maps": ["open"]}
    valid = {("goal", "open", seed): _episode() for seed in eq.SEEDS}
    assert len(eq.compare_rows(valid, valid, scope)) == 3
    aliases = {(arm, scenario, float(seed)): row for (arm, scenario, seed), row in valid.items()}
    with pytest.raises(ValueError, match="receipt identity refused: noncanonical seed"):
        eq.compare_rows(aliases, aliases, scope)


@pytest.mark.parametrize("seeds", [[1001.0, 1002, 1003], ["1001", 1002, 1003], [True, 1002, 1003]])
def test_seed_slice_refuses_coercible_ids(seeds):
    assert producer._check_seed_range([1001, 1002, 1003]) == [1001, 1002, 1003]
    with pytest.raises(ValueError, match="receipt identity refused: noncanonical seed"):
        producer._check_seed_range(seeds)


@pytest.mark.parametrize("fault", ["map_case", "float_seed"])
def test_sealed_sources_refuse_noncanonical_ids(exact_source_sweeps, fault):
    from scripts.ci import receipt_provenance
    from tests.validation.receipt_scheduler_fixture import NODE

    _, _, head = exact_source_sweeps
    source = json.loads((head / "execution_main.json").read_text())["head_sha"]

    def read():
        return receipt_provenance.verified_raw_rows(
            head, "11", NODE, source_sha=source, arm_role="head"
        )

    assert len(read()) == 2142
    if fault == "map_case":
        _relabel_sweeps((head,), field="maps", old="map-0", new="MAP-0")
    else:
        _change_identity(head, "seed", 1001.0)
        _refresh_equivalence_binding(head, validate_identity=False)
    before = _sealed_bytes((head,))
    with pytest.raises(ValueError, match="receipt identity refused: noncanonical"):
        read()
    assert _sealed_bytes((head,)) == before


def test_campaign_directory_and_scenario_identity_are_normalized(tmp_path):
    from robot_sf.benchmark.camera_ready._config import _parse_planner_specs, _sanitize_name
    from robot_sf.training.scenario_loader import select_scenario
    from scripts.ci import receipt_provenance

    planners = _parse_planner_specs(
        [{"key": " Goal ", "algo": "goal"}, {"key": "goal", "algo": "goal"}], base_dir=tmp_path
    )
    assert [planner.key for planner in planners] == ["Goal", "goal"]
    assert [_sanitize_name(planner.key) for planner in planners] == ["goal", "goal"]
    with pytest.raises(ValueError, match="receipt identity refused: noncanonical arm"):
        receipt_provenance.canonical_id(planners[0].key, "arm")
    scenarios = [{"name": "open"}, {"name": "OPEN"}]
    assert select_scenario(scenarios, "open") is scenarios[0]
    assert select_scenario(scenarios, "OPEN") is scenarios[0]
