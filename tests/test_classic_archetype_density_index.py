"""Consistency tests for the classic archetype density / tier index (issue #3725).

The index file ``configs/scenarios/archetypes/classic_density_tier_index.yaml`` is a
machine-readable, *derived* description of the current density-tier coverage and
pedestrian spawn semantics of the classic interaction archetypes. These tests
re-derive the same facts directly from the ``classic_*.yaml`` configs (and the
``classic_interactions.yaml`` includes) and fail closed if the index drifts from
the configs or omits a config / tier. They do not run a simulator and do not
assert anything about benchmark results.

What is guarded:
  * every in-directory ``classic_*.yaml`` config (and every density tier inside
    it) is covered by the index (missing-coverage guard, both directions);
  * every classic archetype config is either admitted by
    ``classic_interactions.yaml`` or explicitly marked ``evaluation: excluded``
    or ``evaluation: planned`` with a reason (issue #4971);
  * each tier's ``ped_density`` and ``density_advisory`` match the config;
  * each config's ``spawn_mode`` / ``in_matrix`` flags match what the configs and
    ``classic_interactions.yaml`` actually say;
  * the marker-spawn semantics: ``spawn_mode: markers`` always means
    ``ped_density == 0.0`` plus ``density_advisory: zero_baseline_route_spawn``
    (the overloaded ``density=0`` placeholder, not an empty scene);
  * the aggregate ``summary`` counts match the recomputed values.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
ARCHETYPES_DIR = ROOT / "configs" / "scenarios" / "archetypes"
INDEX_FILE = ARCHETYPES_DIR / "classic_density_tier_index.yaml"
MATRIX_FILE = ROOT / "configs" / "scenarios" / "classic_interactions.yaml"

ZERO_DENSITY_ADVISORY = "zero_baseline_route_spawn"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _classic_config_paths() -> list[Path]:
    """All ``classic_*.yaml`` archetype configs (excluding the index itself)."""
    return sorted(p for p in ARCHETYPES_DIR.glob("classic_*.yaml") if p != INDEX_FILE)


def _derive_spawn_mode(scenario: dict) -> str:
    """Classify a scenario's pedestrian spawn mode from its config fields.

    Mirrors the definitions documented in the index ``spawn_modes`` block.
    """
    sim_cfg = scenario.get("simulation_config", {}) or {}
    metadata = scenario.get("metadata", {}) or {}
    density = sim_cfg.get("ped_density")
    advisory = metadata.get("density_advisory")
    explicit_spawn_mode = metadata.get("spawn_mode")
    if explicit_spawn_mode == "markers":
        return "markers"
    has_scripted = bool(scenario.get("single_pedestrians"))
    if density == 0.0 and advisory == ZERO_DENSITY_ADVISORY:
        raise AssertionError(
            f"{scenario['name']}: marker spawning must use metadata.spawn_mode=markers"
        )
    if has_scripted and density:
        return "scripted_and_route_density"
    return "route_density"


def _derive_config_facts(path: Path) -> dict:
    """Re-derive the per-config facts the index claims, straight from the config."""
    data = _load_yaml(path)
    scenarios = data.get("scenarios", [])
    assert scenarios, f"{path.name}: no scenarios defined"

    archetypes = {s["metadata"]["archetype"] for s in scenarios}
    assert len(archetypes) == 1, f"{path.name}: mixed archetypes {archetypes}"
    spawn_modes = {_derive_spawn_mode(s) for s in scenarios}
    assert len(spawn_modes) == 1, f"{path.name}: mixed spawn modes {spawn_modes}"

    tiers = {}
    for scenario in scenarios:
        metadata = scenario["metadata"]
        tier = metadata["density"]
        assert tier not in tiers, f"{path.name}: duplicate tier {tier!r}"
        tiers[tier] = {
            "scenario": scenario["name"],
            "ped_density": (scenario.get("simulation_config", {}) or {}).get("ped_density"),
            "density_advisory": metadata.get("density_advisory"),
        }
    return {
        "archetype": next(iter(archetypes)),
        "spawn_mode": next(iter(spawn_modes)),
        "deprecated": any(s["metadata"].get("deprecated_alias_for") for s in scenarios),
        "tiers": tiers,
    }


def _matrix_config_filenames() -> set[str]:
    """Config filenames composed by classic_interactions.yaml."""
    data = _load_yaml(MATRIX_FILE)
    return {Path(inc).name for inc in data.get("includes", [])}


def _index_entries_by_config() -> dict[str, dict]:
    index = _load_yaml(INDEX_FILE)
    return {entry["config"]: entry for entry in index["archetypes"]}


def _assert_evaluation_replacement_is_valid(config_path: Path, replacement: object) -> None:
    """Require an explicit replacement to name an archetype config file."""
    assert isinstance(replacement, str) and replacement.strip(), (
        f"{config_path.name}: evaluation_replacement must be a non-empty string."
    )
    assert (ARCHETYPES_DIR / replacement).is_file(), (
        f"{config_path.name}: evaluation_replacement={replacement!r} must name a real file."
    )


def test_index_file_exists_and_is_inert() -> None:
    """Index loads and has no ``scenarios`` key, so scenario loaders ignore it."""
    assert INDEX_FILE.exists(), f"Missing index: {INDEX_FILE}"
    index = _load_yaml(INDEX_FILE)
    assert index["schema"] == "robot_sf.classic_archetype_density_index.v1"
    assert "scenarios" not in index, "index must not look like a scenario config"


def test_every_config_and_tier_is_covered() -> None:
    """Missing-coverage guard: index entries match the configs exactly (both ways)."""
    config_facts = {p.name: _derive_config_facts(p) for p in _classic_config_paths()}
    index_entries = _index_entries_by_config()

    assert set(index_entries) == set(config_facts), (
        "Index config set does not match classic_*.yaml files. "
        f"Only in index: {set(index_entries) - set(config_facts)}; "
        f"only on disk: {set(config_facts) - set(index_entries)}"
    )

    for name, facts in config_facts.items():
        entry = index_entries[name]
        assert entry["archetype"] == facts["archetype"], name
        assert entry["spawn_mode"] == facts["spawn_mode"], name

        index_tiers = {t["tier"]: t for t in entry["tiers"]}
        assert set(index_tiers) == set(facts["tiers"]), (
            f"{name}: tier coverage mismatch "
            f"(index {set(index_tiers)} vs config {set(facts['tiers'])})"
        )
        for tier, cfg_tier in facts["tiers"].items():
            idx_tier = index_tiers[tier]
            assert idx_tier["scenario"] == cfg_tier["scenario"], (name, tier)
            assert idx_tier["ped_density"] == cfg_tier["ped_density"], (name, tier)
            # density_advisory is optional in the index when the config has none.
            assert idx_tier.get("density_advisory") == cfg_tier["density_advisory"], (
                name,
                tier,
            )


def test_in_matrix_flag_matches_includes() -> None:
    """``in_matrix`` must match classic_interactions.yaml's includes list."""
    matrix_files = _matrix_config_filenames()
    for config, entry in _index_entries_by_config().items():
        assert entry["in_matrix"] == (config in matrix_files), config


def test_every_classic_archetype_is_admitted_or_explicitly_dispositioned() -> None:
    """Fail closed when a classic config is outside the standard manifest silently."""
    matrix_files = _matrix_config_filenames()
    allowed_dispositions = {"excluded", "planned"}

    for config_path in _classic_config_paths():
        config = _load_yaml(config_path)
        if "evaluation_replacement" in config:
            _assert_evaluation_replacement_is_valid(
                config_path,
                config["evaluation_replacement"],
            )

        if config_path.name in matrix_files:
            continue

        disposition = config.get("evaluation")
        reason = config.get("evaluation_reason")
        assert disposition in allowed_dispositions, (
            f"{config_path.name}: not included by {MATRIX_FILE.name}; add it to the manifest or "
            "set evaluation to 'excluded' or 'planned'."
        )
        assert isinstance(reason, str) and reason.strip(), (
            f"{config_path.name}: evaluation={disposition!r} requires a non-empty "
            "evaluation_reason."
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("", "non-empty string"),
        ("   ", "non-empty string"),
        (["classic_cross_trap.yaml"], "non-empty string"),
        ("missing_classic_config.yaml", "must name a real file"),
    ],
)
def test_evaluation_replacement_fails_closed_when_invalid(
    replacement: object,
    message: str,
) -> None:
    """Disposition replacements must be usable archetype config paths."""
    with pytest.raises(AssertionError, match=message):
        _assert_evaluation_replacement_is_valid(
            ARCHETYPES_DIR / "classic_example.yaml",
            replacement,
        )


def test_marker_spawn_density_zero_semantics() -> None:
    """Every ``markers`` tier encodes the overloaded ped_density=0.0 placeholder."""
    marker_tiers = 0
    config_facts = {path.name: _derive_config_facts(path) for path in _classic_config_paths()}
    for config_name, entry in _index_entries_by_config().items():
        if entry["spawn_mode"] != "markers":
            continue
        assert config_facts[config_name]["spawn_mode"] == "markers", config_name
        for tier in entry["tiers"]:
            assert tier["ped_density"] == 0.0, entry["config"]
            assert tier["density_advisory"] == ZERO_DENSITY_ADVISORY, entry["config"]
            marker_tiers += 1
    assert marker_tiers > 0, "expected at least one marker-spawn tier in the density index"


def test_summary_counts_match_recomputation() -> None:
    """Aggregate ``summary`` block matches values recomputed from the configs."""
    index = _load_yaml(INDEX_FILE)
    entries = index["archetypes"]
    summary = index["summary"]

    in_matrix = [e for e in entries if e["in_matrix"]]
    excluded = [e for e in entries if not e["in_matrix"]]

    rows_by_mode: dict[str, int] = {}
    full_triad = low_medium = single_tier = 0
    for entry in in_matrix:
        rows_by_mode[entry["spawn_mode"]] = rows_by_mode.get(entry["spawn_mode"], 0) + len(
            entry["tiers"]
        )
        tiers = {t["tier"] for t in entry["tiers"]}
        if tiers == {"low", "medium", "high"}:
            full_triad += 1
        elif tiers == {"low", "medium"}:
            low_medium += 1
        elif len(tiers) == 1:
            single_tier += 1

    im = summary["in_matrix"]
    assert im["config_files"] == len(in_matrix)
    assert im["graded_scenario_rows"] == sum(len(e["tiers"]) for e in in_matrix)
    assert im["rows_by_spawn_mode"] == rows_by_mode
    assert im["tier_coverage"] == {
        "full_triad": full_triad,
        "low_medium": low_medium,
        "single_tier": single_tier,
    }

    de = summary["deprecated_excluded"]
    assert de["config_files"] == len(excluded)
    assert de["graded_scenario_rows"] == sum(len(e["tiers"]) for e in excluded)


@pytest.mark.parametrize("entry", _index_entries_by_config().values(), ids=lambda e: e["config"])
def test_index_scenarios_reference_real_configs(entry: dict) -> None:
    """Each index scenario name actually appears in its referenced config file."""
    config_path = ARCHETYPES_DIR / entry["config"]
    names = {s["name"] for s in _load_yaml(config_path).get("scenarios", [])}
    for tier in entry["tiers"]:
        assert tier["scenario"] in names, (entry["config"], tier["scenario"])
