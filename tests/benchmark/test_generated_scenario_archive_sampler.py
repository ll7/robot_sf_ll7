"""Tests for #4932 deterministic rare-event sampling over generated archives."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from robot_sf.benchmark.scenario_generation.archive_sampler import (
    ArchiveSamplingSpec,
    GeneratedScenarioArchiveSamplingError,
    run_archive_sampling,
    sample_generated_archive,
)
from robot_sf.benchmark.scenario_generation.segment_extraction import extract_critical_segment

if TYPE_CHECKING:
    from pathlib import Path


def _entry(episode_id: str, clearance_m: float) -> dict[str, Any]:
    return extract_critical_segment(
        {
            "episode_id": episode_id,
            "seed": 4932,
            "source_map": "maps/svg_maps/classic_crossing.svg",
            "steps": [
                {
                    "time_s": 0.0,
                    "robot": {"position": [0.0, 0.0]},
                    "pedestrians": [{"position": [3.0, 0.0]}],
                },
                {
                    "time_s": 1.0,
                    "robot": {"position": [1.0, 0.0]},
                    "pedestrians": [{"position": [1.0 + clearance_m, 0.0]}],
                },
                {
                    "time_s": 2.0,
                    "robot": {"position": [2.0, 0.0]},
                    "pedestrians": [{"position": [5.0, 0.0]}],
                },
            ],
        }
    )


def _spec(*, seed: int = 4932, sample_size: int = 2) -> ArchiveSamplingSpec:
    return ArchiveSamplingSpec.from_payload(
        {
            "schema_version": "generated-scenario-rare-event-sampling.v1",
            "seed": seed,
            "sample_size": sample_size,
            "sampler": {
                "type": "criticality_weighted_without_replacement.v1",
                "metric": "min_clearance_m",
                "direction": "lower_is_more_critical",
                "clearance_floor_m": 0.05,
                "exponent": 1.0,
            },
            "claim_boundary": "generated scenario hypotheses only",
        }
    )


def _write_config_and_archive(tmp_path: Path) -> tuple[Path, Path, Path]:
    archive_path = tmp_path / "archive.yaml"
    archive_payload = {
        "schema_version": "generated-scenario-catalog.v1",
        "metadata": {
            "source": "auto_generated",
            "required_manual_review": True,
            "benchmark_evidence": False,
        },
        "entries": [_entry("low", 0.1), _entry("medium", 0.5), _entry("high", 1.5)],
    }
    archive_path.write_text(yaml.safe_dump(archive_payload, sort_keys=True), encoding="utf-8")
    output_path = tmp_path / "selection.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "generated-scenario-rare-event-sampling.v1",
                "seed": 4932,
                "source_archive": archive_path.as_posix(),
                "sample_size": 2,
                "sampler": {
                    "type": "criticality_weighted_without_replacement.v1",
                    "metric": "min_clearance_m",
                    "direction": "lower_is_more_critical",
                    "clearance_floor_m": 0.05,
                    "exponent": 1.0,
                },
                "output_path": output_path.as_posix(),
                "claim_boundary": "generated scenario hypotheses only",
            }
        ),
        encoding="utf-8",
    )
    return config_path, archive_path, output_path


def test_fixed_seed_is_deterministic_and_archive_order_independent() -> None:
    """Stable ids define draw order, so YAML record ordering cannot change selection."""

    entries = [_entry("low", 0.1), _entry("medium", 0.5), _entry("high", 1.5)]
    first = sample_generated_archive(entries, spec=_spec())
    second = sample_generated_archive(list(reversed(entries)), spec=_spec())

    assert first == second
    assert len({row["scenario_id"] for row in first}) == 2
    assert [row["selection_rank"] for row in first] == [1, 2]
    assert all(row["entry"]["metadata"]["benchmark_evidence"] is False for row in first)


def test_lower_clearance_receives_higher_selection_frequency() -> None:
    """Across deterministic seeds, criticality weighting favors the lower-clearance record."""

    critical = _entry("critical", 0.1)
    ordinary = _entry("ordinary", 1.0)
    selected_counts = {critical["scenario_id"]: 0, ordinary["scenario_id"]: 0}
    for seed in range(200):
        selected = sample_generated_archive(
            [ordinary, critical], spec=_spec(seed=seed, sample_size=1)
        )
        selected_counts[selected[0]["scenario_id"]] += 1

    assert selected_counts[critical["scenario_id"]] > 160
    assert selected_counts[ordinary["scenario_id"]] < 40


@pytest.mark.parametrize("entries", [[], ()])
def test_empty_archive_fails_closed(entries: list[Any] | tuple[Any, ...]) -> None:
    """An empty archive never degrades to an empty successful selection."""

    with pytest.raises(GeneratedScenarioArchiveSamplingError, match="archive is empty"):
        sample_generated_archive(entries, spec=_spec(sample_size=1))


def test_undersized_and_malformed_archives_fail_closed() -> None:
    """Sampling cannot silently replace records or skip malformed provenance."""

    entry = _entry("only", 0.2)
    with pytest.raises(GeneratedScenarioArchiveSamplingError, match="exceeds archive size"):
        sample_generated_archive([entry], spec=_spec(sample_size=2))

    malformed = deepcopy(entry)
    malformed["metadata"]["benchmark_evidence"] = True
    with pytest.raises(GeneratedScenarioArchiveSamplingError, match="entry 0 is invalid"):
        sample_generated_archive([malformed], spec=_spec(sample_size=1))

    non_finite = deepcopy(entry)
    non_finite["criticality"]["source_metrics"]["min_clearance_m"] = float("nan")
    with pytest.raises(GeneratedScenarioArchiveSamplingError, match="must be finite"):
        sample_generated_archive([non_finite], spec=_spec(sample_size=1))


def test_run_writes_archive_checksum_sampling_provenance_and_governance(tmp_path: Path) -> None:
    """The standalone path records its exact input, controls, draws, weights, and claim boundary."""

    config_path, archive_path, output_path = _write_config_and_archive(tmp_path)
    result = run_archive_sampling(config_path)
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted == result
    assert result["schema_version"] == "generated-scenario-rare-event-selection.v1"
    assert (
        result["source_archive"]["sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    )
    assert result["source_archive"]["entry_count"] == 3
    assert result["sampler"]["id"] == "criticality_weighted_without_replacement.v1"
    assert all("random_draw" in row and "selection_key" in row for row in result["selected"])
    assert result["governance"] == {
        "required_manual_review": True,
        "benchmark_evidence": False,
        "scenario_certification": False,
    }

    with pytest.raises(FileExistsError, match="already exists"):
        run_archive_sampling(config_path)


def test_run_resolves_relative_paths_from_config_not_cwd(tmp_path: Path, monkeypatch) -> None:
    """A checked-in sampler config remains runnable outside the repository root."""
    config_path, archive_path, output_path = _write_config_and_archive(tmp_path)
    relative_config = tmp_path / "nested" / "config.yaml"
    relative_config.parent.mkdir()
    relative_config.write_text(
        yaml.safe_dump(
            {
                **yaml.safe_load(config_path.read_text(encoding="utf-8")),
                "source_archive": "../archive.yaml",
                "output_path": "../selection.json",
            }
        ),
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = run_archive_sampling(relative_config)

    assert result["source_archive"]["path"] == archive_path.as_posix()
    assert output_path.is_file()


def test_run_rejects_missing_or_wrong_schema_archive(tmp_path: Path) -> None:
    """Missing files and non-generated catalogs are explicit blockers."""

    config_path, archive_path, _output_path = _write_config_and_archive(tmp_path)
    archive_path.unlink()
    with pytest.raises(GeneratedScenarioArchiveSamplingError, match="does not exist"):
        run_archive_sampling(config_path)

    archive_path.write_text("schema_version: unrelated.v1\nentries: []\n", encoding="utf-8")
    with pytest.raises(GeneratedScenarioArchiveSamplingError, match="schema_version"):
        run_archive_sampling(config_path)


def test_run_rejects_archive_with_benchmark_evidence_metadata(tmp_path: Path) -> None:
    """A catalog that claims benchmark status cannot enter this hypothesis-only sampler."""

    config_path, archive_path, _output_path = _write_config_and_archive(tmp_path)
    archive = yaml.safe_load(archive_path.read_text(encoding="utf-8"))
    archive["metadata"]["benchmark_evidence"] = True
    archive_path.write_text(yaml.safe_dump(archive), encoding="utf-8")

    with pytest.raises(GeneratedScenarioArchiveSamplingError, match="benchmark_evidence"):
        run_archive_sampling(config_path)
