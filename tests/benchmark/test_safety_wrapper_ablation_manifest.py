"""Tests for the dry-run safety-wrapper factorial-ablation manifest (issue #3501)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.safety_wrapper_ablation_manifest import (
    SAFETY_WRAPPER_ABLATION_SCHEMA,
    SAFETY_WRAPPER_MODE_DISABLED,
    SAFETY_WRAPPER_MODE_ENABLED,
    SAFETY_WRAPPER_MODE_FIELD,
    WRAPPER_OFF_ARM,
    WRAPPER_ON_ARM,
    ManifestOptions,
    build_safety_wrapper_ablation_manifest,
    check_factorial_ablation,
    check_factorial_ablation_rows,
    load_safety_wrapper_ablation_config,
    write_safety_wrapper_ablation_manifest,
)
from robot_sf.robot.safety_wrapper import SAFETY_WRAPPER_SCHEMA

_CONFIG_PATH = "configs/research/safety_wrapper_ablation_v1.yaml"

if TYPE_CHECKING:
    from pathlib import Path


def _repo_config() -> dict[str, object]:
    return load_safety_wrapper_ablation_config(_CONFIG_PATH)


def _options() -> ManifestOptions:
    return ManifestOptions(config_path=_CONFIG_PATH, git_head="abc1234")


def test_manifest_factorizes_planners_over_wrapper_off_on() -> None:
    """Every planner gets exactly the off and on arm; cells carry the shared paired seeds."""
    config = _repo_config()
    manifest = build_safety_wrapper_ablation_manifest(config, options=_options())

    assert manifest["schema_version"] == SAFETY_WRAPPER_ABLATION_SCHEMA
    assert manifest["safety_wrapper_schema"] == SAFETY_WRAPPER_SCHEMA
    assert manifest["status"] == "manifest_dry_run_only"
    assert manifest["evidence_status"] == "not_benchmark_evidence"

    planners = manifest["planner_groups"]
    assert manifest["cell_count"] == len(planners) * 2
    # Each planner appears once per arm.
    for planner in planners:
        arms_for_planner = sorted(
            cell["wrapper_arm"] for cell in manifest["cells"] if cell["planner"] == planner
        )
        assert arms_for_planner == sorted([WRAPPER_OFF_ARM, WRAPPER_ON_ARM])
    # Paired seeds are applied identically to every cell.
    for cell in manifest["cells"]:
        assert cell["seeds"] == manifest["seeds"]
        assert cell[SAFETY_WRAPPER_MODE_FIELD] == (
            SAFETY_WRAPPER_MODE_ENABLED
            if cell["wrapper_arm"] == WRAPPER_ON_ARM
            else SAFETY_WRAPPER_MODE_DISABLED
        )

    check = manifest["factorial_check"]
    assert check["complete"] is True
    assert check["arms_are_off_on"] is True
    assert check["off_on_enabled"] is True
    assert check["seeds_paired_across_arms"] is True
    assert check["expected_cell_count"] == len(planners) * 2


def test_manifest_echoes_predeclared_wrapper_thresholds_as_provenance() -> None:
    """The on arm echoes the fixed, predeclared SafetyWrapperConfig thresholds."""
    manifest = build_safety_wrapper_ablation_manifest(_repo_config(), options=_options())

    arms = {arm["key"]: arm for arm in manifest["wrapper_arms"]}
    off_arm = arms[WRAPPER_OFF_ARM]
    on_arm = arms[WRAPPER_ON_ARM]

    assert off_arm["enabled"] is False
    assert off_arm["baseline"] is True
    assert off_arm["wrapper_config"] is None
    assert off_arm[SAFETY_WRAPPER_MODE_FIELD] == SAFETY_WRAPPER_MODE_DISABLED

    assert on_arm["enabled"] is True
    assert on_arm["baseline"] is False
    assert on_arm["thresholds_source"] == "predeclared_fixed_no_per_planner_tuning"
    assert on_arm["wrapper_config"] == {
        "pedestrian_caution_radius_m": 2.0,
        "capped_speed_m_s": 0.5,
        "ttc_veto_threshold_s": 1.0,
        "clearance_veto_m": 0.3,
    }
    assert on_arm[SAFETY_WRAPPER_MODE_FIELD] == SAFETY_WRAPPER_MODE_ENABLED
    assert on_arm["runtime_binding_status"] == "unresolved_runtime_binding"
    assert manifest["event_ledger_target"] == 3482


def test_manifest_claim_boundary_prevents_benchmark_or_paper_claims() -> None:
    """The dry-run manifest must not be worded as evidence."""
    manifest = build_safety_wrapper_ablation_manifest(_repo_config(), options=_options())

    claim_boundary = manifest["claim_boundary"]
    assert "dry-run factorial-ablation manifest only" in claim_boundary
    assert "not benchmark evidence" in claim_boundary
    assert "not a mitigation-effectiveness result" in claim_boundary
    assert "not paper-facing evidence" in claim_boundary
    assert manifest["dry_run"] is True


def test_config_rejects_arm_that_breaks_off_on_factorization() -> None:
    """An enabled off arm breaks the wrapper off/on contrast and must be rejected."""
    config = _repo_config()
    arms = {arm["key"]: arm for arm in config["wrapper_arms"]}
    arms[WRAPPER_OFF_ARM]["enabled"] = True

    with pytest.raises(ValueError, match="wrapper_off.*enabled: false"):
        build_safety_wrapper_ablation_manifest(config, options=_options())


def test_config_rejects_non_positive_wrapper_threshold() -> None:
    """On-arm thresholds must construct a real SafetyWrapperConfig (positive thresholds)."""
    config = _repo_config()
    arms = {arm["key"]: arm for arm in config["wrapper_arms"]}
    arms[WRAPPER_ON_ARM]["config"]["capped_speed_m_s"] = 0.0

    with pytest.raises(ValueError, match="capped_speed_m_s must be > 0"):
        build_safety_wrapper_ablation_manifest(config, options=_options())


def test_config_rejects_unpaired_seeds() -> None:
    """Duplicate seeds break the paired-seed contract across arms."""
    config = _repo_config()
    config["fixed_scope"]["seeds"] = [111, 111, 112]

    with pytest.raises(ValueError, match="seeds must be unique"):
        build_safety_wrapper_ablation_manifest(config, options=_options())


def test_config_rejects_non_mapping_wrapper_arm() -> None:
    """A non-mapping entry in wrapper_arms is rejected cleanly (no raw TypeError)."""
    config = _repo_config()
    config["wrapper_arms"].append("not-a-mapping")

    with pytest.raises(ValueError, match="each entry in wrapper_arms must be a mapping"):
        build_safety_wrapper_ablation_manifest(config, options=_options())


def test_check_factorial_ablation_flags_missing_arm() -> None:
    """The checker reports incomplete factorization when an arm is missing."""
    only_off = [{"key": WRAPPER_OFF_ARM, "enabled": False, "baseline": True}]
    report = check_factorial_ablation(["orca"], only_off, [111, 112])

    assert report["complete"] is False
    assert report["arms_are_off_on"] is False


def test_manifest_json_output_is_deterministic(tmp_path: Path) -> None:
    """Repeated builds and writes with the same inputs produce byte-identical JSON."""
    config = _repo_config()
    options = _options()

    first = build_safety_wrapper_ablation_manifest(config, options=options)
    second = build_safety_wrapper_ablation_manifest(config, options=options)

    assert first == second
    assert json.dumps(first, indent=2, sort_keys=True) == json.dumps(
        second, indent=2, sort_keys=True
    )

    first_path = write_safety_wrapper_ablation_manifest(first, tmp_path / "first")
    second_path = write_safety_wrapper_ablation_manifest(second, tmp_path / "second")

    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")


def _ablation_row(wrapper_arm: str, seed: int = 111) -> dict[str, object]:
    wrapper_mode = (
        SAFETY_WRAPPER_MODE_ENABLED
        if wrapper_arm == WRAPPER_ON_ARM
        else SAFETY_WRAPPER_MODE_DISABLED
    )
    return {
        "study_id": "issue_3501_safety_wrapper_ablation_v1",
        "planner": "orca",
        "wrapper_arm": wrapper_arm,
        SAFETY_WRAPPER_MODE_FIELD: wrapper_mode,
        "scenario_id": "crossing_basic",
        "seed": seed,
        "software_commit": "abc1234",
        "event_ledger": {"schema_version": "EpisodeEventLedger.v1"},
        "metric_values": {"exact_collision_probability": 0.0},
        "wrapper_intervention_rate": 0.0,
    }


def test_manifest_declares_row_contract_for_later_result_checker() -> None:
    """Manifest publishes fields required before any wrapper on/off comparison."""
    manifest = build_safety_wrapper_ablation_manifest(_repo_config(), options=_options())
    row_contract = manifest["row_contract"]

    assert row_contract["required_fields"] == manifest["result_contract"]["required_outputs"]
    assert row_contract["pairing_key_fields"] == ["planner", "scenario_id", "seed"]
    assert row_contract["expected_wrapper_arms"] == [WRAPPER_OFF_ARM, WRAPPER_ON_ARM]
    assert "exactly one wrapper_off row and one wrapper_on row" in row_contract["pairing_rule"]


def test_check_factorial_ablation_rows_accepts_exact_paired_rows() -> None:
    """Result rows are complete only with one off/on arm per planner/scenario/seed."""
    rows = [_ablation_row(WRAPPER_OFF_ARM), _ablation_row(WRAPPER_ON_ARM)]

    report = check_factorial_ablation_rows(rows)

    assert report["complete"] is True
    assert report["row_count"] == 2
    assert report["pair_count"] == 1
    assert report["missing_required_fields"] == []
    assert report["invalid_provenance_fields"] == []
    assert report["incomplete_pairs"] == []
    assert report["pair_provenance_mismatches"] == []


def test_check_factorial_ablation_rows_rejects_arm_mode_mismatch() -> None:
    """Wrapper arm and opt-in mode provenance must agree before comparison."""
    off_row = _ablation_row(WRAPPER_OFF_ARM)
    on_row = _ablation_row(WRAPPER_ON_ARM)
    on_row[SAFETY_WRAPPER_MODE_FIELD] = SAFETY_WRAPPER_MODE_DISABLED

    report = check_factorial_ablation_rows([off_row, on_row])

    assert report["complete"] is False
    assert report["invalid_provenance_fields"] == [
        {"row_index": 1, "fields": [SAFETY_WRAPPER_MODE_FIELD]}
    ]


def test_check_factorial_ablation_rows_rejects_mixed_pair_provenance() -> None:
    """Paired off/on rows must come from the same study and software commit."""
    off_row = _ablation_row(WRAPPER_OFF_ARM)
    on_row = _ablation_row(WRAPPER_ON_ARM)
    on_row["study_id"] = "different_study"
    on_row["software_commit"] = "def5678"

    report = check_factorial_ablation_rows([off_row, on_row])

    assert report["complete"] is False
    assert report["pair_provenance_mismatches"] == [
        {
            "pairing_key": {"planner": "orca", "scenario_id": "crossing_basic", "seed": 111},
            "fields": ["study_id", "software_commit"],
        }
    ]


def test_check_factorial_ablation_rows_rejects_missing_provenance_and_unpaired_arm() -> None:
    """Missing provenance fields or one-arm rows fail closed before comparison."""
    row = _ablation_row(WRAPPER_ON_ARM)
    del row["software_commit"]

    report = check_factorial_ablation_rows([row])

    assert report["complete"] is False
    assert report["missing_required_fields"] == [{"row_index": 0, "fields": ["software_commit"]}]
    assert report["incomplete_pairs"] == [
        {
            "pairing_key": {"planner": "orca", "scenario_id": "crossing_basic", "seed": 111},
            "wrapper_arms": [WRAPPER_ON_ARM],
        }
    ]


def test_check_safety_wrapper_ablation_rows_cli_fails_malformed_provenance(
    tmp_path: Path,
) -> None:
    """CLI rejects pair-complete rows missing usable evidence fields."""
    import subprocess
    import sys

    rows = [_ablation_row(WRAPPER_OFF_ARM), _ablation_row(WRAPPER_ON_ARM)]
    rows[0]["event_ledger"] = {"schema_version": "WrongLedger.v1"}
    rows[1]["wrapper_intervention_rate"] = -0.1
    rows_path = tmp_path / "malformed_rows.jsonl"
    rows_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/check_safety_wrapper_ablation_rows.py",
            "--rows",
            str(rows_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["complete"] is False
    assert report["incomplete_pairs"] == []
    assert report["invalid_provenance_fields"] == [
        {"row_index": 0, "fields": ["event_ledger"]},
        {"row_index": 1, "fields": ["wrapper_intervention_rate"]},
    ]


def test_check_factorial_ablation_rows_rejects_malformed_provenance_values() -> None:
    """Pair-complete rows still fail closed when provenance values are unusable."""
    off_row = _ablation_row(WRAPPER_OFF_ARM)
    off_row.update(
        {
            "software_commit": "",
            "event_ledger": {"schema_version": "WrongLedger.v1"},
            "metric_values": {},
            "wrapper_intervention_rate": 1.25,
        }
    )
    on_row = _ablation_row(WRAPPER_ON_ARM)
    on_row["seed"] = True

    report = check_factorial_ablation_rows([off_row, on_row])

    assert report["complete"] is False
    assert report["invalid_provenance_fields"] == [
        {
            "row_index": 0,
            "fields": [
                "software_commit",
                "event_ledger",
                "metric_values",
                "wrapper_intervention_rate",
            ],
        },
        {"row_index": 1, "fields": ["seed"]},
    ]


def test_check_factorial_ablation_rows_rejects_duplicate_or_unknown_arm() -> None:
    """Duplicate or non-contract wrapper arms cannot be compared."""
    rows = [
        _ablation_row(WRAPPER_OFF_ARM),
        _ablation_row(WRAPPER_OFF_ARM),
        _ablation_row("wrapper_auto"),
    ]

    report = check_factorial_ablation_rows(rows)

    assert report["complete"] is False
    assert report["unexpected_wrapper_arms"] == ["wrapper_auto"]
    assert report["duplicate_pair_rows"] == [
        {
            "pairing_key": {"planner": "orca", "scenario_id": "crossing_basic", "seed": 111},
            "wrapper_arm": WRAPPER_OFF_ARM,
            "count": 2,
        }
    ]


def test_load_safety_wrapper_ablation_rows_reads_jsonl_and_json_list(tmp_path: Path) -> None:
    """The public checker accepts benchmark-style JSONL and compact JSON fixtures."""
    from robot_sf.benchmark.safety_wrapper_ablation_manifest import (
        load_safety_wrapper_ablation_rows,
    )

    off_row = _ablation_row(WRAPPER_OFF_ARM)
    on_row = _ablation_row(WRAPPER_ON_ARM)
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in [off_row, on_row]) + "\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "rows.json"
    json_path.write_text(json.dumps([off_row, on_row], sort_keys=True), encoding="utf-8")

    assert load_safety_wrapper_ablation_rows(jsonl_path) == [off_row, on_row]
    assert load_safety_wrapper_ablation_rows(json_path) == [off_row, on_row]


def test_check_safety_wrapper_ablation_rows_cli_passes_and_writes_report(
    tmp_path: Path,
) -> None:
    """CLI returns success only for rows with exact off/on pairs."""
    import subprocess
    import sys

    rows_path = tmp_path / "paired_rows.jsonl"
    rows_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in [_ablation_row(WRAPPER_OFF_ARM), _ablation_row(WRAPPER_ON_ARM)]
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/check_safety_wrapper_ablation_rows.py",
            "--rows",
            str(rows_path),
            "--out",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["complete"] is True
    assert report["pair_count"] == 1


def test_check_safety_wrapper_ablation_rows_cli_fails_unpaired_rows(tmp_path: Path) -> None:
    """CLI fails closed before any one-arm row can be compared."""
    import subprocess
    import sys

    rows_path = tmp_path / "unpaired_rows.jsonl"
    rows_path.write_text(
        json.dumps(_ablation_row(WRAPPER_ON_ARM), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/check_safety_wrapper_ablation_rows.py",
            "--rows",
            str(rows_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["complete"] is False
    assert report["incomplete_pairs"] == [
        {
            "pairing_key": {"planner": "orca", "scenario_id": "crossing_basic", "seed": 111},
            "wrapper_arms": [WRAPPER_ON_ARM],
        }
    ]
