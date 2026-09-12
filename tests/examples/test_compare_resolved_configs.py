"""Tests for the resolved-config drift example (issue #8904)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = REPO_ROOT / "examples" / "fixtures" / "config_drift"
_DRIVER = REPO_ROOT / "examples" / "advanced" / "38_compare_resolved_configs.py"


def _load_driver():
    """Load the example driver module from its repository path."""

    spec = importlib.util.spec_from_file_location("compare_resolved_configs", _DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load_driver()


def _write_pair(
    tmp_path: Path, left: str, right: str, base: str | None = None
) -> tuple[Path, Path]:
    """Write a base/left/right fixture triple and return the left/right paths."""

    if base is not None:
        (tmp_path / "base.yaml").write_text(base, encoding="utf-8")
    left_path = tmp_path / "left.yaml"
    right_path = tmp_path / "right.yaml"
    left_path.write_text(left, encoding="utf-8")
    right_path.write_text(right, encoding="utf-8")
    return left_path, right_path


def test_semantic_and_unknown_changes_make_pair_not_comparable() -> None:
    """The fixture pair exposes semantic, unknown, and non-identity changes."""

    report = driver.compare_resolved_configs(
        _FIXTURES / "left.yaml",
        _FIXTURES / "right.yaml",
    )

    assert report["schema"] == "resolved_config_drift.v1"
    assert report["verdict"] == "not_comparable"
    assert set(report["reason_codes"]) == {"semantic_difference", "unknown_difference"}
    assert report["identity_equivalent"] is False

    classes = {change["path"]: change for change in report["changes"]}
    assert classes["seed"]["change_class"] == "semantic"
    assert classes["seed"]["contributes_to_identity"] is True
    assert classes["seed"]["left_origin"].endswith("base.yaml")
    assert classes["seed"]["right_origin"].endswith("right.yaml")
    assert classes["training.hidden_sizes"]["change_class"] == "unknown"
    assert classes["scenario.ped_density_by_difficulty"]["change_class"] == "unknown"
    assert classes["notes"]["change_class"] == "provenance"
    assert classes["display_name"]["change_class"] == "presentation_only"
    assert classes["execution.output_dir"]["change_class"] == "execution_environment"
    assert all(
        classes[path]["contributes_to_identity"] is False
        for path in ("notes", "display_name", "execution.output_dir")
    )


def test_provenance_only_change_is_comparable(tmp_path: Path) -> None:
    """A pair that differs only in bookkeeping fields remains comparable."""

    base = "seed: 7\ntraining:\n  learning_rate: 0.0003\n"
    left = "base_config: base.yaml\nnotes: left notes\n"
    right = "base_config: base.yaml\nnotes: right notes\ncommit: abc123\n"
    left_path, right_path = _write_pair(tmp_path, left, right, base)

    report = driver.compare_resolved_configs(left_path, right_path)

    assert report["verdict"] == "comparable"
    assert report["reason_codes"] == []
    assert report["identity_equivalent"] is True
    assert {change["change_class"] for change in report["changes"]} == {"provenance"}


def test_identical_resolved_configs_are_identical() -> None:
    """Resolving the same fixture twice yields the identical verdict."""

    report = driver.compare_resolved_configs(_FIXTURES / "left.yaml", _FIXTURES / "left.yaml")

    assert report["verdict"] == "identical"
    assert report["changes"] == []
    assert report["identity_equivalent"] is True


def test_incompatible_config_families_fail_closed(tmp_path: Path) -> None:
    """Different declared config families are never comparable."""

    left = "algo: ppo\nseed: 7\n"
    right = "algo: sac\nseed: 7\n"
    left_path, right_path = _write_pair(tmp_path, left, right)

    report = driver.compare_resolved_configs(left_path, right_path)

    assert report["verdict"] == "not_comparable"
    assert "incompatible_config_families" in report["reason_codes"]


def test_duplicate_keys_fail_closed(tmp_path: Path) -> None:
    """Duplicate YAML keys are rejected instead of silently keeping the last value."""

    left, right = _write_pair(
        tmp_path,
        "seed: 7\nseed: 8\n",
        "seed: 7\n",
    )

    with pytest.raises(driver.ConfigDriftError) as exc_info:
        driver.compare_resolved_configs(left, right)

    assert exc_info.value.reason_code == "duplicate_key"


def test_unresolved_interpolation_fails_closed(tmp_path: Path) -> None:
    """Interpolation placeholders are not compared as if resolved."""

    left, right = _write_pair(
        tmp_path,
        "output_dir: ${ROBOT_SF_OUTPUT}\n",
        "output_dir: output/train\n",
    )

    with pytest.raises(driver.ConfigDriftError) as exc_info:
        driver.compare_resolved_configs(left, right)

    assert exc_info.value.reason_code == "unresolved_interpolation"


def test_path_escape_fails_closed(tmp_path: Path) -> None:
    """Values that traverse out of their configuration root are rejected."""

    left, right = _write_pair(
        tmp_path,
        "map_file: ../../etc/passwd\n",
        "map_file: maps/svg_maps/debug_06.svg\n",
    )

    with pytest.raises(driver.ConfigDriftError) as exc_info:
        driver.compare_resolved_configs(left, right)

    assert exc_info.value.reason_code == "path_escape"


def test_missing_config_fails_closed(tmp_path: Path) -> None:
    """A missing input path is refused with a stable reason code."""

    right = tmp_path / "right.yaml"
    right.write_text("seed: 7\n", encoding="utf-8")

    with pytest.raises(driver.ConfigDriftError) as exc_info:
        driver.compare_resolved_configs(tmp_path / "missing.yaml", right)

    assert exc_info.value.reason_code == "missing_config"


def test_invalid_schema_fails_closed(tmp_path: Path) -> None:
    """A non-mapping config is refused instead of coerced."""

    left, right = _write_pair(
        tmp_path,
        "- not\n- a\n- mapping\n",
        "seed: 7\n",
    )

    with pytest.raises(driver.ConfigDriftError) as exc_info:
        driver.compare_resolved_configs(left, right)

    assert exc_info.value.reason_code in {"invalid_schema", "invalid_yaml"}


def test_absolute_paths_are_sanitized(tmp_path: Path) -> None:
    """Absolute private paths never appear verbatim in the report."""

    absolute = tmp_path / "private" / "run"
    left, right = _write_pair(
        tmp_path,
        f"output_dir: {absolute}\n",
        "output_dir: output/train/right\n",
    )

    report = driver.compare_resolved_configs(left, right)
    payload = json.dumps(report)

    assert str(tmp_path) not in payload
    assert "<abs>/run" in payload


def test_report_is_deterministic() -> None:
    """Repeated comparisons produce equal reports with sorted changes."""

    first = driver.compare_resolved_configs(_FIXTURES / "left.yaml", _FIXTURES / "right.yaml")
    second = driver.compare_resolved_configs(_FIXTURES / "left.yaml", _FIXTURES / "right.yaml")

    assert first == second
    paths = [change["path"] for change in first["changes"]]
    assert paths == sorted(paths)


def test_main_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI returns 1 for incomparable pairs and 2 for refused inputs."""

    exit_code = driver.main(
        [
            "--left",
            str(_FIXTURES / "left.yaml"),
            "--right",
            str(_FIXTURES / "right.yaml"),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["verdict"] == "not_comparable"

    exit_code = driver.main(
        ["--left", str(_FIXTURES / "missing.yaml"), "--right", str(_FIXTURES / "left.yaml")]
    )
    error_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert error_payload["error"] == "missing_config"
