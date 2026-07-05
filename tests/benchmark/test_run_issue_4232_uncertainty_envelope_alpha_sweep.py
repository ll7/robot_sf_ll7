"""Tests for issue #4232 diagnostic uncertainty-envelope alpha sweep runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/benchmark/run_issue_4232_uncertainty_envelope_alpha_sweep.py"
PACKET = REPO_ROOT / "configs/benchmarks/issue_4232_uncertainty_envelope_claim_packet.yaml"

_SPEC = importlib.util.spec_from_file_location("_issue_4232_alpha_sweep", SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _fake_scenarios(_path: Path) -> list[dict[str, object]]:
    return [
        {
            "id": "crossing",
            "name": "crossing",
            "map_file": "maps/svg_maps/classic_crossing.svg",
            "metadata": {"scenario_family": "classic_crossing"},
            "simulation_config": {},
        }
    ]


def _fake_run_map_batch(
    scenarios: list[dict[str, object]],
    out_path: str | Path,
    *_args: object,
    **_kwargs: object,
) -> dict[str, object]:
    scenario = scenarios[0]
    alpha = float(scenario["simulation_config"]["pedestrian_uncertainty_alpha_mps"])  # type: ignore[index]
    enabled = bool(scenario["simulation_config"]["pedestrian_uncertainty_envelope_enabled"])  # type: ignore[index]
    record = {
        "scenario_id": scenario["id"],
        "seed": 111,
        "success": True,
        "collision": False,
        "metrics": {
            "near_misses": 0,
            "mean_clearance": 0.8 + alpha,
            "path_efficiency": 0.7,
            "runtime_seconds": 0.05 + alpha,
        },
        "planner_runtime": {
            "pedestrian_uncertainty_envelope": {
                "enabled": enabled,
                "alpha_mps": alpha,
                "effective_radius_used_by_planner": enabled and alpha > 0.0,
                "envelope_activation_count": 2 if enabled and alpha > 0.0 else 0,
            }
        },
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return {"written": 1}


def test_issue_4232_diagnostic_runner_writes_compact_diagnostic_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner executes paired local arms and keeps rows diagnostic-only by default."""
    calls: list[dict[str, object]] = []

    def _tracking_run_map_batch(
        scenarios: list[dict[str, object]],
        out_path: str | Path,
        *args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append({"scenario": scenarios[0], "out_path": str(out_path), "kwargs": kwargs})
        return _fake_run_map_batch(scenarios, out_path, *args, **kwargs)

    monkeypatch.setattr(_MODULE, "load_scenario_matrix", _fake_scenarios)
    monkeypatch.setattr(_MODULE.map_runner, "run_map_batch", _tracking_run_map_batch)

    report = _MODULE.run_diagnostic(
        _MODULE.parse_args(
            [
                "--packet",
                str(PACKET),
                "--output-dir",
                str(tmp_path / "issue_4232"),
                "--max-scenarios",
                "1",
                "--max-seeds",
                "1",
            ]
        )
    )

    assert report["ok"] is True
    assert report["row_count"] == 3
    assert report["default_row_status"] == "diagnostic_only"
    assert len(calls) == 3
    assert all("raw_episode_jsonl" in call["out_path"] for call in calls)

    rows_path = tmp_path / "issue_4232" / "compact_alpha_sweep_rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))["rows"]
    assert {row["alpha_arm_key"] for row in rows} == {
        "envelope_off_alpha_0",
        "envelope_on_alpha_0",
        "envelope_on_alpha_0p10",
    }
    assert {row["row_status"] for row in rows} == {"diagnostic_only"}
    nonzero = next(row for row in rows if row["alpha_arm_key"] == "envelope_on_alpha_0p10")
    assert nonzero["diagnostics"]["effective_radius_used_by_planner"] is True

    claim_readiness = (
        tmp_path / "issue_4232" / "compact_evidence" / "claim_readiness.md"
    ).read_text(encoding="utf-8")
    assert "not ready for benchmark-strength" in claim_readiness


@pytest.mark.parametrize("row_status", ["diagnostic_only", "failed", "blocked", "not_available"])
def test_activation_diagnostics_keep_missing_telemetry_unknown(row_status: str) -> None:
    """Nonzero-alpha rows without telemetry stay unknown instead of synthesizing activation."""
    diagnostics = _MODULE._activation_diagnostics(
        {},
        arm={
            "pedestrian_uncertainty_alpha_mps": 0.1,
            "pedestrian_uncertainty_envelope_enabled": True,
        },
        row_status=row_status,
    )

    assert diagnostics == {
        "envelope_activation_count": None,
        "effective_radius_used_by_planner": None,
    }


def test_activation_diagnostics_keep_empty_envelope_mapping_unknown() -> None:
    """An empty envelope telemetry mapping must not synthesize activation from arm intent."""
    diagnostics = _MODULE._activation_diagnostics(
        {"planner_runtime": {"pedestrian_uncertainty_envelope": {}}},
        arm={
            "pedestrian_uncertainty_alpha_mps": 0.1,
            "pedestrian_uncertainty_envelope_enabled": True,
        },
        row_status="diagnostic_only",
    )

    assert diagnostics == {
        "envelope_activation_count": None,
        "effective_radius_used_by_planner": None,
    }


def test_activation_diagnostics_preserve_valid_runtime_telemetry() -> None:
    """Well-formed runtime telemetry is preserved without coercion."""
    diagnostics = _MODULE._activation_diagnostics(
        {
            "planner_runtime": {
                "pedestrian_uncertainty_envelope": {
                    "effective_radius_used_by_planner": True,
                    "envelope_activation_count": 3,
                }
            }
        },
        arm={
            "pedestrian_uncertainty_alpha_mps": 0.1,
            "pedestrian_uncertainty_envelope_enabled": True,
        },
        row_status="successful_evidence",
    )

    assert diagnostics == {
        "envelope_activation_count": 3,
        "effective_radius_used_by_planner": True,
    }


def test_activation_diagnostics_read_map_runner_algorithm_metadata() -> None:
    """Real episode rows carry planner runtime diagnostics under algorithm_metadata."""
    diagnostics = _MODULE._activation_diagnostics(
        {
            "algorithm_metadata": {
                "planner_runtime": {
                    "pedestrian_uncertainty_envelope": {
                        "effective_radius_used_by_planner": True,
                        "envelope_activation_count": 4,
                    }
                }
            }
        },
        arm={
            "pedestrian_uncertainty_alpha_mps": 0.1,
            "pedestrian_uncertainty_envelope_enabled": True,
        },
        row_status="diagnostic_only",
    )

    assert diagnostics == {
        "envelope_activation_count": 4,
        "effective_radius_used_by_planner": True,
    }


def test_activation_diagnostics_keep_malformed_fields_unknown() -> None:
    """Malformed telemetry fields remain unknown while valid telemetry stays preserved."""
    diagnostics = _MODULE._activation_diagnostics(
        {
            "planner_runtime": {
                "pedestrian_uncertainty_envelope": {
                    "effective_radius_used_by_planner": "yes",
                    "envelope_activation_count": 3,
                }
            }
        },
        arm={
            "pedestrian_uncertainty_alpha_mps": 0.1,
            "pedestrian_uncertainty_envelope_enabled": True,
        },
        row_status="successful_evidence",
    )

    assert diagnostics == {
        "envelope_activation_count": 3,
        "effective_radius_used_by_planner": None,
    }


def test_activation_diagnostics_keep_alpha_zero_behavior_stable() -> None:
    """Alpha-zero arms still omit activation diagnostics entirely."""
    diagnostics = _MODULE._activation_diagnostics(
        {},
        arm={
            "pedestrian_uncertainty_alpha_mps": 0.0,
            "pedestrian_uncertainty_envelope_enabled": True,
        },
        row_status="diagnostic_only",
    )

    assert diagnostics == {}


def test_issue_4232_diagnostic_runner_keeps_missing_activation_telemetry_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing runtime telemetry for enabled nonzero alpha stays unknown end-to-end."""

    def _missing_nonzero_telemetry_run_map_batch(
        scenarios: list[dict[str, object]],
        out_path: str | Path,
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        scenario = scenarios[0]
        alpha = float(scenario["simulation_config"]["pedestrian_uncertainty_alpha_mps"])  # type: ignore[index]
        enabled = bool(scenario["simulation_config"]["pedestrian_uncertainty_envelope_enabled"])  # type: ignore[index]
        record = {
            "scenario_id": scenario["id"],
            "seed": 111,
            "success": True,
            "collision": False,
            "metrics": {
                "near_misses": 0,
                "mean_clearance": 0.8 + alpha,
                "path_efficiency": 0.7,
                "runtime_seconds": 0.05 + alpha,
            },
        }
        if not (enabled and alpha > 0.0):
            record["planner_runtime"] = {
                "pedestrian_uncertainty_envelope": {
                    "enabled": enabled,
                    "alpha_mps": alpha,
                    "effective_radius_used_by_planner": enabled and alpha > 0.0,
                    "envelope_activation_count": 2 if enabled and alpha > 0.0 else 0,
                }
            }
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        return {"written": 1}

    monkeypatch.setattr(_MODULE, "load_scenario_matrix", _fake_scenarios)
    monkeypatch.setattr(
        _MODULE.map_runner,
        "run_map_batch",
        _missing_nonzero_telemetry_run_map_batch,
    )

    report = _MODULE.run_diagnostic(
        _MODULE.parse_args(
            [
                "--packet",
                str(PACKET),
                "--output-dir",
                str(tmp_path / "issue_4232"),
                "--max-scenarios",
                "1",
                "--max-seeds",
                "1",
            ]
        )
    )

    assert report["ok"] is True

    rows_path = tmp_path / "issue_4232" / "compact_alpha_sweep_rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))["rows"]
    nonzero = next(row for row in rows if row["alpha_arm_key"] == "envelope_on_alpha_0p10")
    assert nonzero["diagnostics"] == {
        "envelope_activation_count": None,
        "effective_radius_used_by_planner": None,
    }

    activation = json.loads(
        (
            tmp_path / "issue_4232" / "compact_evidence" / "envelope_activation_diagnostics.json"
        ).read_text(encoding="utf-8")
    )
    assert activation["status_counts"]["unknown_mechanism_activation"] == 1
    activation_by_arm = {
        row["alpha_arm_key"]: row["activation_status"] for row in activation["rows"]
    }
    assert activation_by_arm["envelope_on_alpha_0p10"] == "unknown_mechanism_activation"


def test_issue_4232_diagnostic_runner_fails_unknown_alpha_arm(tmp_path: Path) -> None:
    """Runner rejects alpha arms not pre-registered in the packet."""
    args = _MODULE.parse_args(
        [
            "--packet",
            str(PACKET),
            "--output-dir",
            str(tmp_path / "issue_4232"),
            "--alpha-arm",
            "not_registered",
        ]
    )
    with pytest.raises(_MODULE.DiagnosticRunError, match="unknown alpha arm"):
        _MODULE.run_diagnostic(args)
