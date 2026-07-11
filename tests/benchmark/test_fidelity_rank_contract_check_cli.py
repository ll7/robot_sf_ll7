"""Tests for the issue #4401 successor slice: standalone rank-identifiability contract check.

The merged #4420/#4446 slices defined the ``runtime_rank_identifiability_recheck``
post-run contract and its in-process checker. This successor slice adds the
operator-facing, re-runnable CLI (``--fixed-scope-check-rank-contract``) that
validates an *already-produced* ``fidelity_rank_stability_report.json`` against
the registered contract and exits fail-closed when the report does not satisfy
it. That makes "claim promotion remains blocked unless the post-run report
passes" an independently verifiable gate without re-running any campaign.

These tests run no episode. They build small deterministic report fixtures (the
same shapes ``analyze_fidelity_sensitivity`` emits) and exercise the new CLI
mode in-process through ``main([...])``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "research" / "fidelity_sensitivity_v1.yaml"


def _load_campaign_runner() -> ModuleType:
    module_path = REPO_ROOT / "scripts" / "benchmark" / "run_fidelity_sensitivity_campaign.py"
    spec = importlib.util.spec_from_file_location(
        "fidelity_rank_contract_check_runner", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load campaign runner module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fidelity_rank_contract_check_runner"] = module
    spec.loader.exec_module(module)
    return module


campaign_runner = _load_campaign_runner()


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_report(report: dict, path: Path) -> Path:
    """Write a report fixture in the shape the standalone report writer emits.

    The checker only reads ``rank_identifiable`` / ``rank_identifiability_reason``,
    so the fixture preserves the report's own schema version from
    ``analyze_fidelity_sensitivity``.
    """
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _first_json_object(stdout: str) -> dict:
    """Extract the first balanced JSON object from a stdout string.

    The CLI prints a pretty-printed (indent=2) JSON packet followed by a
    plain-text status line, so line-based parsing is unsafe. This scans for the
    first balanced ``{...}`` object instead.
    """
    import json as _json

    start = stdout.index("{")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(stdout)):
        ch = stdout[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return _json.loads(stdout[start : idx + 1])
    raise AssertionError("no balanced JSON object found in stdout")


def _identifiable_report() -> dict:
    """A report whose primary metric is identifiable (varies across planners)."""
    from robot_sf.benchmark.fidelity_rank_stability import analyze_fidelity_sensitivity

    nominal = {"planner_a": {"snqi": 0.9}, "planner_b": {"snqi": 0.4}}
    report = analyze_fidelity_sensitivity(
        nominal, {}, primary_metric="snqi", drift_metrics=["snqi"]
    ).to_dict()
    # Sanity: this fixture is genuinely identifiable.
    assert report["rank_identifiable"] is True
    return report


def _non_identifiable_report() -> dict:
    """A report whose primary metric is all-tied (zero variance, non-identifiable)."""
    from robot_sf.benchmark.fidelity_rank_stability import analyze_fidelity_sensitivity

    tied = {"planner_a": {"snqi": 0.0}, "planner_b": {"snqi": 0.0}}
    report = analyze_fidelity_sensitivity(
        tied, {}, primary_metric="snqi", drift_metrics=["snqi"]
    ).to_dict()
    assert report["rank_identifiable"] is False
    return report


# ---------------------------------------------------------------------------
# select_rank_identifiability_contract_spec: shared spec selection
# ---------------------------------------------------------------------------


def test_select_spec_returns_registered_rank_contract() -> None:
    """The shared selector finds the rank-identifiability contract in a fresh plan."""
    plan = campaign_runner.build_fixed_scope_run_plan(
        _config(),
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="test-head",
    )
    spec = campaign_runner.select_rank_identifiability_contract_spec(plan)
    assert spec is not None
    assert spec["id"] == campaign_runner.RANK_IDENTIFIABILITY_CONTRACT_ID
    assert spec["threshold"] == "non_zero_variance_and_rank_identifiable"
    assert spec["blocks_claims_when_failed"] is True


def test_select_spec_returns_none_when_absent() -> None:
    """A plan without the contract yields None, not an error (fail-closed downstream)."""
    plan = campaign_runner.build_fixed_scope_run_plan(
        _config(),
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="test-head",
    )
    plan["post_run_contract_specs"] = []
    assert campaign_runner.select_rank_identifiability_contract_spec(plan) is None


# ---------------------------------------------------------------------------
# CLI: --fixed-scope-check-rank-contract
# ---------------------------------------------------------------------------


def test_cli_passes_when_report_is_identifiable(tmp_path: Path) -> None:
    """An identifiable report passes the contract and exits zero."""
    report_path = _write_report(
        _identifiable_report(), tmp_path / "fidelity_rank_stability_report.json"
    )
    exit_code = campaign_runner.main(
        [
            "--fixed-scope-check-rank-contract",
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 0


def test_cli_fails_closed_when_report_is_non_identifiable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A zero-variance report fails the contract and exits non-zero (fail-closed)."""
    report_path = _write_report(
        _non_identifiable_report(), tmp_path / "fidelity_rank_stability_report.json"
    )
    exit_code = campaign_runner.main(
        [
            "--fixed-scope-check-rank-contract",
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    # The fail-closed message and the contract id both appear in the output.
    assert "fail-closed" in captured.out
    assert campaign_runner.RANK_IDENTIFIABILITY_CONTRACT_ID in captured.out


def test_cli_fails_closed_when_report_path_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing --report exits non-zero with an actionable message (no episode run)."""
    exit_code = campaign_runner.main(
        [
            "--fixed-scope-check-rank-contract",
            "--report",
            str(tmp_path / "does_not_exist.json"),
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--report not found" in captured.out
    assert "fidelity_rank_stability_report.json" in captured.out


def test_cli_emits_machine_readable_check_packet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The check emits a JSON packet with the registered claim boundary and provenance."""
    report_path = _write_report(
        _identifiable_report(), tmp_path / "fidelity_rank_stability_report.json"
    )
    exit_code = campaign_runner.main(
        [
            "--fixed-scope-check-rank-contract",
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    # The first non-empty JSON object on stdout is the check packet.
    packet = _first_json_object(captured.out)
    assert packet["schema_version"] == campaign_runner.RANK_CONTRACT_CHECK_SCHEMA_VERSION
    assert "not benchmark evidence" in packet["claim_boundary"]
    assert packet["contract_id"] == campaign_runner.RANK_IDENTIFIABILITY_CONTRACT_ID
    assert packet["threshold"] == "non_zero_variance_and_rank_identifiable"
    assert packet["blocks_claims_when_failed"] is True
    assert packet["satisfied"] is True
    assert packet["reason"] is None
    # Default spec provenance (no --plan) is the rebuilt fixed-scope config.
    assert packet["contract_spec_provenance"] == "rebuilt_from_config"
    assert packet["report_rank_identifiable"] is True


def test_cli_reads_contract_spec_from_plan_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--plan pins the contract spec to a serialized run plan's registered spec."""
    plan = campaign_runner.build_fixed_scope_run_plan(
        _config(),
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="test-head",
    )
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    plan_path = campaign_runner.write_fixed_scope_run_plan(plan, plan_dir)
    report_path = _write_report(
        _identifiable_report(), tmp_path / "fidelity_rank_stability_report.json"
    )

    exit_code = campaign_runner.main(
        [
            "--fixed-scope-check-rank-contract",
            "--report",
            str(report_path),
            "--plan",
            str(plan_path),
        ]
    )
    assert exit_code == 0
    packet = _first_json_object(capsys.readouterr().out)
    assert packet["contract_spec_provenance"].startswith("plan_file:")
    assert "fidelity_fixed_scope_run_plan.json" in packet["contract_spec_provenance"]


def test_cli_plan_without_contract_spec_raises(tmp_path: Path) -> None:
    """A --plan that lacks the rank-identifiability spec is a hard error, not a silent pass."""
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    plan_path = plan_dir / "fidelity_fixed_scope_run_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "issue_3207_fidelity_fixed_scope_run_plan.v1",
                "post_run_contract_specs": [],
            }
        ),
        encoding="utf-8",
    )
    report_path = _write_report(
        _identifiable_report(), tmp_path / "fidelity_rank_stability_report.json"
    )
    with pytest.raises(ValueError, match="post_run_contract_specs entry"):
        campaign_runner.main(
            [
                "--fixed-scope-check-rank-contract",
                "--report",
                str(report_path),
                "--plan",
                str(plan_path),
            ]
        )


# ---------------------------------------------------------------------------
# Regression: the execute path still uses the shared selector (no divergence)
# ---------------------------------------------------------------------------


def test_execute_path_uses_shared_selector() -> None:
    """_run_fixed_scope_execute resolves its spec via the shared selector helper."""
    # The execute-path contract gate references the shared selector symbol.
    source = (
        Path(campaign_runner.__file__).read_text(encoding="utf-8")
        if hasattr(campaign_runner, "__file__")
        else ""
    )
    # The execute function's post-run gate must call the shared selector.
    assert "select_rank_identifiability_contract_spec(plan)" in source
