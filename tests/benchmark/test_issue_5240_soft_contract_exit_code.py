"""Issue #5240: a soft SNQI-contract warning (enforcement=warn) must not fail the exit code.

A benchmark campaign that COMPLETES all planner rows but carries a soft contract warning
(``snqi_contract.enforcement == "warn"`` with contract ``status`` of ``warn``/``fail``) must
exit 0 and surface ``soft_contract_warning: true`` plus a ``warnings[]`` entry. Hard
enforcement (``error``/``enforce``) with a contract ``fail`` stays fatal (nonzero exit).

These tests lock both layers named in the issue:
- the exit-code mapping (``campaign_exit_code`` / ``summarize_campaign_outcome``), and
- the CLI wrapper (``scripts/tools/run_camera_ready_benchmark.py``) that maps the returned
  status to the process exit code.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.fallback_policy import campaign_exit_code
from robot_sf.benchmark.snqi.campaign_contract import soft_contract_warning_active
from scripts.tools import run_camera_ready_benchmark

if TYPE_CHECKING:
    from pathlib import Path

_SOFT_CONTRACT_WARNING_FRAGMENT = "soft contract warning"


# --------------------------------------------------------------------------- #
# Layer: the soft-contract-warning decision boundary (the heart of the fix).
# --------------------------------------------------------------------------- #


class TestSoftContractWarningDecision:
    """``soft_contract_warning_active`` encodes the warn-vs-block enforcement boundary."""

    @pytest.mark.parametrize(
        ("enforcement", "contract_status", "expected"),
        [
            # enforcement=warn is SOFT: warn/fail status surface as non-fatal warnings.
            ("warn", "fail", True),
            ("warn", "warn", True),
            # No soft warning when the contract actually passed.
            ("warn", "pass", False),
            # Hard enforcement levels are NEVER soft: a fail stays fatal via snqi_hard_fail.
            ("enforce", "fail", False),
            ("error", "fail", False),
            ("enforce", "warn", False),
            ("error", "warn", False),
            # Hard enforcement with a passing contract is neither soft nor fatal.
            ("enforce", "pass", False),
        ],
    )
    def test_decision_boundary(
        self, enforcement: str, contract_status: str, expected: bool
    ) -> None:
        """Decision boundary matches the issue's warn (soft) vs block (fatal) contract."""
        assert soft_contract_warning_active(enforcement, contract_status) is expected

    def test_hard_enforcement_fail_is_not_demoted_to_soft(self) -> None:
        """Regression guard for issue #5240: enforce/error + fail must NOT become a soft warning.

        If this ever flips to True, the hard-fatal path would be silently demoted to a non-fatal
        warning, re-introducing the orphaned-campaign defect.
        """
        assert soft_contract_warning_active("enforce", "fail") is False
        assert soft_contract_warning_active("error", "fail") is False

    def test_decision_is_case_and_whitespace_insensitive(self) -> None:
        """Config-driven values are normalized before the decision is applied."""
        assert soft_contract_warning_active("  WARN  ", "FAIL") is True
        assert soft_contract_warning_active("Enforce", "Fail") is False


# --------------------------------------------------------------------------- #
# Layer 1: the exit-code mapping must stay 0 for a soft-warning success campaign.
# --------------------------------------------------------------------------- #


def _success_payload_with_soft_warning() -> dict:
    """A benchmark-success campaign result that also carries a soft contract warning."""
    return {
        "campaign_id": "cid",
        "campaign_root": "/tmp/cid",
        "benchmark_success": True,
        "status": "benchmark_success",
        "campaign_execution_status": "completed",
        "evidence_status": "valid",
        "row_status_summary": {
            "successful_evidence_rows": 3,
            "accepted_unavailable_rows": 0,
            "unexpected_failed_rows": 0,
            "fallback_or_degraded_rows": 0,
        },
        "status_reason": "all planner rows were benchmark-success",
        "successful_runs": 3,
        "accepted_unavailable_runs": 0,
        "unexpected_failed_runs": 0,
        "non_success_runs": 0,
        "total_runs": 3,
        "exit_code": 0,
        "soft_contract_warning": True,
        "warnings": [
            "SNQI contract status=fail with snqi_contract.enforcement=warn; "
            "campaign marked with soft contract warning."
        ],
    }


def test_campaign_exit_code_is_zero_for_soft_warning_success() -> None:
    """A soft contract warning on an otherwise-successful campaign must not change the exit code."""
    payload = _success_payload_with_soft_warning()
    assert payload["soft_contract_warning"] is True
    # exit_code 0 is canonical and trusted; the soft warning must not override it.
    assert campaign_exit_code(payload) == 0


def test_campaign_exit_code_is_zero_derived_from_rows_even_without_explicit_exit_code() -> None:
    """Even without an explicit exit_code, all-ok rows derive exit 0 despite the soft warning."""
    payload = _success_payload_with_soft_warning()
    payload.pop("exit_code")
    # Falls through to summarize_campaign_outcome -> all rows ok -> exit_code 0.
    assert campaign_exit_code(payload) == 0


def test_soft_warning_does_not_mask_a_genuine_failure() -> None:
    """A soft contract warning must NOT force exit 0 when rows genuinely failed."""
    payload = _success_payload_with_soft_warning()
    payload["planner_rows"] = [{"status": "ok"}, {"status": "partial-failure"}]
    payload["exit_code"] = 2  # genuine unexpected failure
    payload["benchmark_success"] = False
    # The genuine failure exit code is preserved; the soft warning does not override it.
    assert campaign_exit_code(payload) == 2


# --------------------------------------------------------------------------- #
# Layer 2: the CLI wrapper maps the returned status to the process exit code.
# --------------------------------------------------------------------------- #


def _install_soft_warning_campaign(monkeypatch, tmp_path: Path, payload: dict) -> None:
    """Install mocked config loader + run_campaign for the camera-ready benchmark CLI."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")
    sentinel_cfg = object()

    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "load_campaign_config",
        lambda path: sentinel_cfg if path == config_path else None,
    )
    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "prepare_campaign_preflight",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("prepare_campaign_preflight should not be called in run mode")
        ),
    )
    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "run_campaign",
        lambda cfg, **kw: payload if cfg is sentinel_cfg else {},
    )


def test_wrapper_exit_zero_for_soft_contract_warning_campaign(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """CLI run mode returns exit 0 for a complete campaign carrying a soft contract warning.

    This is the exact regression for issue #5240: a full, expensive result set must not be
    orphaned as 'failed' merely because of a non-fatal soft contract warning.
    """
    payload = _success_payload_with_soft_warning()
    _install_soft_warning_campaign(monkeypatch, tmp_path, payload)

    config_path = tmp_path / "config.yaml"
    exit_code = run_camera_ready_benchmark.main(["--config", str(config_path)])
    assert exit_code == 0

    printed = json.loads(capsys.readouterr().out)
    # The soft warning is surfaced as a top-level flag plus a warnings[] entry.
    assert printed["soft_contract_warning"] is True
    assert printed["benchmark_success"] is True
    assert printed["campaign_execution_status"] == "completed"
    soft_warning_strings = [
        w for w in printed.get("warnings", []) if _SOFT_CONTRACT_WARNING_FRAGMENT in w
    ]
    assert soft_warning_strings, "expected a soft contract warning string in warnings[]"


def test_wrapper_preserves_nonzero_exit_for_genuine_failure_with_soft_warning(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A soft contract warning does not turn a genuine row failure into exit 0."""
    payload = _success_payload_with_soft_warning()
    payload["planner_rows"] = [{"status": "ok"}, {"status": "failed"}]
    payload["exit_code"] = 2
    payload["benchmark_success"] = False
    payload["campaign_execution_status"] = "failed"
    payload["evidence_status"] = "invalid"
    _install_soft_warning_campaign(monkeypatch, tmp_path, payload)

    config_path = tmp_path / "config.yaml"
    exit_code = run_camera_ready_benchmark.main(["--config", str(config_path)])
    assert exit_code == 2
    printed = json.loads(capsys.readouterr().out)
    # Soft warning is still surfaced for observability even on a failed campaign.
    assert printed["soft_contract_warning"] is True


def test_wrapper_hard_contract_failure_stays_fatal(tmp_path: Path, monkeypatch) -> None:
    """Hard enforcement (error/enforce) + contract fail keeps its current fatal behavior.

    snqi_hard_fail raises RuntimeError inside run_campaign; the wrapper must NOT swallow it,
    so the process exits nonzero. Locking the 'do not touch that path' contract from issue #5240.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")
    sentinel_cfg = object()

    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "load_campaign_config",
        lambda path: sentinel_cfg if path == config_path else None,
    )
    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "prepare_campaign_preflight",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unreachable")),
    )

    def _raise_hard_contract_failure(cfg, **kwargs):
        """Simulate the snqi_hard_fail raise path (enforce/error + contract fail)."""
        raise RuntimeError(
            "SNQI contract failed with enforcement=enforce; rank_alignment=0.2000, "
            "outcome_separation=-0.0500."
        )

    monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _raise_hard_contract_failure)

    # The wrapper does not catch the hard-contract RuntimeError, so it propagates -> nonzero exit.
    with pytest.raises(RuntimeError, match="SNQI contract failed with enforcement=enforce"):
        run_camera_ready_benchmark.main(["--config", str(config_path)])
