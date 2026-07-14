"""Tests for the verified-harvest #3216 analysis runner."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analysis/run_issue3216_rank_stability.py"
_SPEC = importlib.util.spec_from_file_location("issue5247_rank_stability", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules["issue5247_rank_stability"] = runner
_SPEC.loader.exec_module(runner)

# Hub-local verified-harvest artifacts (only present on the pinned analysis host).
# Never committed; this test skips elsewhere so the suite stays portable.
_REAL_HARVEST_DIR = Path("~/git/robot_sf_ll7/output/issue3216-13274-harvest").expanduser()
_REAL_CAMPAIGN_DIR = _REAL_HARVEST_DIR / "issue3216_s20_headline_ci"
_REAL_HARVEST_LOG = Path("~/git/context/codex-orchestrator/runtime/harvest_13274.log").expanduser()
_REAL_PLANNER_CONFIG = Path(
    "~/git/robot_sf_ll7/configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml"
).expanduser()
_REAL_HARVEST_PRESENT = (
    _REAL_CAMPAIGN_DIR.is_dir() and _REAL_HARVEST_LOG.is_file() and _REAL_PLANNER_CONFIG.is_file()
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _campaign(root: Path) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True)
    _write_json(
        reports / "campaign_summary.json",
        {
            "campaign": {
                "finished_at_utc": "2026-06-30T12:34:56+00:00",
                "snqi_contract_status": "fail",
            },
            "warnings": [
                "SNQI contract status=fail with snqi_contract.enforcement=warn; "
                "campaign marked with soft contract warning."
            ],
        },
    )
    _write_json(
        reports / "snqi_diagnostics.json",
        {
            "contract_status": "fail",
            "contract_enforcement": "warn",
            "rank_alignment_spearman": 0.2,
            "outcome_separation": 0.1,
            "dominant_component_mean_abs": 0.1,
            "thresholds": {
                "rank_alignment_fail": 0.3,
                "outcome_separation_fail": 0.0,
                "max_component_dominance_fail": 0.27,
            },
        },
    )
    (reports / "seed_episode_rows.csv").write_text(
        "planner_key,scenario_id,seed,success,collision,near_miss,snqi\n"
        "orca,crossing,1,1,0,0,0.8\n"
        "orca,crossing,2,1,0,0,0.9\n"
        "ppo,crossing,1,0,1,0,0.2\n"
        "ppo,crossing,2,0,1,0,0.3\n",
        encoding="utf-8",
    )
    (reports / "scenario_family_breakdown.csv").write_text(
        "scenario_family\ncrossing\n", encoding="utf-8"
    )
    (reports / "campaign_table.csv").write_text(
        "planner_key,execution_mode,status\norca,nominal,successful_evidence\n"
        "ppo,nominal,successful_evidence\n",
        encoding="utf-8",
    )


def test_verified_harvest_runner_writes_reproducible_provenance(tmp_path: Path) -> None:
    """A verified complete campaign delegates analysis and records exact SNQI failure."""

    campaign = tmp_path / "campaign"
    _campaign(campaign)
    harvest_log = tmp_path / "harvest.log"
    harvest_log.write_text("copy done\nVERIFIED_COMPLETE\n", encoding="utf-8")
    planner_config = tmp_path / "planners.yaml"
    planner_config.write_text("planners:\n  - key: orca\n  - key: ppo\n", encoding="utf-8")
    output = tmp_path / "analysis"
    command = [
        sys.executable,
        str(_SCRIPT),
        "--campaign-root",
        str(campaign),
        "--harvest-log",
        str(harvest_log),
        "--planner-config",
        str(planner_config),
        "--output-dir",
        str(output),
        "--bootstrap-samples",
        "12",
        "--rank-resamples",
        "8",
    ]

    first = subprocess.run(command, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    first_hashes = json.loads((output / "analysis_provenance.json").read_text())["output_sha256"]
    second = subprocess.run(command, text=True, capture_output=True, check=False)
    assert second.returncode == 0, second.stderr
    second_payload = json.loads((output / "analysis_provenance.json").read_text())
    assert second_payload["output_sha256"] == first_hashes
    assert second_payload["schema_version"] == "issue_5247_verified_harvest_rank_stability.v2"
    assert second_payload["ranking"] == {
        "profile": "constraints_first",
        "rank_metric": "success",
        "snqi_rank_limitation": (
            "SNQI contract status=fail with snqi_contract.enforcement=warn; failed check(s): "
            "rank_alignment_spearman=0.2 (below fail threshold 0.3)"
        ),
        "snqi_rank_status": "blocked_invalid_metric",
    }
    result = json.loads((output / "result.json").read_text())
    assert result["inputs"]["rank_profile"] == "constraints_first"
    assert result["inputs"]["rank_metric"] == "success"
    assert result["inputs"]["invalid_rank_metric_reason"] is None
    assert {claim["decision"] for claim in result["adjacent_rank_claims"]} == {"diagnostic_only"}
    failure = second_payload["snqi_contract_failure"]
    assert failure["campaign_finished_at_utc"] == "2026-06-30T12:34:56+00:00"
    assert failure["enforcement"] == "warn"
    assert failure["failed_checks"] == [
        {
            "check": "rank_alignment_spearman",
            "direction": "below",
            "fail_threshold": 0.3,
            "value": 0.2,
        }
    ]


def test_runner_rejects_harvest_without_completion_marker(tmp_path: Path) -> None:
    """The runner never analyzes a partial or unverified harvest."""

    campaign = tmp_path / "campaign"
    _campaign(campaign)
    harvest_log = tmp_path / "harvest.log"
    harvest_log.write_text("copy done\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--campaign-root",
            str(campaign),
            "--harvest-log",
            str(harvest_log),
            "--output-dir",
            str(tmp_path / "analysis"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "VERIFIED_COMPLETE" in result.stderr
    assert not (tmp_path / "analysis").exists()


def test_runner_rejects_nonfinite_snqi_diagnostics(tmp_path: Path) -> None:
    """A failed contract with a non-finite diagnostic cannot become provenance."""
    campaign = tmp_path / "campaign"
    _campaign(campaign)
    diagnostics_path = campaign / "reports" / "snqi_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["rank_alignment_spearman"] = float("nan")
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    harvest_log = tmp_path / "harvest.log"
    harvest_log.write_text("VERIFIED_COMPLETE\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--campaign-root",
            str(campaign),
            "--harvest-log",
            str(harvest_log),
            "--output-dir",
            str(tmp_path / "analysis"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "must be finite numeric values" in result.stderr


@pytest.mark.skipif(
    not _REAL_HARVEST_PRESENT, reason="real job-13274 harvest not present on this host"
)
class TestRealHarvestReproducibility:
    """Reproduce the verified job-13274 analysis identically on a second run.

    This exercises the runner against the ALREADY-COMPLETE 8,640-episode campaign
    (Slurm job 13274, salvaged 2026-07-11) exactly as issue #5247 requires: it
    verifies the harvest-completion marker, delegates to the canonical #3216 CLI,
    and produces constraints-first success rank-stability artifacts. The run must
    be deterministic -- identical output hashes on re-invocation -- which is the
    issue's core ``Verify`` gate over real data. Outputs are written to tmp_path
    only; nothing under the harvested tree is mutated.
    """

    def test_real_harvest_analysis_is_deterministic(self, tmp_path: Path) -> None:
        first = tmp_path / "run1"
        second = tmp_path / "run2"
        command = [
            sys.executable,
            str(_SCRIPT),
            "--campaign-root",
            str(_REAL_CAMPAIGN_DIR),
            "--harvest-log",
            str(_REAL_HARVEST_LOG),
            "--planner-config",
            str(_REAL_PLANNER_CONFIG),
        ]

        def run_against(out_dir: Path) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [*command, "--output-dir", str(out_dir)],
                text=True,
                capture_output=True,
                check=False,
            )

        run_a = run_against(first)
        assert run_a.returncode == 0, run_a.stderr
        assert (first / "result.json").exists()
        run_b = run_against(second)
        assert run_b.returncode == 0, run_b.stderr

        first_prov = json.loads((first / "analysis_provenance.json").read_text(encoding="utf-8"))
        second_prov = json.loads((second / "analysis_provenance.json").read_text(encoding="utf-8"))
        assert first_prov["output_sha256"] == second_prov["output_sha256"]
        for artifact in ("result.json", "report.md"):
            h_a = _sha256(first / artifact)
            h_b = _sha256(second / artifact)
            assert h_a == h_b, f"{artifact} differs across runs"

        # The real SNQI soft-contract failure must be captured exactly.
        failure = first_prov["snqi_contract_failure"]
        assert failure["contract_status"] == "fail"
        assert failure["enforcement"] == "warn"
        assert any(
            c["check"] == "rank_alignment_spearman" and c["value"] < c["fail_threshold"]
            for c in failure["failed_checks"]
        )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
