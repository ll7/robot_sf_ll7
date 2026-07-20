"""Tests for the issue #5248 salvaged trace-capable h600 registration checker."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.failure_mechanism_taxonomy import (
    REQUIRED_MECHANISM_FIELDS as REQUIRED_MECHANISM_FIELDS_TUNNEL,
)
from scripts.validation.check_issue_5248_salvaged_trace_rerun import (
    BLOCKED_STATUS,
    READY_STATUS,
    _load_trace_contract,
    _public_path,
    build_registration_receipt,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION_CONFIG = (
    REPO_ROOT / "configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml"
)

# Host-local, verified harvest root for job 13334. Present only on the issue's
# declared execution host; absent on CI / other hosts by design.
HARVEST_CAMPAIGN_ROOT = Path(
    "/home/luttkule/git/robot_sf_ll7/output/issue4206-13334-harvest"
    "/issue4206_trace_capable_h600_rerun_20260704"
)


def test_real_harvest_blocks_on_trace_label_floor_not_missing_inputs() -> None:
    """Regression guard for issue #5248 successor slice.

    When the verified job-13334 harvest is present (issue-declared host only),
    the checker must read both source artifacts and confirm 6,480 completed
    episodes. With no post-hoc derivation sidecar supplied it must still block,
    because the campaign's own ``seed_episode_rows.csv`` carries ``unknown``
    mechanism labels: the campaign IS trace-capable (its episodes carry
    ``simulation_step_trace``), but the trace-verified labels are a post-hoc
    derivation (issue #4831 sidecar) that lives outside the run tree, not a
    write-time field. It must NOT block on missing inputs.
    """

    if not HARVEST_CAMPAIGN_ROOT.is_dir():
        pytest.skip(
            "harvest_not_available: verified job-13334 harvest absent on this host",
        )

    receipt = build_registration_receipt(
        campaign_root=HARVEST_CAMPAIGN_ROOT,
        job_id="13334",
        expected_total_episodes=6480,
        preregistration_config=PREREGISTRATION_CONFIG,
        generated_at="2026-07-14T175754Z",
    )

    assert receipt["status"] == BLOCKED_STATUS
    assert receipt["campaign"]["total_episodes_observed"] == 6480
    assert receipt["campaign"]["campaign_execution_status"] == "completed"
    assert receipt["campaign"]["episode_row_count"] == 6480
    # Source artifacts were actually read, not missing.
    assert set(receipt["source_files"]) == {
        "reports/campaign_summary.json",
        "reports/seed_episode_rows.csv",
    }
    # The blocker is the real trace-label floor, never a missing-input error.
    assert any("trace-verified labeled fraction" in blocker for blocker in receipt["blockers"])
    assert not any("cannot read" in blocker for blocker in receipt["blockers"])


def _write_campaign(
    root: Path,
    *,
    episode_count: int = 3,
    completed: bool = True,
    trace_labeled_rows: int = 2,
    omit_field: str | None = None,
    success_values: list[str] | None = None,
) -> Path:
    """Write the smallest camera-ready campaign fixture accepted by the checker.

    ``success_values`` sets each row's write-time ``success`` outcome (``"0.0"``
    = failure episode, ``"1.0""`` = success episode). It defaults to all failures
    so the failure-episode denominator equals the all-rows denominator, which
    preserves the pre-#5779 fixture semantics. Tests that exercise the
    denominator correction pass a mix of failure and success outcomes.
    """

    reports = root / "reports"
    reports.mkdir(parents=True)
    (reports / "campaign_summary.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "total_episodes": episode_count,
                    "campaign_execution_status": "completed" if completed else "failed",
                    "status": "accepted_unavailable_only",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if success_values is None:
        success_values = ["0.0"] * episode_count
    fields = [
        "episode_id",
        "success",
        "mechanism_schema_version",
        "mechanism_label",
        "mechanism_confidence",
        "mechanism_evidence_mode",
        "mechanism_evidence_uri",
        "mechanism_case_id",
        "mechanism_caveat",
    ]
    if omit_field is not None:
        fields.remove(omit_field)
    with (reports / "seed_episode_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(episode_count):
            trace_labeled = index < trace_labeled_rows
            row = {
                "episode_id": f"ep-{index}",
                "success": success_values[index],
                "mechanism_schema_version": "failure_mechanism_taxonomy.v1",
                "mechanism_label": "static_deadlock_or_local_minimum"
                if trace_labeled
                else "not_derivable",
                "mechanism_confidence": "observed_mechanism" if trace_labeled else "unknown",
                "mechanism_evidence_mode": "paired_trace" if trace_labeled else "unknown",
                "mechanism_evidence_uri": f"trace://fixture/{index}" if trace_labeled else "",
                "mechanism_case_id": f"fixture-{index}",
                "mechanism_caveat": "",
            }
            writer.writerow({field: row[field] for field in fields})
    return root


def _receipt(root: Path, **kwargs: object) -> dict:
    """Build a deterministic fixture receipt."""

    return build_registration_receipt(
        campaign_root=root,
        job_id="fixture-job",
        expected_total_episodes=3,
        preregistration_config=PREREGISTRATION_CONFIG,
        generated_at="2026-07-11T00:00:00+00:00",
        **kwargs,
    )


def test_complete_campaign_with_trace_rows_is_ready(tmp_path: Path) -> None:
    """A completed campaign above the registered trace-label floor is ready."""

    receipt = _receipt(_write_campaign(tmp_path / "campaign"))

    assert receipt["status"] == READY_STATUS
    assert receipt["blockers"] == []
    assert receipt["campaign"]["episode_row_count"] == 3
    assert receipt["trace_labels"]["trace_labeled_rows"] == 2
    assert receipt["trace_labels"]["trace_labeled_fraction"] == 2 / 3
    assert set(receipt["source_files"]) == {
        "reports/campaign_summary.json",
        "reports/seed_episode_rows.csv",
    }


def test_incomplete_campaign_is_blocked_without_promoting_status(tmp_path: Path) -> None:
    """A non-completed execution state cannot be salvaged by trace rows alone."""

    receipt = _receipt(_write_campaign(tmp_path / "campaign", completed=False))

    assert receipt["status"] == BLOCKED_STATUS
    assert any("campaign_execution_status" in blocker for blocker in receipt["blockers"])


def test_episode_total_mismatch_is_blocked(tmp_path: Path) -> None:
    """A completed campaign cannot register when summary and requested totals disagree."""

    receipt = _receipt(_write_campaign(tmp_path / "campaign", episode_count=2))

    assert receipt["status"] == BLOCKED_STATUS
    assert any("campaign.total_episodes" in blocker for blocker in receipt["blockers"])
    assert any("seed_episode_rows.csv" in blocker for blocker in receipt["blockers"])


def test_insufficient_trace_labels_are_blocked(tmp_path: Path) -> None:
    """Rows below the preregistered trace-label threshold fail closed."""

    receipt = _receipt(_write_campaign(tmp_path / "campaign", trace_labeled_rows=1))

    assert receipt["status"] == BLOCKED_STATUS
    assert any("trace-verified labeled fraction" in blocker for blocker in receipt["blockers"])


def test_missing_mechanism_field_is_blocked(tmp_path: Path) -> None:
    """The registration cannot accept an episode table missing taxonomy fields."""

    receipt = _receipt(_write_campaign(tmp_path / "campaign", omit_field="mechanism_evidence_uri"))

    assert receipt["status"] == BLOCKED_STATUS
    assert any("mechanism_evidence_uri" in blocker for blocker in receipt["blockers"])


def test_trace_contract_rejects_boolean_labeled_fraction(tmp_path: Path) -> None:
    """A boolean is not a numeric preregistration coverage threshold."""
    config = tmp_path / "preregistration.yaml"
    config.write_text(
        PREREGISTRATION_CONFIG.read_text(encoding="utf-8").replace(
            "min_trace_verified_labeled_fraction: 0.5",
            "min_trace_verified_labeled_fraction: true",
        ),
        encoding="utf-8",
    )

    try:
        _load_trace_contract(config)
    except ValueError as exc:
        assert "minimum trace-labeled fraction" in str(exc)
    else:
        raise AssertionError("expected boolean threshold to fail closed")


def test_cli_writes_blocked_receipt_and_nonzero_exit(tmp_path: Path, capsys) -> None:
    """The CLI leaves a reviewable receipt even when source registration is blocked."""

    campaign = _write_campaign(tmp_path / "campaign", trace_labeled_rows=0)
    output_dir = tmp_path / "receipt"

    exit_code = main(
        [
            "--campaign-root",
            str(campaign),
            "--job-id",
            "fixture-job",
            "--expected-total-episodes",
            "3",
            "--preregistration-config",
            str(PREREGISTRATION_CONFIG),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            "2026-07-11T00:00:00+00:00",
        ]
    )

    assert exit_code == 2
    assert "status: blocked_campaign_registration" in capsys.readouterr().out
    payload = json.loads((output_dir / "registration.json").read_text(encoding="utf-8"))
    assert payload["status"] == BLOCKED_STATUS
    assert (output_dir / "registration.md").is_file()


# ---------------------------------------------------------------------------
# Post-hoc mechanism-label sidecar overlay (#4831 trace-verified derivation).
# The campaign's own rows are pre-derivation (always ``unknown``); the derived
# trace-verified labels live in a separate sidecar CSV keyed by ``episode_id``.
# ---------------------------------------------------------------------------


def _write_unlabeled_campaign(
    root: Path,
    *,
    episode_ids: list[str],
    success_values: list[str] | None = None,
) -> Path:
    """Write a completed campaign whose raw episode rows are all unlabeled.

    Mirrors the real job-13334 shape: every row carries a valid taxonomy schema
    block but ``unknown`` labels, because mechanism labels are a post-hoc
    derivation. Each row carries a unique ``episode_id`` for sidecar joining.
    ``success_values`` sets each row's write-time outcome (``"0.0"`` failure /
    ``"1.0""`` success); it defaults to all failures so the failure-episode
    denominator equals the all-rows denominator unless a test opts into a mix.
    """

    reports = root / "reports"
    reports.mkdir(parents=True)
    (reports / "campaign_summary.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "total_episodes": len(episode_ids),
                    "campaign_execution_status": "completed",
                    "status": "accepted_unavailable_only",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if success_values is None:
        success_values = ["0.0"] * len(episode_ids)
    fields = ["episode_id", "success", *REQUIRED_MECHANISM_FIELDS_TUNNEL]
    with (reports / "seed_episode_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode_id, success in zip(episode_ids, success_values, strict=True):
            writer.writerow(
                {
                    "episode_id": episode_id,
                    "success": success,
                    "mechanism_schema_version": "failure_mechanism_taxonomy.v1",
                    "mechanism_label": "not_derivable",
                    "mechanism_confidence": "unknown",
                    "mechanism_evidence_mode": "unknown",
                    "mechanism_evidence_uri": "",
                    "mechanism_case_id": "",
                    "mechanism_caveat": "not_derivable_from_episode_record",
                }
            )
    return root


def _write_sidecar(
    path: Path,
    *,
    labeled: list[str],
    weak: list[str] | None = None,
    drop_field: str | None = None,
    reorder_header: bool = False,
) -> Path:
    """Write a #4831-style derived mechanism-label sidecar CSV.

    ``labeled`` episode ids receive a trace-verified ``supported_hypothesis``
    label; ``weak`` ids receive a ``weak_hypothesis`` label (below the accepted
    floor). A leading ``#`` comment line mirrors the real builder's review marker.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode_id",
        "scenario_id",
        "planner_key",
        "source_run",
        "mechanism_schema_version",
        "mechanism_label",
        "mechanism_confidence",
        "mechanism_evidence_mode",
        "mechanism_evidence_uri",
        "mechanism_case_id",
        "mechanism_caveat",
    ]
    if drop_field is not None and drop_field in fields:
        fields.remove(drop_field)
    if reorder_header:
        fields = [
            "scenario_id",
            "episode_id",
            *[field for field in fields if field != "episode_id"],
        ]
    weak = weak or []
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# AI-GENERATED NEEDS-REVIEW\n")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for episode_id in labeled + weak:
            is_weak = episode_id in weak
            row = {
                "episode_id": episode_id,
                "scenario_id": "fixture",
                "planner_key": "fixture",
                "source_run": "fixture__differential_drive",
                "mechanism_schema_version": "failure_mechanism_taxonomy.v1",
                "mechanism_label": "time_budget_artifact"
                if is_weak
                else "static_deadlock_or_local_minimum",
                "mechanism_confidence": "weak_hypothesis" if is_weak else "supported_hypothesis",
                "mechanism_evidence_mode": "paired_trace",
                "mechanism_evidence_uri": f"trace://fixture/{episode_id}",
                "mechanism_case_id": episode_id,
                "mechanism_caveat": "" if not is_weak else "weak",
            }
            writer.writerow({field: row[field] for field in fields})
    return path


def _receipt_with_sidecar(campaign: Path, sidecar: Path) -> dict:
    """Build a fixture receipt that overlays a sidecar on an unlabeled campaign."""

    return build_registration_receipt(
        campaign_root=campaign,
        job_id="fixture-job",
        expected_total_episodes=4,
        preregistration_config=PREREGISTRATION_CONFIG,
        generated_at="2026-07-15T00:00:00+00:00",
        mechanism_sidecar=sidecar,
    )


def test_sidecar_overlay_promotes_unlabeled_failures_above_floor(tmp_path: Path) -> None:
    """A fully-unlabeled campaign below the floor becomes ready via the sidecar."""

    episode_ids = [f"ep-{i}" for i in range(4)]
    campaign = _write_unlabeled_campaign(tmp_path / "campaign", episode_ids=episode_ids)
    # 3 of 4 episodes (0.75) get accepted-confidence labels via the sidecar.
    sidecar = _write_sidecar(tmp_path / "side.csv", labeled=episode_ids[:3])

    receipt = _receipt_with_sidecar(campaign, sidecar)

    assert receipt["status"] == READY_STATUS
    assert receipt["blockers"] == []
    # Without the sidecar the raw rows are all unknown (0.0); the sidecar raised it.
    assert receipt["trace_labels"]["trace_labeled_rows_without_sidecar"] == 0
    assert receipt["trace_labels"]["trace_labeled_fraction_without_sidecar"] == 0.0
    assert receipt["trace_labels"]["trace_labeled_rows"] == 3
    assert receipt["trace_labels"]["trace_labeled_fraction"] == 0.75


def test_sidecar_accepts_comment_and_reordered_header(tmp_path: Path) -> None:
    """Leading review comments cannot make a valid reordered CSV look malformed."""
    episode_ids = [f"ep-{i}" for i in range(4)]
    campaign = _write_unlabeled_campaign(tmp_path / "campaign", episode_ids=episode_ids)
    sidecar = _write_sidecar(tmp_path / "side.csv", labeled=episode_ids, reorder_header=True)

    receipt = _receipt_with_sidecar(campaign, sidecar)

    assert receipt["status"] == READY_STATUS
    assert receipt["trace_labels"]["trace_labeled_rows"] == 4


def test_sidecar_overlay_excludes_weak_hypothesis_from_floor(tmp_path: Path) -> None:
    """``weak_hypothesis`` labels count as derived but not as trace-verified."""

    episode_ids = [f"ep-{i}" for i in range(4)]
    campaign = _write_unlabeled_campaign(tmp_path / "campaign", episode_ids=episode_ids)
    # 2 accepted-confidence + 2 weak: 0.50 of the rows are weak, only 0.50 are
    # trace-verified, which meets the 0.5 floor exactly.
    sidecar = _write_sidecar(tmp_path / "side.csv", labeled=episode_ids[:2], weak=episode_ids[2:])

    receipt = _receipt_with_sidecar(campaign, sidecar)

    assert receipt["status"] == READY_STATUS
    assert receipt["trace_labels"]["trace_labeled_rows"] == 2
    assert receipt["trace_labels"]["trace_labeled_fraction"] == 0.5


def test_sidecar_overlay_below_floor_still_blocks(tmp_path: Path) -> None:
    """A sidecar that lifts coverage but not past the floor keeps it blocked."""

    episode_ids = [f"ep-{i}" for i in range(4)]
    campaign = _write_unlabeled_campaign(tmp_path / "campaign", episode_ids=episode_ids)
    sidecar = _write_sidecar(tmp_path / "side.csv", labeled=episode_ids[:1])

    receipt = _receipt_with_sidecar(campaign, sidecar)

    assert receipt["status"] == BLOCKED_STATUS
    assert any("trace-verified labeled fraction" in b for b in receipt["blockers"])
    assert receipt["trace_labels"]["trace_labeled_rows"] == 1
    assert receipt["trace_labels"]["trace_labeled_fraction"] == 0.25


def test_sidecar_overlay_only_counts_matching_episode_ids(tmp_path: Path) -> None:
    """Sidecar labels for episode_ids absent from the campaign do not count."""

    episode_ids = [f"ep-{i}" for i in range(4)]
    campaign = _write_unlabeled_campaign(tmp_path / "campaign", episode_ids=episode_ids)
    # Sidecar labels episodes that are NOT in the campaign.
    sidecar = _write_sidecar(tmp_path / "side.csv", labeled=["ghost-1", "ghost-2", "ghost-3"])

    receipt = _receipt_with_sidecar(campaign, sidecar)

    assert receipt["status"] == BLOCKED_STATUS
    assert receipt["trace_labels"]["trace_labeled_rows"] == 0
    assert receipt["mechanism_sidecar"]["rows_matched_in_campaign"] == 0


def test_malformed_sidecar_missing_field_is_a_blocker(tmp_path: Path) -> None:
    """A sidecar missing a required taxonomy field fails closed, not silently."""

    episode_ids = [f"ep-{i}" for i in range(4)]
    campaign = _write_unlabeled_campaign(tmp_path / "campaign", episode_ids=episode_ids)
    sidecar = _write_sidecar(
        tmp_path / "side.csv", labeled=episode_ids, drop_field="mechanism_evidence_uri"
    )

    receipt = _receipt_with_sidecar(campaign, sidecar)

    assert receipt["status"] == BLOCKED_STATUS
    assert any(
        "mechanism sidecar" in b and "mechanism_evidence_uri" in b for b in receipt["blockers"]
    )


def test_sidecar_receipt_records_provenance(tmp_path: Path) -> None:
    """The receipt records the sidecar path, checksum, row count, and match count."""

    episode_ids = [f"ep-{i}" for i in range(4)]
    campaign = _write_unlabeled_campaign(tmp_path / "campaign", episode_ids=episode_ids)
    sidecar = _write_sidecar(tmp_path / "side.csv", labeled=episode_ids[:3])

    receipt = _receipt_with_sidecar(campaign, sidecar)

    sidecar_block = receipt["mechanism_sidecar"]
    assert sidecar_block is not None
    assert sidecar_block["path"].endswith("side.csv")
    assert isinstance(sidecar_block["sha256"], str) and len(sidecar_block["sha256"]) == 64
    assert sidecar_block["row_count"] == 3
    assert sidecar_block["rows_matched_in_campaign"] == 3
    assert sidecar_block["load_error"] is None
    # path is repo-relative (no host prefix embedded)
    assert "/home/" not in sidecar_block["path"]


def test_public_path_does_not_strip_external_directory_names(tmp_path: Path) -> None:
    """Only paths under the repository root receive a repository-relative key."""
    external_path = tmp_path / "docs" / "artifact.csv"

    assert _public_path(external_path) == "artifact.csv"


def test_no_sidecar_omits_sidecar_block_and_keeps_legacy_fraction(tmp_path: Path) -> None:
    """Without a sidecar the receipt has no sidecar block and keeps legacy behavior."""

    receipt = _receipt(_write_campaign(tmp_path / "campaign"))

    assert receipt["mechanism_sidecar"] is None
    assert receipt["trace_labels"]["trace_labeled_rows"] == 2
    assert receipt["trace_labels"]["trace_labeled_fraction"] == 2 / 3
    assert receipt["trace_labels"]["trace_labeled_fraction_without_sidecar"] == 2 / 3
    # Issue #5779: the receipt records the denominator correction and both ratios.
    assert receipt["trace_labels"]["minimum_labeled_fraction_denominator"] == "failure_episodes"
    assert "denominator_correction" in receipt["trace_labels"]
    assert receipt["trace_labels"]["total_row_count"] == 3
    # Default fixture has all failure outcomes, so both denominators coincide.
    assert receipt["trace_labels"]["failure_episode_count"] == 3
    assert receipt["trace_labels"]["trace_labeled_fraction_all_rows"] == 2 / 3


# ---------------------------------------------------------------------------
# Issue #5779 structural denominator correction: the trace-verified labeled
# fraction floor is measured over *failure episodes* (rows whose write-time
# ``success`` outcome is below 1.0), not all campaign rows. Success episodes
# structurally carry no failure-mechanism label, so an all-rows denominator
# made the floor unreachable by construction for any campaign with a healthy
# success rate. These tests pin the denominator logic on portable fixtures.
# ---------------------------------------------------------------------------


def test_failure_episode_denominator_excludes_success_episodes(tmp_path: Path) -> None:
    """Success episodes are excluded from the gating denominator.

    A campaign with 2 labeled failure episodes, 2 unlabeled failure episodes,
    and 6 success episodes: the all-rows fraction is 2/10 = 0.2 (below the 0.5
    floor), but the failure-episode fraction is 2/4 = 0.5 (meets the floor). The
    gate must use the failure-episode fraction and the campaign must be ready,
    while the receipt still reports the 0.2 all-rows ratio for transparency.
    """

    # Rows 0-1: labeled failures. Rows 2-3: unlabeled failures. Rows 4-9: successes.
    success_values = ["0.0", "0.0", "0.0", "0.0", "1.0", "1.0", "1.0", "1.0", "1.0", "1.0"]
    campaign = _write_campaign(
        tmp_path / "campaign",
        episode_count=10,
        trace_labeled_rows=2,
        success_values=success_values,
    )

    receipt = build_registration_receipt(
        campaign_root=campaign,
        job_id="fixture-job",
        expected_total_episodes=10,
        preregistration_config=PREREGISTRATION_CONFIG,
        generated_at="2026-07-20T000000Z",
    )

    assert receipt["status"] == READY_STATUS
    assert receipt["blockers"] == []
    trace_labels = receipt["trace_labels"]
    assert trace_labels["total_row_count"] == 10
    assert trace_labels["failure_episode_count"] == 4
    assert trace_labels["trace_labeled_rows"] == 2
    # The gate uses the failure-episode denominator.
    assert trace_labels["trace_labeled_fraction"] == 0.5
    # The all-rows ratio is still reported for transparency.
    assert trace_labels["trace_labeled_fraction_all_rows"] == 0.2


def test_all_rows_denominator_would_block_but_failure_denominator_passes(
    tmp_path: Path,
) -> None:
    """The denominator correction flips a blocked (all-rows) verdict to ready.

    Mirrors the job-13334 shape: a small campaign where labeled failures are a
    minority of all rows but a majority of failure episodes. Under the old
    all-rows denominator this blocked; under the corrected failure-episode
    denominator it is ready. The receipt records both ratios so a reviewer can
    see exactly what the denominator correction changed.
    """

    # 3 labeled failures, 1 unlabeled failure, 8 successes: 3/12 = 0.25 all rows,
    # 3/4 = 0.75 failure episodes.
    success_values = ["0.0", "0.0", "0.0", "0.0"] + ["1.0"] * 8
    campaign = _write_campaign(
        tmp_path / "campaign",
        episode_count=12,
        trace_labeled_rows=3,
        success_values=success_values,
    )

    receipt = build_registration_receipt(
        campaign_root=campaign,
        job_id="fixture-job",
        expected_total_episodes=12,
        preregistration_config=PREREGISTRATION_CONFIG,
        generated_at="2026-07-20T000000Z",
    )

    assert receipt["status"] == READY_STATUS
    trace_labels = receipt["trace_labels"]
    assert trace_labels["trace_labeled_fraction"] == 0.75
    assert trace_labels["trace_labeled_fraction_all_rows"] == 0.25
    assert trace_labels["failure_episode_count"] == 4


def test_sidecar_failure_denominator_excludes_success_episodes(tmp_path: Path) -> None:
    """The failure-episode denominator applies with the sidecar overlay too.

    A fully-unlabeled campaign with 4 failures + 4 successes: the sidecar labels
    3 of the 4 failures at accepted confidence. All-rows fraction is 3/8 = 0.375
    (below 0.5), but the failure-episode fraction is 3/4 = 0.75 (meets 0.5), so
    the campaign is ready and the success episodes do not penalize the floor.
    """

    episode_ids = [f"ep-{i}" for i in range(8)]
    # First 4 are failures, last 4 are successes.
    success_values = ["0.0"] * 4 + ["1.0"] * 4
    campaign = _write_unlabeled_campaign(
        tmp_path / "campaign", episode_ids=episode_ids, success_values=success_values
    )
    # Sidecar labels the first 3 failure episodes (ep-0, ep-1, ep-2).
    sidecar = _write_sidecar(tmp_path / "side.csv", labeled=episode_ids[:3])

    receipt = build_registration_receipt(
        campaign_root=campaign,
        job_id="fixture-job",
        expected_total_episodes=8,
        preregistration_config=PREREGISTRATION_CONFIG,
        generated_at="2026-07-20T000000Z",
        mechanism_sidecar=sidecar,
    )

    assert receipt["status"] == READY_STATUS
    trace_labels = receipt["trace_labels"]
    assert trace_labels["failure_episode_count"] == 4
    assert trace_labels["trace_labeled_rows"] == 3
    assert trace_labels["trace_labeled_fraction"] == 0.75
    assert trace_labels["trace_labeled_fraction_all_rows"] == 0.375


def test_failure_denominator_below_floor_still_blocks(tmp_path: Path) -> None:
    """A low failure-episode fraction (not all-rows) still fails closed."""

    # 4 failures, 1 labeled -> 0.25 failure-episode fraction, below the 0.5 floor.
    success_values = ["0.0", "0.0", "0.0", "0.0"] + ["1.0"] * 6
    campaign = _write_campaign(
        tmp_path / "campaign",
        episode_count=10,
        trace_labeled_rows=1,
        success_values=success_values,
    )

    receipt = build_registration_receipt(
        campaign_root=campaign,
        job_id="fixture-job",
        expected_total_episodes=10,
        preregistration_config=PREREGISTRATION_CONFIG,
        generated_at="2026-07-20T000000Z",
    )

    assert receipt["status"] == BLOCKED_STATUS
    blocker = next(b for b in receipt["blockers"] if "trace-verified labeled fraction" in b)
    assert "failure-episode denominator" in blocker
    assert "1 of 4 failure episodes" in blocker
    assert receipt["trace_labels"]["trace_labeled_fraction"] == 0.25


def test_missing_success_column_degrades_to_all_rows_denominator(tmp_path: Path) -> None:
    """A campaign without a ``success`` column conservatively counts every row.

    A missing/unparseable write-time outcome is treated as a failure episode so a
    malformed campaign degrades to the stricter all-rows denominator rather than
    silently shrinking the failure-episode floor.
    """

    reports = tmp_path / "campaign" / "reports"
    reports.mkdir(parents=True)
    (reports / "campaign_summary.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "total_episodes": 4,
                    "campaign_execution_status": "completed",
                    "status": "accepted_unavailable_only",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # No ``success`` column: every row is conservatively a failure episode.
    fields = list(REQUIRED_MECHANISM_FIELDS_TUNNEL)
    with (reports / "seed_episode_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(4):
            trace_labeled = index < 1
            writer.writerow(
                {
                    "mechanism_schema_version": "failure_mechanism_taxonomy.v1",
                    "mechanism_label": "static_deadlock_or_local_minimum"
                    if trace_labeled
                    else "not_derivable",
                    "mechanism_confidence": "observed_mechanism" if trace_labeled else "unknown",
                    "mechanism_evidence_mode": "paired_trace" if trace_labeled else "unknown",
                    "mechanism_evidence_uri": f"trace://fixture/{index}" if trace_labeled else "",
                    "mechanism_case_id": f"fixture-{index}",
                    "mechanism_caveat": "",
                }
            )

    receipt = build_registration_receipt(
        campaign_root=tmp_path / "campaign",
        job_id="fixture-job",
        expected_total_episodes=4,
        preregistration_config=PREREGISTRATION_CONFIG,
        generated_at="2026-07-20T000000Z",
    )

    trace_labels = receipt["trace_labels"]
    # No success column -> every row counted as a failure -> both denominators 4.
    assert trace_labels["failure_episode_count"] == 4
    assert trace_labels["total_row_count"] == 4
    assert trace_labels["trace_labeled_fraction"] == 0.25
    assert trace_labels["trace_labeled_fraction_all_rows"] == 0.25
    assert receipt["status"] == BLOCKED_STATUS


def test_denominator_field_read_from_contract(tmp_path: Path) -> None:
    """The checker reads and honors the declared denominator from the contract."""

    modes, fraction, denominator = _load_trace_contract(PREREGISTRATION_CONFIG)
    assert modes == {
        "paired_trace",
        "deterministic_replay",
        "direct_probe",
        "root_cause",
    }
    assert fraction == 0.5
    assert denominator == "failure_episodes"


def test_trace_contract_rejects_missing_denominator(tmp_path: Path) -> None:
    """A contract that omits the denominator declaration fails closed."""
    import yaml

    payload = yaml.safe_load(PREREGISTRATION_CONFIG.read_text(encoding="utf-8"))
    del payload["required_outputs"]["failure_mechanism"][
        "min_trace_verified_labeled_fraction_denominator"
    ]
    config = tmp_path / "preregistration.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    try:
        _load_trace_contract(config)
    except ValueError as exc:
        assert "min_trace_verified_labeled_fraction_denominator" in str(exc)
    else:
        raise AssertionError("expected missing denominator to fail closed")


REAL_MECHANISM_SIDECAR = REPO_ROOT.joinpath(
    "docs/context/evidence/issue_4831_trace_verified_failure_mechanisms/mechanism_labels.csv"
)


def test_job_13334_denominator_correction_makes_campaign_ready() -> None:
    """Real job-13334 harvest + the real #4831 sidecar is READY after #5779.

    Regression guard for issue #5779 (denominator correction). The campaign IS
    trace-capable (its episodes carry ``simulation_step_trace``), and #4831
    derived trace-verified labels for all 2,612 failure episodes. Under the old
    all-rows denominator the accepted-confidence fraction (~0.373 over 6,480
    rows) sat below the 0.5 floor and blocked. Under the corrected
    failure-episode denominator the accepted-confidence fraction (~0.925 over
    ~2,612 failure episodes) meets the 0.5 floor, so the campaign is ready for
    the issue #4206 reanalysis. The receipt must report BOTH ratios so a
    reviewer can see exactly what the denominator correction changed. Skips
    when the verified harvest or #4831 sidecar is absent on this host.
    """

    if not HARVEST_CAMPAIGN_ROOT.is_dir() or not REAL_MECHANISM_SIDECAR.is_file():
        pytest.skip(
            "verified job-13334 harvest or #4831 sidecar absent on this host "
            "[harvest_or_sidecar_not_available]"
        )

    receipt = build_registration_receipt(
        campaign_root=HARVEST_CAMPAIGN_ROOT,
        job_id="13334",
        expected_total_episodes=6480,
        preregistration_config=PREREGISTRATION_CONFIG,
        generated_at="2026-07-16T125218Z",
        mechanism_sidecar=REAL_MECHANISM_SIDECAR,
    )

    assert receipt["campaign"]["episode_row_count"] == 6480
    assert receipt["campaign"]["campaign_execution_status"] == "completed"
    # Without the sidecar the raw rows are all unknown.
    assert receipt["trace_labels"]["trace_labeled_fraction_without_sidecar"] == 0.0
    trace_labels = receipt["trace_labels"]
    # The denominator correction flips the verdict: gating fraction is measured
    # over failure episodes and clears the 0.5 floor.
    assert trace_labels["minimum_labeled_fraction_denominator"] == "failure_episodes"
    assert trace_labels["failure_episode_count"] > 0
    assert trace_labels["trace_labeled_fraction"] >= 0.5
    assert receipt["status"] == READY_STATUS
    assert receipt["blockers"] == []
    # The all-rows ratio is still reported for transparency and stays ~0.373.
    assert 0.35 < trace_labels["trace_labeled_fraction_all_rows"] < 0.45
    assert trace_labels["trace_labeled_rows"] == 2415
    assert receipt["mechanism_sidecar"]["row_count"] == 2612
    assert receipt["mechanism_sidecar"]["load_error"] is None
