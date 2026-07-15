"""Contract tests for the issue #5592 cross-matrix agreement builder."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.validation import build_issue_5592_cross_matrix_agreement as builder

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_5592_cross_matrix_preregistration.yaml"
ROSTER_SIGNATURE = builder._roster_signature(builder._load_packet(PACKET))
PRIMARY_OUTPUT = "cross_matrix_agreement.csv"
REQUIRED_COLUMNS = [
    "matrix",
    "structural_class",
    "rank",
    "reference_rank",
    "rank_delta",
    "agreement_status",
    "caveat",
]

REFERENCE_RANKING = f"""structural_class,rank,roster_signature
constraint_first_hybrid,1,{ROSTER_SIGNATURE}
learned_policy,2,{ROSTER_SIGNATURE}
predictive,3,{ROSTER_SIGNATURE}
baseline_reactive,4,{ROSTER_SIGNATURE}
"""

CANDIDATE_RANKING = f"""structural_class,rank,roster_signature
constraint_first_hybrid,1,{ROSTER_SIGNATURE}
learned_policy,3,{ROSTER_SIGNATURE}
predictive,2,{ROSTER_SIGNATURE}
baseline_reactive,4,{ROSTER_SIGNATURE}
"""


def _write_ranking(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_packet_and_comparison_contract_are_satisfied() -> None:
    """The builder accepts the merged pre-registration packet and its contract."""
    packet = builder._load_packet(PACKET)
    builder._validate_comparison_contract(packet)
    assert packet["schema_version"] == "issue_5592_cross_matrix_preregistration.v1"


def test_blocked_when_rankings_absent(tmp_path: Path) -> None:
    """No campaign ran: emit blocked_missing_matrix rows, never a fabricated ranking."""
    out = tmp_path / "out"
    summary = builder.build_packet(
        packet_path=PACKET,
        reference_ranking_path=None,
        candidate_ranking_path=None,
        output_dir=out,
        generated_at="2026-07-14T00:00:00+00:00",
    )
    assert summary["status"] == "blocked_missing_matrix"
    assert summary["reference_ranking_present"] is False
    assert summary["candidate_ranking_present"] is False
    assert summary["next_action"] is not None
    report = (out / builder.INTEGRATION_REPORT).read_text(encoding="utf-8")
    assert "reference (`classic_interactions`) structural ranking CSV" in report
    assert "candidate (`atomic_topology`) structural ranking CSV" in report
    assert "New blockers introduced by this slice" in report
    assert "None. This report only consolidates" in report

    rows = list((out / PRIMARY_OUTPUT).read_text(encoding="utf-8").splitlines())
    header = rows[0].split(",")
    assert header == REQUIRED_COLUMNS
    body = [line for line in rows[1:] if line]
    assert len(body) == len(builder.STRUCTURAL_CLASS_ORDER)
    for line in body:
        # agreement_status is the 6th CSV column
        assert line.split(",")[5] == "blocked_missing_matrix"


def test_ready_emits_real_agreement_and_disagreement(tmp_path: Path) -> None:
    """With both rankings present, rank flips are reported (not hidden in a merge)."""
    out = tmp_path / "out"
    summary = builder.build_packet(
        packet_path=PACKET,
        reference_ranking_path=_write_ranking(tmp_path, "ref.csv", REFERENCE_RANKING),
        candidate_ranking_path=_write_ranking(tmp_path, "cand.csv", CANDIDATE_RANKING),
        output_dir=out,
        generated_at="2026-07-14T00:00:00+00:00",
    )
    assert summary["status"] == "ready"
    assert summary["disagreement_row_count"] == 2

    lines = [
        line
        for line in (out / PRIMARY_OUTPUT).read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("matrix")
    ]
    by_class = {line.split(",")[1]: line.split(",") for line in lines}
    # learned_policy: candidate rank 3 vs reference rank 2 -> delta +1 disagreement
    learned = by_class["learned_policy"]
    assert learned[2] == "3" and learned[3] == "2" and learned[4] == "1"
    assert learned[5] == "disagreement"
    # constraint_first_hybrid agrees at rank 1
    assert by_class["constraint_first_hybrid"][5] == "agreement"


def test_rejects_incomplete_or_invalid_ranking(tmp_path: Path) -> None:
    """A present file must contain a complete, unique structural ranking."""
    incomplete = _write_ranking(
        tmp_path,
        "incomplete.csv",
        f"structural_class,rank,roster_signature\nconstraint_first_hybrid,1,{ROSTER_SIGNATURE}\n",
    )
    try:
        builder.build_packet(
            packet_path=PACKET,
            reference_ranking_path=incomplete,
            candidate_ranking_path=incomplete,
            output_dir=tmp_path / "out",
            generated_at="2026-07-14T00:00:00+00:00",
        )
    except builder.BuildError as exc:
        assert "exactly the four structural classes" in str(exc)
    else:
        raise AssertionError("incomplete ranking must fail closed")


def test_rejects_incomparable_planner_roster(tmp_path: Path) -> None:
    """Rankings from a different preregistered roster cannot be compared."""
    incompatible = _write_ranking(
        tmp_path,
        "incompatible.csv",
        CANDIDATE_RANKING.replace(ROSTER_SIGNATURE, "0" * 64),
    )
    try:
        builder.build_packet(
            packet_path=PACKET,
            reference_ranking_path=incompatible,
            candidate_ranking_path=incompatible,
            output_dir=tmp_path / "out",
            generated_at="2026-07-14T00:00:00+00:00",
        )
    except builder.BuildError as exc:
        assert "incompatible planner roster" in str(exc)
    else:
        raise AssertionError("incomparable roster must fail closed")


def test_artifact_set_is_complete_and_checksummed(tmp_path: Path) -> None:
    """The durable evidence contract includes the consolidated integration report."""
    out = tmp_path / "out"
    builder.build_packet(
        packet_path=PACKET,
        reference_ranking_path=_write_ranking(tmp_path, "ref.csv", REFERENCE_RANKING),
        candidate_ranking_path=_write_ranking(tmp_path, "cand.csv", CANDIDATE_RANKING),
        output_dir=out,
        generated_at="2026-07-14T00:00:00+00:00",
    )
    for name in (
        "README.md",
        "metadata.json",
        PRIMARY_OUTPUT,
        builder.INTEGRATION_REPORT,
        "SHA256SUMS",
    ):
        assert (out / name).exists(), f"missing durable artifact: {name}"
    sums = (out / "SHA256SUMS").read_text(encoding="utf-8")
    # Every durable artifact except SHA256SUMS is represented in the checksum file.
    assert PRIMARY_OUTPUT in sums
    assert "metadata.json" in sums
    assert builder.INTEGRATION_REPORT in sums
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["issue"] == 5592
    assert metadata["source_commit"] not in ("", "unknown")
    assert metadata["roster_signature"] == ROSTER_SIGNATURE


def test_ready_report_records_no_remaining_input_blocker(tmp_path: Path) -> None:
    """A complete pair keeps the report useful without claiming beyond two matrices."""
    out = tmp_path / "out"
    builder.build_packet(
        packet_path=PACKET,
        reference_ranking_path=_write_ranking(tmp_path, "ref.csv", REFERENCE_RANKING),
        candidate_ranking_path=_write_ranking(tmp_path, "cand.csv", CANDIDATE_RANKING),
        output_dir=out,
        generated_at="2026-07-14T00:00:00+00:00",
    )
    report = (out / builder.INTEGRATION_REPORT).read_text(encoding="utf-8")
    assert "None for the artifact builder" in report
    assert "not a general-purpose generalization guarantee" in report
    assert "Review `cross_matrix_agreement.csv`" in report


def test_report_treats_explicit_null_contract_sections_as_empty(tmp_path: Path) -> None:
    """Null YAML sections do not crash the diagnostic report writer."""
    out = tmp_path / "out"
    out.mkdir()
    builder._write_integration_report(
        out,
        packet={
            "reference_contract": None,
            "candidate_contract": None,
            "pairing_contract": None,
            "planner_roster": {"structural_classes": {"constraint_first_hybrid": None}},
        },
        packet_path=PACKET,
        status="blocked_missing_matrix",
        source_commit="0" * 40,
        reference_present=False,
        candidate_present=False,
        next_action="Provide both ranking inputs.",
    )

    report = (out / builder.INTEGRATION_REPORT).read_text(encoding="utf-8")

    assert "Reference matrix: `None`" in report
    assert "Candidate scenarios, in frozen order:" in report
