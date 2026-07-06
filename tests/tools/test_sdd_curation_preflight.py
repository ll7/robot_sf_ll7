"""Tests for the SDD curation readiness preflight (issue #1126).

These exercise the curation-step gate on importer fixtures only. They assert the fail-closed
contract: fixture/proxy annotations are never marked benchmark-promotable, and missing SDD blocks
benchmark evidence regardless of whether the importer could parse a candidate file.
"""

from __future__ import annotations

import json
import math
import shlex
from typing import TYPE_CHECKING

import pytest

from scripts.tools import import_sdd_scenarios, manage_external_data, sdd_curation_preflight

if TYPE_CHECKING:
    from pathlib import Path


def _write_sdd_fixture(path: Path) -> None:
    """Write a tiny valid SDD-format annotation fixture (two qualifying Pedestrian tracks)."""
    path.write_text(
        "\n".join(
            [
                "1 0 0 10 10 0 0 0 0 Pedestrian",
                "1 5 0 15 10 5 0 0 0 Pedestrian",
                "1 10 0 20 10 10 0 0 0 Pedestrian",
                "1 15 0 25 10 15 0 0 0 Pedestrian",
                "2 100 100 110 110 0 0 0 0 Pedestrian",
                "2 100 105 110 115 5 0 0 0 Pedestrian",
                "2 100 110 110 120 10 0 0 0 Pedestrian",
                "2 100 115 110 125 15 0 0 0 Pedestrian",
                "3 0 0 10 10 0 1 0 0 Pedestrian",  # lost -> filtered
                "4 0 0 10 10 0 0 0 0 Biker",  # wrong label -> filtered
            ]
        ),
        encoding="utf-8",
    )


def _write_quoted_sdd_fixture(path: Path) -> None:
    """Write tiny valid SDD-format annotation fixture with quoted labels."""
    path.write_text(
        "\n".join(
            [
                '1 0 0 10 10 0 0 0 0 "Pedestrian"',
                '1 5 0 15 10 5 0 0 0 "Pedestrian"',
                '1 10 0 20 10 10 0 0 0 "Pedestrian"',
                '1 15 0 25 10 15 0 0 0 "Pedestrian"',
                '2 100 100 110 110 0 0 0 0 "Pedestrian"',
                '2 100 105 110 115 5 0 0 0 "Pedestrian"',
                '2 100 110 110 120 10 0 0 0 "Pedestrian"',
                '2 100 115 110 125 15 0 0 0 "Pedestrian"',
                '3 0 0 10 10 0 1 0 0 "Pedestrian"',  # lost -> filtered
                '4 0 0 10 10 0 0 0 0 "Biker"',  # wrong label -> filtered
            ]
        ),
        encoding="utf-8",
    )


def test_probe_accepts_user_facing_label_for_quoted_sdd_rows(tmp_path: Path) -> None:
    """Probe should accept unquoted labels for quoted SDD annotation rows."""
    annotations = tmp_path / "annotations.txt"
    _write_quoted_sdd_fixture(annotations)

    probe = sdd_curation_preflight.probe_annotation_file(
        annotations,
        label="Pedestrian",
        min_track_points=4,
        max_pedestrians=4,
    )

    assert probe["label"] == "Pedestrian"
    assert probe["usable_label_points"] == 8
    blocked = sdd_curation_preflight.probe_annotation_file(
        annotations,
        label='"Pedestrian"',
        min_track_points=99,
        max_pedestrians=4,
    )
    assert "usable 'Pedestrian' points" in blocked["blockers"][0]
    assert probe["usable_track_count"] == 2
    assert probe["selection_satisfiable"] is True
    assert probe["blockers"] == []


def test_probe_normalizes_legacy_quoted_label_argument(tmp_path: Path) -> None:
    """Probe report should display the canonical unquoted label."""
    annotations = tmp_path / "annotations.txt"
    _write_quoted_sdd_fixture(annotations)

    probe = sdd_curation_preflight.probe_annotation_file(
        annotations,
        label='"Pedestrian"',
        min_track_points=4,
        max_pedestrians=4,
    )

    assert probe["label"] == "Pedestrian"
    assert probe["usable_label_points"] == 8


def test_probe_counts_usable_tracks_after_filtering(tmp_path: Path) -> None:
    """Probe should count only label-matching, non-lost tracks meeting min_track_points."""
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)

    probe = sdd_curation_preflight.probe_annotation_file(
        annotations, label="Pedestrian", min_track_points=4, max_pedestrians=4
    )
    assert probe["exists"] is True
    assert probe["usable_track_count"] == 2  # tracks 1 and 2; track 3 lost, track 4 wrong label
    assert probe["usable_label_points"] == 8
    assert probe["selection_satisfiable"] is True
    assert probe["blockers"] == []


def test_probe_missing_file_is_blocker(tmp_path: Path) -> None:
    """A missing annotation file fails closed with a blocker, not an exception."""
    probe = sdd_curation_preflight.probe_annotation_file(
        tmp_path / "nope.txt", label="Pedestrian", min_track_points=8, max_pedestrians=4
    )
    assert probe["exists"] is False
    assert probe["selection_satisfiable"] is False
    assert probe["blockers"]


def test_probe_insufficient_track_length_is_blocker(tmp_path: Path) -> None:
    """A label present but with too-short tracks is not curation-satisfiable."""
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)

    probe = sdd_curation_preflight.probe_annotation_file(
        annotations, label="Pedestrian", min_track_points=100, max_pedestrians=4
    )
    assert probe["usable_track_count"] == 0
    assert probe["selection_satisfiable"] is False
    assert probe["blockers"]


def test_missing_sdd_blocks_benchmark_promotion_even_with_valid_fixture(tmp_path: Path) -> None:
    """Core fail-closed contract: a parseable fixture must NOT become benchmark evidence."""
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)

    proxy_gate = {
        "mode": manage_external_data.SDD_MODE_PROXY,
        "dataset_backed": False,
        "availability": {"state": "missing"},
        "reason": "SDD is not staged.",
        "staging_dir": str(tmp_path),
    }
    probe = sdd_curation_preflight.probe_annotation_file(
        annotations, label="Pedestrian", min_track_points=4, max_pedestrians=4
    )
    report = sdd_curation_preflight.classify_curation_readiness(proxy_gate, probe)

    assert report["dataset_backed"] is False
    assert report["benchmark_promotion_allowed"] is False
    assert report["evidence_status"] == sdd_curation_preflight.EVIDENCE_PROXY
    assert report["output_classification"] == sdd_curation_preflight.OUTPUT_PROXY_ONLY
    # The fixture parses, so curation may still run as a schema smoke -- just not be promoted.
    assert report["curation_runnable"] is True
    assert report["blockers"]


def test_missing_sdd_without_probe_is_blocked(tmp_path: Path) -> None:
    """With no candidate annotation and no staged SDD, the gate is blocked."""
    proxy_gate = {
        "mode": manage_external_data.SDD_MODE_PROXY,
        "dataset_backed": False,
        "availability": {"state": "missing"},
        "reason": "SDD is not staged.",
        "staging_dir": str(tmp_path),
    }
    report = sdd_curation_preflight.classify_curation_readiness(proxy_gate, None)
    assert report["benchmark_promotion_allowed"] is False
    assert report["evidence_status"] == sdd_curation_preflight.EVIDENCE_PROXY
    assert report["output_classification"] == sdd_curation_preflight.OUTPUT_BLOCKED


def test_dataset_backed_with_valid_probe_is_benchmark_candidate(tmp_path: Path) -> None:
    """A staged/validated SDD plus a satisfiable probe unlocks benchmark candidacy."""
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)

    backed_gate = {
        "mode": manage_external_data.SDD_MODE_DATASET_BACKED,
        "dataset_backed": True,
        "availability": {"state": "dataset_backed"},
        "reason": "SDD is staged and validated.",
        "staging_dir": str(tmp_path),
    }
    probe = sdd_curation_preflight.probe_annotation_file(
        annotations, label="Pedestrian", min_track_points=4, max_pedestrians=4
    )
    report = sdd_curation_preflight.classify_curation_readiness(backed_gate, probe)

    assert report["dataset_backed"] is True
    assert report["benchmark_promotion_allowed"] is True
    assert report["evidence_status"] == sdd_curation_preflight.EVIDENCE_BENCHMARK_CANDIDATE
    assert (
        report["output_classification"] == sdd_curation_preflight.OUTPUT_BENCHMARK_READY_CANDIDATE
    )
    assert report["blockers"] == []


def test_dataset_backed_without_probe_is_not_benchmark_candidate(tmp_path: Path) -> None:
    """A pinned SDD tree still needs a selected annotation probe before promotion."""
    backed_gate = {
        "mode": manage_external_data.SDD_MODE_DATASET_BACKED,
        "dataset_backed": True,
        "availability": {"state": "dataset_backed"},
        "reason": "SDD staged validated.",
        "staging_dir": str(tmp_path),
    }

    report = sdd_curation_preflight.classify_curation_readiness(backed_gate, None)

    assert report["benchmark_promotion_allowed"] is False
    assert report["evidence_status"] == sdd_curation_preflight.EVIDENCE_BLOCKED
    assert report["output_classification"] == sdd_curation_preflight.OUTPUT_BLOCKED
    assert "no candidate annotation was probed" in report["blockers"][0]


def test_dataset_backed_with_bad_probe_is_not_promotable(tmp_path: Path) -> None:
    """Even with staged SDD, an unsatisfiable probe must not be benchmark-promotable."""
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)

    backed_gate = {
        "mode": manage_external_data.SDD_MODE_DATASET_BACKED,
        "dataset_backed": True,
        "availability": {"state": "dataset_backed"},
        "reason": "SDD is staged and validated.",
        "staging_dir": str(tmp_path),
    }
    probe = sdd_curation_preflight.probe_annotation_file(
        annotations, label="Pedestrian", min_track_points=100, max_pedestrians=4
    )
    report = sdd_curation_preflight.classify_curation_readiness(backed_gate, probe)

    assert report["dataset_backed"] is True
    assert report["benchmark_promotion_allowed"] is False
    assert report["evidence_status"] == sdd_curation_preflight.EVIDENCE_BLOCKED
    assert report["output_classification"] == sdd_curation_preflight.OUTPUT_BLOCKED


def test_cli_require_benchmark_ready_fails_closed_when_unstaged(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The CLI exits non-zero under --require-benchmark-ready when SDD is not staged.

    Forces the unstaged staging gate so the assertion is deterministic regardless of the local
    machine's SDD staging state, while still exercising the real CLI parse/probe/exit-code path.
    """
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)
    monkeypatch.setattr(
        manage_external_data,
        "resolve_sdd_scenario_prior_mode",
        lambda *, manifest_path=None: {
            "mode": manage_external_data.SDD_MODE_PROXY,
            "dataset_backed": False,
            "availability": {"state": "missing"},
            "reason": "SDD is not staged.",
            "staging_dir": str(tmp_path),
        },
    )
    exit_code = sdd_curation_preflight.main(
        [
            "--annotation",
            str(annotations),
            "--min-track-points",
            "4",
            "--require-benchmark-ready",
            "--json",
        ]
    )
    assert exit_code == 3
    out = capsys.readouterr().out
    assert '"benchmark_promotion_allowed": false' in out


def test_decision_packet_preserves_proxy_blocker(tmp_path: Path) -> None:
    """Decision packet records proxy-only ceiling, not benchmark readiness."""
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)
    proxy_gate = {
        "mode": manage_external_data.SDD_MODE_PROXY,
        "dataset_backed": False,
        "availability": {"state": "missing"},
        "reason": "SDD not staged.",
        "staging_dir": str(tmp_path),
    }
    probe = sdd_curation_preflight.probe_annotation_file(
        annotations, label="Pedestrian", min_track_points=4, max_pedestrians=4
    )
    report = sdd_curation_preflight.classify_curation_readiness(proxy_gate, probe)

    packet = sdd_curation_preflight.build_decision_packet(
        report,
        annotation=annotations,
        label="Pedestrian",
        min_track_points=4,
        max_pedestrians=4,
        dataset_id="sdd_test_scene",
        output_dir=tmp_path / "derived",
    )

    assert packet["schema"] == sdd_curation_preflight.DECISION_PACKET_SCHEMA
    assert packet["readiness"]["benchmark_promotion_allowed"] is False
    assert packet["readiness"]["output_classification"] == sdd_curation_preflight.OUTPUT_PROXY_ONLY
    assert packet["raw_data_policy"]["raw_sdd_committed"] is False
    assert "--dataset-id sdd_test_scene" in packet["required_next_commands"]["import"]


def test_cli_writes_decision_packet_without_benchmark_claim(tmp_path: Path, monkeypatch) -> None:
    """CLI packet output is metadata-only and keeps missing SDD fail-closed."""
    annotations = tmp_path / "annotations.txt"
    packet_path = tmp_path / "packet.json"
    _write_sdd_fixture(annotations)
    monkeypatch.setattr(
        manage_external_data,
        "resolve_sdd_scenario_prior_mode",
        lambda *, manifest_path=None: {
            "mode": manage_external_data.SDD_MODE_PROXY,
            "dataset_backed": False,
            "availability": {"state": "missing"},
            "reason": "SDD not staged.",
            "staging_dir": str(tmp_path),
        },
    )

    exit_code = sdd_curation_preflight.main(
        [
            "--annotation",
            str(annotations),
            "--min-track-points",
            "4",
            "--decision-dataset-id",
            "sdd_test_scene",
            "--write-decision-packet",
            str(packet_path),
            "--json",
        ]
    )

    assert exit_code == 0
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["claim_boundary"].startswith("decision packet only")
    assert packet["readiness"]["evidence_status"] == sdd_curation_preflight.EVIDENCE_PROXY
    assert packet["readiness"]["benchmark_promotion_allowed"] is False
    assert not (tmp_path / "derived").exists()


def _parse_importer_command(import_command: str, monkeypatch) -> object:
    """Parse the generated import command with the importer's own argparse parser.

    ``import_sdd_scenarios.parse_args`` reads ``sys.argv`` directly, so we strip the
    ``uv run python <script>`` prefix and stage the remaining tokens as argv.
    """
    tokens = shlex.split(import_command)
    assert tokens[:3] == ["uv", "run", "python"]
    assert tokens[3].endswith("import_sdd_scenarios.py")
    monkeypatch.setattr("sys.argv", ["import_sdd_scenarios.py", *tokens[4:]])
    return import_sdd_scenarios.parse_args()


def test_decision_packet_import_command_is_runnable_by_importer(
    tmp_path: Path, monkeypatch
) -> None:
    """The generated import command must parse against the real importer CLI (issue #1126).

    Regression guard: the importer requires ``--annotations``/``--out-dir``/``--meters-per-pixel``.
    An earlier packet emitted ``--annotation``/``--output-dir`` and omitted the scale, so the
    handoff command failed closed at argparse. We parse the generated command with the importer's
    own parser to prove a curator can run it verbatim.
    """
    annotations = tmp_path / "annotations.txt"
    _write_sdd_fixture(annotations)
    report = sdd_curation_preflight.classify_curation_readiness(
        {"mode": manage_external_data.SDD_MODE_PROXY, "dataset_backed": False}, None
    )

    packet = sdd_curation_preflight.build_decision_packet(
        report,
        annotation=annotations,
        label="Pedestrian",
        min_track_points=8,
        max_pedestrians=4,
        dataset_id="sdd_test_scene",
        output_dir=tmp_path / "derived",
        meters_per_pixel=0.0417,
    )

    import_command = packet["required_next_commands"]["import"]
    # Uses the importer's real flag names, not the earlier broken --annotation/--output-dir.
    assert "--annotations" in import_command
    assert "--out-dir" in import_command
    assert "--annotation " not in import_command
    assert "--output-dir" not in import_command

    parsed = _parse_importer_command(import_command, monkeypatch)
    assert str(parsed.annotations) == str(annotations)
    assert str(parsed.out_dir) == str(tmp_path / "derived")
    assert parsed.meters_per_pixel == 0.0417
    assert parsed.dataset_id == "sdd_test_scene"
    assert packet["curation_parameters"]["meters_per_pixel"] == 0.0417


def test_decision_packet_records_scale_placeholder_when_unset(tmp_path: Path, monkeypatch) -> None:
    """Without a scene scale the packet keeps a fill-in placeholder and records meters_per_pixel=None.

    The scale is scene-specific and unknown until BYO annotations are staged, so the command must
    surface an explicit ``<meters-per-pixel>`` token rather than silently dropping the required flag.
    """
    report = sdd_curation_preflight.classify_curation_readiness(
        {"mode": manage_external_data.SDD_MODE_PROXY, "dataset_backed": False}, None
    )

    packet = sdd_curation_preflight.build_decision_packet(
        report,
        annotation=None,
        label="Pedestrian",
        min_track_points=8,
        max_pedestrians=4,
        dataset_id="sdd_first_real_candidate",
        output_dir=tmp_path / "derived",
    )

    import_command = packet["required_next_commands"]["import"]
    assert "--meters-per-pixel <meters-per-pixel>" in import_command
    assert packet["curation_parameters"]["meters_per_pixel"] is None
    # Once the curator fills the placeholder scale, the command parses as a valid float.
    filled = import_command.replace("<meters-per-pixel>", "0.05").replace(
        "<staged-sdd>/<scene>/<video>/annotations.txt", str(tmp_path / "a.txt")
    )
    parsed = _parse_importer_command(filled, monkeypatch)
    assert parsed.meters_per_pixel == 0.05


def test_cli_decision_meters_per_pixel_is_recorded(tmp_path: Path, monkeypatch) -> None:
    """The --decision-meters-per-pixel flag threads the scene scale into the packet."""
    annotations = tmp_path / "annotations.txt"
    packet_path = tmp_path / "packet.json"
    _write_sdd_fixture(annotations)
    monkeypatch.setattr(
        manage_external_data,
        "resolve_sdd_scenario_prior_mode",
        lambda *, manifest_path=None: {
            "mode": manage_external_data.SDD_MODE_PROXY,
            "dataset_backed": False,
            "availability": {"state": "missing"},
            "reason": "SDD not staged.",
            "staging_dir": str(tmp_path),
        },
    )

    exit_code = sdd_curation_preflight.main(
        [
            "--annotation",
            str(annotations),
            "--min-track-points",
            "4",
            "--write-decision-packet",
            str(packet_path),
            "--decision-meters-per-pixel",
            "0.0417",
        ]
    )

    assert exit_code == 0
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["curation_parameters"]["meters_per_pixel"] == 0.0417
    assert "--meters-per-pixel 0.0417" in packet["required_next_commands"]["import"]


@pytest.mark.parametrize("bad_scale", [0.0, -0.05, float("nan"), float("inf")])
def test_decision_packet_rejects_invalid_meters_per_pixel(tmp_path: Path, bad_scale: float) -> None:
    """A non-positive or non-finite scale must fail closed at packet build (issue #1126).

    The importer requires ``--meters-per-pixel > 0``; a NaN/inf slips past that ``<= 0`` guard and
    would produce garbage geometry. Reject it here so the generated handoff command is never emitted
    with an unrunnable/garbage scale.
    """
    report = sdd_curation_preflight.classify_curation_readiness(
        {"mode": manage_external_data.SDD_MODE_PROXY, "dataset_backed": False}, None
    )
    with pytest.raises(ValueError, match="finite value > 0"):
        sdd_curation_preflight.build_decision_packet(
            report,
            annotation=None,
            label="Pedestrian",
            min_track_points=8,
            max_pedestrians=4,
            dataset_id="sdd_first_real_candidate",
            output_dir=tmp_path / "derived",
            meters_per_pixel=bad_scale,
        )
    assert not math.isfinite(bad_scale) or bad_scale <= 0  # guards the parametrization intent


def _benchmark_candidate_readiness() -> dict[str, object]:
    return {
        "benchmark_promotion_allowed": True,
        "output_classification": sdd_curation_preflight.OUTPUT_BENCHMARK_READY_CANDIDATE,
    }


def test_smoke_decision_rejects_timeout_candidate_for_closure() -> None:
    """#1126 timeout smoke stays exploratory-only and requests benchmark-ready follow-up."""
    decision = sdd_curation_preflight.classify_smoke_decision(
        _benchmark_candidate_readiness(),
        [
            {
                "horizon": 80,
                "successful_jobs": 1,
                "failed_jobs": 0,
                "success": False,
                "timeout": True,
                "collisions": 0,
            },
            {
                "horizon": 384,
                "successful_jobs": 1,
                "failed_jobs": 0,
                "success": False,
                "timeout": True,
                "collisions": 0,
            },
        ],
        generated_artifacts_load=True,
    )

    assert decision["classification"] == sdd_curation_preflight.SMOKE_EXPLORATORY_ONLY
    assert decision["recommended_next_action"] == "tune_or_select_benchmark_ready_candidate"
    assert decision["exploratory_only"] is True
    assert decision["benchmark_ready"] is False
    assert any("timed out" in reason for reason in decision["reasons"])


def test_smoke_decision_promotes_clean_success_candidate() -> None:
    """Cleanly loaded, successful smoke can be promoted as benchmark-ready candidate."""
    decision = sdd_curation_preflight.classify_smoke_decision(
        _benchmark_candidate_readiness(),
        [
            {
                "horizon": 384,
                "successful_jobs": 1,
                "failed_jobs": 0,
                "success": True,
                "timeout": False,
                "collisions": 0,
            }
        ],
        generated_artifacts_load=True,
    )

    assert decision["classification"] == sdd_curation_preflight.SMOKE_BENCHMARK_READY
    assert decision["recommended_next_action"] == "promote_benchmark_ready_candidate"
    assert decision["benchmark_ready"] is True


def test_smoke_decision_fails_closed_when_artifacts_do_not_load() -> None:
    """Smoke classification must not hide generated artifact load failures."""
    decision = sdd_curation_preflight.classify_smoke_decision(
        _benchmark_candidate_readiness(),
        [
            {
                "horizon": 384,
                "successful_jobs": 1,
                "failed_jobs": 0,
                "success": True,
                "timeout": False,
                "collisions": 0,
            }
        ],
        generated_artifacts_load=False,
    )

    assert decision["classification"] == sdd_curation_preflight.SMOKE_BLOCKED
    assert decision["recommended_next_action"] == "fix_import_or_smoke_execution"
    assert any("did not load" in reason for reason in decision["reasons"])
