"""Tests for the campaign-atlas generator (scripts/tools/generate_campaign_atlas.py).

Pins the contract from issue #5301:
- scans ``docs/context/evidence/*/campaign/campaign_manifest.json``;
- one table row per campaign with neutral provenance fields;
- tolerates missing fields: a malformed manifest becomes an ``INCOMPLETE`` row,
  never an exception;
- adding a fake registry dir in a tmpdir produces the expected row;
- ``--check`` is CI-friendly (passes when up to date, nonzero when stale);
- output is deterministic (sorted, no timestamps, stable across regenerations).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Make ``scripts.tools.*`` importable without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tools.generate_campaign_atlas import (  # noqa: E402
    COLUMNS,
    CampaignRow,
    generate,
    main,
    render_atlas,
    scan_campaigns,
)

_EVIDENCE = "docs/context/evidence"
# The atlas is rendered as if it lives at its canonical repo-relative path; this
# keeps link targets stable and mirrors how the real CLI is invoked.
_ATLAS_OUTPUT = Path("docs/benchmarks/CAMPAIGN_ATLAS.md")


# --- fixtures -------------------------------------------------------------


def _make_git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo rooted at tmp_path and return it."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True, capture_output=True
    )
    return tmp_path


def _write_manifest(  # noqa: PLR0913
    repo: Path,
    campaign_dir: str,
    *,
    campaign_id: str = "demo_campaign_001",
    commit: str = "1a2b3c4d5e6f7g8h9i0j",
    scenario_matrix: str = "configs/scenarios/demo.yaml",
    scenario_matrix_hash: str = "abcdef123456",
    created_at_utc: str = "2026-05-04T15:12:17.036639Z",
    remote: str = "https://github.com/ll7/robot_sf_ll7",
    total_episodes: int | None = 1008,
    extra_fields: dict | None = None,
) -> Path:
    """Write a well-formed campaign manifest + optional summary + sha256."""
    rel = f"{_EVIDENCE}/{campaign_dir}"
    (repo / rel / "campaign").mkdir(parents=True, exist_ok=True)
    (repo / rel / "reports").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "benchmark-camera-ready-campaign.v1",
        "campaign_id": campaign_id,
        "created_at_utc": created_at_utc,
        "scenario_matrix": scenario_matrix,
        "scenario_matrix_hash": scenario_matrix_hash,
        "git": {"commit": commit, "branch": "demo", "remote": remote},
        "repository_url": remote,
    }
    if extra_fields:
        manifest.update(extra_fields)
    manifest_path = repo / rel / "campaign" / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # reports referenced by the atlas key-reports column
    for name in ("campaign_report.md", "matrix_summary.json", "campaign_summary.json"):
        (repo / rel / "reports" / name).write_text("{}\n", encoding="utf-8")
    if total_episodes is not None:
        summary = {
            "campaign": {
                "campaign_id": campaign_id,
                "total_episodes": total_episodes,
            }
        }
        (repo / rel / "reports" / "campaign_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    # manifest.sha256 attests the bundle contents
    (repo / rel / "manifest.sha256").write_text("deadbeef  campaign/campaign_manifest.json\n")
    return manifest_path


def _build_registry(tmp_path: Path) -> Path:
    """Build a tmp repo with two valid campaigns for hermetic end-to-end tests."""
    repo = _make_git_repo(tmp_path)
    _write_manifest(
        repo,
        "alpha_demo_2026-05-01",
        campaign_id="alpha_campaign_20260501",
        commit="1111111111111111111111111111111111111111",
        total_episodes=336,
        created_at_utc="2026-05-01T10:00:00Z",
    )
    _write_manifest(
        repo,
        "beta_demo_2026-05-02",
        campaign_id="beta_campaign_20260502",
        commit="2222222222222222222222222222222222222222",
        total_episodes=672,
    )
    return repo


def _chdir(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run from inside the tmp repo so repo-relative paths resolve, and return it."""
    monkeypatch.chdir(repo)
    return repo


def _scan(repo: Path) -> list[CampaignRow]:
    """Scan the tmp repo's evidence root using a repo-relative path."""
    return scan_campaigns(Path(_EVIDENCE))


# --- scanning -------------------------------------------------------------


def test_scan_finds_only_dirs_with_campaign_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only directories with ``campaign/campaign_manifest.json`` become rows."""
    repo = _build_registry(tmp_path)
    _chdir(repo, monkeypatch)
    # A non-campaign evidence bundle (no manifest) must be ignored.
    (repo / f"{_EVIDENCE}/issue_9999_summary_only").mkdir(parents=True)
    (repo / f"{_EVIDENCE}/issue_9999_summary_only/summary.json").write_text("{}\n")

    rows = _scan(repo)

    ids = {row.campaign_id for row in rows}
    assert ids == {"alpha_campaign_20260501", "beta_campaign_20260502"}
    # The summary-only bundle is not a campaign, so it is absent.
    assert all("issue_9999" not in r.campaign_dir_name for r in rows)


def test_scan_rows_sorted_by_campaign_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows are deterministically sorted by campaign id."""
    repo = _build_registry(tmp_path)
    _chdir(repo, monkeypatch)
    rows = _scan(repo)
    ids = [row.campaign_id for row in rows]
    assert ids == sorted(ids)
    assert ids == ["alpha_campaign_20260501", "beta_campaign_20260502"]


def test_scan_empty_when_root_missing(tmp_path: Path) -> None:
    """A missing evidence root yields an empty row list, not an error."""
    assert scan_campaigns(tmp_path / "does_not_exist") == []


def test_resolved_row_has_expected_neutral_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A well-formed campaign resolves every neutral provenance field."""
    repo = _build_registry(tmp_path)
    _chdir(repo, monkeypatch)
    rows = {r.campaign_dir_name: r for r in _scan(repo)}
    row = rows["alpha_demo_2026-05-01"]
    assert row.status == "OK"
    assert row.incomplete_reasons == ()
    assert row.campaign_id == "alpha_campaign_20260501"
    assert row.date == "2026-05-01"
    assert row.git_commit == "1111111111111111111111111111111111111111"
    assert row.scenario_matrix == "configs/scenarios/demo.yaml"
    assert row.scenario_matrix_hash == "abcdef123456"
    assert row.episodes == "336"
    assert row.manifest_sha256_present is True
    assert row.evidence_rel == f"{_EVIDENCE}/alpha_demo_2026-05-01"
    # Key reports are repo-relative POSIX paths in priority order.
    assert row.reports == (
        f"{_EVIDENCE}/alpha_demo_2026-05-01/reports/campaign_report.md",
        f"{_EVIDENCE}/alpha_demo_2026-05-01/reports/matrix_summary.json",
        f"{_EVIDENCE}/alpha_demo_2026-05-01/reports/campaign_summary.json",
    )


def test_missing_total_episodes_renders_em_dash_but_still_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing episode count shows ``—`` without marking the row INCOMPLETE."""
    repo = _make_git_repo(tmp_path)
    _chdir(repo, monkeypatch)
    _write_manifest(repo, "no_episodes_demo", total_episodes=None)
    rows = _scan(repo)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "OK"
    assert row.episodes == "—"


def test_missing_manifest_sha256_renders_no_but_still_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent ``manifest.sha256`` shows ``no`` without failing the row."""
    repo = _make_git_repo(tmp_path)
    _chdir(repo, monkeypatch)
    _write_manifest(repo, "no_sha_demo")
    (repo / f"{_EVIDENCE}/no_sha_demo/manifest.sha256").unlink()
    rows = _scan(repo)
    assert len(rows) == 1
    assert rows[0].status == "OK"
    assert rows[0].manifest_sha256_present is False


# --- tolerance: malformed manifests --------------------------------------


def test_malformed_json_manifest_produces_incomplete_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt JSON manifest becomes an INCOMPLETE row, not an exception."""
    repo = _make_git_repo(tmp_path)
    _chdir(repo, monkeypatch)
    manifest = _write_manifest(repo, "broken_demo")
    manifest.write_text("{ not valid json }}}", encoding="utf-8")
    rows = _scan(repo)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "INCOMPLETE"
    assert any("malformed" in reason for reason in row.incomplete_reasons)
    # The row still carries the directory name as a fallback identifier.
    assert row.campaign_id == "broken_demo"


def test_manifest_not_a_json_object_produces_incomplete_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A JSON array manifest becomes an INCOMPLETE row, not an exception."""
    repo = _make_git_repo(tmp_path)
    _chdir(repo, monkeypatch)
    manifest = _write_manifest(repo, "array_demo")
    manifest.write_text("[1, 2, 3]\n", encoding="utf-8")
    rows = _scan(repo)
    assert len(rows) == 1
    assert rows[0].status == "INCOMPLETE"
    assert "manifest is not a JSON object" in rows[0].incomplete_reasons


def test_missing_core_fields_produce_incomplete_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing campaign_id / commit / scenario_matrix flag INCOMPLETE with reasons."""
    repo = _make_git_repo(tmp_path)
    _chdir(repo, monkeypatch)
    _write_manifest(
        repo,
        "gappy_demo",
        campaign_id="",
        commit="",
        scenario_matrix="",
        extra_fields={"git": {"branch": "x"}},  # remove commit-bearing git block
    )
    rows = _scan(repo)
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "INCOMPLETE"
    reason_text = "; ".join(row.incomplete_reasons)
    assert "missing campaign_id" in reason_text
    assert "missing git commit" in reason_text
    assert "missing scenario_matrix" in reason_text


# --- rendering -----------------------------------------------------------


def test_render_includes_one_row_per_campaign_and_all_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rendered atlas has a header plus exactly one row per campaign."""
    repo = _build_registry(tmp_path)
    _chdir(repo, monkeypatch)
    rows = _scan(repo)
    rendered = render_atlas(rows, atlas_output=_ATLAS_OUTPUT)
    lines = [line for line in rendered.splitlines() if line.startswith("| ")]
    # First two ``| ...`` lines are the header + separator; then one row each.
    assert lines[0] == "| " + " | ".join(COLUMNS) + " |"
    assert set(lines[1]) <= set("| -")
    data_rows = lines[2:]
    assert len(data_rows) == 2
    assert all("alpha_campaign_20260501" in r or "beta_campaign_20260502" in r for r in data_rows)


def test_render_commit_cell_links_short_hash_to_web(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The commit cell renders a short hash linking to the full commit URL."""
    repo = _build_registry(tmp_path)
    _chdir(repo, monkeypatch)
    rows = _scan(repo)
    rendered = render_atlas(rows, atlas_output=_ATLAS_OUTPUT)
    assert (
        "[`111111111111`](https://github.com/ll7/robot_sf_ll7/commit/"
        "1111111111111111111111111111111111111111)" in rendered
    )


def test_render_normalizes_git_remote_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``remote`` ending in ``.git`` is normalized for the commit URL."""
    repo = _make_git_repo(tmp_path)
    _chdir(repo, monkeypatch)
    _write_manifest(
        repo,
        "suffix_demo",
        remote="https://github.com/ll7/robot_sf_ll7.git",
        commit="abc123abc123abc123abc123abc123abc123abcd",
    )
    rows = _scan(repo)
    rendered = render_atlas(rows, atlas_output=_ATLAS_OUTPUT)
    assert (
        "https://github.com/ll7/robot_sf_ll7/commit/abc123abc123abc123abc123abc123abc123abcd"
        in rendered
    )
    assert ".git/commit/" not in rendered


def test_render_report_links_are_relative_to_atlas_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report links resolve relative to the atlas file (e.g. ``../context/...``)."""
    repo = _build_registry(tmp_path)
    _chdir(repo, monkeypatch)
    rows = _scan(repo)
    rendered = render_atlas(rows, atlas_output=_ATLAS_OUTPUT)
    assert "](../context/evidence/alpha_demo_2026-05-01/reports/campaign_report.md)" in rendered


def test_render_escape_pipe_in_cell() -> None:
    """A pipe in any surfaced string is escaped so it cannot break the table."""
    row = CampaignRow(
        campaign_dir_name="pipe_demo",
        evidence_rel=f"{_EVIDENCE}/pipe_demo",
        status="INCOMPLETE",
        campaign_id="a|b",
        date="2026-05-04",
        git_commit="",
        scenario_matrix="",
        scenario_matrix_hash="",
        episodes="—",
        reports=(),
        manifest_sha256_present=False,
        incomplete_reasons=("missing campaign_id|extra",),
        remote_base="https://github.com/ll7/robot_sf_ll7",
    )
    rendered = render_atlas([row], atlas_output=_ATLAS_OUTPUT)
    data_line = [line for line in rendered.splitlines() if line.startswith("| ")][2]
    # The literal ``|`` inside the campaign id/reason must be escaped.
    assert "a\\|b" in data_line
    assert "missing campaign_id\\|extra" in data_line


def test_render_counts_ok_and_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The summary line reports correct OK/INCOMPLETE counts."""
    repo = _make_git_repo(tmp_path)
    _chdir(repo, monkeypatch)
    _write_manifest(repo, "ok_demo", campaign_id="ok_one")
    _write_manifest(repo, "broken_demo", campaign_id="ok_two")
    (repo / f"{_EVIDENCE}/broken_demo/campaign/campaign_manifest.json").write_text(
        "garbage", encoding="utf-8"
    )
    rows = _scan(repo)
    rendered = render_atlas(rows, atlas_output=_ATLAS_OUTPUT)
    assert "Campaigns indexed: **2**" in rendered
    assert "OK: **1**, INCOMPLETE: **1**" in rendered


def test_render_no_timestamps_in_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The atlas body is free of wall-clock timestamps for diff-stability."""
    import re

    repo = _build_registry(tmp_path)
    _chdir(repo, monkeypatch)
    rows = _scan(repo)
    rendered = render_atlas(rows, atlas_output=_ATLAS_OUTPUT)
    # No standalone ``Generated ...`` style wall-clock lines.
    assert not re.search(r"generated at|run at|today|now:", rendered, re.IGNORECASE)


# --- determinism ---------------------------------------------------------


def test_generate_is_deterministic_across_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two generations from the same registry produce byte-identical output."""
    repo = _build_registry(tmp_path)
    _chdir(repo, monkeypatch)
    first, n1 = generate(Path(_EVIDENCE), _ATLAS_OUTPUT)
    second, n2 = generate(Path(_EVIDENCE), _ATLAS_OUTPUT)
    assert n1 == n2 == 2
    assert first == second


# --- CLI main(): write + check -------------------------------------------


def test_main_writes_output_and_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` without ``--check`` writes the atlas and returns 0."""
    repo = _build_registry(tmp_path)
    _chdir(repo, monkeypatch)
    out = repo / "out" / "ATLAS.md"
    rc = main(["--evidence-root", _EVIDENCE, "--output", str(out)])
    assert rc == 0
    assert out.is_file()
    assert "Campaigns indexed: **2**" in out.read_text(encoding="utf-8")


def test_main_check_passes_when_up_to_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--check`` returns 0 when the committed file matches generation."""
    repo = _build_registry(tmp_path)
    _chdir(repo, monkeypatch)
    out = repo / "ATLAS.md"
    main(["--evidence-root", _EVIDENCE, "--output", str(out)])
    rc = main(["--evidence-root", _EVIDENCE, "--output", str(out), "--check"])
    assert rc == 0


def test_main_check_fails_when_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--check`` returns 1 when the committed file differs from generation."""
    repo = _build_registry(tmp_path)
    _chdir(repo, monkeypatch)
    out = repo / "ATLAS.md"
    main(["--evidence-root", _EVIDENCE, "--output", str(out)])
    out.write_text(out.read_text(encoding="utf-8") + "\nSTALE\n", encoding="utf-8")
    rc = main(["--evidence-root", _EVIDENCE, "--output", str(out), "--check"])
    assert rc == 1
    assert "stale" in capsys.readouterr().err.lower()


def test_main_check_fails_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--check`` returns 1 when the atlas file does not exist yet."""
    repo = _build_registry(tmp_path)
    _chdir(repo, monkeypatch)
    out = repo / "does_not_exist.md"
    rc = main(["--evidence-root", _EVIDENCE, "--output", str(out), "--check"])
    assert rc == 1


def test_main_check_fails_when_registry_grew(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newly-added campaign makes a previously-current atlas stale."""
    repo = _build_registry(tmp_path)
    _chdir(repo, monkeypatch)
    out = repo / "ATLAS.md"
    main(["--evidence-root", _EVIDENCE, "--output", str(out)])
    # Add a third campaign after the first commit.
    _write_manifest(
        repo,
        "gamma_demo_2026-05-03",
        campaign_id="gamma_campaign_20260503",
        commit="3333333333333333333333333333333333333333",
    )
    rc = main(["--evidence-root", _EVIDENCE, "--output", str(out), "--check"])
    assert rc == 1
