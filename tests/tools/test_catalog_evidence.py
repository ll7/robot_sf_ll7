"""Tests for the evidence-catalog backlog proposer (scripts/tools/catalog_evidence.py).

The tool drains the issue #3014 cataloging backlog by proposing additive
``docs/context/catalog.yaml`` rows for uncovered evidence bundles.  These tests
pin the safety contract: deterministic inference, conservative
``needs-human-review`` routing, additive-only/stable ``--apply``, and
idempotence.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.tools.catalog_evidence import (
    CatalogProposal,
    ProposedEntry,
    _representative_member,
    _validate_entry_vocabulary,
    apply_proposal,
    build_proposal,
    infer_area,
    infer_freshness,
    infer_status,
    main,
    render_dry_run,
    render_entry_block,
)

_EVIDENCE = "docs/context/evidence"


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


def _git_add_commit(repo_root: Path) -> None:
    """Stage and commit everything in the tmp git repo."""
    subprocess.run(["git", "add", "-A"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"], cwd=repo_root, check=True, capture_output=True
    )


def _write(repo_root: Path, rel: str, text: str) -> Path:
    """Write a file under the repo, creating parents."""
    path = repo_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


_CATALOG_HEADER = (
    "version: 1\n"
    "status_values:\n"
    "  current: x\n"
    "  historical: x\n"
    "  superseded: x\n"
    "  evidence: x\n"
    "  proposal: x\n"
    "freshness_values:\n"
    "  maintained: x\n"
    "  dated: x\n"
    "  policy: x\n"
    "  evidence: x\n"
    "entries:\n"
)


def _build_sample_repo(tmp_path: Path) -> Path:
    """Build a repo exercising every classification branch of the proposer."""
    repo = _make_git_repo(tmp_path)

    # Clean benchmark bundle (area via "seed" keyword).
    _write(repo, f"{_EVIDENCE}/issue_500_seed_sensitivity_2026-01-01/summary.json", '{"ok": 1}\n')
    # Clean adversarial bundle (specific topic family).
    _write(repo, f"{_EVIDENCE}/issue_501_adversarial_smoke_2026-01-01/report.json", '{"ok": 1}\n')
    # Clean standalone learned-policy file (area via "lidar_ppo").
    _write(repo, f"{_EVIDENCE}/issue_1662_lidar_ppo_smoke_summary.json", '{"ok": 1}\n')
    # Ambiguous area: no keyword matches -> needs-human-review.
    _write(repo, f"{_EVIDENCE}/issue_503_zzz_unknown_topic_2026-01-01/summary.json", '{"ok": 1}\n')
    # Benchmark name, but the only evidence file references output/ -> fail-closed review.
    _write(
        repo,
        f"{_EVIDENCE}/issue_504_seed_dirty_2026-01-01/summary.json",
        '{"checkpoint": "output/tmp/model.pt"}\n',
    )
    # Already covered bundle -> must NOT be re-proposed.
    _write(repo, f"{_EVIDENCE}/issue_505_already_seed_2026-01-01/summary.json", '{"ok": 1}\n')

    catalog = (
        _CATALOG_HEADER
        + f"  - path: {_EVIDENCE}/issue_505_already_seed_2026-01-01/summary.json\n"
        + "    status: evidence\n"
        + "    freshness: evidence\n"
        + "    area: benchmark_evidence\n"
    )
    _write(repo, "docs/context/catalog.yaml", catalog)
    _git_add_commit(repo)
    return repo


# --- inference ------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("issue_1_adversarial_x", "adversarial_search"),
        ("issue_2_carla_replay_y", "carla_external_sim"),
        ("issue_3_predictive_same_seed", "predictive_planner"),
        ("issue_4_orca_residual_bc_smoke", "learned_policy"),
        ("issue_5_seed_sensitivity", "benchmark_evidence"),
        ("issue_6_zzz_unknown_topic", None),
    ],
)
def test_infer_area_keyword_map(name: str, expected: str | None) -> None:
    """Area inference is deterministic, ordered, and returns None when ambiguous."""
    assert infer_area(Path(f"{_EVIDENCE}/{name}")) == expected


def test_infer_status_and_freshness_inside_evidence_dir() -> None:
    """Anchors under docs/context/evidence/ classify as tracked evidence."""
    bundle = Path(f"{_EVIDENCE}/issue_9_example")
    assert infer_status(bundle) == "evidence"
    assert infer_freshness(bundle) == "evidence"


def test_infer_status_and_freshness_outside_evidence_dir() -> None:
    """Anchors outside the evidence root are not confidently classifiable."""
    other = Path("configs/benchmarks/issue_9_example.yaml")
    assert infer_status(other) is None
    assert infer_freshness(other) is None


# --- representative file selection ---------------------------------------


def test_representative_member_prefers_summary_then_skips_dirty(tmp_path: Path) -> None:
    """Selection prefers canonical proof names and skips output/ pointing files."""
    repo = _make_git_repo(tmp_path)
    bundle = Path(f"{_EVIDENCE}/issue_700_seed_x")
    readme = _write(repo, f"{bundle.as_posix()}/README.md", "clean\n")
    summary = _write(repo, f"{bundle.as_posix()}/summary.json", '{"ok": 1}\n')
    members = [readme, summary]
    # Make repo-relative paths (as the tool consumes them).
    rel_members = [m.relative_to(repo) for m in members]
    chosen = _representative_member(rel_members, repo)
    assert chosen == Path(f"{bundle.as_posix()}/summary.json")


def test_representative_member_returns_none_when_all_dirty(tmp_path: Path) -> None:
    """When every member trips the evidence content scan, route to review."""
    repo = _make_git_repo(tmp_path)
    bundle = Path(f"{_EVIDENCE}/issue_701_seed_x")
    dirty = _write(repo, f"{bundle.as_posix()}/summary.json", '{"p": "output/tmp/x.pt"}\n')
    chosen = _representative_member([dirty.relative_to(repo)], repo)
    assert chosen is None


# --- proposal building ----------------------------------------------------


def test_build_proposal_classifies_and_routes(tmp_path: Path) -> None:
    """End-to-end proposal: confident entries plus conservative review routing."""
    repo = _build_sample_repo(tmp_path)
    proposal = build_proposal(repo)

    proposed_by_area = {entry.area for entry in proposal.proposed}
    proposed_paths = {entry.path for entry in proposal.proposed}

    # Three clean, classifiable bundles are proposed.
    assert {"benchmark_evidence", "adversarial_search", "learned_policy"} <= proposed_by_area
    assert f"{_EVIDENCE}/issue_500_seed_sensitivity_2026-01-01/summary.json" in proposed_paths
    assert f"{_EVIDENCE}/issue_1662_lidar_ppo_smoke_summary.json" in proposed_paths

    # Every proposed entry uses canonical evidence status/freshness.
    assert all(e.status == "evidence" and e.freshness == "evidence" for e in proposal.proposed)

    # The already-cataloged bundle is never re-proposed.
    assert all("issue_505_already_seed" not in p for p in proposed_paths)

    review_bundles = {item.bundle: item.reason for item in proposal.needs_human_review}
    # Ambiguous area routed to review (not guessed).
    ambiguous = f"{_EVIDENCE}/issue_503_zzz_unknown_topic_2026-01-01"
    assert ambiguous in review_bundles
    assert "area" in review_bundles[ambiguous]
    # Dirty evidence routed to review (fail-closed), despite a matching area keyword.
    dirty = f"{_EVIDENCE}/issue_504_seed_dirty_2026-01-01"
    assert dirty in review_bundles
    assert "clean" in review_bundles[dirty]


def test_build_proposal_uses_custom_catalog_path(tmp_path: Path) -> None:
    """A custom catalog path controls which evidence bundles are considered covered."""
    repo = _make_git_repo(tmp_path)
    uncovered = f"{_EVIDENCE}/issue_600_seed_uncovered_2026-01-01/summary.json"
    covered_only_by_custom = f"{_EVIDENCE}/issue_601_seed_custom_2026-01-01/summary.json"
    _write(repo, uncovered, '{"ok": 1}\n')
    _write(repo, covered_only_by_custom, '{"ok": 1}\n')
    _write(repo, "docs/context/catalog.yaml", _CATALOG_HEADER)
    custom_catalog = Path("docs/context/custom_catalog.yaml")
    _write(
        repo,
        custom_catalog.as_posix(),
        _CATALOG_HEADER
        + f"  - path: {covered_only_by_custom}\n"
        + "    status: evidence\n"
        + "    freshness: evidence\n"
        + "    area: benchmark_evidence\n",
    )
    _git_add_commit(repo)

    proposal = build_proposal(repo, catalog_path=custom_catalog)

    proposed_paths = {entry.path for entry in proposal.proposed}
    assert uncovered in proposed_paths
    assert covered_only_by_custom not in proposed_paths


def test_proposal_is_path_sorted(tmp_path: Path) -> None:
    """Proposed entries are sorted by referenced path for stable review."""
    repo = _build_sample_repo(tmp_path)
    proposal = build_proposal(repo)
    paths = [entry.path for entry in proposal.proposed]
    assert paths == sorted(paths)


# --- apply: additive, stable, idempotent ----------------------------------


def test_apply_is_additive_and_preserves_existing_rows(tmp_path: Path) -> None:
    """--apply only appends; the original catalog text is a byte-for-byte prefix."""
    repo = _build_sample_repo(tmp_path)
    catalog_file = repo / "docs/context/catalog.yaml"
    before = catalog_file.read_text(encoding="utf-8")

    proposal = build_proposal(repo)
    written = apply_proposal(proposal, catalog_file)
    after = catalog_file.read_text(encoding="utf-8")

    assert written == len(proposal.proposed) > 0
    # Additive-only: the prior content is an exact prefix of the new content.
    assert after.startswith(before)
    appended = after[len(before) :]
    # Every appended line is a new entry field; nothing else changed.
    assert "- path:" in appended
    assert after.count("issue_505_already_seed") == 1  # existing row untouched, not duplicated


def test_apply_is_idempotent(tmp_path: Path) -> None:
    """Re-running after apply proposes nothing new for the now-covered bundles."""
    repo = _build_sample_repo(tmp_path)
    catalog_file = repo / "docs/context/catalog.yaml"

    first = build_proposal(repo)
    apply_proposal(first, catalog_file)

    second = build_proposal(repo)
    # All previously-proposed bundles are now covered -> none re-proposed.
    assert second.proposed == []
    # The unresolved review queue is unchanged (still needs a human).
    assert {i.bundle for i in second.needs_human_review} == {
        i.bundle for i in first.needs_human_review
    }


def test_apply_noop_when_nothing_proposed(tmp_path: Path) -> None:
    """Applying an empty proposal writes nothing and reports zero."""
    repo = _make_git_repo(tmp_path)
    _write(repo, "docs/context/catalog.yaml", _CATALOG_HEADER.rstrip("\n") + " []\n")
    _git_add_commit(repo)
    catalog_file = repo / "docs/context/catalog.yaml"
    before = catalog_file.read_text(encoding="utf-8")

    proposal = build_proposal(repo)
    assert apply_proposal(proposal, catalog_file) == 0
    assert catalog_file.read_text(encoding="utf-8") == before


# --- vocabulary safety ----------------------------------------------------


def test_render_entry_block_matches_indentation() -> None:
    """Rendered rows use the catalog's 2/4-space indentation and field order."""
    block = render_entry_block(
        ProposedEntry(
            bundle="b",
            path=f"{_EVIDENCE}/issue_1_seed/summary.json",
            status="evidence",
            freshness="evidence",
            area="benchmark_evidence",
        )
    )
    assert block == (
        f"  - path: {_EVIDENCE}/issue_1_seed/summary.json\n"
        "    status: evidence\n"
        "    freshness: evidence\n"
        "    area: benchmark_evidence\n"
    )


def test_render_dry_run_uses_custom_catalog_path() -> None:
    """The dry-run diff header names the selected catalog path."""
    proposal = CatalogProposal(
        proposed=[
            ProposedEntry(
                bundle=f"{_EVIDENCE}/issue_1_seed",
                path=f"{_EVIDENCE}/issue_1_seed/summary.json",
                status="evidence",
                freshness="evidence",
                area="benchmark_evidence",
            )
        ],
        needs_human_review=[],
    )

    rendered = render_dry_run(proposal, catalog_rel=Path("docs/context/custom_catalog.yaml"))

    assert rendered.splitlines()[:2] == [
        "--- docs/context/custom_catalog.yaml (entries: additions)",
        "+++ docs/context/custom_catalog.yaml",
    ]


def test_main_uses_custom_catalog_for_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI uses --catalog for both proposal discovery and dry-run headers."""
    repo = _make_git_repo(tmp_path)
    uncovered = f"{_EVIDENCE}/issue_610_seed_uncovered_2026-01-01/summary.json"
    covered_only_by_custom = f"{_EVIDENCE}/issue_611_seed_custom_2026-01-01/summary.json"
    _write(repo, uncovered, '{"ok": 1}\n')
    _write(repo, covered_only_by_custom, '{"ok": 1}\n')
    _write(repo, "docs/context/catalog.yaml", _CATALOG_HEADER)
    custom_catalog = Path("docs/context/custom_catalog.yaml")
    _write(
        repo,
        custom_catalog.as_posix(),
        _CATALOG_HEADER
        + f"  - path: {covered_only_by_custom}\n"
        + "    status: evidence\n"
        + "    freshness: evidence\n"
        + "    area: benchmark_evidence\n",
    )
    _git_add_commit(repo)

    assert main(["--repo-root", str(repo), "--catalog", custom_catalog.as_posix()]) == 0

    output = capsys.readouterr().out
    assert output.splitlines()[:2] == [
        "--- docs/context/custom_catalog.yaml (entries: additions)",
        "+++ docs/context/custom_catalog.yaml",
    ]
    assert uncovered in output
    assert covered_only_by_custom not in output


def test_validate_entry_vocabulary_rejects_non_canonical() -> None:
    """The defensive guard refuses to emit non-canonical status/freshness."""
    with pytest.raises(ValueError, match="status"):
        _validate_entry_vocabulary(
            ProposedEntry(bundle="b", path="p", status="bogus", freshness="evidence", area="x")
        )
    with pytest.raises(ValueError, match="freshness"):
        _validate_entry_vocabulary(
            ProposedEntry(bundle="b", path="p", status="evidence", freshness="bogus", area="x")
        )
