"""Tests for the changed-path docs/evidence integrity check (issue #3476)."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from scripts.dev.check_docs_evidence_integrity import check_files, main

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


def _write_catalog(root: Path, paths: list[str]) -> None:
    """Write a minimal context catalog registering evidence paths."""
    catalog = root / "docs/context/catalog.yaml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    entries = "\n".join(
        f"  - path: {path}\n    status: evidence\n    freshness: evidence" for path in paths
    )
    catalog.write_text(
        "version: 1\n"
        "status_values:\n"
        "  evidence: Evidence pointer or manifest.\n"
        "freshness_values:\n"
        "  evidence: Evidence pointer.\n"
        "entries:\n"
        f"{entries}\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """Return sha256 digest for a test fixture file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_valid_files_pass(tmp_path: Path) -> None:
    """Well-formed JSON/YAML/resolvable relative link produce no problems."""
    (tmp_path / "target.md").write_text("# target\n", encoding="utf-8")
    (tmp_path / "data.json").write_text('{"ok": true}\n', encoding="utf-8")
    (tmp_path / "data.yaml").write_text("schema_version: 1\nitems: [a, b]\n", encoding="utf-8")
    (tmp_path / "note.md").write_text(
        "See [target](./target.md) [home](https://example.com) and [anchor](#section).\n",
        encoding="utf-8",
    )

    problems = check_files(["data.json", "data.yaml", "note.md"], root=tmp_path)

    assert problems == []


def test_invalid_json_is_reported(tmp_path: Path) -> None:
    """Malformed JSON in a changed evidence file must be flagged."""
    (tmp_path / "broken.json").write_text("{not valid json}", encoding="utf-8")

    problems = check_files(["broken.json"], root=tmp_path)

    assert len(problems) == 1
    assert "invalid JSON" in problems[0]


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    """Malformed YAML in a changed catalogue/manifest file must be flagged."""
    (tmp_path / "broken.yaml").write_text("key: [unclosed\n", encoding="utf-8")

    problems = check_files(["broken.yaml"], root=tmp_path)

    assert len(problems) == 1
    assert "invalid YAML" in problems[0]


def test_broken_relative_link_is_reported(tmp_path: Path) -> None:
    """Explicit relative links must resolve."""
    (tmp_path / "note.md").write_text("[missing](./missing.md)\n", encoding="utf-8")

    problems = check_files(["note.md"], root=tmp_path)

    assert len(problems) == 1
    assert "broken repo-local link" in problems[0]


def test_relative_link_cannot_escape_repo(tmp_path: Path) -> None:
    """Relative Markdown links cannot escape the checkout."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "docs").mkdir()
    (root / "docs/note.md").write_text("[escape](../../outside.md)\n", encoding="utf-8")

    problems = check_files(["docs/note.md"], root=root)

    assert len(problems) == 1
    assert "escapes repository" in problems[0]


def test_link_with_anchor_fragment_resolves_to_file(tmp_path: Path) -> None:
    """Only the file part of an anchor link is validated."""
    (tmp_path / "target.md").write_text("# target\n", encoding="utf-8")
    (tmp_path / "note.md").write_text("[sec](./target.md#section)\n", encoding="utf-8")

    assert check_files(["note.md"], root=tmp_path) == []


def test_changed_evidence_file_must_be_catalog_registered(tmp_path: Path) -> None:
    """A changed evidence file without a catalog entry must fail."""
    summary = tmp_path / "docs/context/evidence/issue_999_missing/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"status": "ok"}\n', encoding="utf-8")
    _write_catalog(tmp_path, [])

    problems = check_files([summary.relative_to(tmp_path).as_posix()], root=tmp_path)

    assert any("not registered" in problem for problem in problems)


def test_changed_evidence_file_checksum_must_match(tmp_path: Path) -> None:
    """A changed evidence file covered by a checksum manifest must match it."""
    bundle = tmp_path / "docs/context/evidence/issue_999_checksums"
    bundle.mkdir(parents=True)
    summary = bundle / "summary.json"
    summary.write_text('{"status": "ok"}\n', encoding="utf-8")
    manifest = bundle / "SHA256SUMS"
    manifest.write_text(
        f"{'0' * 64}  {summary.relative_to(tmp_path).as_posix()}\n",
        encoding="utf-8",
    )
    _write_catalog(
        tmp_path,
        [
            summary.relative_to(tmp_path).as_posix(),
            manifest.relative_to(tmp_path).as_posix(),
        ],
    )

    problems = check_files([summary.relative_to(tmp_path).as_posix()], root=tmp_path)

    assert any("checksum mismatch" in problem for problem in problems)


def test_changed_checksum_manifest_validates_targets(tmp_path: Path) -> None:
    """A changed checksum manifest validates referenced file checksums."""
    bundle = tmp_path / "docs/context/evidence/issue_999_manifest"
    bundle.mkdir(parents=True)
    summary = bundle / "summary.json"
    summary.write_text('{"status": "ok"}\n', encoding="utf-8")
    manifest = bundle / "SHA256SUMS"
    manifest.write_text(
        f"{_sha256(summary)}  {summary.relative_to(tmp_path).as_posix()}\n",
        encoding="utf-8",
    )
    _write_catalog(
        tmp_path,
        [
            summary.relative_to(tmp_path).as_posix(),
            manifest.relative_to(tmp_path).as_posix(),
        ],
    )

    problems = check_files([manifest.relative_to(tmp_path).as_posix()], root=tmp_path)

    assert problems == []


def test_bare_name_manifest_verifies_packet_local_file(tmp_path: Path) -> None:
    """A bare SHA256SUMS entry must verify the packet's own file, not a repo-root twin.

    Regression for issue #4317: packets generated with ``sha256sum *`` list bare
    filenames (for example ``README.md``). A repo-root file with the same name must
    not shadow the packet-local file during verification.
    """
    (tmp_path / "README.md").write_text("# repo root readme\n", encoding="utf-8")

    bundle = tmp_path / "docs/context/evidence/issue_999_collision"
    bundle.mkdir(parents=True)
    readme = bundle / "README.md"
    readme.write_text("# packet readme\n", encoding="utf-8")
    manifest = bundle / "SHA256SUMS"
    manifest.write_text(f"{_sha256(readme)}  README.md\n", encoding="utf-8")
    _write_catalog(
        tmp_path,
        [
            readme.relative_to(tmp_path).as_posix(),
            manifest.relative_to(tmp_path).as_posix(),
        ],
    )

    problems = check_files([manifest.relative_to(tmp_path).as_posix()], root=tmp_path)

    assert problems == []


def test_bare_name_mismatch_names_packet_local_target(tmp_path: Path) -> None:
    """A wrong bare-name hash must fail against the packet-local file, not repo root."""
    (tmp_path / "README.md").write_text("# repo root readme\n", encoding="utf-8")

    bundle = tmp_path / "docs/context/evidence/issue_999_collision_bad"
    bundle.mkdir(parents=True)
    readme = bundle / "README.md"
    readme.write_text("# packet readme\n", encoding="utf-8")
    manifest = bundle / "SHA256SUMS"
    manifest.write_text(f"{'0' * 64}  README.md\n", encoding="utf-8")
    _write_catalog(
        tmp_path,
        [
            readme.relative_to(tmp_path).as_posix(),
            manifest.relative_to(tmp_path).as_posix(),
        ],
    )

    problems = check_files([manifest.relative_to(tmp_path).as_posix()], root=tmp_path)

    assert any("checksum mismatch" in problem for problem in problems)
    # The mismatch must name the packet-local README, not the repo-root twin.
    assert any(str(readme) in problem for problem in problems)
    assert not any(
        problem.endswith(f"checksum mismatch for {tmp_path / 'README.md'}") for problem in problems
    )


def test_changed_bare_name_packet_file_is_validated(tmp_path: Path) -> None:
    """Changing a packet-local file finds its adjacent bare-name manifest entry (issue #4317)."""
    (tmp_path / "README.md").write_text("# repo root readme\n", encoding="utf-8")

    bundle = tmp_path / "docs/context/evidence/issue_999_changed_bare"
    bundle.mkdir(parents=True)
    readme = bundle / "README.md"
    readme.write_text("# packet readme\n", encoding="utf-8")
    manifest = bundle / "SHA256SUMS"
    manifest.write_text(f"{_sha256(readme)}  README.md\n", encoding="utf-8")
    _write_catalog(
        tmp_path,
        [
            readme.relative_to(tmp_path).as_posix(),
            manifest.relative_to(tmp_path).as_posix(),
        ],
    )

    problems = check_files([readme.relative_to(tmp_path).as_posix()], root=tmp_path)

    assert problems == []


def test_catalog_change_validates_registered_paths(tmp_path: Path) -> None:
    """Changing catalog.yaml validates that registered paths exist."""
    _write_catalog(tmp_path, ["docs/context/evidence/issue_999_missing/summary.json"])

    problems = check_files(["docs/context/catalog.yaml"], root=tmp_path)

    assert any("does not exist" in problem for problem in problems)


def test_catalog_directory_entry_is_accepted(tmp_path: Path) -> None:
    """An evidence bundle registered as a directory must not be flagged missing."""
    bundle = tmp_path / "docs/context/evidence/issue_999_bundle"
    bundle.mkdir(parents=True)
    (bundle / "summary.json").write_text('{"status": "ok"}\n', encoding="utf-8")
    _write_catalog(tmp_path, [bundle.relative_to(tmp_path).as_posix()])

    problems = check_files(["docs/context/catalog.yaml"], root=tmp_path)

    assert not any("does not exist" in problem for problem in problems)


def test_file_inside_directory_registered_bundle_counts_as_registered(tmp_path: Path) -> None:
    """A changed file inside a directory-registered bundle is treated as registered."""
    bundle = tmp_path / "docs/context/evidence/issue_999_bundle"
    bundle.mkdir(parents=True)
    summary = bundle / "summary.json"
    summary.write_text('{"status": "ok"}\n', encoding="utf-8")
    _write_catalog(tmp_path, [bundle.relative_to(tmp_path).as_posix()])

    problems = check_files([summary.relative_to(tmp_path).as_posix()], root=tmp_path)

    assert not any("not registered" in problem for problem in problems)


def test_readme_summary_classification_drift_is_reported(tmp_path: Path) -> None:
    """README evidence status must not disagree with summary.json fields."""
    bundle = tmp_path / "docs/context/evidence/issue_999_drift"
    bundle.mkdir(parents=True)
    readme = bundle / "README.md"
    readme.write_text(
        "# Evidence\n\n## Evidence Status\n- `result_classification`: `positive_result`\n",
        encoding="utf-8",
    )
    summary = bundle / "summary.json"
    summary.write_text(
        json.dumps({"result_classification": "negative_result"}) + "\n",
        encoding="utf-8",
    )
    _write_catalog(
        tmp_path,
        [
            readme.relative_to(tmp_path).as_posix(),
            summary.relative_to(tmp_path).as_posix(),
        ],
    )

    problems = check_files([readme.relative_to(tmp_path).as_posix()], root=tmp_path)

    assert any("disagrees" in problem for problem in problems)


def test_cited_command_and_config_paths_must_exist(tmp_path: Path) -> None:
    """Changed docs verify practical script/config paths cited in commands."""
    note = tmp_path / "docs/context/issue_999.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "Run `uv run python scripts/missing.py --config configs/missing.yaml`.\n",
        encoding="utf-8",
    )

    problems = check_files([note.relative_to(tmp_path).as_posix()], root=tmp_path)

    assert sum("cited command/config path" in problem for problem in problems) == 2


def test_output_flag_paths_are_not_required_to_exist(tmp_path: Path) -> None:
    """Paths handed to an output flag are created by the command, not inputs."""
    script = tmp_path / "scripts/tools/create_scenario.py"
    script.parent.mkdir(parents=True)
    script.write_text("# stub\n", encoding="utf-8")

    note = tmp_path / "docs/context/issue_998.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "Run `uv run python scripts/tools/create_scenario.py "
        "--output configs/scenarios/single/draft_review.yaml`.\n",
        encoding="utf-8",
    )

    problems = check_files([note.relative_to(tmp_path).as_posix()], root=tmp_path)

    assert not any("cited command/config path" in problem for problem in problems)


def test_artifact_registry_paths_may_be_missing(tmp_path: Path) -> None:
    """A self-declared artifact-presence registry may list not-yet-existing artifacts."""
    registry = tmp_path / "configs/research/registry.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        "schema_version: research-package-registry.v1\n"
        "packages:\n"
        "  - id: pkg\n"
        "    title: Pkg\n"
        "    required_artifacts:\n"
        "      - scripts/tools/not_yet_built.py\n",
        encoding="utf-8",
    )

    problems = check_files([registry.relative_to(tmp_path).as_posix()], root=tmp_path)

    assert not any("cited command/config path" in problem for problem in problems)


def test_non_registry_yaml_still_enforces_cited_paths(tmp_path: Path) -> None:
    """A YAML file without the registry schema marker still requires cited paths to exist."""
    config = tmp_path / "configs/research/other.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("entrypoint: scripts/tools/not_yet_built.py\n", encoding="utf-8")

    problems = check_files([config.relative_to(tmp_path).as_posix()], root=tmp_path)

    assert any("cited command/config path" in problem for problem in problems)


def test_warn_only_mode_does_not_fail_on_problems(
    tmp_path: Path, capsys, monkeypatch: MonkeyPatch
) -> None:
    """Advisory mode exits 0 and emits GitHub warnings."""
    monkeypatch.setattr("scripts.dev.check_docs_evidence_integrity._repo_root", lambda: tmp_path)
    (tmp_path / "broken.json").write_text("{not valid}", encoding="utf-8")

    exit_code = main(["--files", "broken.json", "--warn-only"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "::warning::" in out
    assert "invalid JSON" in out


def test_blocking_mode_fails_on_problems(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Without --warn-only the same problem fails closed."""
    monkeypatch.setattr("scripts.dev.check_docs_evidence_integrity._repo_root", lambda: tmp_path)
    (tmp_path / "broken.json").write_text("{not valid}", encoding="utf-8")

    assert main(["--files", "broken.json"]) == 1


def test_split_list_config_does_not_yield_phantom_dash(tmp_path: Path) -> None:
    """A YAML or markdown document splitting --config and its value doesn't yield '-'."""
    config_file = tmp_path / "configs/training/learned_risk_model_v1.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("model: learned_risk\n", encoding="utf-8")

    note = tmp_path / "docs/context/issue_4692.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "command:\n  - --config\n  - configs/training/learned_risk_model_v1.yaml\n",
        encoding="utf-8",
    )

    problems = check_files([note.relative_to(tmp_path).as_posix()], root=tmp_path)
    assert problems == []


def test_split_list_config_still_detects_genuine_missing_path(tmp_path: Path) -> None:
    """A split-list config command still reports the config if it does not exist."""
    note = tmp_path / "docs/context/issue_4692.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "command:\n  - --config\n  - configs/training/missing_config_file.yaml\n",
        encoding="utf-8",
    )

    problems = check_files([note.relative_to(tmp_path).as_posix()], root=tmp_path)
    assert len(problems) == 1
    assert "configs/training/missing_config_file.yaml" in problems[0]
