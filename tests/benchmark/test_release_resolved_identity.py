"""End-to-end contract tests for non-self-referential release identity resolution."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from robot_sf.benchmark import release_acceptance, release_doctor, release_protocol
from robot_sf.benchmark.release_protocol import (
    RELEASE_IDENTITY_TEMPLATE_SCHEMA_VERSION,
    load_release_campaign_config,
    load_release_manifest,
    verify_resolved_release_identity,
    write_resolved_release_identity,
)
from robot_sf.benchmark.release_tag_identity import derive_sha_tag
from robot_sf.benchmark.zenodo_publisher import build_release_binding
from scripts.tools import resolve_benchmark_release_identity as identity_cli
from scripts.tools import run_benchmark_release

if TYPE_CHECKING:
    from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_canonical_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n",
        encoding="utf-8",
    )


def _release_template_repository(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Release Fixture")
    _git(repo, "config", "user.email", "release-fixture@example.invalid")
    (repo / ".gitignore").write_text("output/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "fixture: initialize ignored output")

    scenarios = repo / "scenarios.yaml"
    _write_yaml(scenarios, [{"name": f"scenario_{index:02d}"} for index in range(48)])
    seed_sets = repo / "seed_sets.yaml"
    _write_yaml(seed_sets, {"paper_eval_s30": list(range(111, 141))})
    for name in ("suite.yaml", "route.yaml", "comparability.yaml"):
        _write_yaml(repo / name, {"schema_version": "fixture.v1"})
    (repo / "CITATION.cff").write_text("cff-version: 1.2.0\n", encoding="utf-8")
    (repo / "RELEASE.md").write_text("# Fixture release\n", encoding="utf-8")
    planner_config = repo / "planner.yaml"
    _write_yaml(planner_config, {"schema_version": "fixture-planner.v1"})

    planner_keys = [f"planner_{index:02d}" for index in range(14)]
    campaign = repo / "campaign.yaml"
    _write_yaml(
        campaign,
        {
            "name": "future_s30_h600_fixture",
            "paper_facing": True,
            "paper_profile_version": "paper-matrix-v1",
            "paper_interpretation_profile": "baseline-ready-core",
            "scenario_matrix": "scenarios.yaml",
            "comparability_mapping": "comparability.yaml",
            "route_clearance_certifications": "route.yaml",
            "seed_policy": {
                "mode": "seed-set",
                "seed_set": "paper_eval_s30",
                "seed_sets_path": "seed_sets.yaml",
            },
            "workers": 1,
            "horizon": 600,
            "dt": 0.1,
            "checkpoint_provenance_enforcement": "error",
            "kinematics_matrix": ["differential_drive"],
            "release_tag": "{{release_tag}}",
            "doi": "{{version_doi}}",
            "planners": [
                {
                    "key": key,
                    "algo": "goal",
                    "planner_group": "core",
                    "socnav_missing_prereq_policy": "fail-fast",
                    **({"algo_config": planner_config.name} if index == 0 else {}),
                }
                for index, key in enumerate(planner_keys)
            ],
        },
    )

    metadata_template = repo / "zenodo_metadata.template.json"
    metadata_template.write_text(
        json.dumps(
            {
                "metadata": {
                    "title": "Future benchmark fixture",
                    "upload_type": "dataset",
                    "access_right": "open",
                    "license": "GPL-3.0-only",
                    "creators": [{"name": "Release Fixture"}],
                    "description": (
                        "SNQI is advisory and supplies no ranking claim. "
                        "source={{source_sha}} concept={{concept_doi}} "
                        "version={{version_doi}}"
                    ),
                    "related_identifiers": [
                        {
                            "identifier": (
                                "https://github.com/ll7/robot_sf_ll7/releases/tag/{{release_tag}}"
                            ),
                            "relation": "isSupplementTo",
                            "scheme": "url",
                        },
                        {
                            "identifier": (
                                "https://github.com/ll7/robot_sf_ll7/commit/{{source_sha}}"
                            ),
                            "relation": "isDerivedFrom",
                            "scheme": "url",
                        },
                    ],
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    template = repo / "release.template.yaml"
    _write_yaml(
        template,
        {
            "schema_version": "benchmark-release-manifest.v0.2",
            "benchmark_protocol_version": "0.1.0",
            "release_id": "{{release_tag}}",
            "release_tag": "{{release_tag}}",
            "maturity": "pre-1.0",
            "release_kind": "benchmark-data",
            "latest_main_base_commit": "a" * 40,
            "canonical_campaign_config": "campaign.yaml",
            "campaign_config_sha256": _sha256(campaign),
            "expected_paper_profile_version": "paper-matrix-v1",
            "expected_paper_interpretation_profile": "baseline-ready-core",
            "matrix": {"expected_episode_cells": 20160, "horizon_steps": 600},
            "publication": {
                "channel": "direct_zenodo_benchmark_dataset",
                "concept_doi": "{{concept_doi}}",
                "version_doi": "{{version_doi}}",
                "metadata_path": metadata_template.name,
                "metadata_sha256": _sha256(metadata_template),
            },
            "scenario": {
                "matrix_path": scenarios.name,
                "matrix_sha256": _sha256(scenarios),
                "suite_policy_path": "suite.yaml",
                "suite_policy_sha256": _sha256(repo / "suite.yaml"),
                "route_certification_path": "route.yaml",
                "route_certification_sha256": _sha256(repo / "route.yaml"),
            },
            "seed_policy": {
                "mode": "seed-set",
                "seed_set": "paper_eval_s30",
                "seeds": [],
                "seed_sets_path": seed_sets.name,
                "seed_sets_sha256": _sha256(seed_sets),
                "resolved_seeds": list(range(111, 141)),
            },
            "metrics": {"snqi_claim_policy": "advisory_no_ranking"},
            "planners": {
                "keys": planner_keys,
                "groups": dict.fromkeys(planner_keys, "core"),
            },
            "kinematics": {"matrix": ["differential_drive"]},
            "artifacts": {"required_paths": ["reports/campaign_summary.json"]},
            "provenance": {
                "repository_url": "https://github.com/ll7/robot_sf_ll7",
                "doi": "{{version_doi}}",
            },
            "citation_path": "CITATION.cff",
            "release_checklist_path": "RELEASE.md",
            "identity_resolution": {
                "schema_version": RELEASE_IDENTITY_TEMPLATE_SCHEMA_VERSION,
            },
        },
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture: freeze tracked release template")
    source_commit = _git(repo, "rev-parse", "HEAD")
    return repo, template, source_commit


def _identity_inputs(repo: Path, template: Path, source_commit: str) -> dict[str, object]:
    return {
        "template_path": template,
        "source_commit": source_commit,
        "release_tag": derive_sha_tag("paper-matrix-v2-h600-s30", source_commit),
        "concept_doi": "10.5281/zenodo.99000001",
        "version_doi": "10.5281/zenodo.99000002",
        "repository_root": repo,
    }


def test_clean_candidate_generates_and_verifies_byte_identical_resolved_identity(
    tmp_path: Path,
) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    release_tag = derive_sha_tag("paper-matrix-v2-h600-s30", source_commit)
    output = repo / "output" / "release" / "release_identity.resolved.json"
    inputs = _identity_inputs(repo, template, source_commit)

    write_resolved_release_identity(output_path=output, **inputs)
    first_identity_bytes = output.read_bytes()
    first_metadata_bytes = (output.parent / "zenodo_metadata.resolved.json").read_bytes()
    second_identity = write_resolved_release_identity(output_path=output, **inputs)

    assert output.read_bytes() == first_identity_bytes
    assert (output.parent / "zenodo_metadata.resolved.json").read_bytes() == first_metadata_bytes
    verified = verify_resolved_release_identity(output, repository_root=repo)
    loaded = load_release_manifest(output, repository_root=repo)
    campaign = load_release_campaign_config(loaded, repository_root=repo)
    publication_binding = build_release_binding(loaded)
    assert verified.source_sha == source_commit
    assert loaded.source_sha == source_commit
    assert loaded.release_tag == release_tag
    assert campaign.release_tag == release_tag
    assert campaign.doi == "10.5281/zenodo.99000002"
    assert second_identity["resolved_manifest"]["source_sha"] == source_commit
    planner_identities = second_identity["resolved_manifest"]["planners"]["config_identities"]
    assert planner_identities[0] == {
        "key": "planner_00",
        "algo": "goal",
        "path": "planner.yaml",
        "sha256": _sha256(repo / "planner.yaml"),
    }
    assert len(planner_identities) == 14
    assert all(item["path"] is None for item in planner_identities[1:])
    assert publication_binding["source_tag"].endswith(release_tag)
    assert publication_binding["metadata_sha256"] == _sha256(
        output.parent / "zenodo_metadata.resolved.json"
    )
    assert second_identity["resolved_manifest"]["provenance"]["metadata_path"] == (
        "output/release/zenodo_metadata.resolved.json"
    )
    assert second_identity["resolved_manifest"]["canonical_campaign_config_sha256"] == _sha256(
        repo / "campaign.yaml"
    )
    assert second_identity["resolved_manifest"]["identity_resolution"] == {
        "schema_version": "benchmark-release-resolved-identity.v1",
        "template_path": template.name,
        "template_sha256": _sha256(template),
        "metadata_template_path": "zenodo_metadata.template.json",
        "metadata_template_sha256": _sha256(repo / "zenodo_metadata.template.json"),
    }
    assert _git(repo, "status", "--porcelain", "--untracked-files=normal") == ""


def test_generation_rejects_an_existing_tag_collision(tmp_path: Path) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    release_tag = derive_sha_tag("paper-matrix-v2-h600-s30", source_commit)
    _git(repo, "tag", release_tag, "HEAD^")

    with pytest.raises(ValueError, match="tag already exists"):
        write_resolved_release_identity(
            template_path=template,
            output_path=repo / "output" / "collision" / "release_identity.resolved.json",
            source_commit=source_commit,
            release_tag=release_tag,
            concept_doi="10.5281/zenodo.99000001",
            version_doi="10.5281/zenodo.99000002",
            repository_root=repo,
        )


def test_verification_accepts_existing_exact_published_tag(tmp_path: Path) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    inputs = _identity_inputs(repo, template, source_commit)
    output = repo / "output" / "published" / "release_identity.resolved.json"
    write_resolved_release_identity(output_path=output, **inputs)
    _git(repo, "tag", str(inputs["release_tag"]), source_commit)

    verified = verify_resolved_release_identity(output, repository_root=repo)

    assert verified.source_sha == source_commit
    with pytest.raises(ValueError, match="tag already exists"):
        write_resolved_release_identity(output_path=output, **inputs)


def test_generation_rejects_dirty_tree_and_wrong_commit(tmp_path: Path) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    inputs = _identity_inputs(repo, template, source_commit)
    output = repo / "output" / "dirty" / "release_identity.resolved.json"
    (repo / "dirty.txt").write_text("not frozen\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not clean"):
        write_resolved_release_identity(output_path=output, **inputs)

    (repo / "dirty.txt").unlink()
    inputs["source_commit"] = _git(repo, "rev-parse", "HEAD^")
    inputs["release_tag"] = derive_sha_tag("paper-matrix-v2-h600-s30", str(inputs["source_commit"]))
    with pytest.raises(ValueError, match="does not match source_commit"):
        write_resolved_release_identity(output_path=output, **inputs)

    inputs["source_commit"] = "f" * 40
    inputs["release_tag"] = derive_sha_tag("paper-matrix-v2-h600-s30", "f" * 40)
    with pytest.raises(ValueError, match="not reachable"):
        write_resolved_release_identity(output_path=output, **inputs)


@pytest.mark.parametrize(
    "release_tag",
    [
        "paper-matrix-v2-h600-s30-" + "b" * 40,
        "paper-matrix-v2-h600-s30-semantic",
        "paper-matrix-v2-h600-s30-{source}-{source}",
        " paper-matrix-v2-h600-s30-{source}",
    ],
)
def test_generation_rejects_noncanonical_or_colliding_tag_representations(
    tmp_path: Path, release_tag: str
) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    supplied = release_tag.format(source=source_commit)
    inputs = _identity_inputs(repo, template, source_commit)
    inputs["release_tag"] = supplied

    with pytest.raises(ValueError, match="canonical full-SHA suffix"):
        write_resolved_release_identity(
            output_path=repo / "output" / "tag" / "release_identity.resolved.json",
            **inputs,
        )


def test_verification_rejects_stale_identity_metadata_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    output = repo / "output" / "stale" / "release_identity.resolved.json"
    write_resolved_release_identity(
        output_path=output,
        **_identity_inputs(repo, template, source_commit),
    )
    original_identity = output.read_bytes()
    metadata = output.parent / "zenodo_metadata.resolved.json"
    original_metadata = metadata.read_bytes()

    payload = json.loads(original_identity)
    payload["resolved_manifest"]["matrix"]["horizon_steps"] = 599
    _write_canonical_json(output, payload)
    with pytest.raises(ValueError, match="stale or non-canonical"):
        verify_resolved_release_identity(output, repository_root=repo)

    output.write_bytes(original_identity + b"\n")
    with pytest.raises(ValueError, match="stale or non-canonical"):
        verify_resolved_release_identity(output, repository_root=repo)

    output.write_bytes(original_identity)
    metadata.write_bytes(original_metadata.replace(b"Future", b"Stale ", 1))
    with pytest.raises(ValueError, match="metadata is stale"):
        verify_resolved_release_identity(output, repository_root=repo)


def test_verification_rejects_post_freeze_mutation_and_alternate_checkout(
    tmp_path: Path,
) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    output = repo / "output" / "freeze" / "release_identity.resolved.json"
    write_resolved_release_identity(
        output_path=output,
        **_identity_inputs(repo, template, source_commit),
    )
    scenarios = repo / "scenarios.yaml"
    scenarios.write_text(scenarios.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not clean"):
        verify_resolved_release_identity(output, repository_root=repo)

    _git(repo, "restore", "scenarios.yaml")
    planner_config = repo / "planner.yaml"
    planner_config.write_text(
        planner_config.read_text(encoding="utf-8") + "# changed after freeze\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not clean"):
        verify_resolved_release_identity(output, repository_root=repo)

    _git(repo, "restore", "planner.yaml")
    _git(repo, "commit", "--allow-empty", "-qm", "fixture: alternate checkout")
    with pytest.raises(ValueError, match="does not match source_commit"):
        verify_resolved_release_identity(output, repository_root=repo)


def test_generation_rechecks_cleanliness_after_resolving_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    output = repo / "output" / "late-mutation" / "release_identity.resolved.json"
    original_build = release_protocol._build_resolved_release_identity

    def _mutating_build(**kwargs: Any) -> Any:
        result = original_build(**kwargs)
        scenarios = repo / "scenarios.yaml"
        scenarios.write_text(
            scenarios.read_text(encoding="utf-8") + "# mutated after resolution\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(release_protocol, "_build_resolved_release_identity", _mutating_build)

    with pytest.raises(ValueError, match="not clean"):
        write_resolved_release_identity(
            output_path=output,
            **_identity_inputs(repo, template, source_commit),
        )
    assert not output.exists()


def test_generation_rejects_symlinked_template_input(tmp_path: Path) -> None:
    repo, template, _ = _release_template_repository(tmp_path)
    linked_metadata = repo / "linked_metadata.json"
    linked_metadata.symlink_to(repo / "zenodo_metadata.template.json")
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    payload["publication"]["metadata_path"] = linked_metadata.name
    _write_yaml(template, payload)
    _git(repo, "add", "release.template.yaml", "linked_metadata.json")
    _git(repo, "commit", "-qm", "fixture: tracked symlink escape probe")
    source_commit = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="symlink"):
        write_resolved_release_identity(
            output_path=repo / "output" / "symlink" / "release_identity.resolved.json",
            **_identity_inputs(repo, template, source_commit),
        )


def test_generation_rejects_invalid_resolved_zenodo_metadata(tmp_path: Path) -> None:
    repo, template, _ = _release_template_repository(tmp_path)
    metadata_path = repo / "zenodo_metadata.template.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata["metadata"].pop("access_right")
    _write_canonical_json(metadata_path, metadata)
    template_payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    template_payload["publication"]["metadata_sha256"] = _sha256(metadata_path)
    _write_yaml(template, template_payload)
    _git(repo, "add", template.name, metadata_path.name)
    _git(repo, "commit", "-qm", "fixture: invalid Zenodo metadata")
    source_commit = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="access_right=open"):
        write_resolved_release_identity(
            output_path=repo / "output" / "metadata" / "release_identity.resolved.json",
            **_identity_inputs(repo, template, source_commit),
        )


def test_generation_requires_campaign_publication_identity_slots(tmp_path: Path) -> None:
    repo, template, _ = _release_template_repository(tmp_path)
    campaign_path = repo / "campaign.yaml"
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    campaign["release_tag"] = "stale-semantic-tag"
    _write_yaml(campaign_path, campaign)
    template_payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    template_payload["campaign_config_sha256"] = _sha256(campaign_path)
    _write_yaml(template, template_payload)
    _git(repo, "add", template.name, campaign_path.name)
    _git(repo, "commit", "-qm", "fixture: concrete campaign publication identity")
    source_commit = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="campaign release_tag must use"):
        write_resolved_release_identity(
            output_path=repo / "output" / "campaign" / "release_identity.resolved.json",
            **_identity_inputs(repo, template, source_commit),
        )


def test_generation_rejects_symlinked_or_escaped_output(tmp_path: Path) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    inputs = _identity_inputs(repo, template, source_commit)
    linked_output = repo / "output" / "linked-identity.json"
    linked_output.parent.mkdir(parents=True)
    linked_output.symlink_to(repo / "output" / "redirected-identity.json")

    with pytest.raises(ValueError, match="symlink"):
        write_resolved_release_identity(output_path=linked_output, **inputs)
    with pytest.raises(ValueError, match="contained"):
        write_resolved_release_identity(output_path=tmp_path / "escaped.json", **inputs)


def test_generation_requires_both_outputs_to_be_git_ignored(tmp_path: Path) -> None:
    repo, template, _ = _release_template_repository(tmp_path)
    (repo / ".gitignore").write_text("release_identity.resolved.json\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "fixture: ignore only the identity filename")
    source_commit = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="metadata output must be Git-ignored"):
        write_resolved_release_identity(
            output_path=repo / "release_identity.resolved.json",
            **_identity_inputs(repo, template, source_commit),
        )


def test_verification_validates_publication_coordinates_before_reproduction(
    tmp_path: Path,
) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    output = repo / "output" / "coordinates" / "release_identity.resolved.json"
    write_resolved_release_identity(
        output_path=output,
        **_identity_inputs(repo, template, source_commit),
    )
    payload = json.loads(output.read_bytes())
    payload["publication"]["version_doi"] = "10.5281/zenodo.not-a-reservation"
    _write_canonical_json(output, payload)

    with pytest.raises(ValueError, match="version_doi must be an exact reserved Zenodo DOI"):
        verify_resolved_release_identity(output, repository_root=repo)


def test_cold_checkout_reproduces_and_verifies_identity(tmp_path: Path) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    output = repo / "output" / "release" / "release_identity.resolved.json"
    write_resolved_release_identity(
        output_path=output,
        **_identity_inputs(repo, template, source_commit),
    )
    cold = tmp_path / "cold"
    _git(repo, "worktree", "add", "--detach", "-q", str(cold), source_commit)
    cold_output = cold / output.relative_to(repo)
    cold_output.parent.mkdir(parents=True)
    shutil.copy2(output, cold_output)
    shutil.copy2(output.parent / "zenodo_metadata.resolved.json", cold_output.parent)

    manifest = verify_resolved_release_identity(cold_output, repository_root=cold)
    original_manifest = load_release_manifest(output, repository_root=repo)
    cold_campaign = load_release_campaign_config(original_manifest, repository_root=cold)

    assert manifest.source_sha == source_commit
    assert manifest.identity_template_path == cold / template.name
    assert cold_campaign.scenario_matrix_path == cold / "scenarios.yaml"
    assert cold_output.read_bytes() == output.read_bytes()


def test_generate_and_verify_command_reports_exact_artifact_digests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    output = repo / "output" / "cli" / "release_identity.resolved.json"
    release_tag = derive_sha_tag("paper-matrix-v2-h600-s30", source_commit)

    assert (
        identity_cli.main(
            [
                "generate",
                "--template",
                str(template),
                "--output",
                str(output),
                "--source-commit",
                source_commit,
                "--release-tag",
                release_tag,
                "--concept-doi",
                "10.5281/zenodo.99000001",
                "--version-doi",
                "10.5281/zenodo.99000002",
                "--repository-root",
                str(repo),
            ]
        )
        == 0
    )
    generated = json.loads(capsys.readouterr().out)
    assert generated["status"] == "generated"
    assert generated["identity_sha256"] == _sha256(output)

    assert (
        identity_cli.main(
            [
                "verify",
                "--identity",
                str(output),
                "--repository-root",
                str(repo),
            ]
        )
        == 0
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["status"] == "verified"
    assert verified["source_commit"] == source_commit


def test_public_runner_preflight_consumes_the_verified_resolved_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    output = repo / "output" / "runner" / "release_identity.resolved.json"
    identity = write_resolved_release_identity(
        output_path=output,
        **_identity_inputs(repo, template, source_commit),
    )
    monkeypatch.setattr(release_protocol, "get_repository_root", lambda: repo)
    monkeypatch.setattr(run_benchmark_release, "get_repository_root", lambda: repo)
    monkeypatch.setattr(run_benchmark_release, "_current_source_commit", lambda: source_commit)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", lambda _cfg: None)
    monkeypatch.setattr(
        run_benchmark_release,
        "prepare_campaign_preflight",
        lambda *_args, **_kwargs: {
            "campaign_id": "fixture",
            "campaign_root": repo / "output" / "campaign",
            "validate_config_path": repo / "output" / "validate.json",
            "preview_scenarios_path": repo / "output" / "preview.json",
            "matrix_summary_json_path": repo / "output" / "matrix.json",
            "matrix_summary_csv_path": repo / "output" / "matrix.csv",
        },
    )

    exit_code = run_benchmark_release.main(["--manifest", str(output), "--mode", "preflight"])

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["manifest_validation"]["status"] == "valid"
    assert result["resolved_manifest"] == identity["resolved_manifest"]
    assert result["resolved_manifest"]["provenance"]["source_sha"] == source_commit


def test_doctor_and_acceptance_resolve_the_same_identity_bound_campaign(
    tmp_path: Path,
) -> None:
    repo, template, source_commit = _release_template_repository(tmp_path)
    output = repo / "output" / "consumers" / "release_identity.resolved.json"
    inputs = _identity_inputs(repo, template, source_commit)
    write_resolved_release_identity(output_path=output, **inputs)

    doctor_check, doctor_manifest, doctor_campaign = release_doctor._manifest_check(
        output,
        20160,
        repository_root=repo,
    )
    acceptance_campaign, blockers = release_acceptance._full_release_campaign_config(
        doctor_manifest,
        None,
        repo,
    )
    scenario_ids, seeds, axis_blockers = release_acceptance._resolve_expected_matrix_axes(
        doctor_manifest,
        acceptance_campaign,
        repo,
    )

    assert doctor_check.status == "pass"
    assert doctor_campaign.release_tag == inputs["release_tag"]
    assert doctor_campaign.doi == inputs["version_doi"]
    assert blockers == []
    assert axis_blockers == []
    assert len(scenario_ids) == 48
    assert seeds == tuple(range(111, 141))


def test_acceptance_resolves_axes_and_planners_from_resolved_manifest(
    tmp_path: Path,
) -> None:
    """Exercise the acceptance fallback that loads the verified campaign in-process."""
    # These imports are normally initialized while pytest is collecting modules.  Reloading
    # here keeps the release-facing import contract itself inside the measured test path.
    importlib.reload(release_acceptance)
    repo, template, source_commit = _release_template_repository(tmp_path)
    output = repo / "output" / "acceptance" / "release_identity.resolved.json"
    write_resolved_release_identity(
        output_path=output,
        **_identity_inputs(repo, template, source_commit),
    )
    manifest = load_release_manifest(output, repository_root=repo)

    scenario_ids, seeds, axis_blockers = release_acceptance._resolve_expected_matrix_axes(
        manifest,
        None,
        repo,
    )
    planner_candidates, planner_blockers = release_acceptance._full_release_planner_candidates(
        manifest,
        None,
        repo,
    )

    assert axis_blockers == []
    assert planner_blockers == []
    assert len(scenario_ids) == 48
    assert seeds == tuple(range(111, 141))
    assert len(release_acceptance._full_release_planner_items(planner_candidates)) == 14

    legacy_manifest = replace(manifest, resolved_identity_path=None)
    legacy_scenario_ids, legacy_seeds, legacy_axis_blockers = (
        release_acceptance._resolve_expected_matrix_axes(legacy_manifest, None, repo)
    )
    legacy_planners, legacy_planner_blockers = release_acceptance._full_release_planner_candidates(
        legacy_manifest,
        None,
        repo,
    )
    assert legacy_axis_blockers == []
    assert legacy_planner_blockers == []
    assert legacy_scenario_ids == scenario_ids
    assert legacy_seeds == seeds
    assert len(release_acceptance._full_release_planner_items(legacy_planners)) == 14
