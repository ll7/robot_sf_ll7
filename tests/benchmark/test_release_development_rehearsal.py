"""Development identities exercise real frozen inputs without release eligibility.

Test-value gate: protects opt-in development materialization; reverting the
overlay fails the CLI admission; sealed-source tests cover no development
identity; public CLI and committed real inputs need no production test seam.
"""
# robot-sf-test-lane: slow -- real-input source freezes and public identity materialization
# seed-holdout: synthetic-fixture begin

import hashlib
import json
from dataclasses import fields, replace
from datetime import UTC, datetime

import pytest

from robot_sf.benchmark import (
    release_acceptance,
    release_doctor,
    release_tag_identity,
    zenodo_publisher,
)
from robot_sf.benchmark import release_protocol as protocol
from robot_sf.benchmark import runtime_smoke_admission as smoke
from robot_sf.benchmark import spawn_preflight as spawn
from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.evidence.writers import write_json
from scripts.analysis import _distribution_admission
from scripts.tools import resolve_benchmark_release_identity as cli
from tests.benchmark.test_release_campaign_authority import (
    IDENTITY_TEMPLATE,
)
from tests.benchmark.test_release_campaign_authority import no_execution as _no_execution
from tests.benchmark.test_release_campaign_authority import sealed_repository as _sealed_repository
from tests.benchmark.test_sealed_source_pins import git, materialize, worker_stub

no_execution = _no_execution
sealed_repository = _sealed_repository


def generate(repo, seeds="1001,1002,1003"):
    """Use the public resolver with real source-bound D-083 bytes."""
    head = git(repo, "rev-parse", "HEAD")
    output = repo / "output/development/release_identity.resolved.json"
    code = cli.main(
        [
            "generate",
            "--development-rehearsal",
            "--development-seeds",
            seeds,
            "--template",
            str(repo / IDENTITY_TEMPLATE),
            "--output",
            str(output),
            "--source-commit",
            head,
            "--release-tag",
            f"development-rehearsal-{head}",
            "--concept-doi",
            "10.5281/zenodo.99000001",
            "--version-doi",
            "10.5281/zenodo.99000002",
            "--repository-root",
            str(repo),
        ]
    )
    return code, output


def test_public_development_identity_round_trip(sealed_repository):
    """Only seeds and diagnostic marker change; canonical scientific bytes remain."""
    repo = sealed_repository
    code, output = generate(repo)
    assert code == 0
    assert cli.main(["verify", "--identity", str(output), "--repository-root", str(repo)]) == 0
    identity = json.loads(output.read_text())
    manifest = protocol.load_release_manifest(output, repository_root=repo)
    assert identity["resolved_manifest"]["release_kind"] == "development_rehearsal"
    assert manifest.resolved_seeds == (1001, 1002, 1003)
    assert manifest.expected_episode_cells == 2016
    assert manifest.release_kind == "development_rehearsal"
    sealed = materialize(repo, "main")
    diagnostic_cfg = protocol.load_release_campaign_config(manifest, repository_root=repo)
    release_cfg = protocol.load_release_campaign_config(sealed, repository_root=repo)
    differences = {
        field.name
        for field in fields(diagnostic_cfg)
        if getattr(diagnostic_cfg, field.name) != getattr(release_cfg, field.name)
    }
    assert differences == {"seed_policy", "release_tag"}
    assert diagnostic_cfg.snqi_v2_spec is None  # D-083 excludes uncalibrated SNQI.
    scenarios = _load_campaign_scenarios(diagnostic_cfg, repository_root=repo)
    assert len(scenarios) == 48
    assert protocol._resolved_seed_inventory(scenarios) == [1001, 1002, 1003]
    assert set(protocol.resolve_release_horizon_budgets(manifest, diagnostic_cfg).values()) == {
        400,
        500,
        600,
        650,
        700,
    }


@pytest.mark.parametrize(
    "seeds",
    [
        "111",
        "140",
        "50036",
        "1031",
        "1000",
        "1001,1001",
        "50036,50140,50331,50403,50813,51339,51709,51767,52094,52175,52257,52671,52850,52971,53020,53198,53239,53636,53671,53779,55022,55379,55568,56170,56966,57077,57113,57494,57943,59019",
    ],
)
def test_public_rehearsal_refuses_non_development_inventory(sealed_repository, seeds, capsys):
    """Reject retired/sealed inventories before any identity or environment exists."""
    code, output = generate(sealed_repository, seeds)
    assert code == 2
    assert "only unique development seeds 1001-1030" in capsys.readouterr().out
    assert not output.exists()


def test_rehearsal_marker_cannot_be_stripped(sealed_repository):
    """Removing the marker or changing its digest-covered seed tuple invalidates verification."""
    code, output = generate(sealed_repository)
    assert code == 0
    payload = json.loads(output.read_text())
    payload["resolved_manifest"]["release_kind"] = "benchmark-data"
    write_json(output, payload)
    with pytest.raises(ValueError, match="stale or non-canonical"):
        protocol.verify_resolved_release_identity(output, repository_root=sealed_repository)


def test_development_guard_reaches_only_recording_workers(sealed_repository, monkeypatch):
    """Use the same spawn guard/worker path, with exact dev inventory, without execution."""
    code, output = generate(sealed_repository)
    assert code == 0
    manifest = protocol.load_release_manifest(output, repository_root=sealed_repository)
    reached = worker_stub(monkeypatch)
    report = spawn.run_manifest_preflight(manifest, workers=1, source_commit=manifest.source_sha)
    assert report["status"] == "valid", report
    assert reached == [[1001, 1002, 1003]] * 48
    reached.clear()
    for seeds in ((111, 112, 113), (50036, 50140, 50331)):
        mutated = replace(
            manifest,
            resolved_seeds=seeds,
            seed_policy={**manifest.seed_policy, "seeds": list(seeds)},
        )
        with pytest.raises(ValueError, match="development seeds|retired evaluation"):
            spawn.guard_manifest_execution(mutated, source_commit=manifest.source_sha)
    assert reached == []


@pytest.mark.parametrize(
    "admission", ["sealed", "full", "mint", "doi", "tag", "comparator", "runtime-smoke", "metadata"]
)
def test_rehearsal_cannot_acquire_release_status(sealed_repository, admission):
    """A verified identity is refused explicitly at every release authority boundary."""
    repo = sealed_repository
    code, output = generate(repo)
    assert code == 0
    manifest = protocol.load_release_manifest(output, repository_root=repo)
    if admission == "sealed":
        assert (
            "development rehearsal cannot be a sealed release"
            == protocol.sealed_seed_execution_problem(
                manifest, (50036, 50140), repository_root=repo
            )
        )
    elif admission == "full":
        report = release_acceptance.validate_full_benchmark_release_acceptance(
            repo / "output/missing", manifest=manifest
        )
        assert report["benchmark_success"] is False
        assert report["blockers"] == [
            "development rehearsal cannot satisfy full release acceptance"
        ]
    elif admission == "mint":
        report, _, _ = release_doctor._manifest_check(output, 20160, repository_root=repo)
        assert report.status == "fail"
        assert "development rehearsal cannot be minted" in report.summary
    elif admission == "doi":
        with pytest.raises(
            zenodo_publisher.ZenodoPublisherError, match="development rehearsal cannot"
        ):
            zenodo_publisher.build_release_binding(manifest)
    elif admission == "tag":
        assert release_tag_identity.check_canonical_source_tag(
            manifest.release_tag, manifest.source_sha
        ) == ["development rehearsal cannot become a release tag"]
        assert release_tag_identity.check_tag_source_consistency(
            manifest.release_tag, manifest.source_sha
        ) == ["development rehearsal cannot become a release tag"]
    elif admission == "runtime-smoke":
        from robot_sf.benchmark.runtime_smoke_admission import (
            RuntimeSmokeAdmissionError,
            validate_runtime_smoke_result,
        )

        result = repo / "output/smoke/release/release_result.json"
        result.parent.mkdir(parents=True)
        write_json(result, {"release_kind": "development_rehearsal"})
        with pytest.raises(
            RuntimeSmokeAdmissionError,
            match="development rehearsal cannot satisfy release runtime smoke",
        ):
            validate_runtime_smoke_result(
                result,
                repo_root=repo,
                expected_source_commit=manifest.source_sha,
                expected_planner_keys=manifest.planner_keys,
            )
    elif admission == "metadata":
        with pytest.raises(
            zenodo_publisher.ZenodoPublisherError, match="development rehearsal cannot reserve"
        ):
            zenodo_publisher.load_dataset_metadata(manifest.metadata_path)
        metadata = json.loads(manifest.metadata_path.read_text())["metadata"]
        with pytest.raises(
            zenodo_publisher.ZenodoPublisherError, match="development rehearsal cannot reserve"
        ):
            zenodo_publisher.reserve(None, metadata)
    else:
        with pytest.raises(
            ValueError, match="development rehearsal cannot enter comparator release mode"
        ):
            _distribution_admission.admit(
                repo / "missing-baseline.tar.gz",
                repo / "output/missing",
                successor_manifest=output,
                successor_manifest_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                successor_source_root=repo,
            )


def test_rehearsal_publication_contract_refuses_release(tmp_path):
    """An otherwise accepted publication fixture cannot promote a diagnostic bundle."""
    from robot_sf.benchmark.release_publication_contract import (
        validate_release_publication_contract,
    )
    from tests.benchmark.test_release_publication_contract import _build_fixture

    campaign, bundle = _build_fixture(tmp_path)
    result_path = campaign / "release/release_result.json"
    payload = json.loads(result_path.read_text())
    payload["release_kind"] = "development_rehearsal"
    write_json(result_path, payload)
    report = validate_release_publication_contract(campaign, bundle, expected_release_tag="v1")
    assert report["status"] == "blocked"
    assert report["blockers"] == ["development rehearsal cannot be published as a release"]


def test_shared_acceptance_counts_development_cells_without_promoting(sealed_repository):
    """Real D-083 axes are checked by the common validator; absent rows never pass."""
    code, output = generate(sealed_repository)
    assert code == 0
    manifest = protocol.load_release_manifest(output, repository_root=sealed_repository)
    cfg = protocol.load_release_campaign_config(manifest, repository_root=sealed_repository)
    report = release_acceptance.validate_development_rehearsal_acceptance(
        sealed_repository / "output/missing",
        manifest=manifest,
        campaign_config=cfg,
        source_repository_root=sealed_repository,
    )
    assert report["schema_version"] == "benchmark-development-rehearsal-acceptance.v1"
    assert report["expected_episode_cells"] == 2016
    assert report["expected_scenario_count"] == 48
    assert report["expected_seed_count"] == 3
    assert report["status"] == "invalid"
    assert report["benchmark_success"] is False


def test_shared_exporter_preserves_non_release_marker(tmp_path):
    """The real exporter signs the diagnostic payload and validator detects marker loss."""
    from robot_sf.benchmark.artifact_publication import (
        PublicationPreflightError,
        export_publication_bundle,
        verify_publication_bundle_preflight,
    )
    from tests.benchmark.test_artifact_publication import _make_run

    run = tmp_path / "run"
    _make_run(run, with_video=False)
    (run / "episodes/episodes.jsonl").unlink()
    episode_path = run / "runs/goal__differential_drive/episodes.jsonl"
    episode_path.parent.mkdir(parents=True)
    write_json(
        episode_path,
        {
            "episode_id": "ep-1",
            "event_ledger": {"software_commit": "abc123"},
        },
        indent=None,
    )
    (run / "release").mkdir()
    write_json(
        run / "release/release_result.json",
        {
            "benchmark_release": {"release_kind": "development_rehearsal"},
            "release_kind": "development_rehearsal",
            "release_eligible": False,
            "release_benchmark_success": False,
        },
    )
    bundle = export_publication_bundle(
        run,
        tmp_path / "publication",
        release_tag="development-rehearsal-" + "a" * 40,
        doi="10.5281/zenodo.99000002",
    )
    payload = json.loads(bundle.manifest_path.read_text())
    assert payload["release_kind"] == "development_rehearsal"
    assert payload["release_eligible"] is False
    report = verify_publication_bundle_preflight(
        bundle.bundle_dir, require_release_reconciliation=False
    )
    assert report["status"] == "pass"
    assert report["release_eligible"] is False
    payload.pop("release_kind")
    write_json(bundle.manifest_path, payload)
    with pytest.raises(PublicationPreflightError, match="development rehearsal marker differs"):
        verify_publication_bundle_preflight(bundle.bundle_dir, require_release_reconciliation=False)


@pytest.mark.parametrize("defect", [None, "digest", "release_flag", "incomplete", "wrong_seeds"])
def test_development_smoke_keeps_shared_admission_and_digest_checks(
    sealed_repository, monkeypatch, defect
):
    """A recording row validator exercises receipt binding without a simulation seam."""
    repo = sealed_repository
    code, identity = generate(repo, "1001" if defect != "wrong_seeds" else "1001,1002")
    assert code == 0
    manifest = protocol.load_release_manifest(identity, repository_root=repo)
    root = repo / "output/smoke"
    result_path = root / "release/release_result.json"
    result_path.parent.mkdir(parents=True)
    staging = repo / "output/staging.json"
    write_json(staging, {"status": "staged"})
    write_json(root / "run_meta.json", {"finished_at_utc": datetime.now(UTC).isoformat()})
    write_json(
        result_path,
        {
            "benchmark_release": {"manifest_path": identity.relative_to(repo).as_posix()},
            "release_kind": "development_rehearsal",
            "development_runtime_smoke": True,
            "diagnostic_success": True,
            "release_eligible": defect == "release_flag",
            "release_benchmark_success": False,
            "release_exit_code": 0,
            "campaign_id": "dev-smoke",
            "checkpoint_staging_receipt": {
                "path": staging.relative_to(repo).as_posix(),
                "sha256": "bad"
                if defect == "digest"
                else hashlib.sha256(staging.read_bytes()).hexdigest(),
            },
        },
    )
    calls = []

    def rows(campaign_root, **kwargs):
        calls.append((campaign_root, kwargs["manifest"].resolved_seeds))
        return {
            "status": "invalid" if defect == "incomplete" else "valid",
            "blockers": ["missing rows"] if defect == "incomplete" else [],
        }

    def checkpoints(cfg, path, **kwargs):
        calls.append((path, cfg.seed_policy.seeds, kwargs["campaign_config_path"]))

    monkeypatch.setattr(smoke, "validate_development_rehearsal_acceptance", rows)
    monkeypatch.setattr(smoke, "validate_checkpoint_staging_receipt", checkpoints)
    args = {
        "repo_root": repo,
        "expected_source_commit": manifest.source_sha,
        "expected_planner_keys": manifest.planner_keys,
        "development_rehearsal": True,
    }
    if defect:
        with pytest.raises(
            smoke.RuntimeSmokeAdmissionError,
            match="digest mismatch|admission failed|identity/source/roster mismatch",
        ):
            smoke.validate_runtime_smoke_result(result_path, **args)
    else:
        receipt = smoke.validate_runtime_smoke_result(result_path, **args)
        assert receipt["status"] == "admitted_diagnostic"
        assert receipt["release_eligible"] is False
        assert receipt["episode_cells"] == 672
        assert calls == [
            (root, (1001,)),
            (staging, (1001,), manifest.canonical_campaign_config_path),
        ]


def test_development_comparator_uses_same_pinned_runtime(sealed_repository, monkeypatch):
    """Diagnostic comparison changes the inventory passed to the existing runtime resolver."""
    repo = sealed_repository
    code, identity = generate(repo)
    assert code == 0
    manifest = protocol.load_release_manifest(identity, repository_root=repo)
    root = repo / "output/comparator"
    root.mkdir()
    write_json(root / "campaign_manifest.json", {"campaign_id": "rehearsal"})
    calls = []

    def pinned(*args, **kwargs):
        calls.append((args, kwargs))
        return "config-hash", "scenario-hash", {"arm": "runtime"}, {"arm": "scope"}, {("slot",)}

    monkeypatch.setattr(_distribution_admission.admission, "_runtime_successor_identity", pinned)
    digest = hashlib.sha256(identity.read_bytes()).hexdigest()
    verified = _distribution_admission._verified_development_successor(
        identity, digest, repo, root, {}
    )
    assert verified["source_commit"] == manifest.source_sha
    assert verified["expected_slots"] == {("slot",)}
    assert calls[0][0][:2] == (repo, manifest.source_sha)
    assert calls[0][1]["development_rehearsal_seeds"] == (1001, 1002, 1003)
    assert calls[0][1]["publication_identity"] == {
        "release_tag": manifest.release_tag,
        "doi": manifest.doi,
    }


# seed-holdout: synthetic-fixture end
