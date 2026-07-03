"""Tests for the external data setup assistant."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from scripts.tools import manage_external_data

if TYPE_CHECKING:
    from pathlib import Path


def _init_git_repo(path: Path, *, gitignore: str = "") -> None:
    """Create a small git repo for git-ignore staging checks."""
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    if gitignore:
        (path / ".gitignore").write_text(gitignore, encoding="utf-8")


def _write_sdd_fixture(path: Path) -> None:
    """Create a minimal SDD annotation fixture."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "annotations.txt").write_text(
        "1 0 0 10 10 0 0 0 0 Pedestrian\n",
        encoding="utf-8",
    )


def _write_socnavbench_eth_fixture(path: Path) -> None:
    """Create a minimal SocNavBench ETH layout fixture for metadata-only tests."""
    mesh_dir = path / "sd3dis" / "stanford_building_parser_dataset" / "mesh" / "ETH"
    traversible_dir = path / "sd3dis" / "stanford_building_parser_dataset" / "traversibles" / "ETH"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    traversible_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / "mesh.obj").write_text("# synthetic fixture, not official data\n", encoding="utf-8")
    (traversible_dir / "data.pkl").write_bytes(b"synthetic fixture, not official data")


def _write_atc_fixture(path: Path) -> None:
    """Create minimal ATC-shaped fixture; not official data."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "atc-20121024.csv").write_text(
        "time,person_id,x,y,z,velocity,angle,facing_angle\n0,1,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    (path / "README.md").write_text("Synthetic fixture for registry tests.\n", encoding="utf-8")


def _write_ind_fixture(path: Path) -> None:
    """Create minimal inD-shaped fixture; not official data."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "00_tracks.csv").write_text(
        "trackId,frame,xCenter,yCenter\n1,0,0,0\n", encoding="utf-8"
    )
    (path / "TERMS.txt").write_text("Synthetic terms placeholder for tests.\n", encoding="utf-8")


def test_registry_covers_initial_required_asset_groups() -> None:
    """The first slice should cover SDD, SocNavBench, and AMV provenance assets."""
    asset_ids = {asset.asset_id for asset in manage_external_data.list_assets()}

    assert "sdd" in asset_ids
    assert "socnavbench-s3dis-eth" in asset_ids
    assert "socnavbench-control" in asset_ids
    assert "amv-calibration" in asset_ids


def test_registry_covers_alyassi_table_7_asset_slice() -> None:
    """Issue #4224 registers the Alyassi Table 7 dataset candidates as manual assets."""
    expected = {
        "atc-pedestrian",
        "eth-ucy-trajectories",
        "ind-crossings",
        "crowdbot",
        "scand-demos",
    }
    assets = {asset.asset_id: asset for asset in manage_external_data.list_assets()}

    assert expected <= set(assets)
    for asset_id in expected:
        asset = assets[asset_id]
        assert asset.source_url
        assert asset.license_url
        assert asset.license_note
        assert asset.access_note
        assert asset.shared_root_subpath.as_posix()
        assert asset.related_issues
        assert asset.auto_download_allowed is False
        assert asset.expected_tree_sha256 is None
        assert asset.expected_tree_sha256_status == "pending_first_staging"


def test_missing_license_gated_asset_fails_closed(tmp_path: Path) -> None:
    """Missing gated data should report why without attempting fallback."""
    report = manage_external_data.check_asset("sdd", source_path=tmp_path / "missing")

    assert report["ok"] is False
    assert report["status"] == "missing"
    assert report["availability"] == {
        "schema": "robot_sf_external_data_availability.v1",
        "state": "missing",
        "mode": manage_external_data.SDD_MODE_PROXY,
        "dataset_backed": False,
        "validated": False,
        "proxy_only": True,
    }
    assert report["auto_download_allowed"] is False
    assert "official acquisition" in report["action"]


def test_unset_external_data_root_preserves_repo_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the shared-root env var, asset paths remain repo-local."""
    monkeypatch.delenv(manage_external_data.EXTERNAL_DATA_ROOT_ENV, raising=False)
    asset = manage_external_data._get_asset("sdd")

    assert manage_external_data.external_data_root() is None
    assert manage_external_data.resolve_asset_local_path(asset) == asset.expected_local_path


def test_external_data_root_overrides_asset_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shared data root should replace repo-local default paths for all registered assets."""
    shared_root = tmp_path / "robot_sf_external_data"
    monkeypatch.setenv(manage_external_data.EXTERNAL_DATA_ROOT_ENV, str(shared_root))

    assert manage_external_data.resolve_asset_local_path_by_id("sdd") == shared_root / "sdd"
    assert (
        manage_external_data.resolve_asset_local_path_by_id("socnavbench-control")
        == shared_root / "socnavbench"
    )
    assert (
        manage_external_data.resolve_asset_local_path_by_id("socnavbench-s3dis-eth")
        == shared_root / "socnavbench"
    )
    assert (
        manage_external_data.resolve_asset_local_path_by_id("amv-calibration")
        == shared_root / "amv_calibration"
    )
    assert (
        manage_external_data.resolve_asset_local_path_by_id("atc-pedestrian")
        == shared_root / "atc_pedestrian"
    )
    assert (
        manage_external_data.resolve_asset_local_path_by_id("eth-ucy-trajectories")
        == shared_root / "eth_ucy_trajectories"
    )
    assert (
        manage_external_data.resolve_asset_local_path_by_id("ind-crossings")
        == shared_root / "ind_crossings"
    )
    assert (
        manage_external_data.resolve_asset_local_path_by_id("crowdbot") == shared_root / "crowdbot"
    )
    assert (
        manage_external_data.resolve_asset_local_path_by_id("scand-demos")
        == shared_root / "scand_demos"
    )


def test_check_uses_external_data_root_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default checks should validate the shared-root copy when configured."""
    shared_root = tmp_path / "robot_sf_external_data"
    _write_sdd_fixture(shared_root / "sdd")
    monkeypatch.setenv(manage_external_data.EXTERNAL_DATA_ROOT_ENV, str(shared_root))

    report = manage_external_data.check_asset("sdd")

    assert report["ok"] is True
    assert report["availability"]["state"] == "staged"
    assert report["availability"]["mode"] == manage_external_data.SDD_MODE_PROXY
    assert report["availability"]["proxy_only"] is True
    assert report["source_path"] == str((shared_root / "sdd").resolve())
    assert report["expected_local_path"] == str(shared_root.resolve() / "sdd")
    assert report["default_local_path"].endswith("output/external_data/sdd")
    assert report["external_data_root"] == str(shared_root.resolve())


def test_stage_defaults_to_external_data_root_when_source_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage subcommand path should be worktree-portable without per-worktree symlinks."""
    shared_root = tmp_path / "robot_sf_external_data"
    _write_sdd_fixture(shared_root / "sdd")
    manifest_path = tmp_path / "sdd.provenance.json"
    monkeypatch.setenv(manage_external_data.EXTERNAL_DATA_ROOT_ENV, str(shared_root))

    manifest = manage_external_data.stage_asset("sdd", manifest_out=manifest_path)

    assert manifest_path.is_file()
    assert manifest["local_path"] == str((shared_root / "sdd").resolve())
    assert manifest["availability"]["state"] == "validated"
    assert manifest["availability"]["mode"] == manage_external_data.SDD_MODE_PROXY
    assert manifest["availability"]["validated"] is True
    assert manifest["availability"]["dataset_backed"] is False
    assert manifest["validation_command"].endswith(
        f"check sdd --source {(shared_root / 'sdd').resolve()}"
    )


def test_cli_list_and_check_report_external_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI JSON output should expose the shared-root path for routing/debugging."""
    shared_root = tmp_path / "robot_sf_external_data"
    _write_sdd_fixture(shared_root / "sdd")
    monkeypatch.setenv(manage_external_data.EXTERNAL_DATA_ROOT_ENV, str(shared_root))

    list_result = subprocess.run(
        [sys.executable, "scripts/tools/manage_external_data.py", "--json", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    list_payload = json.loads(list_result.stdout)
    sdd_entry = next(entry for entry in list_payload if entry["asset_id"] == "sdd")
    assert sdd_entry["expected_local_path"] == str(shared_root.resolve() / "sdd")
    assert sdd_entry["availability"]["state"] == "staged"

    check_result = subprocess.run(
        [sys.executable, "scripts/tools/manage_external_data.py", "--json", "check", "sdd"],
        check=True,
        capture_output=True,
        text=True,
    )
    check_payload = json.loads(check_result.stdout)
    assert check_payload["ok"] is True
    assert check_payload["availability"]["state"] == "staged"
    assert check_payload["source_path"] == str((shared_root / "sdd").resolve())


def test_cli_list_and_explain_include_alyassi_metadata() -> None:
    """List/explain JSON expose license/access/checksum-pending metadata."""
    list_result = subprocess.run(
        [sys.executable, "scripts/tools/manage_external_data.py", "--json", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    list_payload = json.loads(list_result.stdout)
    atc_entry = next(entry for entry in list_payload if entry["asset_id"] == "atc-pedestrian")

    assert atc_entry["license_url"] == "https://dil.atr.jp/crest2010_HRI/ATC_dataset/"
    assert atc_entry["expected_tree_sha256"] is None
    assert atc_entry["expected_tree_sha256_status"] == "pending_first_staging"
    assert atc_entry["auto_download_allowed"] is False

    explain_result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/manage_external_data.py",
            "--json",
            "explain",
            "scand-demos",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    explain_payload = json.loads(explain_result.stdout)

    assert explain_payload["asset_id"] == "scand-demos"
    assert explain_payload["license_url"].startswith("https://dataverse.tdl.org/")
    assert explain_payload["expected_tree_sha256_status"] == "pending_first_staging"
    assert explain_payload["required_paths"]


def test_download_for_gated_asset_is_rejected() -> None:
    """Download must fail closed when redistribution/download terms are not encoded."""
    with pytest.raises(manage_external_data.ExternalDataError, match="license-gated"):
        manage_external_data.download_asset("sdd")


def test_alyassi_table_7_downloads_are_rejected() -> None:
    """Issue #4224 assets are registry/manual-staging only, not auto-downloadable."""
    for asset_id in (
        "atc-pedestrian",
        "eth-ucy-trajectories",
        "ind-crossings",
        "crowdbot",
        "scand-demos",
    ):
        with pytest.raises(manage_external_data.ExternalDataError, match="license-gated"):
            manage_external_data.download_asset(asset_id)


def test_alyassi_table_7_missing_check_fails_closed(tmp_path: Path) -> None:
    """Missing local Table 7 assets report manual acquisition action."""
    report = manage_external_data.check_asset("atc-pedestrian", source_path=tmp_path / "missing")

    assert report["ok"] is False
    assert report["status"] == "missing"
    assert report["availability"]["state"] == "missing"
    assert report["expected_tree_sha256"] is None
    assert report["expected_tree_sha256_status"] == "pending_first_staging"
    assert "official acquisition" in report["action"]


def test_stage_writes_manifest_for_alyassi_direct_style_asset(tmp_path: Path) -> None:
    """ATC fixture staging records checksum metadata without raw content leakage."""
    _init_git_repo(tmp_path, gitignore="external/atc/\n")
    source_root = tmp_path / "external" / "atc"
    _write_atc_fixture(source_root)
    manifest_path = tmp_path / "manifests" / "atc.provenance.json"

    manifest = manage_external_data.stage_asset(
        "atc-pedestrian",
        source_path=source_root,
        manifest_out=manifest_path,
        repo_root=tmp_path,
    )

    assert manifest_path.is_file()
    assert manifest["asset_id"] == "atc-pedestrian"
    assert manifest["license_url"]
    assert manifest["tree_sha256"]
    assert manifest["expected_tree_sha256"] is None
    assert manifest["expected_tree_sha256_status"] == "pending_first_staging"
    assert manifest["sample_files"]
    assert all("0,1,0,0" not in json.dumps(sample) for sample in manifest["sample_files"])
    assert manage_external_data.check_provenance_manifest("atc-pedestrian", manifest_path)["ok"]


def test_stage_writes_manifest_for_alyassi_byo_asset(tmp_path: Path) -> None:
    """BYO/research-request assets still get local provenance when staged."""
    _init_git_repo(tmp_path, gitignore="external/ind/\n")
    source_root = tmp_path / "external" / "ind"
    _write_ind_fixture(source_root)
    manifest_path = tmp_path / "manifests" / "ind.provenance.json"

    manifest = manage_external_data.stage_asset(
        "ind-crossings",
        source_path=source_root,
        manifest_out=manifest_path,
        repo_root=tmp_path,
    )

    assert manifest["asset_id"] == "ind-crossings"
    assert manifest["license_url"] == "https://levelxdata.com/ind-dataset/"
    assert manifest["matched_required_paths"] == ["00_tracks.csv", "TERMS.txt"]
    assert manage_external_data.check_provenance_manifest("ind-crossings", manifest_path)["ok"]


def test_stage_rejects_unignored_repo_local_raw_data(tmp_path: Path) -> None:
    """Repo-local raw files must be gitignored before a manifest can bless them."""
    _init_git_repo(tmp_path)
    source_root = tmp_path / "external" / "sdd"
    source_root.mkdir(parents=True)
    (source_root / "annotations.txt").write_text(
        "1 0 0 10 10 0 0 0 0 Pedestrian\n",
        encoding="utf-8",
    )

    with pytest.raises(manage_external_data.ExternalDataError, match="not covered by gitignore"):
        manage_external_data.stage_asset(
            "sdd",
            source_path=source_root,
            manifest_out=tmp_path / "manifest.json",
            repo_root=tmp_path,
        )


def test_stage_writes_small_manifest_for_ignored_sdd_path(tmp_path: Path) -> None:
    """Manual staging should validate, checksum, and write compact provenance."""
    _init_git_repo(tmp_path, gitignore="external/sdd/\n")
    source_root = tmp_path / "external" / "sdd"
    source_root.mkdir(parents=True)
    (source_root / "annotations.txt").write_text(
        "1 0 0 10 10 0 0 0 0 Pedestrian\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifests" / "sdd.provenance.json"

    manifest = manage_external_data.stage_asset(
        "sdd",
        source_path=source_root,
        manifest_out=manifest_path,
        repo_root=tmp_path,
    )

    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload == manifest
    assert payload["asset_id"] == "sdd"
    assert payload["file_count"] == 1
    assert payload["total_size_bytes"] > 0
    assert payload["tree_sha256"]
    assert payload["checksum_policy"].startswith("aggregate_tree_sha256")
    assert payload["validation_command"].endswith(f"check sdd --source {source_root.resolve()}")
    assert payload["sample_files"][0]["path"] == "annotations.txt"
    assert "Pedestrian" not in json.dumps(payload)


def test_socnavbench_eth_provenance_check_accepts_staged_manifest(tmp_path: Path) -> None:
    """ETH provenance readiness requires source, license, checksum, and path metadata."""
    _init_git_repo(tmp_path, gitignore="external/socnavbench/\n")
    source_root = tmp_path / "external" / "socnavbench"
    _write_socnavbench_eth_fixture(source_root)
    manifest_path = tmp_path / "manifests" / "socnavbench-s3dis-eth.provenance.json"

    manage_external_data.stage_asset(
        "socnavbench-s3dis-eth",
        source_path=source_root,
        manifest_out=manifest_path,
        repo_root=tmp_path,
    )
    report = manage_external_data.check_provenance_manifest("socnavbench-s3dis-eth", manifest_path)

    assert report["ok"] is True
    assert report["status"] == "ready"
    assert report["missing_metadata"] == []
    assert report["tree_sha256"]


@pytest.mark.parametrize(
    ("field", "expected_missing"),
    [
        ("source_url", "source_url"),
        ("license_note", "license_note"),
        ("tree_sha256", "tree_sha256"),
        ("sample_files", "sample_files[].sha256"),
    ],
)
def test_socnavbench_eth_provenance_check_fails_missing_metadata(
    tmp_path: Path, field: str, expected_missing: str
) -> None:
    """ETH provenance readiness fails closed for missing URI/license/checksum metadata."""
    manifest_path = tmp_path / "socnavbench-s3dis-eth.provenance.json"
    manifest = {
        "schema": "robot_sf_external_data_manifest.v1",
        "asset_id": "socnavbench-s3dis-eth",
        "source_url": "https://github.com/CMU-TBD/SocNavBench",
        "license_note": "External licensed data not redistributed by Robot SF.",
        "tree_sha256": "0" * 64,
        "sample_files": [{"path": "sd3dis/example", "sha256": "1" * 64}],
        "matched_required_paths": [
            "sd3dis/stanford_building_parser_dataset/mesh/ETH",
            "sd3dis/stanford_building_parser_dataset/traversibles/ETH/data.pkl",
        ],
    }
    manifest.pop(field)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = manage_external_data.check_provenance_manifest("socnavbench-s3dis-eth", manifest_path)

    assert report["ok"] is False
    assert report["status"] == "incomplete_metadata"
    assert expected_missing in report["missing_metadata"]


def test_provenance_check_fails_closed_on_non_dict_json(tmp_path: Path) -> None:
    """A manifest whose JSON root is not an object must fail closed, not crash."""
    manifest_path = tmp_path / "socnavbench-s3dis-eth.provenance.json"
    manifest_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

    report = manage_external_data.check_provenance_manifest("socnavbench-s3dis-eth", manifest_path)

    assert report["ok"] is False
    assert report["status"] == "invalid_json"


def test_provenance_check_rejects_incomplete_path_coverage(tmp_path: Path) -> None:
    """A non-empty matched-path list that misses a required group must not pass."""
    manifest_path = tmp_path / "socnavbench-s3dis-eth.provenance.json"
    manifest = {
        "schema": "robot_sf_external_data_manifest.v1",
        "asset_id": "socnavbench-s3dis-eth",
        "source_url": "https://github.com/CMU-TBD/SocNavBench",
        "license_note": "External licensed data not redistributed by Robot SF.",
        "tree_sha256": "0" * 64,
        "sample_files": [{"path": "sd3dis/example", "sha256": "1" * 64}],
        # Only the traversible group is present; the required ETH mesh group is missing.
        "matched_required_paths": [
            "sd3dis/stanford_building_parser_dataset/traversibles/ETH/data.pkl"
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = manage_external_data.check_provenance_manifest("socnavbench-s3dis-eth", manifest_path)

    assert report["ok"] is False
    assert report["status"] == "incomplete_metadata"
    assert "matched_required_paths" in report["missing_metadata"]


def test_provenance_check_rejects_unrelated_matched_paths(tmp_path: Path) -> None:
    """An unrelated non-empty matched-path list must not satisfy the readiness gate."""
    manifest_path = tmp_path / "socnavbench-s3dis-eth.provenance.json"
    manifest = {
        "schema": "robot_sf_external_data_manifest.v1",
        "asset_id": "socnavbench-s3dis-eth",
        "source_url": "https://github.com/CMU-TBD/SocNavBench",
        "license_note": "External licensed data not redistributed by Robot SF.",
        "tree_sha256": "0" * 64,
        "sample_files": [{"path": "sd3dis/example", "sha256": "1" * 64}],
        "matched_required_paths": ["totally/unrelated/path"],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = manage_external_data.check_provenance_manifest("socnavbench-s3dis-eth", manifest_path)

    assert report["ok"] is False
    assert report["status"] == "incomplete_metadata"
    assert "matched_required_paths" in report["missing_metadata"]


def test_cli_provenance_check_reports_missing_socnavbench_eth_metadata(tmp_path: Path) -> None:
    """CLI provenance-check reports incomplete ETH metadata with nonzero exit."""
    manifest_path = tmp_path / "socnavbench-s3dis-eth.provenance.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "robot_sf_external_data_manifest.v1",
                "asset_id": "socnavbench-s3dis-eth",
                "license_note": "External licensed data not redistributed by Robot SF.",
                "matched_required_paths": [
                    "sd3dis/stanford_building_parser_dataset/traversibles/ETH/data.pkl"
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/manage_external_data.py",
            "--json",
            "provenance-check",
            "socnavbench-s3dis-eth",
            "--manifest",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "incomplete_metadata"
    assert "source_url" in payload["missing_metadata"]
    assert "tree_sha256" in payload["missing_metadata"]


def test_socnavbench_control_check_accepts_wayptnav_layout(tmp_path: Path) -> None:
    """Control-pipeline assets should validate from the expected wayptnav directory."""
    (tmp_path / "wayptnav_data").mkdir()
    (tmp_path / "wayptnav_data" / "README.md").write_text("local fixture\n", encoding="utf-8")

    report = manage_external_data.check_asset("socnavbench-control", source_path=tmp_path)

    assert report["ok"] is True
    assert report["matched_required_paths"] == ["wayptnav_data"]


def test_socnavbench_s3dis_rejects_empty_required_mesh_directory(tmp_path: Path) -> None:
    """Required asset directories should not pass when only the directory shell exists."""
    mesh_dir = tmp_path / "sd3dis" / "stanford_building_parser_dataset" / "mesh" / "ETH"
    mesh_dir.mkdir(parents=True)
    traversible = (
        tmp_path
        / "sd3dis"
        / "stanford_building_parser_dataset"
        / "traversibles"
        / "ETH"
        / "data.pkl"
    )
    traversible.parent.mkdir(parents=True)
    traversible.write_bytes(b"fixture")

    report = manage_external_data.check_asset("socnavbench-s3dis-eth", source_path=tmp_path)

    assert report["ok"] is False
    assert report["status"] == "incomplete"
    assert "sd3dis/stanford_building_parser_dataset/mesh/ETH" in report["missing_required_paths"]


def test_amv_calibration_check_accepts_one_source_format(tmp_path: Path) -> None:
    """AMV calibration provenance can be represented by one accepted local source file."""
    (tmp_path / "accepted_source.json").write_text(
        '{"source_class": "official_spec", "note": "fixture only"}\n',
        encoding="utf-8",
    )

    report = manage_external_data.check_asset("amv-calibration", source_path=tmp_path)

    assert report["ok"] is True
    assert report["matched_required_paths"] == ["accepted_source.json"]


def test_amv_calibration_check_reports_source_alternatives(tmp_path: Path) -> None:
    """Missing AMV calibration input should point to the source alternative group."""
    report = manage_external_data.check_asset("amv-calibration", source_path=tmp_path)

    assert report["ok"] is False
    assert report["status"] == "incomplete"
    assert report["missing_required_paths"] == [
        "source: one of **/*.json, **/*.yaml, **/*.yml, **/*.csv, **/*.pdf"
    ]
