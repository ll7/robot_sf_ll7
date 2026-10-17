#!/usr/bin/env python3
"""Recompute compact #10112 diagnostics from preserved custody, without episodes.

Raw data and bundles are external inputs, identified by SHA-256 in the proofs.
Only small, portable projections and review sidecars enter the evidence tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tarfile
from pathlib import Path

import yaml

from robot_sf.benchmark.artifact_publication import verify_publication_bundle_preflight
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.snqi.v2_binding import bind_acquired_anchors
from robot_sf.benchmark.snqi.v2_reports import _source_scenario_algorithms, score_episode
from robot_sf.benchmark.snqi.v2_spec import load_snqi_v2_spec
from robot_sf.evidence.writers import write_review_sidecar

SOURCE = "d56092ed9d4b442f9dfb99e31010a9f5a8666547"
NAMES = (
    "rehearsal-d56092ed-proof.json",
    "calibration-d56092ed-grid-proof.json",
    "actual-anchor-binding-d56092ed-preflight-cwd-fixed.json",
    "dev1003-d56092ed-proof.json",
)


def digest(path: Path) -> str:
    """Return a source file digest.

    Returns:
        SHA-256 of the preserved bytes.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_archive(bundle: Path, archive: Path):
    """Verify every archived member against the preserved bundle.

    Returns:
        Equivalence status and verified regular-file count.
    """
    expected = {str(p.relative_to(bundle)): digest(p) for p in bundle.rglob("*") if p.is_file()}
    observed = {}
    with tarfile.open(archive, "r:gz") as packed:
        for member in packed.getmembers():
            relative = Path(member.name).relative_to(bundle.name)
            assert not relative.is_absolute() and ".." not in relative.parts
            assert member.isdir() or member.isfile()
            if member.isfile():
                assert str(relative) not in observed
                with packed.extractfile(member) as stream:
                    observed[str(relative)] = hashlib.sha256(stream.read()).hexdigest()
    assert observed == expected
    return {"status": "pass", "files_byte_equal": len(observed)}


def check_grid(root: Path, seeds: set[int], expected: int, scenarios_per_arm: int = 48):
    """Check the complete unique producer grid and sidecar hashes.

    Returns:
        Source-bound scalar records and their original episode files.
    """
    records = []
    files = sorted(root.glob("runs/**/episodes.jsonl"))
    assert len(files) == 14, len(files)
    for path in files:
        sidecar = json.loads(path.with_name(path.name + ".provenance.json").read_bytes())
        assert sidecar["run"]["repo_commit"] == SOURCE
        matching = [
            entry for entry in sidecar["raw_artifacts"] if Path(entry["path"]).name == path.name
        ]
        assert len(matching) == 1 and matching[0]["sha256"] == digest(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        cells = {(row["scenario_id"], row["seed"]) for row in rows}
        assert len(rows) == scenarios_per_arm * len(seeds) == len(cells)
        assert {row["seed"] for row in rows} == seeds
        assert len({row["scenario_id"] for row in rows}) == scenarios_per_arm
        assert all(
            row["git_hash"] == SOURCE and row["result_provenance"]["repo_commit"] == SOURCE
            for row in rows
        )
        records.extend(
            (row, sidecar["campaign_identity"]["algorithm"], path.parent.name) for row in rows
        )
    assert len(records) == expected
    return records, files


def portable(payload):
    """Remove locator-only private paths while retaining hashes and source identities.

    Returns:
        A recursively projected JSON value.
    """
    if isinstance(payload, dict):
        return {key: portable(value) for key, value in payload.items() if key != "bundle_dir"}
    if isinstance(payload, list):
        return [portable(value) for value in payload]
    if isinstance(payload, str) and payload.startswith("/home/"):
        raise ValueError("unexpected private locator in public proof")
    return payload


def emit(output: Path, name: str, payload) -> None:
    """Write a portable compact proof and register its hash-bound review sidecar."""
    path = output / name
    path.write_text(json.dumps(portable(payload), indent=2, sort_keys=True) + "\n")
    write_review_sidecar(path)


def regenerate(args) -> None:
    """Verify producer bytes, recompute all scores and regenerate the four raw proofs."""
    repo, artifacts, smoke, output = (
        args.repository_root,
        args.custody_root,
        args.smoke_root,
        args.output_dir,
    )
    calibration, campaign = artifacts / "calibration", artifacts / "campaign"
    anchors = artifacts / "acquired-anchors.json"
    cal_rows, _ = check_grid(calibration, {1001, 1002}, 1344)
    rows, files = check_grid(campaign, {1001, 1002, 1003}, 2016)
    anchor_doc = json.loads(anchors.read_bytes())
    assert anchor_doc["calibration"]["source_commit"] == SOURCE
    assert anchor_doc["calibration"]["seeds"] == [1001, 1002]
    spec = load_snqi_v2_spec(
        repo / "configs/benchmarks/snqi_v2/weights.v2.0.json",
        anchors,
        repo / "configs/benchmarks/snqi_v2/family.v2.0.yaml",
    )
    manifest = json.loads((campaign / "campaign_manifest.json").read_bytes())
    routes = {
        f"{p['key']}__differential_drive": _source_scenario_algorithms(p, repo)
        for p in manifest["planners"]
    }
    values, routed = [], 0
    for row, algorithm, identity in rows:
        expected = routes[identity].get(row["scenario_id"], algorithm)
        assert row["algo"] == expected
        routed += expected != algorithm
        recomputed = score_episode(row, spec, expected_algorithm=expected)["metrics"]
        value = row["metrics"]["snqi_v2"]
        assert not isinstance(value, bool) and math.isfinite(value)
        assert math.isclose(value, recomputed["snqi_v2"], rel_tol=0, abs_tol=1e-12)
        assert row["metrics"]["snqi_v2_terms"] == recomputed["snqi_v2_terms"]
        assert row["metrics"]["snqi_v2_force_provenance"] == recomputed["snqi_v2_force_provenance"]
        values.append(value)
    result = json.loads((campaign / "release/release_result.json").read_bytes())
    assert result["release_eligible"] is False
    assert result["resolved_manifest"]["source_sha"] == SOURCE
    assert result["release_status"] == "development_rehearsal_passed"
    bundle = artifacts / Path(result["publication_bundle"]["bundle_dir"]).name
    archive = artifacts / Path(result["publication_bundle"]["archive_path"]).name
    preflight = verify_publication_bundle_preflight(bundle)
    assert (
        preflight["status"] == "pass" and not preflight["violations"] and not preflight["warnings"]
    )
    archive_check = verify_archive(bundle, archive)
    for path in files:
        assert digest(bundle / "payload" / path.relative_to(campaign)) == digest(path)
    emit(
        output,
        NAMES[0],
        {
            "source": SOURCE,
            "calibration_rows": len(cal_rows),
            "scored_rows": len(rows),
            "calibration_seeds": [1001, 1002],
            "rehearsal_seeds": [1001, 1002, 1003],
            "anchors_sha256": digest(anchors),
            "archive_sha256": digest(archive),
            "snqi_v2_min": min(values),
            "snqi_v2_max": max(values),
            "snqi_v2_recomputed_rows": len(values),
            "source_declared_routed_rows": routed,
            "bundle_raw_rows_byte_equal": True,
            "release_eligible": False,
            "publication_preflight": preflight,
            "archive_byte_equivalence": archive_check,
        },
    )
    emit(
        output,
        NAMES[1],
        {
            "source": SOURCE,
            "calibration_rows": len(cal_rows),
            "arms": anchor_doc["calibration"]["arms"],
            "scenarios_per_arm": 48,
            "seeds": [1001, 1002],
            "anchors_sha256": digest(anchors),
            "anchors": anchor_doc,
            "producer_hashes_byte_verified": True,
            "diagnostic_only": True,
        },
    )
    cfg = load_campaign_config(
        repo
        / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml",
        repository_root=repo,
    )
    for raw_root in (calibration, None):
        bound = bind_acquired_anchors(
            cfg,
            calibration_root=raw_root,
            anchors_path=anchors,
            source_commit=SOURCE,
            diagnostic=True,
        )
        assert bound.snqi_v2_spec.hashes["anchors"] == digest(anchors)
    emit(
        output,
        NAMES[2],
        {
            "source": SOURCE,
            "rehearsal_seeds": [1001, 1002, 1003],
            "anchors_sha256": digest(anchors),
            "upper_anchors": {
                term: entry["upper"] for term, entry in anchor_doc["anchors"].items()
            },
            "strict_raw_custody_attachment": "passed",
            "artifact_only_attachment": "passed",
            "diagnostic": True,
            "episodes_executed": 0,
            "scientific_admission": False,
        },
    )
    contract = json.loads((smoke / "dev1003-public-contract.json").read_bytes())
    smoke_result = json.loads((smoke / "campaign/release/release_result.json").read_bytes())
    smoke_config_path = smoke_result["resolved_manifest"]["canonical_campaign_config"]
    smoke_config_hash = digest(repo / smoke_config_path)
    assert contract["source_commit"] == SOURCE and contract["status"] == "admitted"
    assert contract["planner_arms"] == contract["episode_cells"] == 14
    assert contract["episodes_excluded"] == contract["fallback_or_degraded_rows"] == 0
    check_grid(smoke / "campaign", {1003}, 14, 1)
    assert digest(smoke / "campaign/release/release_result.json") == contract["result_sha256"]
    assert (
        digest(smoke / "runtime_smoke_checkpoint_staging_receipt.json")
        == contract["checkpoint_receipt_sha256"]
    )
    # Inventory captured at the acquisition handoff; extra copied logs are not producer claims.
    inventory = json.loads(args.smoke_inventory.read_bytes())
    for item in inventory:
        assert digest(smoke / item["path"]) == item["sha256"]
    # This is a diagnostic projection, not an active campaign receipt. Retain
    # provenance as notes: campaign_id triggers registry commit/config lookup.
    # The original receipt and producer/config bytes were verified above.
    contract["producer_commit_note"] = contract.pop("source_commit")
    contract["campaign_id_note"] = contract.pop("campaign_id")
    emit(
        output,
        NAMES[3],
        {
            "source": SOURCE,
            "actual_rows": 14,
            "actual_seeds": [1003],
            "public_contract": contract,
            "config_path_note": smoke_config_path,
            "config_sha256": smoke_config_hash,
            "raw_sidecars_byte_verified": True,
            "files_preserved": len(inventory),
            "scientific_admission": False,
        },
    )


def differences(old, new, prefix=""):
    """Compare the union of object fields, preserving absent-to-bound transitions.

    Returns:
        Complete leaf-field difference ledger.
    """
    if isinstance(old, dict) and isinstance(new, dict):
        return [
            row
            for key in sorted(old.keys() | new.keys())
            for row in differences(old.get(key), new.get(key), f"{prefix}.{key}" if prefix else key)
        ]
    return [] if old == new else [{"field": prefix, "old": old, "new": new}]


def identity_projection(args) -> None:
    """Regenerate main identity differences and the four doorway-specific transitions."""
    old = json.loads(args.identity_old.read_bytes())
    new = json.loads(args.identity_new.read_bytes())
    main = differences(old, new)
    assert len(main) == 18
    # Verify the four literal source hashes without changing the exact identity ledger.
    assets = new["resolved_manifest"]["metrics"]["snqi_v2_binding"]
    for asset in assets.values():
        source_bytes = subprocess.check_output(["git", "show", f"{SOURCE}:{asset['path']}"])
        assert hashlib.sha256(source_bytes).hexdigest() == asset["sha256"]
    emit(args.output_dir, "identity-differences.json", main)
    old_template = yaml.safe_load(args.doorway_old.read_bytes())
    new_template = yaml.safe_load(args.doorway_new.read_bytes())

    def doorway_view(template, config):
        metrics = template["metrics"]
        return {
            "canonical_campaign_config": template["canonical_campaign_config"],
            "canonical_campaign_name": config["name"],
            "snqi_weights": {
                "path": metrics.get("snqi_weights_path"),
                "sha256": metrics.get("snqi_weights_sha256"),
            },
            "snqi_baseline": {
                "path": metrics.get("snqi_baseline_path"),
                "sha256": metrics.get("snqi_baseline_sha256"),
            },
        }

    doorway = differences(
        doorway_view(old_template, yaml.safe_load(args.doorway_config_old.read_bytes())),
        doorway_view(new_template, yaml.safe_load(args.doorway_config_new.read_bytes())),
    )
    assert len(doorway) == 6  # two canonical fields plus two legacy path/hash pairs
    emit(
        args.output_dir,
        "doorway-identity-differences.json",
        {
            "producer_source": SOURCE,
            "main_difference_fields": 18,
            "doorway_additional_field_groups": 4,
            "doorway_additional_leaf_fields": doorway,
            "snqi_v2_binding": {
                name: {
                    **asset,
                    "location": "Tracked producer template; resolve relative to configs/benchmarks/releases at d56092ed.",
                }
                for name, asset in new_template["metrics"]["snqi_v2_binding"].items()
            },
            "claim_boundary": "Template/config transitions only; regenerate concrete doorway identity at named freeze.",
        },
    )


def main() -> None:
    """Parse external custody inputs; never execute a planner or environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    raw = sub.add_parser("raw")
    for name in ("repository-root", "custody-root", "smoke-root", "smoke-inventory"):
        raw.add_argument("--" + name, type=Path, required=True)
    identity = sub.add_parser("identity")
    for name in (
        "identity-old",
        "identity-new",
        "doorway-old",
        "doorway-new",
        "doorway-config-old",
        "doorway-config-new",
    ):
        identity.add_argument("--" + name, type=Path, required=True)
    project = sub.add_parser("project")
    project.add_argument("--input-dir", type=Path, required=True)
    project.add_argument("--names", nargs="+", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "raw":
        regenerate(args)
    elif args.mode == "identity":
        identity_projection(args)
    else:
        for name in args.names:
            if Path(name).name != name:
                raise ValueError("projection name must be a filename")
            content = (args.input_dir / name).read_text()
            if any(
                token in content for token in ("/home/", "imech", "auxme", "seed-policy-deviation")
            ):
                raise ValueError("private locator or obsolete seed framing in public projection")
            target = args.output_dir / name
            target.write_text(content)
            write_review_sidecar(target)


if __name__ == "__main__":
    main()
