"""Distribution admission built from the paired comparator's pinned validators."""

import hashlib
import json
from pathlib import Path

from robot_sf.benchmark.metric_definitions import require_uniform_metric_schema
from scripts.analysis import compare_release_0_0_7_to_0_0_8 as admission


def admit(  # noqa: C901, PLR0912 - independent source, schema and coverage gates
    baseline_bundle: Path,
    successor_root: Path,
    *,
    successor_manifest: Path,
    successor_manifest_sha256: str,
    successor_source_root: Path,
    diagnostic_partial: bool = False,
    snqi_v2_anchors: Path | None = None,
) -> tuple[dict, dict, dict]:
    """Read verified release rows; diagnostics relax coverage only, never binding."""
    successor_payload = json.loads(successor_manifest.read_text())
    rehearsal = successor_payload.get("development_rehearsal") is not None
    if rehearsal and not diagnostic_partial:
        raise ValueError("development rehearsal cannot enter comparator release mode")
    digest = admission._verify_sha256(
        baseline_bundle, admission.BASELINE_SHA256, label="0.0.7 bundle SHA-256"
    )
    source = admission._archive_source_commit(baseline_bundle)
    if source != admission.BASELINE_SOURCE:
        raise ValueError("0.0.7 bundle source mismatch")
    if admission._bundle_campaign_id(baseline_bundle) != admission.BASELINE_CAMPAIGN:
        raise ValueError("0.0.7 campaign mismatch")
    scenario = admission._bundle_scenario_identity(baseline_bundle)
    if scenario != {
        "manifest": admission.EXPECTED_SUCCESSOR_SCENARIO_MANIFEST,
        "sha256": admission.EXPECTED_SUCCESSOR_SCENARIO_MANIFEST_SHA256,
    }:
        raise ValueError("0.0.7 scenario identity mismatch")
    old = admission._bundle_rows(baseline_bundle)
    new, duplicates, duplicate_rows = admission._root_rows(successor_root)
    if rehearsal:
        verified = _verified_development_successor(
            successor_manifest,
            successor_manifest_sha256,
            successor_source_root,
            successor_root,
            new,
        )
    else:
        verified = admission._verified_successor_manifest(
            successor_manifest,
            successor_manifest_sha256,
            successor_source_root,
            new,
            snqi_v2_anchors=snqi_v2_anchors,
        )
    identity = admission._root_identity(successor_root, verified)
    expected = verified["expected_slots"]
    if duplicates or duplicate_rows or set(new) - expected:
        raise ValueError("duplicate or extra successor slots")
    for slot, row in old.items():
        if row["_source_commit"] != source:
            raise ValueError("0.0.7 row source mismatch")
    for slot, row in new.items():
        if row["_source_commit"] != identity["source_commit"]:
            raise ValueError("0.0.8 row source differs from campaign manifest")
        admission._validate_row_runner_hashes(slot, row)
        admission._validate_successor_row(
            slot, row, verified["runtime_rows"], identity["source_commit"]
        )
    schemas = [require_uniform_metric_schema(rows.values()) for rows in (old, new)]
    if schemas != ["robot-sf-metrics.v1", "robot-sf-metrics.v2"]:
        raise ValueError("release schemas must be v1 for 0.0.7 and v2 for 0.0.8")
    from scripts.analysis.compare_release_distributions import check_slots

    check_slots(old, release="0.0.7", diagnostic_partial=diagnostic_partial)
    check_slots(new, release="0.0.8", diagnostic_partial=diagnostic_partial)
    if not diagnostic_partial and set(new) != expected:
        raise ValueError("missing successor slots")
    failures = admission._root_failure_slots(successor_root)
    if diagnostic_partial:
        # Classify every observed probe; coverage is explicitly unadmitted.
        from robot_sf.benchmark.infeasible_probe_safe_failure import (
            PROBE_SCENARIO_IDS,
            classify_probe_slots,
        )

        probes = {slot for slot in new if slot[2] in PROBE_SCENARIO_IDS}
        probe = (
            classify_probe_slots(probes, {slot: new[slot] for slot in probes}, failures)
            if probes
            else None
        )
        if probe is not None and probe["status"] == "fail_admission":
            raise ValueError("diagnostic probe rows fail safe-failure admission")
    else:
        probe = admission._probe_gate(new, expected, duplicates, failures)
        if probe is None:
            raise ValueError("release requires the separate probe gate")
    map_hashes = {}
    for release_rows, commit in ((old, source), (new, identity["source_commit"])):
        for row in release_rows.values():
            name = row["_definition"]["scenario"].get("map_file")
            if not isinstance(name, str) or not name:
                raise ValueError("scenario lacks map_file for definition fingerprint")
            cache_key = commit, name
            if cache_key not in map_hashes:
                map_hashes[cache_key] = hashlib.sha256(
                    admission._source_bytes(successor_source_root, commit, name)
                ).hexdigest()
            row["_map_sha256"] = map_hashes[cache_key]
    row_digests = {
        str(p.relative_to(successor_root)): admission._sha256(p)
        for p in sorted(successor_root.glob("runs/*/episodes.jsonl"))
    }
    return (
        old,
        new,
        {
            "tooling": admission._tooling_identity(),
            "baseline": {
                "path": str(baseline_bundle),
                "sha256": digest,
                "source_commit": source,
                "campaign_id": admission.BASELINE_CAMPAIGN,
                "scenario": scenario,
            },
            "successor": {
                "path": str(successor_root),
                **identity,
                "manifest_sha256": successor_manifest_sha256,
                "row_sha256": row_digests,
            },
            "metric_schema_versions": schemas,
            "probe": {"coverage_admitted": not diagnostic_partial, "summary": probe},
            "release_eligible": not rehearsal,
        },
    )


def _verified_development_successor(path, digest, source_root, campaign_root, rows):
    """Bind a rehearsal to its verified public identity and the same pinned runtime."""
    from robot_sf.benchmark.release_protocol import is_development_rehearsal, load_release_manifest

    source_root = source_root.resolve()
    admission._verify_sha256(
        path,
        admission._hex_digest(digest, "successor manifest digest"),
        label="successor manifest SHA-256",
    )
    manifest = load_release_manifest(path, repository_root=source_root)
    if not is_development_rehearsal(manifest):
        raise ValueError("development comparator requires a verified rehearsal identity")
    campaign = json.loads((campaign_root / "campaign_manifest.json").read_text())
    config_path = manifest.canonical_campaign_config_path.relative_to(source_root).as_posix()
    config_hash, scenario_hash, runtime_rows, scoped_hashes, expected_slots = (
        admission._runtime_successor_identity(
            source_root,
            manifest.source_sha,
            config_path,
            rows,
            publication_identity={"release_tag": manifest.release_tag, "doi": manifest.doi},
            development_rehearsal_seeds=manifest.resolved_seeds,
        )
    )
    return {
        "source_commit": manifest.source_sha,
        "campaign_id": campaign["campaign_id"],
        "campaign_config": {
            "path": config_path,
            "sha256": manifest.campaign_config_sha256,
            "runtime_hash": config_hash,
        },
        "scenario_matrix": {
            "path": manifest.scenario_matrix_path.relative_to(source_root).as_posix(),
            "sha256": manifest.scenario_matrix_sha256,
            "runtime_hash": scenario_hash,
        },
        "runtime_rows": runtime_rows,
        "scoped_hashes": scoped_hashes,
        "expected_slots": expected_slots,
        "planner_keys": sorted(manifest.planner_keys),
        "manifest_sha256": digest,
    }
