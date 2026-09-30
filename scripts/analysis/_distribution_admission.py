"""Distribution admission built from the paired comparator's pinned validators."""

import hashlib
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
) -> tuple[dict, dict, dict]:
    """Read verified release rows; diagnostics relax coverage only, never binding."""
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
    verified = admission._verified_successor_manifest(
        successor_manifest, successor_manifest_sha256, successor_source_root, new
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
        },
    )
