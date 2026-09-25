"""Build and verify a checksum-bound #9671 observer bundle from frozen traces.

This is a diagnostic admission gate. It never edits a producer JSONL, sidecar,
release archive, or 0.0.7 record. The manifest is meaningful only with its
separately recorded SHA-256 and the exact source/config/observer inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.validation import check_issue_9671_trace_reexport as trace_checker

BASELINE_REPORT_SHA256 = "e1637fa907d87f8a5456ee0f3367524e8e335b480c1d2bd5162215f08a3f7ffd"
BASELINE_COUNTS = {"match": 2, "mismatch": 0, "no_release_row": 18}
SCHEMA = "issue-9671-force-observer-bundle.v1"
SIDECAR_SCHEMA = "issue-9671-robot-force-observer.v1"
FROZEN_SOURCE_FILE_SHA256 = {
    "robot_sf/ped_npc/ped_robot_force.py": "73ec0e40987f1a2636e970dcc3640377070af784346b55c08484692fbee1c70e",
    "robot_sf/sim/simulator.py": "d4751f53f973716f55646d8c414335b5bba7381c6b27e124b873820d99535698",
    "robot_sf/benchmark/map_runner/map_runner_episode.py": "7bfece1ee1337ff9803f79aa57b1019f92d9dd40c7aedfc30b0ab6cd350de82d",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label(name: str, path: Path, campaign_root: Path) -> str:
    """Use relocation-stable names and reject files outside the campaign tree."""
    try:
        relative = path.resolve(strict=True).relative_to(campaign_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"artifact is outside its campaign tree: {path}") from exc
    return f"{name}/{relative.as_posix()}"


def _canonical_row_sha(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _rows(paths: list[Path]) -> dict[tuple[str, str, int], dict[str, Any]]:
    indexed = {}
    ids = set()
    for path in paths:
        for row in trace_checker._rows(path.read_bytes().splitlines()):
            key = trace_checker._key(row)
            episode_id = row.get("episode_id")
            if key in indexed or not isinstance(episode_id, str) or episode_id in ids:
                raise ValueError(f"duplicate or absent episode identity: {path} {key}")
            indexed[key] = row
            ids.add(episode_id)
    return indexed


def _vectors(value: Any, count: int) -> list[list[float]]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or not all(trace_checker._vector2(item) for item in value)
    ):
        raise ValueError("observer force/state series missing, non-finite, or wrong shape")
    return value


def _positions(peds: list[dict[str, Any]]) -> list[list[float]]:
    return [ped["position"] for ped in peds]


def _validate_sidecar(  # noqa: C901, PLR0912, PLR0915 - independent custody assertions
    sidecar: dict[str, Any],
    row: dict[str, Any],
    *,
    campaign_id: str,
    job_id: str,
    config_sha256: str,
    observer_sha256: str,
) -> None:
    """Check every force slot against a specific immutable episode row."""
    provenance = sidecar.get("observer_provenance") or {}
    trace = (row.get("algorithm_metadata") or {}).get("simulation_step_trace") or {}
    reset = (trace.get("reset") or {}).get("pedestrians")
    steps = trace.get("steps")
    if not isinstance(reset, list) or not isinstance(steps, list) or not steps:
        raise ValueError("raw episode lacks complete reset/step trace")
    actors = [ped.get("actor_id") for ped in reset]
    if not actors or any(not isinstance(actor, str) or not actor for actor in actors):
        raise ValueError("raw reset actor IDs missing")
    if len(set(actors)) != len(actors):
        raise ValueError("raw reset actor IDs duplicated")
    if (
        sidecar.get("schema_version") != SIDECAR_SCHEMA
        or sidecar.get("episode_id") != row.get("episode_id")
        or sidecar.get("scenario_id") != row.get("scenario_id")
        or sidecar.get("seed") != row.get("seed")
        or sidecar.get("algorithm") != row.get("algo")
        or sidecar.get("episode_record_canonical_sha256") != _canonical_row_sha(row)
        or sidecar.get("actor_ids") != actors
        or provenance.get("frozen_source_commit") != trace_checker.SOURCE_SHA
        or provenance.get("frozen_source_file_sha256") != FROZEN_SOURCE_FILE_SHA256
        or provenance.get("diagnostic_config_sha256") != config_sha256
        or provenance.get("observer_sha256") != observer_sha256
        or provenance.get("campaign_id") != campaign_id
        or provenance.get("job_id") != job_id
        or not isinstance(provenance.get("pid"), str)
        or not provenance["pid"].isdigit()
        or not isinstance(sidecar.get("steps"), list)
        or len(sidecar["steps"]) != len(steps)
        or row.get("steps") != len(steps)
    ):
        raise ValueError(f"observer sidecar identity mismatch: {row.get('episode_id')}")
    prior = _positions(reset)
    prior_robot = (trace.get("reset") or {}).get("robot", {}).get("position")
    if not trace_checker._vector2(prior_robot):
        raise ValueError("raw reset robot position missing")
    roster = None
    component_configs = None
    for index, (sample, raw_step) in enumerate(zip(sidecar["steps"], steps, strict=True)):
        count = len(actors)
        if (
            sample.get("step") != index
            or raw_step.get("step") != index
            or sample.get("actor_ids") != actors
            or [ped.get("actor_id") for ped in raw_step["pedestrians"]] != actors
            or sample.get("step_entry_positions") != prior
            or sample.get("force_input_positions") != prior
            or sample.get("post_step_positions") != _positions(raw_step["pedestrians"])
            or sample.get("total_forces") != raw_step["planner"]["ammv"]["pedestrian_force_vectors"]
        ):
            raise ValueError(f"observer step/actor/raw-force binding mismatch: {index}")
        _vectors(sample.get("step_entry_positions"), count)
        _vectors(sample.get("force_input_positions"), count)
        _vectors(sample.get("post_step_positions"), count)
        _vectors(sample.get("total_forces"), count)
        robot_forces = _vectors(sample.get("robot_forces"), count)
        if len({tuple(position) for position in prior}) != count:
            raise ValueError(f"ambiguous coincident actor positions at step {index}")
        ids = sample.get("component_ids")
        components = sample.get("component_inputs")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids)
            or not isinstance(components, list)
            or len(components) != len(ids)
        ):
            raise ValueError(f"observer component roster invalid at step {index}")
        if roster is None:
            roster = ids
        elif ids != roster:
            raise ValueError(f"observer component roster changed at step {index}")
        totals = [[0.0, 0.0] for _ in actors]
        configs = []
        for component in components:
            if (
                component.get("positions") != prior
                or component.get("robot_position") != prior_robot
            ):
                raise ValueError(f"observer component inputs misbound at step {index}")
            multipliers = component.get("multipliers")
            if multipliers is not None and (
                not isinstance(multipliers, list)
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    for value in multipliers
                )
            ):
                raise ValueError(f"observer multipliers invalid at step {index}")
            if component.get("multipliers_applied") is not (
                multipliers is not None and len(multipliers) == count
            ):
                raise ValueError(f"observer multiplier application mismatch at step {index}")
            config = component.get("config")
            if not isinstance(config, dict) or not {
                "is_active",
                "robot_radius",
                "activation_threshold",
                "force_multiplier",
            } <= set(config):
                raise ValueError(f"observer component config missing at step {index}")
            configs.append(config)
            forces = _vectors(component.get("forces"), count)
            for actor_index, force in enumerate(forces):
                totals[actor_index][0] += force[0]
                totals[actor_index][1] += force[1]
        if totals != robot_forces:
            raise ValueError(f"observer robot-force sum differs at step {index}")
        if component_configs is None:
            component_configs = configs
        elif configs != component_configs:
            raise ValueError(f"observer component config changed at step {index}")
        prior = _positions(raw_step["pedestrians"])
        prior_robot = raw_step["robot"]["position"]


def _trace_state(row: dict[str, Any]) -> dict[str, Any]:
    trace = row["algorithm_metadata"]["simulation_step_trace"]
    return {
        "reset": trace["reset"],
        "steps": [
            {
                "step": step["step"],
                "time_s": step["time_s"],
                "robot": step["robot"],
                "pedestrians": step["pedestrians"],
                "total_forces": step["planner"]["ammv"]["pedestrian_force_vectors"],
            }
            for step in trace["steps"]
        ],
    }


def build_manifest(  # noqa: C901, PLR0912, PLR0915 - custody gate checks disjoint artifacts
    spec: dict[str, Any],
    *,
    expected: set[tuple[str, str, int]] = trace_checker.EXPECTED_TUPLES,
    expected_baseline_report_sha256: str | None = BASELINE_REPORT_SHA256,
    expected_baseline_counts: dict[str, int] | None = BASELINE_COUNTS,
    expected_archive_sha256: str | None = trace_checker.ARCHIVE_SHA256,
    expected_config_sha256: dict[str, str] | None = None,
    expected_effective_hash: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate all inputs and construct a deterministic diagnostic manifest."""
    if set(spec.get("campaigns") or {}) != {"headon_group", "doorway"}:
        raise ValueError("exactly the two diagnostic campaigns are required")
    archive = Path(spec["archive"])
    baseline_path = Path(spec["baseline_report"])
    baseline_sha = _sha(baseline_path)
    if (
        expected_baseline_report_sha256 is not None
        and baseline_sha != expected_baseline_report_sha256
    ):
        raise ValueError("baseline diagnostic comparison SHA-256 mismatch")
    baseline = json.loads(baseline_path.read_text())
    if (
        baseline.get("source_commit") != trace_checker.SOURCE_SHA
        or baseline.get("release_archive_sha256") != _sha(archive)
        or (
            expected_baseline_counts is not None
            and baseline.get("comparison_counts") != expected_baseline_counts
        )
    ):
        raise ValueError("baseline comparison identity/count mismatch")
    baseline_paths = [Path(value) for value in spec["baseline_traces"]]
    if sorted(_sha(path) for path in baseline_paths) != sorted(
        baseline["trace_inputs_sha256"].values()
    ):
        raise ValueError("baseline raw JSONL byte hashes differ from comparison receipt")
    baseline_rows = _rows(baseline_paths)
    if set(baseline_rows) != expected:
        raise ValueError("baseline 20-tuple inventory mismatch")
    observer_path = Path(spec["observer_path"])
    observer_sha = _sha(observer_path)
    if observer_sha != spec["observer_sha256"]:
        raise ValueError("staged observer byte hash mismatch")
    campaigns = spec["campaigns"]
    configs = {name: Path(item["config"]) for name, item in campaigns.items()}
    manifests = {name: Path(item["campaign_manifest"]) for name, item in campaigns.items()}
    campaign_ids = {name: item["campaign_id"] for name, item in campaigns.items()}
    trace_paths = [Path(path) for item in campaigns.values() for path in item["traces"]]
    trace_owner = {
        str(path): name for name, item in campaigns.items() for path in map(Path, item["traces"])
    }
    if len(trace_owner) != len(trace_paths):
        raise ValueError("raw JSONL assigned to multiple campaigns")
    producer_owner = {
        str(path.with_name(path.name + ".provenance.json")): name
        for name, item in campaigns.items()
        for path in map(Path, item["traces"])
    }
    release_report = trace_checker.check(
        archive,
        trace_paths,
        configs,
        manifests,
        expected=expected,
        expected_archive_sha256=expected_archive_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_effective_hash=expected_effective_hash,
        expected_campaign_ids=campaign_ids,
    )
    if (
        release_report["release_archive_sha256"] != _sha(archive)
        or set(release_report["trace_inputs_sha256"]) != {str(path) for path in trace_paths}
        or set(release_report["producer_manifests_sha256"]) != set(producer_owner)
        or any(
            _sha(Path(path)) != digest
            for path, digest in release_report["trace_inputs_sha256"].items()
        )
        or any(
            _sha(Path(path)) != digest
            for path, digest in release_report["producer_manifests_sha256"].items()
        )
        or any(
            _sha(configs[name]) != binding["config_sha256"]
            or _sha(manifests[name]) != binding["campaign_manifest_sha256"]
            for name, binding in release_report["diagnostic_inputs"].items()
        )
    ):
        raise ValueError("trace comparison input bytes changed during bundle construction")
    new_rows = _rows(trace_paths)
    if set(new_rows) != expected:
        raise ValueError("observer trace tuple inventory mismatch")
    receipt_rows = []
    sidecar_hashes: dict[str, str] = {}
    startup_receipts: dict[str, dict[str, Any]] = {}
    for name, item in sorted(campaigns.items()):
        job_id = str(item["job_id"])
        if not job_id.isdigit() or not item["campaign_id"]:
            raise ValueError(f"invalid campaign/job identity: {name}")
        startup = json.loads(Path(item["startup_receipt"]).read_text())
        startup_ids = startup.get("identities") or {}
        if (
            startup.get("schema_version") != "robot-sf-slurm-startup.v2"
            or startup.get("status") != "started"
            or startup_ids.get("job_id") != job_id
            or startup_ids.get("campaign") != item["campaign_id"]
            or startup_ids.get("config") != trace_checker.DIAGNOSTIC_CONFIG[name]
            or startup_ids.get("public_commit") != trace_checker.SOURCE_SHA
            or not startup_ids.get("queue_id")
            or not startup_ids.get("submission_id")
        ):
            raise ValueError(f"Slurm startup identity mismatch: {name}")
        startup_receipts[name] = startup
        selected = set(release_report["diagnostic_inputs"][name]["tuples"])
        sidecar_dir = Path(item["sidecar_dir"])
        paths = sorted(sidecar_dir.rglob("*.robot-force.json"))
        wanted_ids = {new_rows[key]["episode_id"] for key in selected}
        if set(paths) != {
            sidecar_dir / f"{episode_id}.robot-force.json" for episode_id in wanted_ids
        }:
            raise ValueError(f"missing or extra observer sidecars: {name}")
        for key in sorted(selected):
            row = new_rows[key]
            path = sidecar_dir / f"{row['episode_id']}.robot-force.json"
            sidecar = json.loads(path.read_text())
            _validate_sidecar(
                sidecar,
                row,
                campaign_id=item["campaign_id"],
                job_id=job_id,
                config_sha256=release_report["diagnostic_inputs"][name]["config_sha256"],
                observer_sha256=observer_sha,
            )
            label = _label(name, path, manifests[name].parent)
            sidecar_hashes[label] = _sha(path)
            receipt_rows.append(
                {
                    "planner": key[0],
                    "scenario": key[1],
                    "seed": key[2],
                    "episode_id": row["episode_id"],
                    "raw_row_canonical_sha256": _canonical_row_sha(row),
                    "sidecar_path": label,
                    "sidecar_sha256": sidecar_hashes[label],
                    "step_count": row["steps"],
                    "actor_count": len(sidecar["actor_ids"]),
                    "campaign_id": item["campaign_id"],
                    "job_id": job_id,
                }
            )
    differences = []
    comparison_by_key = {
        (item["planner"], item["scenario"], item["seed"]): item
        for item in release_report["comparisons"]
    }
    for key in sorted(expected):
        old = baseline_rows[key]
        new = new_rows[key]
        changed = [
            field for field in ("episode_id", "status", "steps") if old.get(field) != new.get(field)
        ]
        if _trace_state(old) != _trace_state(new):
            changed.append("recorded_state_or_total_forces")
        if changed:
            differences.append({"tuple": list(key), "fields": changed})
        receipt = next(
            item
            for item in receipt_rows
            if (item["planner"], item["scenario"], item["seed"]) == key
        )
        receipt["baseline_episode_id"] = old["episode_id"]
        receipt["baseline_status"] = old["status"]
        receipt["new_status"] = new["status"]
        receipt["release_comparison"] = comparison_by_key[key]["comparison"]
    return {
        "schema_version": SCHEMA,
        "frozen_source_commit": trace_checker.SOURCE_SHA,
        "release_archive_sha256": release_report["release_archive_sha256"],
        "baseline_report_sha256": baseline_sha,
        "baseline_comparison_counts": baseline["comparison_counts"],
        "observer_sha256": observer_sha,
        "campaigns": {
            name: {
                "campaign_id": item["campaign_id"],
                "job_id": str(item["job_id"]),
                "config_sha256": release_report["diagnostic_inputs"][name]["config_sha256"],
                "effective_config_hash": release_report["diagnostic_inputs"][name][
                    "effective_config_hash"
                ],
                "campaign_manifest_sha256": release_report["diagnostic_inputs"][name][
                    "campaign_manifest_sha256"
                ],
                "startup_receipt_sha256": _sha(Path(item["startup_receipt"])),
                "queue_id": startup_receipts[name]["identities"]["queue_id"],
                "submission_id": startup_receipts[name]["identities"]["submission_id"],
            }
            for name, item in sorted(campaigns.items())
        },
        "raw_jsonl_sha256": {
            _label(trace_owner[path], Path(path), manifests[trace_owner[path]].parent): digest
            for path, digest in release_report["trace_inputs_sha256"].items()
        },
        "producer_manifest_sha256": {
            _label(producer_owner[path], Path(path), manifests[producer_owner[path]].parent): digest
            for path, digest in release_report["producer_manifests_sha256"].items()
        },
        "sidecar_sha256": sidecar_hashes,
        "episodes": sorted(
            receipt_rows, key=lambda item: (item["planner"], item["scenario"], item["seed"])
        ),
        "release_comparison_counts": release_report["comparison_counts"],
        "parity_differences": differences,
        "admission_status": (
            "candidate"
            if not differences and release_report["comparison_counts"]["mismatch"] == 0
            else "diagnostic_mismatch"
        ),
    }


def validate_manifest(
    spec: dict[str, Any], manifest: Path, expected_sha256: str, **kwargs: Any
) -> dict[str, Any]:
    """Cold-read all artifacts and compare the manifest to a separately pinned SHA."""
    if _sha(manifest) != expected_sha256:
        raise ValueError("observer bundle manifest SHA-256 mismatch")
    recorded = json.loads(manifest.read_text())
    fresh = build_manifest(spec, **kwargs)
    if recorded != fresh:
        raise ValueError("observer bundle manifest differs from cold artifacts")
    return fresh


def main() -> int:
    """Write or cold-verify a diagnostic bundle manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write", "validate"))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.mode == "write":
        result = build_manifest(spec)
        if args.manifest.exists():
            raise ValueError("refusing to overwrite an existing observer manifest")
        args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        if not args.manifest_sha256:
            raise ValueError("validate requires separately pinned --manifest-sha256")
        result = validate_manifest(spec, args.manifest, args.manifest_sha256)
    print(
        json.dumps(
            {
                "manifest_sha256": _sha(args.manifest),
                "admission_status": result["admission_status"],
                "release_comparison_counts": result["release_comparison_counts"],
                "parity_differences": result["parity_differences"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["admission_status"] == "candidate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
