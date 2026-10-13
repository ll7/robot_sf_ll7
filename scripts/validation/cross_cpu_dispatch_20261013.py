#!/usr/bin/env python3
"""Reconstruct #10211 as a development-only, preserved cross-CPU diagnostic.

Commands: prepare, run (inside Slurm), compare, manifest. See adjacent PLAN.md.
The acquisition uses the canonical runner, unchanged current-main runtime and
public checkpoint bindings; no archived output is treated as fresh evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "scripts/validation/cross_cpu_20261013/study.json"
SCIENTIFIC = (
    "metrics",
    "metric_values",
    "steps",
    "status",
    "outcome",
    "termination_reason",
    "termination",
    "failure_reason",
    "failure_mechanism",
)


def digest(path):
    """Hash file bytes without loading large evidence files into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    """Write a receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def contract():
    """Validate the development-only seed and budget bounds."""
    cfg = json.loads(STUDY.read_text())
    if not 1001 <= cfg["seed_start"] <= cfg["seed_stop"] <= 1200:
        raise ValueError("seed range outside authorized development interval")
    worst = 2 * len(cfg["campaigns"]) * cfg["workers"] * cfg["campaign_timeout_seconds"] / 3600
    # Reserve ten CPU-hours for identification, staging and canaries.
    if worst + 10 > cfg["cpu_hour_cap"]:
        raise ValueError("worst-case allocation exceeds CPU-hour cap")
    return cfg


def prepare(destination, smoke=False):
    """Derive an isolated campaign config from the specified current-main input."""
    import yaml

    cfg = contract()
    destination.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT / cfg["base_config"]).read_text())
    selected = {p["key"]: p for p in base["planners"]}
    base.update(
        {
            "name": "cross_cpu_20261013_dev_diagnostic",
            "paper_facing": False,
            "seed_policy": {
                "mode": "fixed-list",
                "seeds": list(
                    range(cfg["seed_start"], (cfg["seed_start"] if smoke else cfg["seed_stop"]) + 1)
                ),
            },
            "scenario_candidates": cfg["scenarios"],
            "planners": [selected[arm] for arm in cfg["arms"]],
            "workers": cfg["workers"],
            "resume": False,
            "stop_on_failure": True,
            "export_publication_bundle": False,
            "snqi_v2_spec": None,
            "snqi_weights": None,
            "snqi_baseline": None,
            "bootstrap_seed": 1001,
        }
    )
    base["snqi_contract"]["calibration_seed"] = 1001
    base["snqi_contract"]["enabled"] = False
    for mode in ("default", "pinned", "float64"):
        payload = dict(base)
        if mode == "float64":
            payload["numerical_mode"] = "pinned_float64_v1"
        else:
            payload.pop("numerical_mode", None)
        (destination / f"{mode}.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    write_json(
        destination / "preparation.json",
        {
            "study_sha256": digest(STUDY),
            "base_config_sha256": digest(ROOT / cfg["base_config"]),
            "smoke": smoke,
            "configs": {p.name: digest(p) for p in destination.glob("*.yaml")},
        },
    )


def environment(cfg, mode, context_root):
    """Set dispatch before interpreter startup; remove inherited context overrides."""
    env = dict(os.environ)
    for key in (
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PYTHONSTARTUP",
        "ROBOT_SF_NUMERICAL_MODE",
        *cfg["pinned_env"],
    ):
        env.pop(key, None)
    env.update(cfg["common_env"])
    if mode != "default":
        env.update(cfg["pinned_env"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(STUDY.parent), str(ROOT))),
            "PYTHONNOUSERSITE": "1",
            "XCPU_MODE": mode,
            "XCPU_CONTEXT_ROOT": str(context_root),
            "PROBE_ROOT": str(ROOT),
        }
    )
    return env


def identity():
    """Capture installed distributions, CPU, OS, source and dependency identity."""

    def capture(command):
        return subprocess.check_output(command, text=True, cwd=ROOT).strip()

    inventory = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    return {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "glibc": capture(["getconf", "GNU_LIBC_VERSION"]),
        "lscpu": capture(["lscpu", "--json"]),
        "python": sys.version,
        "executable": sys.executable,
        "commit": capture(["git", "rev-parse", "HEAD"]),
        "source_diff": capture(
            ["git", "diff", "HEAD", "--", "robot_sf", "configs", "fast-pysf", "third_party"]
        ),
        "lock_sha256": digest(ROOT / "uv.lock"),
        "inventory": inventory,
        "inventory_sha256": hashlib.sha256(
            json.dumps(inventory, separators=(",", ":")).encode()
        ).hexdigest(),
        "study_sha256": digest(STUDY),
        "driver_sha256": digest(Path(__file__)),
        "hook_sha256": digest(STUDY.parent / "sitecustomize.py"),
        "slurm": {
            k: v
            for k, v in os.environ.items()
            if k
            in {"SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_JOB_PARTITION"}
        },
    }


def run(destination, config_root, smoke=False, preflight=False, selected_campaign=None):
    """Run each frozen campaign with bounded process-tree time and durable logs."""
    cfg = contract()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("acquisition must run in an sbatch allocation")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != cfg["workers"]:
        raise RuntimeError("allocation must match four episode workers")
    if destination.exists():
        raise FileExistsError("fresh acquisition directory required; preserve failed attempts")
    destination.mkdir(parents=True)
    write_json(destination / "environment.json", identity())
    if selected_campaign is not None and (
        selected_campaign not in cfg["campaigns"] or smoke or preflight
    ):
        raise ValueError("single campaign selection requires a full-study campaign")
    campaigns = (
        ["default-canary"]
        if smoke or preflight
        else ([selected_campaign] if selected_campaign else cfg["campaigns"])
    )
    receipts = []
    for campaign in campaigns:
        mode = campaign.split("-", 1)[0]
        env = environment(cfg, mode, destination / "contexts" / campaign)
        command = [
            sys.executable,
            str(ROOT / "scripts/tools/run_camera_ready_benchmark.py"),
            "--config",
            str(config_root / f"{mode}.yaml"),
            "--output-root",
            str(destination / "campaigns"),
            "--campaign-id",
            campaign,
            "--arm-isolation",
            "subprocess",
            "--log-level",
            "WARNING",
        ]
        if preflight:
            command += ["--mode", "preflight", "--checkpoint-preflight-mode", "enforced_staged"]
        start = time.monotonic()
        with (destination / f"{campaign}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=cfg["campaign_timeout_seconds"] if not smoke else 900)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
        receipts.append(
            {
                "campaign": campaign,
                "command": command,
                "exit_code": code,
                "elapsed_seconds": time.monotonic() - start,
                "config_sha256": digest(config_root / f"{mode}.yaml"),
            }
        )
        write_json(destination / "execution.json", receipts)
        if code:
            raise RuntimeError("campaign failed; inspect preserved log; no automatic retry")
    write_json(destination / "COMPLETE.json", {"campaigns": campaigns, "completed": True})


def canonical(value):
    """Define exact scientific JSON byte equality, including signed zeros."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def records(root, campaign, cfg):
    """Index every native row by explicit arm/scenario/seed, rejecting omissions."""
    indexed = {}
    campaign_root = root / "campaigns" / campaign
    if not campaign_root.exists():
        campaign_root = root / "jobs" / campaign / "campaigns" / campaign
    for arm in cfg["arms"]:
        paths = list(campaign_root.glob(f"**/{arm}/**/episodes.jsonl"))
        if len(paths) != 1:
            # Canonical campaign directories include arm plus kinematics.
            paths = [
                p
                for p in campaign_root.rglob("episodes.jsonl")
                if p.parent.name == f"{arm}__differential_drive"
            ]
        if len(paths) != 1:
            raise ValueError(f"expected one native file for {campaign}/{arm}")
        for line in paths[0].read_text().splitlines():
            row = json.loads(line)
            scenario = row.get("scenario_id")
            seed = row.get("seed")
            if (
                scenario not in cfg["scenarios"]
                or type(seed) is not int
                or not cfg["seed_start"] <= seed <= cfg["seed_stop"]
            ):
                raise ValueError("row outside frozen coordinate inventory")
            key = (arm, scenario, seed)
            if key in indexed or not isinstance(row.get("metrics"), dict) or "steps" not in row:
                raise ValueError("duplicate or incomplete scientific row")
            indexed[key] = row
    expected = {
        (a, s, n)
        for a in cfg["arms"]
        for s in cfg["scenarios"]
        for n in range(cfg["seed_start"], cfg["seed_stop"] + 1)
    }
    if indexed.keys() != expected:
        raise ValueError("missing or unexpected coordinates")
    return indexed


def outcome(row):
    """Retain native event outcomes separately from numeric collision counts."""
    if not isinstance(row.get("outcome"), dict) or "termination_reason" not in row:
        raise ValueError("explicit native outcome and termination are required")
    return {
        k: row[k] for k in ("outcome", "termination_reason", "termination", "status") if k in row
    }


def audit(loaded, destination):
    """Validate native schemas and fail-closed execution/integrity semantics."""
    import jsonschema

    from robot_sf.benchmark.release_acceptance import _status_markers
    from robot_sf.benchmark.spawn_validity import record_has_invalid_spawn

    schema = json.loads((ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    counts = {
        "records": 0,
        "schema_errors": 0,
        "forbidden_status": 0,
        "integrity_contradictions": 0,
        "invalid_spawn": 0,
    }
    fields = set()
    for indexed in loaded.values():
        for (arm, _, _), row in indexed.items():
            counts["records"] += 1
            counts["schema_errors"] += not validator.is_valid(row)
            counts["forbidden_status"] += bool(_status_markers(row, "row", expected_algorithm=arm))
            counts["integrity_contradictions"] += bool(
                row.get("integrity", {}).get("contradictions")
            )
            counts["invalid_spawn"] += record_has_invalid_spawn(row)
            fields.update(row)
    write_json(
        destination / "row_audit.json",
        {"counts": counts, "top_level_fields": sorted(fields), "scientific_fields": SCIENTIFIC},
    )
    if any(v for k, v in counts.items() if k != "records"):
        raise ValueError("native evidence audit failed; inspect row_audit.json")


IDENTITY_FIELDS = (
    "commit",
    "lock_sha256",
    "inventory_sha256",
    "study_sha256",
    "driver_sha256",
    "hook_sha256",
)


def collect(root):
    """Reconcile single-campaign jobs only after every producer has stopped."""
    cfg = contract()
    identities, receipts = [], []
    for campaign in cfg["campaigns"]:
        job = root / "jobs" / campaign
        if json.loads((job / "COMPLETE.json").read_text()).get("campaigns") != [campaign]:
            raise ValueError("incomplete or mismatched single-campaign receipt")
        execution = json.loads((job / "execution.json").read_text())
        if len(execution) != 1 or execution[0]["campaign"] != campaign or execution[0]["exit_code"]:
            raise ValueError("non-success single-campaign receipt")
        identities.append(json.loads((job / "environment.json").read_text()))
        receipts.extend(execution)
        records(root, campaign, cfg)
    for field in (*IDENTITY_FIELDS, "hostname", "kernel", "glibc"):
        if any(i.get(field) != identities[0].get(field) for i in identities):
            raise ValueError(f"within-CPU identity mismatch: {field}")
    write_json(root / "environment.json", identities[0])
    write_json(root / "job_environments.json", identities)
    write_json(root / "execution.json", receipts)
    write_json(root / "COMPLETE.json", {"campaigns": cfg["campaigns"], "completed": True})


def submit(root, config_root, nodes, partitions, qos):
    """Submit exactly eight bounded CPU-only jobs and preserve scheduler receipts."""
    cfg = contract()
    if (root / "submission.json").exists():
        raise FileExistsError("existing submission receipt; no duplicate dispatch")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT):
        raise RuntimeError("freeze and commit the producer before dispatch")
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    receipts = []
    for cpu, node, partition, queue_qos in zip(
        ("amd", "intel"), nodes, partitions, qos, strict=True
    ):
        for campaign in cfg["campaigns"]:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "run",
                "--campaign",
                campaign,
                "--destination",
                str(root / cpu / "jobs" / campaign),
                "--config-root",
                str(config_root),
            ]
            scheduler = [
                "sbatch",
                "--parsable",
                f"--partition={partition}",
                f"--qos={queue_qos}",
                f"--nodelist={node}",
                "--cpus-per-task=4",
                "--mem=16G",
                "--time=04:00:00",
                f"--chdir={ROOT}",
                f"--output={root / 'slurm'}/%j-{cpu}-{campaign}.out",
                "--wrap",
                shlex.join(command),
            ]
            job_id = subprocess.check_output(scheduler, text=True).strip().split(";", 1)[0]
            receipt = {
                "cpu": cpu,
                "campaign": campaign,
                "node": node,
                "job_id": job_id,
                "sbatch": scheduler,
                "config_sha256": digest(config_root / f"{campaign.split('-', 1)[0]}.yaml"),
            }
            receipts.append(receipt)
            write_json(root / "submission.json", receipts)
            detail = subprocess.check_output(["scontrol", "show", "job", job_id, "-o"], text=True)
            (root / "slurm" / f"{job_id}-submission.txt").write_text(detail)
    print(
        json.dumps(
            {"job_ids": [r["job_id"] for r in receipts], "maximum_acquisition_cpu_hours": 128}
        )
    )


def compare(first, second, destination):
    """Compare paired scientific rows and write a complete coordinate ledger."""
    cfg = contract()
    destination.mkdir(parents=True, exist_ok=True)
    identities = [json.loads((r / "environment.json").read_text()) for r in (first, second)]
    for field in IDENTITY_FIELDS:
        if not identities[0].get(field) or identities[0][field] != identities[1].get(field):
            raise ValueError(f"cross-CPU environment identity mismatch: {field}")
    for root in (first, second):
        if json.loads((root / "COMPLETE.json").read_text()).get("campaigns") != cfg["campaigns"]:
            raise ValueError("campaign completion receipt mismatch")
        if any(r["exit_code"] for r in json.loads((root / "execution.json").read_text())):
            raise ValueError("non-success campaign execution receipt")
    loaded = {
        (cpu, c): records(root, c, cfg)
        for cpu, root in (("a", first), ("b", second))
        for c in cfg["campaigns"]
    }
    audit(loaded, destination)
    comparisons = [(f"cpu-{c}", ("a", c), ("b", c)) for c in cfg["campaigns"]]
    comparisons += [
        (f"noise-{cpu}", (cpu, "default-r1"), (cpu, "default-r2")) for cpu in ("a", "b")
    ]
    comparisons += [
        (f"default-vs-{m}-{cpu}", (cpu, "default-r1"), (cpu, f"{m}-r1"))
        for cpu in ("a", "b")
        for m in ("pinned", "float64")
    ]
    summary = {}
    with (destination / "paired_coordinates.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "comparison",
                "arm",
                "scenario",
                "seed",
                "scientific_identical",
                "outcome_identical",
                "scientific_a_sha256",
                "scientific_b_sha256",
            ]
        )
        for name, left, right in comparisons:
            counts = {
                "total": 0,
                "scientific_identical": 0,
                "learned_identical": 0,
                "controls_identical": 0,
                "outcome_differences": 0,
                "per_arm": {},
            }
            for key in sorted(loaded[left]):
                a, b = loaded[left][key], loaded[right][key]
                scientific = [canonical({k: r[k] for k in SCIENTIFIC if k in r}) for r in (a, b)]
                same = scientific[0] == scientific[1]
                same_outcome = canonical(outcome(a)) == canonical(outcome(b))
                arm_counts = counts["per_arm"].setdefault(
                    key[0], {"total": 0, "scientific_identical": 0, "outcome_differences": 0}
                )
                for target in (counts, arm_counts):
                    target["total"] += 1
                    target["scientific_identical"] += same
                    target["outcome_differences"] += not same_outcome
                counts[
                    "learned_identical"
                    if key[0] in {"ppo", "guarded_ppo"}
                    else "controls_identical"
                ] += same
                writer.writerow(
                    [
                        name,
                        *key,
                        int(same),
                        int(same_outcome),
                        *[hashlib.sha256(v.encode()).hexdigest() for v in scientific],
                    ]
                )
            summary[name] = counts
    write_json(destination / "comparison.json", summary)


def manifest(root):
    """Build a portable file manifest excluding itself and unfinished temp writes."""
    entries = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"} and p.suffix != ".tmp"
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{digest(p)}  {p.relative_to(root).as_posix()}\n" for p in entries)
    )
    checksum = digest(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{checksum}  SHA256SUMS\n")
    print(checksum)


def main():
    """Parse a bounded acquisition or offline analysis command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "submit", "run", "collect", "compare", "manifest")
    )
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--campaign", choices=contract()["campaigns"])
    parser.add_argument("--nodes", nargs=2)
    parser.add_argument("--partitions", nargs=2)
    parser.add_argument("--qos", nargs=2)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.destination.resolve(), args.smoke)
    elif args.command == "submit":
        if any(x is None for x in (args.config_root, args.nodes, args.partitions, args.qos)):
            parser.error("submit requires --config-root and two --nodes/--partitions/--qos")
        submit(
            args.destination.resolve(),
            args.config_root.resolve(),
            args.nodes,
            args.partitions,
            args.qos,
        )
    elif args.command == "run":
        if args.config_root is None:
            parser.error("run requires --config-root")
        run(
            args.destination.resolve(),
            args.config_root.resolve(),
            args.smoke,
            args.preflight,
            args.campaign,
        )
    elif args.command == "collect":
        collect(args.destination.resolve())
    elif args.command == "compare":
        if args.first is None or args.second is None:
            parser.error("compare requires --first and --second")
        compare(args.first.resolve(), args.second.resolve(), args.destination.resolve())
    else:
        manifest(args.destination.resolve())


if __name__ == "__main__":
    main()
