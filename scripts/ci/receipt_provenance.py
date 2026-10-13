"""Verify producer seals and bind allocations without changing producer artifacts."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath

BINDING_FILE = "execution_binding.json"


class HardwareIdentityError(ValueError):
    """Hardware or provenance evidence cannot support an outcome comparison."""


def _native_id_values(value: str, field: str) -> tuple[str | None, ...]:
    """Use the actual native directory, roster and scenario identity functions.

    Receipt labels must be portable native directory IDs as well as unchanged
    logical scenario IDs. Keep all native projections to detect collisions in
    either namespace; do not reproduce their normalization algorithms here.
    """
    from robot_sf.benchmark.camera_ready import _config
    from robot_sf.training import scenario_loader
    from scripts.validation import run_empty_world_sweep

    directory = _config._sanitize_name(value)
    if field == "arm":
        roster = run_empty_world_sweep._arm_key(
            {"_source_file": f"runs/{directory}__differential_drive/episodes.jsonl"}
        )
        return directory, roster
    if field in {"map", "scenario"}:
        scenario = {"name": value}
        return (
            directory,
            _config._scenario_name_key(scenario),
            scenario_loader._scenario_runtime_identity(scenario),
        )
    raise HardwareIdentityError("receipt identity refused: unknown ID axis")


def canonical_id(value: object, field: str) -> str:
    """Refuse changes under any native normalizer; never rewrite evidence bytes."""
    if (
        not isinstance(value, str)
        or not value
        or any(normalized != value for normalized in _native_id_values(value, field))
    ):
        raise HardwareIdentityError(f"receipt identity refused: noncanonical {field}")
    return value


def canonical_seed(value: object) -> int:
    """Require an exact integer fixed point of the native seed normalizer."""
    from robot_sf.benchmark.camera_ready import _run_state

    try:
        normalized = _run_state._coerce_identifier(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise HardwareIdentityError("receipt identity refused: noncanonical seed") from exc
    if not isinstance(value, int) or type(value) is not int or normalized != value:
        raise HardwareIdentityError("receipt identity refused: noncanonical seed")
    return value


def validate_seed_ids(seeds: list[int]) -> None:
    """Refuse changed native seed IDs and collisions before counting a slice."""
    checked = [canonical_seed(seed) for seed in seeds]
    if len(set(checked)) != len(checked):
        raise HardwareIdentityError("receipt identity refused: normalized seed collision")


def canonical_scenario_seed(scenario: object, seed: object) -> tuple[str, int]:
    """Validate raw and sidecar identities before indexing or comparing links."""
    return canonical_id(scenario, "scenario"), canonical_seed(seed)


def canonical_pair_key(arm: object, scenario: object, seed: object) -> tuple[str, str, int]:
    """Validate all pair axes before dictionary/set equality can collapse aliases."""
    return canonical_id(arm, "arm"), *canonical_scenario_seed(scenario, seed)


def validate_pair_keys(rows: dict) -> None:
    """Check supplied keyed comparisons before Python's permissive key equality."""
    for key in rows:
        if not isinstance(key, tuple) or len(key) != 3:
            raise HardwareIdentityError("receipt identity refused: noncanonical pair key")
        canonical_pair_key(*key)


def validate_scope_ids(scope: dict) -> None:
    """Refuse native projection collisions and noncanonical labels before counts."""
    for field, label in (("arms", "arm"), ("maps", "map")):
        entries = scope.get(field)
        if not isinstance(entries, list):
            raise HardwareIdentityError(f"receipt identity refused: noncanonical scope {field}")
        if not all(isinstance(entry, str) for entry in entries):
            raise HardwareIdentityError(f"receipt identity refused: noncanonical {label}")
        projections = [_native_id_values(entry, label) for entry in entries]
        for normalized in zip(*projections, strict=True):
            if len(set(normalized)) != len(normalized):
                if len(set(entries)) != len(entries):
                    raise HardwareIdentityError(f"equivalence refused: duplicate scope {field}")
                raise HardwareIdentityError(
                    f"receipt identity refused: noncanonical {label}: normalized scope collision"
                )
        for entry in entries:
            canonical_id(entry, label)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HardwareIdentityError(f"hardware comparison invalid: provenance digest {reason}")


def source_files(root: Path) -> list[Path]:
    """Return exactly the comparison inputs, including both producer sidecars."""
    raw = sorted(root.glob("campaigns/**/episodes.jsonl"))
    return sorted(
        {
            *root.glob("episodes_*.jsonl"),
            *root.glob("execution_*.json"),
            *raw,
            *(path.with_suffix(".jsonl.provenance.json") for path in raw),
            *(path.with_suffix(".jsonl.manifest.json") for path in raw),
        }
        - {root / BINDING_FILE, root / "execution_request.json", root / "execution_scheduler.json"}
    )


def verify_digests(root: Path, expected: dict[str, str]) -> None:
    """Run the repository publication digest verifier against unchanged file bytes."""
    from robot_sf.benchmark.artifact_publication import _preflight_check_checksums

    # The repository verifier consumes canonical payload paths. Materialize
    # copies in an isolated directory; neither raw bytes nor seals are rewritten.
    with tempfile.TemporaryDirectory(prefix="receipt-digests-") as temporary:
        staging = Path(temporary)
        for relative in expected:
            path = PurePosixPath(relative)
            _require(
                bool(relative)
                and not path.is_absolute()
                and ".." not in path.parts
                and path.as_posix() == relative,
                "has an unsafe artifact path",
            )
            source = root / relative
            _require(
                source.is_file() and not any(p.is_symlink() for p in (source, *source.parents)),
                "requires regular source artifacts",
            )
            target = staging / "payload" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        checksums = staging / "checksums.sha256"
        checksums.write_text(
            "".join(f"{digest}  payload/{path}\n" for path, digest in expected.items())
        )
        violations: list[str] = []
        _preflight_check_checksums(
            staging,
            {"files": [{"path": path, "sha256": digest} for path, digest in expected.items()]},
            checksums_path=checksums,
            violations=violations,
        )
        _require(not violations, "mismatch in sealed artifacts")


def _producer_claim(claim: dict, source: str, role: str, job: str, node: str) -> None:
    """Check selected producer claims, without confusing checkpoint/input origins."""
    from robot_sf._execution_context import execution_context_digest
    from scripts.ci.behaviour_receipt import validate_source_row_context

    contexts = [claim]
    for parent in (claim, claim.get("algorithm_metadata", {}), claim.get("result_provenance", {})):
        if isinstance(parent, dict) and isinstance(parent.get("execution_context"), dict):
            contexts.append(parent["execution_context"])
    for context in contexts:
        for field in ("repo_commit", "git_hash", "source_sha", "source_commit"):
            _require(
                field not in context or context[field] == source,
                "producer source contradicts claimed commit",
            )
        for field in ("receipt_arm", "arm_role"):
            _require(
                field not in context or context[field] == role,
                "producer arm role contradicts binding",
            )
        for field in ("job_id", "slurm_job_id"):
            _require(
                field not in context or context[field] == job,
                "producer job contradicts scheduler binding",
            )
        if "execution_context_sha256" in context:
            _require(
                context["execution_context_sha256"]
                == execution_context_digest(
                    {
                        k: v
                        for k, v in context.items()
                        if k not in {"hostname", "execution_context_sha256"}
                    }
                ),
                "producer execution context digest differs",
            )
    try:
        validate_source_row_context(
            {"scheduler_job_id": job, "scheduler_node": node, **claim}, job, node
        )
    except HardwareIdentityError as exc:
        raise HardwareIdentityError(
            "hardware comparison invalid: producer execution context disagrees with scheduler"
        ) from exc


def _producer_identity(
    sidecar: dict,
    raw_rows: list[dict],
    source: str,
    role: str,
    job: str,
    node: str,
    raw_path: Path | None = None,
) -> list[dict]:
    run = sidecar["run"]
    _require(
        run.get("runner") == "map_runner.run_map_batch"
        and sidecar["completeness"].get("status") == "complete"
        and all(
            sidecar["campaign_identity"].get(field) == len(raw_rows)
            for field in ("written", "total_jobs")
        ),
        "producer execution is incomplete or has inconsistent counts",
    )
    _require(run.get("repo_commit") == source, "producer source contradicts claimed commit")
    _producer_claim(run, source, role, job, node)
    context = run.get("execution_context")
    _require(
        isinstance(context, dict)
        and context.get("hostname") == node
        and bool(context.get("cpu_model")),
        "producer execution context differs from scheduler",
    )
    if not isinstance(context, dict):
        raise HardwareIdentityError(
            "hardware comparison invalid: producer execution context missing"
        )
    _producer_claim(context, source, role, job, node)
    _require("execution_context_sha256" in context, "producer execution context digest missing")
    links = sidecar["rows"]
    _require(len(links) == len(raw_rows), "producer row link coverage differs")
    sealed_rows = []
    from scripts.ci import behaviour_receipt as gate
    from scripts.ci import receipt_arm_execution

    for index, (link, raw) in enumerate(zip(links, raw_rows, strict=True)):
        _require(
            type(link.get("jsonl_line")) is int
            and link["jsonl_line"] == index
            and canonical_scenario_seed(link.get("scenario_id"), link.get("seed"))
            == canonical_scenario_seed(raw.get("scenario_id"), raw.get("seed"))
            and link.get("repo_commit") == source,
            "producer row source or link differs",
        )
        _producer_claim(link, source, role, job, node)
        _require(raw.get("git_hash") == source, "producer raw source contradicts claimed commit")
        if raw_path is not None:
            raw = receipt_arm_execution._from_sealed_producer(
                raw, sidecar, raw_path, source, gate.ROOT
            )
        expected = sidecar["campaign_identity"].get("algorithm")
        arm = raw_path.parent.name.split("__", 1)[0] if raw_path is not None else ""
        if (
            arm in receipt_arm_execution.PROFILES
            and raw.get("scenario_id") == receipt_arm_execution.SCENARIO
        ):
            expected = receipt_arm_execution.expected_algorithm(
                arm,
                canonical_id(raw.get("scenario_id"), "scenario"),
                expected,
                row=raw,
                source_sha=source,
                repo_root=gate.ROOT,
            )
        _require(
            raw.get("algo") == expected,
            "producer raw execution algorithm contradicts sealed campaign",
        )
        _producer_claim(raw, source, role, job, node)
        provenance = raw.get("result_provenance")
        if isinstance(provenance, dict):
            _producer_claim(provenance, source, role, job, node)
        sealed_rows.append(raw)
    return sealed_rows


def verify_producer_seals(
    root: Path,
    *,
    source_sha: str | None = None,
    arm_role: str | None = None,
    job: str | None = None,
    node: str | None = None,
) -> dict[Path, list[dict]]:
    """Require both producer seals before returning source-attested decoded rows."""
    from robot_sf.benchmark.result_provenance import validate_result_provenance_manifest

    expected = {}
    sealed_rows = {}
    try:
        for raw in sorted(root.glob("campaigns/**/episodes.jsonl")):
            sidecar = json.loads(raw.with_suffix(".jsonl.provenance.json").read_text())
            validate_result_provenance_manifest(sidecar)
            manifest = json.loads(raw.with_suffix(".jsonl.manifest.json").read_text())
            for entries in (
                sidecar["raw_artifacts"],
                manifest["simulation_run_provenance"]["outputs"],
            ):
                matches = [
                    entry for entry in entries if Path(entry.get("path", "")).name == raw.name
                ]
                _require(len(matches) == 1, "requires a unique raw artifact seal")
                entry = matches[0]
                digest = entry.get("sha256")
                _require(
                    entry.get("artifact_status") == "available"
                    and isinstance(digest, str)
                    and len(digest) == 64
                    and all(c in "0123456789abcdef" for c in digest),
                    "requires an available raw artifact seal",
                )
                relative = raw.relative_to(root).as_posix()
                _require(
                    relative not in expected or expected[relative] == digest, "sidecars disagree"
                )
                expected[relative] = digest
        if expected:
            verify_digests(root, expected)
        if source_sha is not None:
            _require(
                arm_role in {"head", "baseline"} and job is not None and node is not None,
                "producer identity claim missing",
            )
            for raw in sorted(root.glob("campaigns/**/episodes.jsonl")):
                sidecar = json.loads(raw.with_suffix(".jsonl.provenance.json").read_text())
                rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
                if arm_role is not None and job is not None and node is not None:
                    from scripts.ci import receipt_arm_execution

                    arm = raw.parent.name.split("__", 1)[0]
                    if arm in receipt_arm_execution.PROFILES and any(
                        row.get("scenario_id") == receipt_arm_execution.SCENARIO for row in rows
                    ):
                        sidecar = receipt_arm_execution._proof_json_loads(
                            raw.with_suffix(".jsonl.provenance.json").read_text()
                        )
                        receipt_arm_execution._proof_json_loads(
                            raw.with_suffix(".jsonl.manifest.json").read_text()
                        )
                        rows = [
                            receipt_arm_execution._proof_json_loads(line)
                            for line in raw.read_text().splitlines()
                            if line.strip()
                        ]
                    sealed_rows[raw.resolve()] = _producer_identity(
                        sidecar, rows, source_sha, arm_role, job, node, raw
                    )
    except HardwareIdentityError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HardwareIdentityError(
            f"hardware comparison invalid: provenance digest verification failed ({type(exc).__name__})"
        ) from exc
    return sealed_rows


def write_binding(root: Path, job: str, node: str) -> None:
    """Seal a new, separate allocation binding after successful execution."""
    request = json.loads((root / "execution_request.json").read_text())
    source, role = request["source_sha"], request["arm_role"]
    _require(role in {"head", "baseline"}, "producer arm role missing")
    verify_producer_seals(root, source_sha=source, arm_role=role, job=job, node=node)
    files = source_files(root)
    binding = {
        "schema_version": "behaviour-allocation-binding.v2",
        "source_sha": source,
        "arm_role": role,
        "job_id": job,
        "node": node,
        "files": {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    with (root / BINDING_FILE).open("x", encoding="utf-8") as handle:
        json.dump(binding, handle, sort_keys=True)
        handle.write("\n")


def verify_binding(
    root: Path, job: str, node: str, *, source_sha: str | None = None, arm_role: str | None = None
) -> dict[Path, list[dict]]:
    """Verify custody before returning rows attested by both producer seals."""
    try:
        _require(not (root / BINDING_FILE).is_symlink(), "binding is not regular")
        binding = json.loads((root / BINDING_FILE).read_text())
        _require(
            binding.get("schema_version") == "behaviour-allocation-binding.v2"
            and binding.get("job_id") == job
            and binding.get("node") == node,
            "allocation binding differs from scheduler",
        )
        source = binding.get("source_sha") if source_sha is None else source_sha
        role = binding.get("arm_role") if arm_role is None else arm_role
        _require(
            isinstance(source, str)
            and len(source) == 40
            and all(c in "0123456789abcdef" for c in source)
            and role in {"head", "baseline"}
            and binding.get("source_sha") == source
            and binding.get("arm_role") == role,
            "producer source or arm role differs from receipt claim",
        )
        expected = binding["files"]
        _require(isinstance(expected, dict) and bool(expected), "binding has no artifacts")
        _require(
            set(expected) == {p.relative_to(root).as_posix() for p in source_files(root)},
            "binding omits comparison inputs",
        )
        verify_digests(root, expected)
        return verify_producer_seals(root, source_sha=source, arm_role=role, job=job, node=node)
    except HardwareIdentityError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HardwareIdentityError(
            f"hardware comparison invalid: provenance digest verification failed ({type(exc).__name__})"
        ) from exc


def verify_optional_binding(root: Path, job: str | None, node: str | None) -> None:
    """Audit helpers may read unbound historical rows; allocation reads require a seal."""
    if job is None:
        return
    _require(node is not None, "verified allocation node missing")
    if node is not None:
        verify_binding(root, job, node)


def verified_raw_rows(
    root: Path, job: str, node: str, *, source_sha: str, arm_role: str
) -> dict[tuple[str, str, int], dict]:
    """Decode sealed admission sources once, then apply allocation context in memory."""
    from scripts.ci import make_behaviour_receipt as producer
    from scripts.ci import receipt_outcomes
    from scripts.ci.behaviour_receipt import validate_source_row_context

    sealed = verify_binding(root, job, node, source_sha=source_sha, arm_role=arm_role)
    producer._verify_sweep_metadata(root, source_sha)
    rows = {}
    locations = {}
    for path in sorted(root.glob("campaigns/**/episodes.jsonl")):
        arm = path.parent.name.split("__")[0]
        for row in sealed[path.resolve()]:
            key = canonical_pair_key(arm, row["scenario_id"], row["seed"])
            _require(key not in rows, "duplicate source row")
            row.setdefault("scheduler_job_id", job)
            row.setdefault("scheduler_node", node)
            validate_source_row_context(row, job, node)
            rows[key] = row
            locations[key] = path.resolve()
    _require(bool(rows), "source rows missing")
    seen = set()
    for path in sorted(root.glob("episodes_*.jsonl")):
        suite = path.stem.removeprefix("episodes_")
        campaign = root / "campaigns" / f"empty_world_{suite}"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            summary = json.loads(line)
            key = canonical_pair_key(summary["arm"], summary["scenario"], summary["seed"])
            source_path = (campaign / summary["source_file"]).resolve()
            _require(
                source_path.is_relative_to(campaign.resolve())
                and key in rows
                and locations[key] == source_path
                and key not in seen,
                "producer summary/raw coverage or source differs",
            )
            raw = rows[key]
            summary.setdefault("scheduler_job_id", job)
            summary.setdefault("scheduler_node", node)
            validate_source_row_context(summary, job, node)
            _require(
                receipt_outcomes.raw_outcomes(raw) == receipt_outcomes.summary_outcomes(summary),
                "producer summary/raw outcomes differ",
            )
            _producer_claim(summary, source_sha, arm_role, job, node)
            _require(
                "execution_status" not in raw
                or raw["execution_status"] == summary["execution_status"],
                "producer raw execution status contradicts summary",
            )
            raw["execution_status"] = summary["execution_status"]
            seen.add(key)
    _require(seen == set(rows), "producer summary/raw coverage differs")
    return rows
