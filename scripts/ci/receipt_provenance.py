"""Verify producer seals and bind allocations without changing producer artifacts."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath

BINDING_FILE = "execution_binding.json"


class HardwareIdentityError(ValueError):
    """Hardware or provenance evidence cannot support an outcome comparison."""


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


def verify_producer_seals(root: Path) -> None:
    """Require the raw digest in both normal producer sidecars to verify."""
    from robot_sf.benchmark.result_provenance import validate_result_provenance_manifest

    expected = {}
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
    except HardwareIdentityError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        _require(False, f"verification failed ({type(exc).__name__})")


def write_binding(root: Path, job: str, node: str) -> None:
    """Seal a new, separate allocation binding after successful execution."""
    verify_producer_seals(root)
    files = source_files(root)
    binding = {
        "schema_version": "behaviour-allocation-binding.v1",
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


def verify_binding(root: Path, job: str, node: str) -> None:
    """Verify allocation binding coverage and both original producer digest seals."""
    try:
        _require(not (root / BINDING_FILE).is_symlink(), "binding is not regular")
        binding = json.loads((root / BINDING_FILE).read_text())
        _require(
            binding.get("schema_version") == "behaviour-allocation-binding.v1"
            and binding.get("job_id") == job
            and binding.get("node") == node,
            "allocation binding differs from scheduler",
        )
        expected = binding["files"]
        _require(isinstance(expected, dict) and bool(expected), "binding has no artifacts")
        _require(
            set(expected) == {p.relative_to(root).as_posix() for p in source_files(root)},
            "binding omits comparison inputs",
        )
        verify_digests(root, expected)
        verify_producer_seals(root)
    except HardwareIdentityError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        _require(False, f"verification failed ({type(exc).__name__})")


def verify_optional_binding(root: Path, job: str | None, node: str | None) -> None:
    """Audit helpers may read unbound historical rows; allocation reads require a seal."""
    if job is None:
        return
    _require(node is not None, "verified allocation node missing")
    if node is not None:
        verify_binding(root, job, node)


def verified_raw_rows(root: Path, job: str, node: str) -> dict[tuple[str, str, int], dict]:
    """Decode sealed admission sources once, then apply allocation context in memory."""
    from scripts.ci.behaviour_receipt import validate_source_row_context

    verify_binding(root, job, node)
    rows = {}
    for path in sorted(root.glob("campaigns/**/episodes.jsonl")):
        arm = path.parent.name.split("__")[0]
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (arm, str(row["scenario_id"]), int(row["seed"]))
            _require(key not in rows, "duplicate source row")
            row.setdefault("scheduler_job_id", job)
            row.setdefault("scheduler_node", node)
            validate_source_row_context(row, job, node)
            rows[key] = row
    _require(bool(rows), "source rows missing")
    return rows
