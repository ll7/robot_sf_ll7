"""Read-only frozen train evidence and promotion checks; never merge or publish.

The existing behaviour validator remains the owner of scientific coverage and
classification rules. This module adds immutable transport and identity binding.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

from scripts.ci import behaviour_receipt as gate

TRAIN_STATES = ("admitted", "WAITING_FOR_TRAIN", "candidate", "executing", "blocked", "ready")
MAX_ARCHIVE_BYTES = 40 * 1024 * 1024


def digest(value: object) -> str:
    """Hash canonical JSON, shared by frozen manifests and evidence envelopes."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def policy_digest() -> str:
    """Bind the trusted validator, schema and train identity policy bytes."""
    root = Path(__file__).parent
    return digest(
        {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "behaviour_receipt.py",
                "behaviour_receipt.schema.json",
                "frozen_candidate.py",
            )
        }
    )


def git(*args: str) -> str:
    """Read exact Git objects from the gate's selected repository."""
    return subprocess.check_output(["git", *args], cwd=gate.ROOT, text=True).strip()


def require(condition: object, reason: str) -> None:
    """Refuse evidence without silently repairing identity mismatches."""
    if not condition:
        raise gate.TrainEvidenceError(reason)


def validate_manifest(manifest: dict, head: str, base: str) -> None:
    """Verify the frozen source tree and original component review references."""
    require(manifest["schema_version"] == "behaviour-candidate.v1", "manifest schema")
    require(manifest["candidate_sha"] == head, "candidate SHA mismatch")
    require(manifest["base_sha"] == base, "expected base mismatch")
    require(gate._digest(head, 40) and gate._digest(base, 40), "invalid candidate/base SHA")
    require(
        manifest["candidate_tree"] == git("rev-parse", f"{head}^{{tree}}"),
        "candidate tree mismatch",
    )
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, head],
            cwd=gate.ROOT,
            capture_output=True,
            check=False,
        ).returncode
        == 0,
        "base is not candidate ancestor",
    )
    components = manifest["components"]
    require(isinstance(components, list) and bool(components), "component reviews missing")
    seen = set()
    for component in components:
        number, source = component["pr"], component["head_sha"]
        require(
            type(number) is int and number > 0 and number not in seen,
            "duplicate/invalid component PR",
        )
        seen.add(number)
        require(gate._digest(source, 40), "invalid component head")
        # Preserve original PR heads; squash assembly needs a separately reviewed mapping.
        require(
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", source, head],
                cwd=gate.ROOT,
                capture_output=True,
                check=False,
            ).returncode
            == 0,
            "component head not in candidate",
        )
        require(bool(component["reviews"]), "component reviews missing")
        for review in component["reviews"]:
            require(
                review["head_sha"] == source
                and review["verdict"] == "accepted"
                and gate._uri(review["uri"]),
                "component review head/verdict mismatch",
            )
    for key in ("suite_sha256", "config_sha256", "baseline_sha256", "evaluation_policy_sha256"):
        require(gate._digest(manifest[key], 64), "invalid manifest digest")


def validate_evidence(
    envelope: dict,
    reference: dict,
    head: str,
    base: str,
    scope: dict,
    release: str,
    baseline_source: str,
) -> dict:
    """Bind all rows and execution identities before applying the unchanged gate."""
    require(envelope["schema_version"] == "behaviour-candidate-evidence.v1", "evidence schema")
    manifest, receipt = envelope["manifest"], envelope["receipt"]
    require(digest(manifest) == reference["manifest_sha256"], "manifest digest mismatch")
    validate_manifest(manifest, head, base)
    require(manifest["suite_sha256"] == digest(scope), "suite digest mismatch")
    require(manifest["config_sha256"] == digest(envelope["config"]), "config digest mismatch")
    require(manifest["baseline_sha256"] == digest(receipt["baseline"]), "baseline digest mismatch")
    require(
        manifest["evaluation_policy_sha256"] == policy_digest(), "evaluation policy digest mismatch"
    )
    require(envelope["receipt_sha256"] == digest(receipt), "receipt digest mismatch")
    executions = envelope["episode_executions"]
    require(
        envelope["episode_executions_sha256"] == digest(executions), "execution digest mismatch"
    )
    require(isinstance(executions, list), "invalid episode execution inventory")
    keys = [(row["arm"], row["map"], row["seed"]) for row in executions]
    expected = {(row["arm"], row["map"], row["seed"]) for row in receipt["rows"]}
    require(
        len(keys) == len(set(keys)) and set(keys) == expected, "episode execution coverage mismatch"
    )
    require(
        all(
            row["source_sha"] == head and gate._digest(row["execution_sha256"], 64)
            for row in executions
        ),
        "episode execution source/identity mismatch",
    )
    require(
        receipt["scheduler"]["source_sha"] == head
        and receipt["interaction_audit"]["source_sha"] == head,
        "external evidence source must equal candidate",
    )
    import jsonschema

    try:
        gate.validate_receipt(receipt, scope, head, release, baseline_source)
    except jsonschema.ValidationError as exc:
        raise gate.TrainEvidenceError("receipt schema mismatch") from exc
    except ValueError as exc:
        raise gate.TrainEvidenceError(str(exc)) from exc
    return envelope


def load_external(reference: dict, repo: str, head: str) -> dict:
    """Fetch a same-repository immutable Actions artifact by ID, never by name/URL."""
    require(
        set(reference) == {"artifact_id", "sha256", "manifest_sha256"}, "external reference fields"
    )
    require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo), "invalid repository")
    artifact_id = reference["artifact_id"]
    require(type(artifact_id) is int and artifact_id > 0, "invalid artifact ID")
    require(
        all(gate._digest(reference[k], 64) for k in ("sha256", "manifest_sha256")),
        "invalid artifact digest",
    )
    endpoint = f"repos/{repo}/actions/artifacts/{artifact_id}"
    metadata = json.loads(subprocess.check_output(["gh", "api", endpoint]))
    require(
        metadata["id"] == artifact_id and metadata["expired"] is False,
        "artifact expired or wrong ID",
    )
    require(metadata["workflow_run"]["head_sha"] == head, "artifact candidate SHA mismatch")
    require(
        metadata["digest"] == "sha256:" + reference["sha256"], "artifact metadata digest mismatch"
    )
    require(0 < metadata["size_in_bytes"] <= MAX_ARCHIVE_BYTES, "artifact archive size limit")
    raw = subprocess.check_output(["gh", "api", endpoint + "/zip"])
    require(
        len(raw) <= MAX_ARCHIVE_BYTES and hashlib.sha256(raw).hexdigest() == reference["sha256"],
        "artifact byte digest mismatch",
    )
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(
            len(entries) == 1 and entries[0].filename == "evidence.json",
            "artifact must contain only evidence.json",
        )
        require(entries[0].file_size <= gate.MAX_ROWS_BYTES, "evidence size limit")
        return json.loads(archive.read(entries[0]))


@dataclass(frozen=True)
class PromotionIdentity:
    """The source and evidence identities independently supplied by the promoter."""

    head: str
    expected_base: str
    current_main: str
    landed: str
    receipt_sha256: str
    executions_sha256: str


def check_promotion(
    envelope: dict,
    reference: dict,
    identity: PromotionIdentity,
    scope: dict,
    release: str,
    baseline_source: str,
) -> dict:
    """Check a proposed landed Git object; leave the atomic ref update to its owner."""
    head, expected_base = identity.head, identity.expected_base
    current_main, landed = identity.current_main, identity.landed
    receipt_sha256, executions_sha256 = identity.receipt_sha256, identity.executions_sha256
    require(current_main == expected_base, "main moved from expected base")
    validate_evidence(envelope, reference, head, expected_base, scope, release, baseline_source)
    require(gate._digest(landed, 40), "invalid landed SHA")
    require(
        git("rev-parse", f"{landed}^{{tree}}") == envelope["manifest"]["candidate_tree"],
        "landed tree mismatch",
    )
    require(receipt_sha256 == envelope["receipt_sha256"], "promotion receipt digest mismatch")
    require(
        executions_sha256 == envelope["episode_executions_sha256"],
        "promotion execution digest mismatch",
    )
    # A new integration head is a new candidate. A landed mapping is not a verdict transfer.
    return {
        "state": "ready",
        "candidate_sha": head,
        "expected_base": expected_base,
        "landed_sha": landed,
        "landed_tree": envelope["manifest"]["candidate_tree"],
        "receipt_sha256": receipt_sha256,
        "episode_executions_sha256": executions_sha256,
        "execution_source_sha": head,
        "merge_authorized": False,
    }


def write_external(path: Path, payload: dict) -> None:
    """Create evidence once, outside the checkout, and fsync before reporting success."""
    require(
        not path.resolve().is_relative_to(gate.ROOT.resolve()),
        "evidence must be outside source checkout",
    )
    with path.open("x") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def assemble_main(argv: list[str] | None = None) -> int:
    """Assemble already executed and independently reviewed evidence without a commit."""
    parser = argparse.ArgumentParser(description=assemble_main.__doc__)
    for name in ("manifest", "receipt", "config", "executions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text())
        receipt = json.loads(args.receipt.read_text())
        executions = json.loads(args.executions.read_text())
        envelope = {
            "schema_version": "behaviour-candidate-evidence.v1",
            "manifest": manifest,
            "receipt": receipt,
            "config": json.loads(args.config.read_text()),
            "episode_executions": executions,
            "receipt_sha256": digest(receipt),
            "episode_executions_sha256": digest(executions),
        }
        reference = {"manifest_sha256": digest(manifest)}
        release = gate.latest_release(args.repo)
        validate_evidence(
            envelope,
            reference,
            manifest["candidate_sha"],
            manifest["base_sha"],
            json.loads(gate.SCOPE_PATH.read_text()),
            release,
            gate.release_source(release),
        )
        write_external(args.output, envelope)
    except FileExistsError:
        print(json.dumps({"state": "blocked", "reason": "evidence already exists"}))
        return 1
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"state": "blocked", "reason": str(exc)}))
        return 1
    print(
        json.dumps(
            {
                "state": "ready",
                "candidate_sha": manifest["candidate_sha"],
                "manifest_sha256": digest(manifest),
                "receipt_sha256": digest(receipt),
                "episode_executions_sha256": digest(executions),
                "merge_authorized": False,
            }
        )
    )
    return 0


def freeze_main(argv: list[str] | None = None) -> int:
    """Write an immutable manifest outside the source checkout before execution."""
    parser = argparse.ArgumentParser(description=freeze_main.__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expected-base", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.input.read_text())
        validate_manifest(manifest, args.candidate, args.expected_base)
        require(
            manifest["evaluation_policy_sha256"] == policy_digest(),
            "evaluation policy digest mismatch",
        )
        write_external(args.output, manifest)
    except FileExistsError:
        print(json.dumps({"state": "blocked", "reason": "frozen manifest already exists"}))
        return 1
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"state": "blocked", "reason": str(exc)}))
        return 1
    print(
        json.dumps(
            {
                "state": "candidate",
                "candidate_sha": args.candidate,
                "manifest_sha256": digest(manifest),
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Verify frozen evidence or promotion locally; perform no GitHub writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expected-base", required=True)
    parser.add_argument("--landed", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--executions-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        reference = json.loads(args.reference.read_text())
        envelope = load_external(reference, args.repo, args.candidate)
        release = gate.latest_release(args.repo)
        identity = PromotionIdentity(
            args.candidate,
            args.expected_base,
            json.loads(
                subprocess.check_output(["gh", "api", f"repos/{args.repo}/git/ref/heads/main"])
            )["object"]["sha"],
            args.landed,
            args.receipt_sha256,
            args.executions_sha256,
        )
        result = check_promotion(
            envelope,
            reference,
            identity,
            json.loads(gate.SCOPE_PATH.read_text()),
            release,
            gate.release_source(release),
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
    ) as exc:
        print(json.dumps({"state": "blocked", "reason": str(exc), "merge_authorized": False}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
