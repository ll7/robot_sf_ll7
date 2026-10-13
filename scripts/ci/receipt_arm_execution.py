"""Derive the finite frozen hybrid dispatch allowance from seal-verified sources.

Proof is a Python attribute, never a JSON field. Only the producer seal verifier
mints it; archive admission re-derives it before restoring projected attributes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath

import yaml

from scripts.ci.receipt_provenance import HardwareIdentityError

SCENARIO = "francis2023_leave_group"
DECLARED_ALGORITHM = "hybrid_rule_local_planner"
PROFILES = {
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4": (
        "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4_s30_h600_release_0_0_8_frozen.yaml",
        "28aafe6352543e4c4d24a146bd3ba7260105362052505af9308f6d45d9d94458",
    ),
    "scenario_adaptive_hybrid_orca_v2_collision_guard_v4": (
        "configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard_v4_s30_h600_release_0_0_8_frozen.yaml",
        "54435e7fd39e592d4684fe40f804ab1c080f39925dceb0781dd68c1ce172ef94",
    ),
}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HardwareIdentityError(f"receipt arm execution proof invalid: {reason}")


@dataclass(frozen=True)
class _Proof:
    arm: str
    scenario: str
    source: str
    path: str
    sha256: str
    algorithm: str


class _VerifiedExecution(dict):
    """A source-verifier-only attribute survives copies without changing JSON."""

    __slots__ = ("_receipt_execution",)

    def __init__(self, row: dict, proof: _Proof) -> None:
        super().__init__(row)
        self._receipt_execution = proof


def carry_execution_proof(source: dict, replacement: dict) -> dict:
    """Copy only an already minted verifier attribute, never a JSON proof key."""
    if isinstance(source, _VerifiedExecution):
        return _VerifiedExecution(replacement, source._receipt_execution)
    return replacement


@lru_cache(maxsize=128)
def _profile_algorithm(repo: str, source: str, path: str, sha256: str) -> str:
    """Read the immutable regular Git blob at a full claimed commit point."""
    _require(
        len(source) == 40 and all(c in "0123456789abcdef" for c in source),
        "source must be a full commit SHA",
    )
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--verify", f"{source}^{{commit}}"], cwd=repo
            )
            .decode()
            .strip()
        )
        _require(commit == source, "source is not the claimed commit point")
        tree = subprocess.check_output(["git", "ls-tree", "-z", source, "--", path], cwd=repo)
        entries = tree.rstrip(b"\0").split(b"\0")
        _require(len(entries) == 1 and b"\t" in entries[0], "profile Git blob missing")
        header, name = entries[0].split(b"\t", 1)
        parts = header.split()
        _require(
            len(parts) == 3 and parts[:2] == [b"100644", b"blob"] and name == path.encode(),
            "profile is not the exact regular 100644 Git blob",
        )
        blob = subprocess.check_output(["git", "cat-file", "blob", parts[2].decode()], cwd=repo)
    except (OSError, subprocess.CalledProcessError) as error:
        raise HardwareIdentityError(
            "receipt arm execution proof invalid: source Git blob unavailable"
        ) from error
    _require(hashlib.sha256(blob).hexdigest() == sha256, "published profile Git bytes changed")
    profile = yaml.safe_load(blob)
    if not isinstance(profile, dict):
        raise HardwareIdentityError(
            "receipt arm execution proof invalid: published profile mapping missing"
        )
    _require(profile.get("algo") == DECLARED_ALGORITHM, "published default algorithm differs")
    overrides = profile.get("scenario_algo_overrides")
    if not isinstance(overrides, dict):
        raise HardwareIdentityError(
            "receipt arm execution proof invalid: published scenario override missing"
        )
    _require(set(overrides) == {SCENARIO}, "published scenario override differs")
    override = overrides[SCENARIO]
    _require(
        isinstance(override, dict) and override.get("algo") == "orca",
        "published execution algorithm differs",
    )
    return override["algo"]


def _orca_execution_identity(row: dict) -> None:
    """Bind the finite explicit identity leaves emitted by the real ORCA producer."""
    metadata = row.get("algorithm_metadata")
    _require(
        isinstance(metadata, dict) and metadata.get("algorithm") == "orca",
        "explicit ORCA execution identity differs: algorithm_metadata.algorithm",
    )
    _require(
        "algorithm" not in row or row["algorithm"] == "orca",
        "explicit ORCA execution identity differs: algorithm",
    )
    identities = {
        ("algorithm_metadata", "canonical_algorithm"): "orca",
        ("algorithm_metadata", "planner_contract", "planner_id"): "orca",
        ("algorithm_metadata", "planner_kinematics", "adapter_name"): "ORCAPlannerAdapter",
        ("algorithm_metadata", "planner_runtime", "planner_type"): "ORCAPlannerAdapter",
        ("algorithm_metadata", "policy_semantics"): "orca_adapter",
        (
            "algorithm_metadata",
            "paired_effect_metric_producer",
            "pairing",
            "key",
            "planner",
        ): "orca",
        ("event_ledger", "planner"): "orca",
        ("scenario_params", "algo"): "orca",
    }
    for path, expected in identities.items():
        current = row
        for component in path:
            if not isinstance(current, dict):
                _require(False, "explicit ORCA execution identity differs: " + ".".join(path))
                break
            _require(
                component in current, "explicit ORCA execution identity differs: " + ".".join(path)
            )
            if component not in current:
                break
            current = current[component]
        else:
            _require(
                current == expected, "explicit ORCA execution identity differs: " + ".".join(path)
            )


def _proof_json_loads(raw: str) -> dict:
    """Reject ambiguous object keys only for the new finite attestation inputs."""

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key in sealed source")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=unique_pairs)


def _from_sealed_producer(
    row: dict, sidecar: dict, raw_path: Path, source: str, repo_root: Path
) -> dict:
    """Mint only inside the producer verifier after both raw seals were checked."""
    from scripts.ci.receipt_provenance import native_row_key

    arm, scenario, _seed = native_row_key(row, raw_path)
    if arm not in PROFILES or scenario != SCENARIO:
        return row
    _require(
        raw_path.name == "episodes.jsonl"
        and raw_path.parent.name == f"{arm}__differential_drive"
        and not raw_path.is_symlink(),
        "raw arm directory differs",
    )
    _require(
        row.get("git_hash") == source and sidecar.get("run", {}).get("repo_commit") == source,
        "producer source differs",
    )
    _require(
        sidecar.get("campaign_identity", {}).get("algorithm") == DECLARED_ALGORITHM,
        "declared arm algorithm differs",
    )
    inputs = sidecar.get("inputs")
    config = inputs.get("algo_config") if isinstance(inputs, dict) else None
    if not isinstance(config, dict):
        raise HardwareIdentityError(
            "receipt arm execution proof invalid: producer config proof missing"
        )
    path, sha256 = PROFILES[arm]
    observed = config.get("path")
    _require(
        isinstance(observed, str)
        and "://" not in observed
        and ".." not in PurePosixPath(observed).parts
        and PurePosixPath(observed).parts[-len(PurePosixPath(path).parts) :]
        == PurePosixPath(path).parts,
        "producer profile path differs from raw arm",
    )
    _require(
        config.get("artifact_status") == "available" and config.get("sha256") == sha256,
        "producer profile unavailable or SHA differs",
    )
    algorithm = _profile_algorithm(str(repo_root.resolve()), source, path, sha256)
    _orca_execution_identity(row)
    return _VerifiedExecution(row, _Proof(arm, SCENARIO, source, path, sha256, algorithm))


def expected_algorithm(
    arm: str, scenario: str, declared: str, *, row: dict, source_sha: str, repo_root: Path
) -> str:
    """Return the canonical default, or the exact independently attested dispatch."""
    from robot_sf.benchmark.algorithm_metadata import canonical_algorithm_name

    algorithm = canonical_algorithm_name(declared)
    if arm in PROFILES:
        _require(algorithm == DECLARED_ALGORITHM, "header cannot redefine a frozen hybrid arm")
    if arm not in PROFILES or scenario != SCENARIO:
        return algorithm
    _require(
        isinstance(row, _VerifiedExecution),
        "sealed source proof missing; JSON claims cannot attest dispatch",
    )
    if not isinstance(row, _VerifiedExecution):
        raise HardwareIdentityError(
            "receipt arm execution proof invalid: sealed source proof missing"
        )
    proof = row._receipt_execution
    _require(
        proof.arm == arm and proof.scenario == scenario and proof.source == source_sha,
        "attested arm/map/source differs",
    )
    path, sha256 = PROFILES[arm]
    _require((proof.path, proof.sha256) == (path, sha256), "attested profile differs")
    return _profile_algorithm(str(repo_root.resolve()), source_sha, path, sha256)
