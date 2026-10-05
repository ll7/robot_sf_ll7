#!/usr/bin/env python3
# AI-GENERATED NEEDS-REVIEW
"""Preserve exact F2 resolver outputs and propose a DOI-bound seed admission."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from robot_sf.evidence.writers import write_json

FREEZE = "66f402ba176b13e45210d0da0b2cf20fcdc0cc02"
ANCHOR_SOURCE = "527eb2509c4375d6ce81091d7d67260d35844310"
ANCHOR_PATH = "docs/context/evidence/2026-10-04_freeze008_f2_calibration/anchors.v2.0.acquired.json"
ANCHOR_SHA = "8d86636bcb33a27bab6ba97318516145112aaebe4a4a9713665ec2e39fbc7349"
RECEIPT_SHA = "cd29d6c9a3213e4dbfabc1b8b8c23305066a6a39c5eb088f54d8f8d539827464"
CONFIG_HASH = "6cd6a57f3583d150d51ac52b7a4a4fd80fa7b077795ac3c7973ef643d3f8fbbd"
REVIEW_REF = "freeze008-box4-independent-review-2026-10-05"
DIGESTS = {
    "main/release_identity.resolved.json": "527f9dc5e9ee3004e93444789472eb25a29438e64741c4c4b7b95a10f0db3e71",
    "doorway/release_identity.resolved.json": "6fce4eb41436213a2c4d9790b74811b3492126297ea654ef89a1f3d94e17e199",
    "main/zenodo_metadata.resolved.json": "e21b48b69916e2ab669759eb485a540b2d3c398d59be3496c6bb1e28fba7790e",
    "doorway/zenodo_metadata.resolved.json": "819335ea5b096d8592b0d5c4c71408e4e3649f39632fd8e165d176480ff4ce37",
}


def sha(data):
    """Hash exact bytes without rewriting resolver output."""
    return hashlib.sha256(data).hexdigest()


def check_identity(root, source, target, kind, d):
    """Require source/DOI/config bindings and preserve its notes receipt."""
    if d["source_commit"] != FREEZE or d["determinism_receipt"]["sha256"] != RECEIPT_SHA:
        raise ValueError("identity source/receipt mismatch")
    for key, expected in [
        ("concept_doi", "10.5281/zenodo.23150471"),
        ("version_doi", "10.5281/zenodo.23150472"),
    ]:
        if d["publication"][key] != expected:
            raise ValueError("reserved DOI mismatch")
    m = d["resolved_manifest"]
    config = subprocess.check_output(
        ["git", "-C", str(root), "show", FREEZE + ":" + m["canonical_campaign_config"]]
    )
    if sha(config) != m["canonical_campaign_config_sha256"]:
        raise ValueError("frozen campaign config mismatch")
    receipt = source / kind / "release_notes_gate.v1.json"
    shutil.copyfile(receipt, target / kind / receipt.name)


def main():
    """Validate frozen custody and preserve the proposed review packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--identity-root", type=Path, default=Path("output/release-008"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    source = args.identity_root if args.identity_root.is_absolute() else root / args.identity_root
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    identities = {}
    for relative, expected in DIGESTS.items():
        data = (source / relative).read_bytes()
        if sha(data) != expected:
            raise ValueError("resolver output digest mismatch: " + relative)
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        if relative.endswith("release_identity.resolved.json"):
            identities[relative.split("/")[0]] = json.loads(data)
    for kind in ("main", "doorway"):
        check_identity(root, source, target, kind, identities[kind])
    acquired = subprocess.check_output(
        ["git", "-C", str(root), "show", ANCHOR_SOURCE + ":" + ANCHOR_PATH]
    )
    if (
        sha(acquired) != ANCHOR_SHA
        or json.loads(acquired)["calibration"]["campaign_config_hash"] != CONFIG_HASH
    ):
        raise ValueError("reviewed acquired anchor mismatch")
    seeds = identities["main"]["resolved_manifest"]["seed_policy"]["resolved_seeds"]
    if (
        seeds != identities["doorway"]["resolved_manifest"]["seed_policy"]["resolved_seeds"]
        or len(set(seeds)) != 30
    ):
        raise ValueError("sealed tuple mismatch")
    decision = {
        "decision": "admitted",
        "public_source_commit": FREEZE,
        "manifest_sha256": DIGESTS["main/release_identity.resolved.json"],
        "campaign_config_sha256": identities["main"]["resolved_manifest"][
            "canonical_campaign_config_sha256"
        ],
        "seeds": seeds,
        "snqi_anchors_sha256": ANCHOR_SHA,
        "review_ref": REVIEW_REF,
        "companion_manifest_sha256": DIGESTS["doorway/release_identity.resolved.json"],
        "companion_campaign_config_sha256": identities["doorway"]["resolved_manifest"][
            "canonical_campaign_config_sha256"
        ],
        "determinism_receipt_sha256": RECEIPT_SHA,
    }
    write_json(target / "evaluation_seed_admission.json", decision)
    for path in sorted(target.rglob("*")):
        if path.is_file() and not path.name.endswith(".review.json"):
            write_json(
                path.with_name(path.name + ".review.json"),
                {
                    "schema_version": "evidence-review-marker.v1",
                    "review_marker": "AI-GENERATED NEEDS-REVIEW",
                    "artifact_path": path.relative_to(root).as_posix(),
                    "artifact_sha256": sha(path.read_bytes()),
                    "preserved_exact_bytes": True,
                },
            )
    print(
        json.dumps(
            {
                "decision_sha256": sha((target / "evaluation_seed_admission.json").read_bytes()),
                "review_ref": REVIEW_REF,
            }
        )
    )


if __name__ == "__main__":
    main()
