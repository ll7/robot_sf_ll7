"""Real-trajectory ingestion and artifact-staging contract.

This package owns the dataset-agnostic, bring-your-own-dataset (BYO) contract for staging real
pedestrian/agent trajectory data (GitHub issue #3065). It defines the manifest schema and a
fail-closed preflight checker. It deliberately does **not** download, copy, or commit any external
dataset; the repository tracks only manifests, checksums, retrieval instructions, and the canonical
conversion shape. Raw external data always stays in a git-ignored local staging location.

See ``docs/context/issue_3065_real_trajectory_ingestion_contract.md`` and
``docs/context/artifact_evidence_vocabulary.md`` (External artifact pointer category) for the
surrounding evidence policy.
"""

from robot_sf.data_ingestion.real_trajectory_contract import (
    MANIFEST_SCHEMA_ID,
    ContractError,
    PreflightIssue,
    PreflightResult,
    load_manifest,
    load_schema,
    run_preflight,
    validate_manifest_structure,
)

__all__ = [
    "MANIFEST_SCHEMA_ID",
    "ContractError",
    "PreflightIssue",
    "PreflightResult",
    "load_manifest",
    "load_schema",
    "run_preflight",
    "validate_manifest_structure",
]
