"""CPU-only schema-compatibility audit for predictive-planner campaign arms (issue #5241).

A gap-prediction comparison campaign halted mid-run because one predictive checkpoint
(``prediction_planner_v2_xl_ego``, job 13194) declared an obstacle-feature schema that the
runtime planner config did not expect -- raising ``ObstacleFeatureSchemaError`` under
``stop_on_failure=true`` and killing 144 jobs. The maintainer decision (do not retrain, do not
write a compat shim) is to make checkpoint<->schema compatibility *checkable before submission*
and to auto-exclude incompatible arms with an explicit manifest note.

This module is the provenance owner for that preflight. For every predictive planner arm in a
camera-ready campaign config it:

1. Resolves the arm checkpoint (from ``predictive_model_id`` via the model registry, or a direct
   ``predictive_checkpoint_path``) -- optionally staging registry artifacts into the durable
   cache, but always CPU-only.
2. Loads *only* the checkpoint metadata / feature spec
   (``torch.load(..., map_location="cpu", weights_only=True)``). It never instantiates the model
   and never touches a GPU: the comparison reads ``config.input_dim`` /
   ``config.feature_schema_name`` / the saved ``feature_schema`` block only.
3. Reuses the *exact* comparison the runtime load path performs
   (``robot_sf.planner.predictive_model.load_predictive_checkpoint`` ->
   ``validate_predictive_feature_schema_metadata``) to classify the arm COMPAT or INCOMPAT.
4. Emits a filtered copy of the campaign config with incompatible arms removed and a
   top-level ``schema_excluded_arms`` provenance list so exclusion is recorded, not silent.

Claim boundary: this is a CPU-only preflight / config-filtering tool. It does not run a
benchmark and is not benchmark evidence by itself. ``COMPAT`` means the checkpoint metadata is
schema-valid for the arm's configured expected schema; it is not a training-quality or
metric-correctness claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.models import resolve_model_path
from robot_sf.planner.obstacle_features import (
    PREDICTIVE_LEGACY_FEATURE_SCHEMA,
    ObstacleFeatureSchemaError,
    infer_predictive_feature_schema,
    validate_predictive_feature_schema_metadata,
    validate_predictive_runtime_feature_schema,
)
from robot_sf.planner.predictive_model import PredictiveModelConfig

if TYPE_CHECKING:
    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec

# Algos whose arms carry a predictive checkpoint that must be schema-audited. ``gap_prediction``
# is the gap-aware predictive planner and reuses the predictive checkpoint + feature schema.
_PREDICTIVE_ALGOS: frozenset[str] = frozenset({"prediction_planner", "gap_prediction"})

# Status values an audited arm can take. Only ``INCOMPAT`` causes arm exclusion; ``COMPAT`` is the
# success state. The remaining statuses are honest "could not classify" outcomes that leave the arm
# in place (the tool never silently drops an arm it could not inspect).
STATUS_COMPAT = "COMPAT"
STATUS_INCOMPAT = "INCOMPAT"
STATUS_NOT_PREDICTIVE = "NOT_PREDICTIVE"
STATUS_NO_CHECKPOINT = "NO_CHECKPOINT"
STATUS_UNRESOLVABLE = "UNRESOLVABLE"
STATUS_UNSTAGED = "UNSTAGED"
STATUS_CORRUPT = "CORRUPT"

# Statuses for which the arm is *actionable* (it is predictive and carries a checkpoint reference
# the audit attempted to classify). Non-predictive / no-checkpoint arms are informational only.
_ACTIONABLE_STATUSES: frozenset[str] = frozenset(
    {STATUS_COMPAT, STATUS_INCOMPAT, STATUS_UNRESOLVABLE, STATUS_UNSTAGED, STATUS_CORRUPT}
)


@dataclass(frozen=True)
class SchemaAuditArmResult:
    """Schema-compatibility outcome for a single predictive campaign arm."""

    arm_key: str
    algo: str
    checkpoint_ref: str
    checkpoint_kind: str  # "model_id" or "checkpoint_path"
    resolved_path: Path | None
    expected_schema: str | None
    checkpoint_schema: str | None
    checkpoint_input_dim: int | None
    status: str
    detail: str
    algo_config_path: Path | None = None


@dataclass(frozen=True)
class SchemaAuditResult:
    """Aggregate schema-audit outcome for a campaign config."""

    config_path: Path
    arms: list[SchemaAuditArmResult] = field(default_factory=list)
    staged: bool = False

    @property
    def incompatible_arms(self) -> list[SchemaAuditArmResult]:
        """Return only the arms classified COMPAT-failing (schema-mismatched)."""
        return [arm for arm in self.arms if arm.status == STATUS_INCOMPAT]

    @property
    def actionable_arms(self) -> list[SchemaAuditArmResult]:
        """Return the arms the audit actually inspected (predictive + has checkpoint)."""
        return [arm for arm in self.arms if arm.status in _ACTIONABLE_STATUSES]

    def to_manifest(self) -> dict[str, Any]:
        """Return a JSON-compatible manifest of the audit (for PR bodies / reports).

        Returns:
            dict[str, Any]: Per-arm schema-audit rows plus a summary.
        """
        return {
            "config_path": str(self.config_path),
            "staged": bool(self.staged),
            "arm_count": len(self.arms),
            "incompatible_arm_count": len(self.incompatible_arms),
            "arms": [
                {
                    "arm_key": arm.arm_key,
                    "algo": arm.algo,
                    "checkpoint_ref": arm.checkpoint_ref,
                    "checkpoint_kind": arm.checkpoint_kind,
                    "resolved_path": str(arm.resolved_path)
                    if arm.resolved_path is not None
                    else None,
                    "expected_schema": arm.expected_schema,
                    "checkpoint_schema": arm.checkpoint_schema,
                    "checkpoint_input_dim": arm.checkpoint_input_dim,
                    "status": arm.status,
                    "detail": arm.detail,
                }
                for arm in self.arms
            ],
        }


def _is_predictive_arm(planner: PlannerSpec) -> bool:
    """Return True when an arm is a predictive-planner family arm carrying a checkpoint config."""
    if not planner.enabled:
        return False
    if planner.algo and planner.algo.strip().lower() in _PREDICTIVE_ALGOS:
        return True
    # Fall back to config inspection so a predictive arm that uses a non-standard algo name but
    # declares a predictive checkpoint is still covered.
    if planner.algo_config_path is None:
        return False
    algo_config = _load_algo_config_dict(planner.algo_config_path)
    return "predictive_model_id" in algo_config or "predictive_checkpoint_path" in algo_config


def _load_algo_config_dict(path: Path) -> dict[str, Any]:
    """Load an arm's algo_config YAML as a mapping (empty when missing/non-mapping).

    Returns:
        dict[str, Any]: Parsed config mapping (empty for a missing file or non-mapping YAML).
    """
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def load_predictive_checkpoint_metadata(
    path: str | Path,
) -> dict[str, Any]:
    """Load ONLY the predictive-checkpoint metadata/feature spec (never instantiate the model).

    This mirrors the metadata read in
    :func:`robot_sf.planner.predictive_model.load_predictive_checkpoint` but stops short of
    instantiating :class:`PredictiveTrajectoryModel` or moving any tensor to a device: it reads
    the ``config`` / ``feature_schema`` payload blocks on CPU with ``weights_only=True`` only.

    Args:
        path: Checkpoint file path.

    Returns:
        dict[str, Any]: A metadata view with keys ``config`` (the saved
        :class:`PredictiveModelConfig` dict), ``feature_schema`` (the saved schema metadata, when
        present), ``epoch`` and ``source_path``.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        RuntimeError: When the payload is not a recognizable predictive checkpoint mapping.
    """
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Predictive checkpoint not found: {checkpoint}")
    # Late import keeps the module importable in torch-optional environments; the audit itself
    # requires torch and surfaces a clear error at call time if it is unavailable.
    import torch  # noqa: PLC0415

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Predictive checkpoint payload is not a mapping: {checkpoint} ({type(payload)!r})"
        )
    return {
        "config": payload.get("config") or {},
        "feature_schema": payload.get("feature_schema"),
        "epoch": payload.get("epoch"),
        "source_path": checkpoint,
    }


def _classify_arm_schema(
    metadata: dict[str, Any],
    *,
    expected_schema_name: str,
) -> tuple[str, str, str | None, int | None]:
    """Classify checkpoint metadata against an expected schema name.

    Reuses the *exact* runtime comparison (``validate_predictive_feature_schema_metadata`` with the
    checkpoint's own ``config.input_dim``), so ``INCOMPAT`` here is the same condition that raises
    ``ObstacleFeatureSchemaError`` when the planner loads the model at runtime.

    Args:
        metadata: Checkpoint metadata view from :func:`load_predictive_checkpoint_metadata`.
        expected_schema_name: The runtime/planner-config expected feature schema name.

    Returns:
        tuple of ``(status, detail, checkpoint_schema_name, checkpoint_input_dim)``.
    """
    config_data = metadata.get("config") or {}
    try:
        config = PredictiveModelConfig(**config_data)
    except TypeError as exc:
        return (
            STATUS_CORRUPT,
            f"checkpoint config is not a valid PredictiveModelConfig: {exc}",
            None,
            None,
        )
    input_dim = int(config.input_dim)

    feature_schema = metadata.get("feature_schema")
    if not isinstance(feature_schema, dict):
        # Mirror load_predictive_checkpoint: infer from input_dim when the checkpoint pre-dates the
        # feature_schema block (e.g. the v1/v2_full checkpoints).
        feature_schema = infer_predictive_feature_schema(input_dim)

    checkpoint_schema_name = str(feature_schema.get("name") or "").strip() or None
    try:
        validate_predictive_feature_schema_metadata(
            feature_schema,
            input_dim=input_dim,
            expected_schema_name=str(expected_schema_name),
        )
        validate_predictive_runtime_feature_schema(feature_schema)
    except ObstacleFeatureSchemaError as exc:
        return (
            STATUS_INCOMPAT,
            (
                f"schema mismatch: checkpoint declares {checkpoint_schema_name!r} "
                f"(input_dim={input_dim}), runtime expects {expected_schema_name!r}: {exc}"
            ),
            checkpoint_schema_name,
            input_dim,
        )
    return (
        STATUS_COMPAT,
        (
            f"checkpoint schema {checkpoint_schema_name!r} (input_dim={input_dim}) "
            f"matches runtime expected {expected_schema_name!r}"
        ),
        checkpoint_schema_name,
        input_dim,
    )


def _resolve_checkpoint_path(
    arm_key: str,
    algo_config: dict[str, Any],
    *,
    stage: bool,
    registry_path: str | Path | None,
    cache_dir: str | Path | None,
) -> tuple[str, str, Path | None, str, str]:
    """Resolve an arm's checkpoint reference to a local path.

    Returns:
        tuple of ``(checkpoint_ref, checkpoint_kind, resolved_path, status, detail)`` where
        ``status`` is empty when resolution succeeded, otherwise one of the non-COMPAT statuses.
    """
    model_id = algo_config.get("predictive_model_id")
    checkpoint_path = algo_config.get("predictive_checkpoint_path")
    if isinstance(model_id, str) and model_id.strip():
        ref = model_id.strip()
        kind = "model_id"
        try:
            path = resolve_model_path(
                ref,
                registry_path=registry_path,
                allow_download=bool(stage),
                cache_dir=cache_dir,
            )
        except FileNotFoundError as exc:
            return (
                ref,
                kind,
                None,
                STATUS_UNSTAGED,
                (f"model_id {ref!r} is not present locally and --stage was not set: {exc}"),
            )
        except (KeyError, RuntimeError, ValueError, OSError) as exc:
            return (
                ref,
                kind,
                None,
                STATUS_UNRESOLVABLE,
                (f"model_id {ref!r} could not be resolved: {type(exc).__name__}: {exc}"),
            )
        if not path.is_file():
            return (
                ref,
                kind,
                None,
                STATUS_UNRESOLVABLE,
                (f"model_id {ref!r} resolved to {path} but no checkpoint file is present"),
            )
        return ref, kind, path, "", ""
    if isinstance(checkpoint_path, str) and checkpoint_path.strip():
        ref = checkpoint_path.strip()
        kind = "checkpoint_path"
        path = Path(ref)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            return ref, kind, None, STATUS_UNRESOLVABLE, f"checkpoint_path file not found: {path}"
        return ref, kind, path, "", ""
    return (
        "",
        "",
        None,
        STATUS_NO_CHECKPOINT,
        (
            f"predictive arm {arm_key!r} declares no predictive_model_id / predictive_checkpoint_path"
        ),
    )


def audit_predictive_checkpoint_schema(
    cfg: CampaignConfig,
    *,
    stage: bool = False,
    registry_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> SchemaAuditResult:
    """Audit every predictive arm's checkpoint schema compatibility (CPU-only).

    Args:
        cfg: A loaded camera-ready campaign config.
        stage: When True, stage registry-backed checkpoints into the durable cache before reading
            metadata (CPU download; still no model instantiation / no GPU). When False (default),
            only arms whose checkpoint is already present locally are classified; unstaged arms are
            reported ``UNSTAGED`` rather than silently skipped.
        registry_path: Optional model-registry path override (useful for tests/fixtures).
        cache_dir: Optional cache directory override for staged downloads.

    Returns:
        SchemaAuditResult: Per-arm schema-compatibility outcome.
    """
    arms: list[SchemaAuditArmResult] = []
    for planner in cfg.planners:
        if not _is_predictive_arm(planner):
            arms.append(
                SchemaAuditArmResult(
                    arm_key=planner.key,
                    algo=planner.algo,
                    checkpoint_ref="",
                    checkpoint_kind="",
                    resolved_path=None,
                    expected_schema=None,
                    checkpoint_schema=None,
                    checkpoint_input_dim=None,
                    status=STATUS_NOT_PREDICTIVE,
                    detail="arm is not a predictive-planner family arm (no checkpoint to audit)",
                    algo_config_path=(
                        planner.algo_config_path if planner.algo_config_path is not None else None
                    ),
                )
            )
            continue

        algo_config_path = (
            planner.algo_config_path if planner.algo_config_path is not None else None
        )
        algo_config = (
            _load_algo_config_dict(algo_config_path) if algo_config_path is not None else {}
        )
        expected_schema_name = str(
            algo_config.get("predictive_feature_schema_name") or PREDICTIVE_LEGACY_FEATURE_SCHEMA
        ).strip()

        ref, kind, resolved, status, detail = _resolve_checkpoint_path(
            planner.key,
            algo_config,
            stage=stage,
            registry_path=registry_path,
            cache_dir=cache_dir,
        )
        if status:
            arms.append(
                SchemaAuditArmResult(
                    arm_key=planner.key,
                    algo=planner.algo,
                    checkpoint_ref=ref,
                    checkpoint_kind=kind,
                    resolved_path=resolved,
                    expected_schema=expected_schema_name,
                    checkpoint_schema=None,
                    checkpoint_input_dim=None,
                    status=status,
                    detail=detail,
                    algo_config_path=algo_config_path,
                )
            )
            continue

        assert resolved is not None  # resolution succeeded with a file
        try:
            metadata = load_predictive_checkpoint_metadata(resolved)
        except (FileNotFoundError, RuntimeError, ValueError, OSError, EOFError) as exc:
            arms.append(
                SchemaAuditArmResult(
                    arm_key=planner.key,
                    algo=planner.algo,
                    checkpoint_ref=ref,
                    checkpoint_kind=kind,
                    resolved_path=resolved,
                    expected_schema=expected_schema_name,
                    checkpoint_schema=None,
                    checkpoint_input_dim=None,
                    status=STATUS_CORRUPT,
                    detail=f"could not read checkpoint metadata: {type(exc).__name__}: {exc}",
                    algo_config_path=algo_config_path,
                )
            )
            continue

        status_c, detail_c, checkpoint_schema, input_dim = _classify_arm_schema(
            metadata, expected_schema_name=expected_schema_name
        )
        arms.append(
            SchemaAuditArmResult(
                arm_key=planner.key,
                algo=planner.algo,
                checkpoint_ref=ref,
                checkpoint_kind=kind,
                resolved_path=resolved,
                expected_schema=expected_schema_name,
                checkpoint_schema=checkpoint_schema,
                checkpoint_input_dim=input_dim,
                status=status_c,
                detail=detail_c,
                algo_config_path=algo_config_path,
            )
        )

    actionable = [arm for arm in arms if arm.status in _ACTIONABLE_STATUSES]
    incompatible = [arm for arm in arms if arm.status == STATUS_INCOMPAT]
    logger.info(
        "Predictive checkpoint schema audit: {} actionable arm(s), {} INCOMPAT, {} unstaged/unresolvable.",
        len(actionable),
        len(incompatible),
        sum(1 for arm in arms if arm.status in {STATUS_UNSTAGED, STATUS_UNRESOLVABLE}),
    )
    return SchemaAuditResult(
        arms=arms, staged=bool(stage), config_path=Path(getattr(cfg, "name", "") or "")
    )


def audit_predictive_checkpoint_schema_from_config(
    config_path: str | Path,
    *,
    stage: bool = False,
    registry_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> SchemaAuditResult:
    """Load a campaign config and run the predictive checkpoint schema audit.

    Args:
        config_path: Path to a camera-ready campaign config YAML.
        stage: When True, stage registry-backed checkpoints into the durable cache first.
        registry_path: Optional model-registry path override.
        cache_dir: Optional cache directory override for staged downloads.

    Returns:
        SchemaAuditResult: Per-arm schema-compatibility outcome.
    """
    from robot_sf.benchmark.camera_ready_campaign import (  # noqa: PLC0415
        load_campaign_config,
    )

    resolved = Path(config_path).resolve()
    cfg = load_campaign_config(resolved)
    result = audit_predictive_checkpoint_schema(
        cfg,
        stage=stage,
        registry_path=registry_path,
        cache_dir=cache_dir,
    )
    # ``name`` on CampaignConfig is the human label; carry the actual config path for provenance.
    return SchemaAuditResult(arms=result.arms, staged=result.staged, config_path=resolved)


def format_schema_audit_table(result: SchemaAuditResult) -> str:
    """Render the per-arm schema-audit table (arm | checkpoint | expected | current | status).

    Args:
        result: The audit result to render.

    Returns:
        str: A fixed-width human-readable audit table plus a summary footer.
    """
    headers = ("arm", "checkpoint", "expected-schema", "current-schema", "status")
    rows: list[tuple[str, str, str, str, str]] = []
    for arm in result.arms:
        rows.append(
            (
                arm.arm_key,
                arm.checkpoint_ref or "-",
                arm.expected_schema or "-",
                arm.checkpoint_schema or "-",
                arm.status,
            )
        )

    def _width(idx: int) -> int:
        return (
            max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        )

    widths = [_width(i) for i in range(len(headers))]
    sep = "  ".join("-" * w for w in widths)
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines = [header_line, sep]
    for row in rows:
        lines.append("  ".join(col.ljust(widths[i]) for i, col in enumerate(row)))
    incompatible = len(result.incompatible_arms)
    actionable = len(result.actionable_arms)
    lines.append("")
    lines.append(
        f"summary: {actionable} predictive arm(s) inspected, {incompatible} INCOMPAT, "
        f"staged={'yes' if result.staged else 'no'}"
    )
    for arm in result.incompatible_arms:
        lines.append(f"  INCOMPAT {arm.arm_key}: {arm.detail}")
    return "\n".join(lines)


def emit_schema_filtered_config(
    config_path: str | Path,
    result: SchemaAuditResult,
    out_path: str | Path,
) -> Path:
    """Write a copy of ``config_path`` with INCOMPAT arms removed and exclusions recorded.

    Only arms the audit classified ``INCOMPAT`` are removed. Arms it could not inspect
    (``UNSTAGED`` / ``UNRESOLVABLE`` / ``CORRUPT``) are left in place and listed under
    ``schema_audit_uninspected_arms`` so a human decides; the tool never silently drops an arm it
    could not read. A ``schema_excluded_arms`` provenance list records each removed arm with its
    checkpoint and mismatch reason.

    Args:
        config_path: Source campaign config path.
        result: The audit result driving the filtering.
        out_path: Destination path for the filtered config.

    Returns:
        Path: The resolved output path that was written.
    """
    source = Path(config_path).resolve()
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Campaign config must be a mapping: {source}")

    # The first-party import is deliberately unguarded: it has no optional or
    # native dependency boundary. Repository-root resolution itself can still
    # fail on unusual filesystem or symlink state, so retain the prior
    # best-effort provenance fallback without reintroducing an import guard.
    try:
        repo_root = get_repository_root()
    except (OSError, RuntimeError):
        repo_root = source.parent
    try:
        source_rel = source.relative_to(repo_root) if repo_root in source.parents else source
    except ValueError:
        source_rel = source

    incompatible_keys = {arm.arm_key for arm in result.incompatible_arms}
    uninspected = [
        arm
        for arm in result.arms
        if arm.status in {STATUS_UNSTAGED, STATUS_UNRESOLVABLE, STATUS_CORRUPT}
    ]

    original_planners = data.get("planners", [])
    kept_planners: list[Any] = []
    for planner in original_planners:
        key = planner.get("key") if isinstance(planner, dict) else None
        if key in incompatible_keys:
            continue
        kept_planners.append(planner)
    data["planners"] = kept_planners

    excluded_records = [
        {
            "key": arm.arm_key,
            "algo": arm.algo,
            "checkpoint_ref": arm.checkpoint_ref,
            "checkpoint_kind": arm.checkpoint_kind,
            "expected_schema": arm.expected_schema,
            "checkpoint_schema": arm.checkpoint_schema,
            "checkpoint_input_dim": arm.checkpoint_input_dim,
            "reason": arm.detail,
        }
        for arm in result.incompatible_arms
    ]
    uninspected_records = [
        {
            "key": arm.arm_key,
            "algo": arm.algo,
            "checkpoint_ref": arm.checkpoint_ref,
            "checkpoint_kind": arm.checkpoint_kind,
            "status": arm.status,
            "reason": arm.detail,
        }
        for arm in uninspected
    ]
    # Provenance header so exclusion is recorded, not silent (issue #5241 decision).
    data["schema_excluded_arms"] = excluded_records
    if uninspected_records:
        data["schema_audit_uninspected_arms"] = uninspected_records
    data["schema_audit"] = {
        "source_config": str(source_rel),
        "tool": "scripts/tools/audit_predictive_checkpoint_schema.py",
        "staged": bool(result.staged),
        "excluded_arm_count": len(excluded_records),
        "uninspected_arm_count": len(uninspected_records),
        "note": (
            "Incompatible predictive arms removed by CPU-only checkpoint-schema audit; "
            "see schema_excluded_arms for per-arm provenance (issue #5241)."
        ),
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out.resolve()


__all__ = [
    "STATUS_COMPAT",
    "STATUS_CORRUPT",
    "STATUS_INCOMPAT",
    "STATUS_NOT_PREDICTIVE",
    "STATUS_NO_CHECKPOINT",
    "STATUS_UNRESOLVABLE",
    "STATUS_UNSTAGED",
    "SchemaAuditArmResult",
    "SchemaAuditResult",
    "audit_predictive_checkpoint_schema",
    "audit_predictive_checkpoint_schema_from_config",
    "emit_schema_filtered_config",
    "format_schema_audit_table",
    "load_predictive_checkpoint_metadata",
]
