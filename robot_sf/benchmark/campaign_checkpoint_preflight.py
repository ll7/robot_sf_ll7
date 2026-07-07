"""Submit-time preflight that verifies benchmark campaign arm checkpoints are resolvable.

Camera-ready campaigns bind each learned-policy arm to a model checkpoint through the arm's
``algo_config`` (a ``model_id`` resolved through the model registry, or a direct ``model_path``).
When the submit worktree's ``output/model_cache`` does not durably contain a required arm
checkpoint, the strict all-rows campaign policy converts a single un-loadable arm into a
whole-campaign failure -- but only *after* the expensive run has already burned compute
(issue #4613: the S30 campaign jobs 13296 and 13301 both failed roughly 14h in on the same missing
PPO ``model_cache`` checkpoint).

This module inspects a campaign config for every enabled arm that declares a checkpoint and fails
fast -- before scenarios load and before ``sbatch`` -- when a checkpoint cannot be resolved. It
mirrors the fail-closed shape of :mod:`robot_sf.benchmark.orca_preflight` and offers two modes:

* ``stage=False`` (default, cheap, network-free): a ``model_id`` is accepted when it is present
  locally OR the registry entry declares a durable remote source (GitHub release / W&B) that
  :func:`robot_sf.models.resolve_model_path` could stage; a direct ``model_path`` is accepted when
  the file exists. This catches unknown / mistyped ``model_id`` values and missing ``model_path``
  files instantly without touching the network, and is safe to call from the always-on campaign
  preflight (including offline preflight-only workflows).
* ``stage=True`` (enforced pre-submit staging): every checkpoint is actually resolved -- downloading
  and checksum-verifying registry artifacts into the durable cache via ``resolve_model_path`` -- so
  the compute node loads a validated file instead of discovering a corrupt or incomplete cache 14h
  in. Ops runs this before ``sbatch`` (see ``scripts/benchmark/preflight_campaign_checkpoints.py``).

Claim boundary: this is a provisioning / fail-closed preflight. It does not run the benchmark, does
not by itself constitute benchmark evidence, and only inspects checkpoint *references* named exactly
``model_id`` / ``model_path`` in an arm's ``algo_config`` (recursively, so a nested prior policy is
covered). Other model-loading side channels (for example ``predictive_foresight_model_id``) are out
of scope and keep their existing runtime behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

from robot_sf.models import get_registry_entry, resolve_model_path

if TYPE_CHECKING:
    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec

# Config keys, at any nesting depth of an arm's algo_config, that name a policy checkpoint.
# ``model_id`` resolves through the registry; ``model_path`` is a direct filesystem reference.
# ``model_id`` wins when both appear at the same mapping level, mirroring
# ``PPOPlanner._load_model`` runtime semantics.
_CHECKPOINT_REFERENCE_KEYS: tuple[str, ...] = ("model_id", "model_path")
# Registry-entry fields that declare a durable remote source ``resolve_model_path`` can stage.
_REMOTE_SOURCE_KEYS: tuple[str, ...] = (
    "github_release",
    "wandb_run_path",
    "wandb_artifact_path",
)


class CampaignCheckpointPreflightError(RuntimeError):
    """Typed campaign checkpoint preflight failure for library-facing callers."""

    def __init__(
        self,
        message: str,
        *,
        arms: tuple[str, ...] = (),
    ) -> None:
        """Store the actionable message plus the failing arm keys."""
        super().__init__(message)
        self.arms = arms


@dataclass(frozen=True)
class ArmCheckpointReference:
    """One checkpoint reference declared by a benchmark campaign arm."""

    planner_key: str
    algo: str
    kind: str  # "model_id" or "model_path"
    value: str
    algo_config_path: Path | None


@dataclass(frozen=True)
class ArmCheckpointResolution:
    """Resolution outcome for a single :class:`ArmCheckpointReference`."""

    reference: ArmCheckpointReference
    resolvable: bool
    status: str
    detail: str
    resolved_path: Path | None = None


def _iter_mapping_checkpoint_keys(node: Any) -> list[tuple[str, str]]:
    """Return ``(kind, value)`` checkpoint references found anywhere in a parsed algo_config.

    Each mapping level contributes at most one reference, preferring ``model_id`` over
    ``model_path`` to mirror runtime loading semantics, and nested mappings/sequences are walked so
    a nested prior-policy checkpoint is still covered.

    Returns:
        list[tuple[str, str]]: ``(kind, value)`` pairs for every checkpoint reference.
    """
    references: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for kind in _CHECKPOINT_REFERENCE_KEYS:
            value = node.get(kind)
            if isinstance(value, str) and value.strip():
                references.append((kind, value.strip()))
                break
        for value in node.values():
            references.extend(_iter_mapping_checkpoint_keys(value))
    elif isinstance(node, (list, tuple)):
        for item in node:
            references.extend(_iter_mapping_checkpoint_keys(item))
    return references


def _load_algo_config(path: Path) -> dict[str, Any]:
    """Load an arm's algo_config YAML into a mapping.

    Returns:
        dict[str, Any]: Parsed config mapping (empty when the file is empty).

    Raises:
        FileNotFoundError: When ``path`` does not resolve to an existing file
            (a missing path or a directory both fail closed here).
        TypeError: When the YAML document is not a mapping.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Planner algo_config not found (or not a file): {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Planner algo_config must be a mapping (YAML dict): {path}")
    return data


def _is_learned_checkpoint_planner(planner: PlannerSpec) -> bool:
    """Return True when a planner is enabled and declares an algo_config to inspect."""
    return bool(planner.enabled) and planner.algo_config_path is not None


def iter_campaign_arm_checkpoint_references(
    cfg: CampaignConfig,
) -> list[ArmCheckpointReference]:
    """Collect every checkpoint reference declared by the enabled arms in a campaign config.

    Args:
        cfg: A loaded camera-ready campaign config.

    Returns:
        list[ArmCheckpointReference]: One entry per ``model_id`` / ``model_path`` reference found in
        an enabled arm's algo_config.

    Raises:
        FileNotFoundError: When an enabled arm names an algo_config file that does not exist.
        TypeError: When an arm's algo_config is not a mapping.
    """
    references: list[ArmCheckpointReference] = []
    for planner in cfg.planners:
        if not _is_learned_checkpoint_planner(planner):
            continue
        config_path = Path(planner.algo_config_path)  # type: ignore[arg-type]
        algo_config = _load_algo_config(config_path)
        for kind, value in _iter_mapping_checkpoint_keys(algo_config):
            references.append(
                ArmCheckpointReference(
                    planner_key=planner.key,
                    algo=planner.algo,
                    kind=kind,
                    value=value,
                    algo_config_path=config_path,
                )
            )
    return references


def _resolve_model_path_reference(
    reference: ArmCheckpointReference,
) -> ArmCheckpointResolution:
    """Resolve a direct ``model_path`` reference against the filesystem.

    Returns:
        ArmCheckpointResolution: ``resolvable`` True only when the file exists.
    """
    path = Path(reference.value)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.is_file():
        return ArmCheckpointResolution(
            reference=reference,
            resolvable=True,
            status="present_local",
            detail=f"model_path file present at {path}",
            resolved_path=path,
        )
    return ArmCheckpointResolution(
        reference=reference,
        resolvable=False,
        status="model_path_missing",
        detail=f"model_path file not found at {path}",
    )


def _entry_has_remote_source(entry: dict[str, Any]) -> bool:
    """Return True when a registry entry declares a durable remote source to stage from."""
    return any(entry.get(key) for key in _REMOTE_SOURCE_KEYS)


def _resolve_model_id_reference_cheap(
    reference: ArmCheckpointReference,
    *,
    registry_path: str | Path | None,
) -> ArmCheckpointResolution:
    """Check a ``model_id`` reference without touching the network.

    A checkpoint is accepted when the registry entry's ``local_path`` exists, or when the entry is
    not ``local_only`` and declares a durable remote source ``resolve_model_path`` could stage.

    Returns:
        ArmCheckpointResolution: The cheap-mode resolution outcome.
    """
    try:
        entry = get_registry_entry(reference.value, registry_path)
    except KeyError:
        return ArmCheckpointResolution(
            reference=reference,
            resolvable=False,
            status="unknown_model_id",
            detail=f"model_id '{reference.value}' is not present in the model registry",
        )
    local_path = entry.get("local_path")
    if local_path:
        resolved = Path(local_path)
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        if resolved.is_file():
            return ArmCheckpointResolution(
                reference=reference,
                resolvable=True,
                status="present_local",
                detail=f"registry local_path present at {resolved}",
                resolved_path=resolved,
            )
    if bool(entry.get("local_only")):
        return ArmCheckpointResolution(
            reference=reference,
            resolvable=False,
            status="local_only_missing",
            detail=(
                f"model_id '{reference.value}' is local_only and its local_path "
                f"({local_path!r}) is not present; it cannot be staged from a remote source"
            ),
        )
    if _entry_has_remote_source(entry):
        return ArmCheckpointResolution(
            reference=reference,
            resolvable=True,
            status="stageable_remote",
            detail=(
                f"model_id '{reference.value}' is not cached locally but declares a durable remote "
                "source; run with stage=True (or the preflight script with --stage) before sbatch "
                "to download and checksum-verify it"
            ),
        )
    return ArmCheckpointResolution(
        reference=reference,
        resolvable=False,
        status="no_resolvable_source",
        detail=(
            f"model_id '{reference.value}' has neither a present local_path nor a durable remote "
            "source (github_release / wandb) to stage from"
        ),
    )


def _resolve_model_id_reference_staged(
    reference: ArmCheckpointReference,
    *,
    registry_path: str | Path | None,
    cache_dir: str | Path | None,
) -> ArmCheckpointResolution:
    """Actually resolve (and stage) a ``model_id`` reference, downloading + verifying as needed.

    Returns:
        ArmCheckpointResolution: ``resolvable`` True only when the checkpoint materializes on disk.
    """
    try:
        path = resolve_model_path(
            reference.value,
            registry_path=registry_path,
            allow_download=True,
            cache_dir=cache_dir,
        )
    except (KeyError, RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
        return ArmCheckpointResolution(
            reference=reference,
            resolvable=False,
            status="stage_failed",
            detail=f"could not stage model_id '{reference.value}': {type(exc).__name__}: {exc}",
        )
    if not path.is_file():
        return ArmCheckpointResolution(
            reference=reference,
            resolvable=False,
            status="stage_missing",
            detail=f"model_id '{reference.value}' resolved to {path} but no checkpoint file is present",
        )
    return ArmCheckpointResolution(
        reference=reference,
        resolvable=True,
        status="staged",
        detail=f"model_id '{reference.value}' staged and verified at {path}",
        resolved_path=path,
    )


def resolve_arm_checkpoint(
    reference: ArmCheckpointReference,
    *,
    stage: bool = False,
    registry_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> ArmCheckpointResolution:
    """Resolve a single arm checkpoint reference.

    Args:
        reference: The checkpoint reference to resolve.
        stage: When True, download and checksum-verify registry artifacts into the durable cache.
            When False, only confirm the checkpoint is present locally or has a stageable remote
            source (no network access).
        registry_path: Optional model-registry path override (useful for tests/fixtures).
        cache_dir: Optional cache directory override for staged downloads.

    Returns:
        ArmCheckpointResolution: The resolution outcome.
    """
    if reference.kind == "model_path":
        return _resolve_model_path_reference(reference)
    if stage:
        return _resolve_model_id_reference_staged(
            reference,
            registry_path=registry_path,
            cache_dir=cache_dir,
        )
    return _resolve_model_id_reference_cheap(reference, registry_path=registry_path)


def _format_failure_lines(failures: list[ArmCheckpointResolution]) -> str:
    """Render one actionable line per failing arm checkpoint.

    Returns:
        str: Newline-joined ``- arm ...`` lines describing each failure.
    """
    lines = []
    for resolution in failures:
        ref = resolution.reference
        config_hint = f" [{ref.algo_config_path}]" if ref.algo_config_path is not None else ""
        lines.append(
            f"  - arm '{ref.planner_key}' (algo={ref.algo}): {ref.kind}='{ref.value}'"
            f"{config_hint} -> {resolution.status}: {resolution.detail}"
        )
    return "\n".join(lines)


def _preflight_failure_message(
    failures: list[ArmCheckpointResolution],
    *,
    stage: bool,
) -> str:
    """Build the actionable fail-closed message naming every unresolvable arm checkpoint.

    Returns:
        str: The full multi-line preflight failure message including the remedy.
    """
    mode = "staging" if stage else "resolvability"
    remedy = (
        "Stage the missing checkpoints into the durable cache before submitting:\n"
        "  uv run python scripts/benchmark/preflight_campaign_checkpoints.py "
        "--config <campaign-config> --stage\n"
        "Or promote the checkpoint to the model registry with a present local_path or a durable "
        "github_release/wandb source (model/registry.yaml). Fix any mistyped model_id.\n"
        "Aborting before starting the benchmark campaign (issue #4613)."
    )
    return (
        f"Campaign checkpoint {mode} preflight failed for "
        f"{len(failures)} arm checkpoint reference(s):\n"
        f"{_format_failure_lines(failures)}\n"
        f"{remedy}"
    )


def check_campaign_arm_checkpoints_preflight(
    cfg: CampaignConfig,
    *,
    stage: bool = False,
    registry_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fail fast when any enabled arm's checkpoint cannot be resolved before a campaign runs.

    When any enabled arm declares a ``model_id`` / ``model_path`` checkpoint that cannot be
    resolved, this function raises :class:`CampaignCheckpointPreflightError` with an actionable
    message naming each failing arm, the unresolvable reference, and the remedy -- turning a 14h
    whole-campaign failure into a fast pre-submit failure (issue #4613).

    Args:
        cfg: A loaded camera-ready campaign config.
        stage: When True, download and checksum-verify registry artifacts into the durable cache
            (enforced pre-submit staging). When False (default), only confirm each checkpoint is
            present locally or has a stageable remote source (no network access).
        registry_path: Optional model-registry path override (useful for tests/fixtures).
        cache_dir: Optional cache directory override for staged downloads.

    Returns:
        dict[str, Any]: A summary with the checked/resolved counts and per-arm resolution status.

    Raises:
        CampaignCheckpointPreflightError: When one or more arm checkpoints are unresolvable.
    """
    references = iter_campaign_arm_checkpoint_references(cfg)
    if not references:
        logger.debug(
            "No arm checkpoints declared in campaign config; checkpoint preflight skipped."
        )
        return {"checked": 0, "resolved": 0, "stage": bool(stage), "arms": []}

    mode = "staging" if stage else "resolvability"
    logger.info(f"Checkpoint {mode} preflight for {len(references)} arm checkpoint reference(s)...")
    resolutions = [
        resolve_arm_checkpoint(
            reference,
            stage=stage,
            registry_path=registry_path,
            cache_dir=cache_dir,
        )
        for reference in references
    ]
    failures = [resolution for resolution in resolutions if not resolution.resolvable]
    if failures:
        message = _preflight_failure_message(failures, stage=stage)
        logger.error(message)
        failing_arms = tuple(sorted({resolution.reference.planner_key for resolution in failures}))
        raise CampaignCheckpointPreflightError(message, arms=failing_arms)

    logger.info(f"Checkpoint {mode} preflight passed for {len(references)} reference(s).")
    return {
        "checked": len(resolutions),
        "resolved": len(resolutions),
        "stage": bool(stage),
        "arms": [
            {
                "planner_key": resolution.reference.planner_key,
                "algo": resolution.reference.algo,
                "kind": resolution.reference.kind,
                "value": resolution.reference.value,
                "status": resolution.status,
                "detail": resolution.detail,
                "algo_config_path": (
                    str(resolution.reference.algo_config_path)
                    if resolution.reference.algo_config_path is not None
                    else None
                ),
                "resolved_path": (
                    str(resolution.resolved_path) if resolution.resolved_path is not None else None
                ),
            }
            for resolution in resolutions
        ],
    }


def check_campaign_arm_checkpoints_preflight_from_config(
    config_path: str | Path,
    *,
    stage: bool = False,
    registry_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load a campaign config from *config_path* and run the checkpoint preflight.

    This entry point is suitable for scripting a pre-``sbatch`` gate without importing the full
    campaign machinery in callers.

    Args:
        config_path: Path to a camera-ready campaign config YAML.
        stage: When True, download and checksum-verify registry artifacts into the durable cache.
        registry_path: Optional model-registry path override.
        cache_dir: Optional cache directory override for staged downloads.

    Returns:
        dict[str, Any]: The preflight summary (see :func:`check_campaign_arm_checkpoints_preflight`).

    Raises:
        CampaignCheckpointPreflightError: When one or more arm checkpoints are unresolvable.
    """
    from robot_sf.benchmark.camera_ready_campaign import (  # noqa: PLC0415
        load_campaign_config,
    )

    cfg = load_campaign_config(config_path)
    return check_campaign_arm_checkpoints_preflight(
        cfg,
        stage=stage,
        registry_path=registry_path,
        cache_dir=cache_dir,
    )


__all__ = [
    "ArmCheckpointReference",
    "ArmCheckpointResolution",
    "CampaignCheckpointPreflightError",
    "check_campaign_arm_checkpoints_preflight",
    "check_campaign_arm_checkpoints_preflight_from_config",
    "iter_campaign_arm_checkpoint_references",
    "resolve_arm_checkpoint",
]
