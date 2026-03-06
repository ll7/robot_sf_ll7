"""Model registry helpers for on-demand policy retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

try:  # pragma: no cover - optional dependency
    import wandb  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    wandb = None  # type: ignore[assignment]

DEFAULT_REGISTRY_PATH = Path("model/registry.yaml")


@dataclass(frozen=True)
class WandbLatestModel:
    """Selection metadata for a latest-model query against W&B."""

    run_id: str
    run_path: str
    run_name: str
    job_type: str | None
    group: str | None
    state: str | None
    created_at: str | None
    file_name: str


def load_registry(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the model registry YAML and return entries indexed by model_id.

    Returns:
        dict[str, dict[str, Any]]: Registry entries keyed by ``model_id``.
    """

    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    entries = data.get("models", [])
    registry: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            logger.warning("Skipping invalid registry entry: {}", entry)
            continue
        model_id = entry.get("model_id")
        if not model_id:
            logger.warning("Skipping registry entry without model_id: {}", entry)
            continue
        if model_id in registry:
            raise ValueError(f"Duplicate model_id in registry: {model_id}")
        registry[model_id] = entry

    return registry


def get_registry_entry(model_id: str, path: str | Path | None = None) -> dict[str, Any]:
    """Return a single registry entry by model_id.

    Returns:
        dict[str, Any]: Registry entry for the requested ``model_id``.
    """

    registry = load_registry(path)
    try:
        return registry[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown model_id '{model_id}' in registry") from exc


def resolve_model_path(
    model_id: str,
    *,
    registry_path: str | Path | None = None,
    allow_download: bool = True,
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve a local model path, downloading from W&B if needed.

    Returns:
        Path: Local filesystem path to the model artifact.
    """

    entry = get_registry_entry(model_id, registry_path)
    local_path = entry.get("local_path")
    if local_path:
        resolved = Path(local_path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        if resolved.exists():
            return resolved

    if not allow_download:
        raise FileNotFoundError(f"Model '{model_id}' not found locally and downloads are disabled.")

    return _download_from_wandb(entry, cache_dir=cache_dir)


def resolve_latest_wandb_model(  # noqa: PLR0913
    *,
    entity: str,
    project: str,
    group: str | None = None,
    job_type: str | None = None,
    name_prefix: str | None = None,
    tags: tuple[str, ...] = (),
    file_name: str = "model.zip",
    allowed_states: tuple[str, ...] = ("finished", "running"),
    cache_dir: str | Path | None = None,
) -> tuple[Path, WandbLatestModel]:
    """Download the newest matching W&B model file and return local path plus selection.

    Returns:
        tuple[Path, WandbLatestModel]: Downloaded local checkpoint path and the W&B run selection
        metadata used to resolve it.
    """

    selection = find_latest_wandb_model(
        entity=entity,
        project=project,
        group=group,
        job_type=job_type,
        name_prefix=name_prefix,
        tags=tags,
        file_name=file_name,
        allowed_states=allowed_states,
    )
    path = _download_from_wandb(
        {
            "model_id": selection.run_name or selection.run_id,
            "wandb_run_path": selection.run_path,
            "wandb_file": selection.file_name,
        },
        cache_dir=cache_dir,
    )
    return path, selection


def find_latest_wandb_model(
    *,
    entity: str,
    project: str,
    group: str | None = None,
    job_type: str | None = None,
    name_prefix: str | None = None,
    tags: tuple[str, ...] = (),
    file_name: str = "model.zip",
    allowed_states: tuple[str, ...] = ("finished", "running"),
) -> WandbLatestModel:
    """Return the newest W&B run that matches the requested filters and file name."""

    if wandb is None:  # pragma: no cover - optional dependency
        raise RuntimeError("W&B not available; cannot query latest model artifact.")

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    normalized_tags = {str(tag).strip() for tag in tags if str(tag).strip()}
    normalized_states = {
        str(state).strip().lower() for state in allowed_states if str(state).strip()
    }
    candidates: list[tuple[datetime, WandbLatestModel]] = []
    for run in runs:
        state = str(getattr(run, "state", "") or "").strip().lower()
        if normalized_states and state not in normalized_states:
            continue
        run_group = str(getattr(run, "group", "") or "").strip() or None
        if group is not None and run_group != group:
            continue
        run_job_type = str(getattr(run, "job_type", "") or "").strip() or None
        if job_type is not None and run_job_type != job_type:
            continue
        run_name = str(getattr(run, "name", "") or "").strip()
        if name_prefix is not None and not run_name.startswith(name_prefix):
            continue
        run_tags = {str(tag).strip() for tag in (getattr(run, "tags", None) or ())}
        if normalized_tags and not normalized_tags.issubset(run_tags):
            continue
        if file_name not in {str(item.name) for item in run.files()}:
            continue
        created_at_raw = getattr(run, "created_at", None)
        created_at = (
            datetime.fromisoformat(str(created_at_raw).replace("Z", "+00:00"))
            if created_at_raw
            else datetime.min.replace(tzinfo=UTC)
        )
        candidates.append(
            (
                created_at,
                WandbLatestModel(
                    run_id=str(getattr(run, "id", "") or ""),
                    run_path=f"{entity}/{project}/{getattr(run, 'id', '')}",
                    run_name=run_name,
                    job_type=run_job_type,
                    group=run_group,
                    state=state or None,
                    created_at=str(created_at_raw) if created_at_raw else None,
                    file_name=file_name,
                ),
            )
        )

    if not candidates:
        raise FileNotFoundError(
            "No W&B run matched latest-model query: "
            f"entity={entity} project={project} group={group} job_type={job_type} "
            f"name_prefix={name_prefix} tags={sorted(normalized_tags)} "
            f"file_name={file_name} allowed_states={sorted(normalized_states)}"
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _download_from_wandb(entry: dict[str, Any], *, cache_dir: str | Path | None) -> Path:
    """Download a model artifact from W&B using metadata stored in the registry.

    Returns:
        Path: Local filesystem path to the downloaded artifact.
    """

    if wandb is None:  # pragma: no cover - optional dependency
        raise RuntimeError("W&B not available; cannot download model artifact.")

    run_path = entry.get("wandb_run_path")
    if not run_path:
        entity = entry.get("wandb_entity")
        project = entry.get("wandb_project")
        run_id = entry.get("wandb_run_id")
        if entity and project and run_id:
            run_path = f"{entity}/{project}/{run_id}"
        else:
            raise ValueError(
                "Registry entry missing wandb_run_path or wandb_entity/project/run_id."
            )

    file_name = entry.get("wandb_file", "model.zip")
    model_id = entry.get("model_id", "unknown-model")
    cache_root = Path(cache_dir) if cache_dir is not None else Path("output/model_cache")
    cache_root = cache_root / model_id
    cache_root.mkdir(parents=True, exist_ok=True)

    cached_path = cache_root / file_name
    if cached_path.exists():
        logger.info("Using cached model artifact: {}", cached_path)
        return cached_path

    logger.info("Downloading model artifact {} from {}", file_name, run_path)
    api = wandb.Api()
    run = api.run(run_path)
    wandb_file = run.file(file_name)
    downloaded_path = wandb_file.download(root=str(cache_root), replace=True)
    if isinstance(downloaded_path, (str, Path)):
        return Path(downloaded_path)
    file_name = getattr(downloaded_path, "name", None)
    if isinstance(file_name, str):
        return Path(file_name)
    raise TypeError(
        "W&B download returned unexpected type; expected path-like or file-like with name.",
    )


def upsert_registry_entry(
    entry: dict[str, Any],
    *,
    registry_path: str | Path | None = None,
) -> None:
    """Insert or update a model registry entry by model_id."""

    model_id = entry.get("model_id")
    if not model_id:
        raise ValueError("Registry entry must include a model_id.")

    path = Path(registry_path) if registry_path is not None else DEFAULT_REGISTRY_PATH
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        data = {}

    models = data.get("models")
    if not isinstance(models, list):
        models = []

    replaced = False
    for idx, existing in enumerate(models):
        if isinstance(existing, dict) and existing.get("model_id") == model_id:
            models[idx] = entry
            replaced = True
            break

    if not replaced:
        models.append(entry)

    data["version"] = data.get("version", 1)
    data["models"] = models

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    logger.info("Updated model registry entry for {}", model_id)
