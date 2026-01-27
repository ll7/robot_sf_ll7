"""Model registry helpers for on-demand policy retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

try:  # pragma: no cover - optional dependency
    import wandb  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    wandb = None  # type: ignore[assignment]

DEFAULT_REGISTRY_PATH = Path("model/registry.yaml")


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
