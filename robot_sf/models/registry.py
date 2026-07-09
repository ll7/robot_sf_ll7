"""Model registry helpers for on-demand policy retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml
from loguru import logger

try:  # pragma: no cover - optional dependency
    import wandb  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    wandb = None  # type: ignore[assignment]

DEFAULT_REGISTRY_PATH = Path("model/registry.yaml")
_LOGGED_CACHED_MODEL_ARTIFACTS: set[Path] = set()
BENCHMARK_PROMOTED_CLAIM_BOUNDARIES = {
    "benchmark_promoted",
    "benchmark_candidate",
}
NON_BENCHMARK_CLAIM_BOUNDARIES = {
    "research_only",
    "smoke_only",
    "legacy_non_track",
    "not_for_benchmark",
}
BENCHMARK_PROMOTION_CLAIM_BOUNDARIES = (
    BENCHMARK_PROMOTED_CLAIM_BOUNDARIES | NON_BENCHMARK_CLAIM_BOUNDARIES
)
BENCHMARK_PROMOTION_REQUIRED_OBSERVATION_FIELDS = {
    "benchmark_track",
    "track_schema_version",
    "observation_level",
    "observation_mode",
    "allowed_observation_keys",
    "goal_encoding",
    "sensor_geometry",
    "privileged_input_status",
}
LEARNED_POLICY_REGISTRY_TAGS = {
    "ppo",
    "learned-policy",
    "learned-checkpoint",
    "rl-policy",
    "imitation-policy",
    "predictive",
}


def _local_only_resolution_error(
    entry: dict[str, Any], *, local_path: str | None
) -> FileNotFoundError:
    """Build a clear resolution error for models intentionally scoped to one machine.

    Returns:
        FileNotFoundError: Resolution error with optional migration guidance.
    """
    model_id = str(entry.get("model_id", "unknown-model"))
    replacement = str(entry.get("replacement_model_id", "") or "").strip()
    message = (
        f"Model '{model_id}' is marked local-only and is unavailable on this machine. "
        f"Expected local path: {local_path or '<unset>'}."
    )
    if replacement:
        message += f" Use '{replacement}' instead."
    return FileNotFoundError(message)


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


@dataclass(frozen=True)
class RegistryIssue:
    """One model-registry validation issue."""

    path: str
    message: str


def _is_missing(value: Any) -> bool:
    """Return whether a registry value is absent for contract validation."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _registry_tags(entry: dict[str, Any]) -> set[str]:
    """Return normalized registry tags."""
    raw_tags = entry.get("tags")
    if not isinstance(raw_tags, list):
        return set()
    return {str(tag).strip().lower() for tag in raw_tags if str(tag).strip()}


def _is_promoted_learned_policy(entry: dict[str, Any]) -> bool:
    """Return whether a registry row is a promoted learned-policy checkpoint."""
    tags = _registry_tags(entry)
    return "promoted" in tags and bool(tags & LEARNED_POLICY_REGISTRY_TAGS)


def _require_benchmark_promotion_field(
    issues: list[RegistryIssue],
    promotion: dict[str, Any],
    field: str,
) -> None:
    """Require one non-empty benchmark-promotion field."""
    if _is_missing(promotion.get(field)):
        issues.append(RegistryIssue(f"benchmark_promotion.{field}", "is required"))


def validate_registry_entry_benchmark_promotion(entry: dict[str, Any]) -> list[RegistryIssue]:
    """Validate observation-track metadata for benchmark promotion.

    Promoted learned-policy checkpoints must declare a benchmark-promotion block with the
    observation-track contract needed to compare their benchmark rows. Research, smoke, legacy, and
    non-benchmark entries can remain in the registry when they declare an explicit claim boundary.

    Returns:
        list[RegistryIssue]: Validation issues, empty when the entry satisfies this contract.
    """
    issues: list[RegistryIssue] = []
    promotion_raw = entry.get("benchmark_promotion")
    if promotion_raw is None:
        if _is_promoted_learned_policy(entry):
            issues.append(
                RegistryIssue(
                    "benchmark_promotion",
                    "promoted learned-policy checkpoints require observation-track metadata",
                )
            )
        return issues
    if not isinstance(promotion_raw, dict) or not promotion_raw:
        return [RegistryIssue("benchmark_promotion", "must be a non-empty mapping")]

    promotion = dict(promotion_raw)
    claim_boundary = promotion.get("claim_boundary")
    if claim_boundary not in BENCHMARK_PROMOTION_CLAIM_BOUNDARIES:
        issues.append(
            RegistryIssue(
                "benchmark_promotion.claim_boundary",
                f"must be one of {', '.join(sorted(BENCHMARK_PROMOTION_CLAIM_BOUNDARIES))}",
            )
        )
        return issues

    if claim_boundary in BENCHMARK_PROMOTED_CLAIM_BOUNDARIES:
        for field in sorted(BENCHMARK_PROMOTION_REQUIRED_OBSERVATION_FIELDS):
            _require_benchmark_promotion_field(issues, promotion, field)
        if not _is_missing(promotion.get("allowed_observation_keys")) and not isinstance(
            promotion.get("allowed_observation_keys"), list
        ):
            issues.append(
                RegistryIssue(
                    "benchmark_promotion.allowed_observation_keys",
                    "must be a non-empty list",
                )
            )
    elif _is_missing(promotion.get("non_benchmark_reason")):
        issues.append(
            RegistryIssue(
                "benchmark_promotion.non_benchmark_reason",
                "is required for non-benchmark claim boundaries",
            )
        )

    return issues


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
    """Resolve a local model path, downloading from a public release or W&B if needed.

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

    if bool(entry.get("local_only")):
        raise _local_only_resolution_error(
            entry,
            local_path=str(local_path) if local_path is not None else None,
        )

    if not allow_download:
        raise FileNotFoundError(f"Model '{model_id}' not found locally and downloads are disabled.")

    if entry.get("github_release"):
        return _download_from_github_release(entry, cache_dir=cache_dir)

    return _download_from_wandb(entry, cache_dir=cache_dir)


def _sha256(path: Path) -> str:
    """Return the SHA256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_from_github_release(entry: dict[str, Any], *, cache_dir: str | Path | None) -> Path:
    """Download a model artifact from a GitHub release asset.

    Returns:
        Path: Local filesystem path to the downloaded artifact.
    """
    release = entry.get("github_release")
    if not isinstance(release, dict):
        raise ValueError("Registry entry github_release must be a mapping.")

    model_id = str(entry.get("model_id", "unknown-model"))
    asset_name = str(release.get("asset_name") or "").strip()
    url = str(release.get("url") or "").strip()
    expected_sha256 = _github_release_expected_sha256(release)
    if not asset_name:
        raise ValueError(f"Registry entry '{model_id}' github_release.asset_name is required.")
    _validate_github_release_asset_name(asset_name, model_id=model_id)
    if not expected_sha256:
        raise ValueError(f"Registry entry '{model_id}' github_release.sha256 is required.")
    if not url:
        url = _github_release_url(release, model_id=model_id, asset_name=asset_name)
    _validate_github_release_url(url, model_id=model_id)

    cache_root = Path(cache_dir) if cache_dir is not None else Path("output/model_cache")
    cache_root = cache_root / model_id
    cache_root.mkdir(parents=True, exist_ok=True)
    cached_path = cache_root / asset_name

    if cached_path.exists() and _cached_release_path_is_valid(cached_path, expected_sha256):
        return cached_path

    logger.info("Downloading model artifact {} from GitHub release {}", asset_name, url)
    try:
        _stream_download_url(url, cached_path)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not download model '{model_id}' from GitHub release asset: {url}"
        ) from exc

    _verify_download_checksum(
        cached_path,
        expected_sha256=expected_sha256,
        model_id=model_id,
        asset_name=asset_name,
    )
    return cached_path


def _github_release_expected_sha256(release: dict[str, Any]) -> str:
    """Return normalized expected SHA256 while preserving non-null YAML scalars."""
    raw_sha256 = release.get("sha256")
    return "" if raw_sha256 is None else str(raw_sha256).strip().lower()


def _github_release_url(
    release: dict[str, Any],
    *,
    model_id: str,
    asset_name: str,
) -> str:
    """Build a GitHub release asset URL from repo/tag/asset fields.

    Returns:
        str: Public GitHub release asset URL.
    """
    repo = str(release.get("repo") or "").strip()
    tag = str(release.get("tag") or "").strip()
    if not repo or not tag:
        raise ValueError(
            f"Registry entry '{model_id}' needs github_release.url or repo/tag/asset_name."
        )
    return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"


def _validate_github_release_asset_name(asset_name: str, *, model_id: str) -> None:
    """Validate that a release asset name cannot escape the cache directory."""
    if Path(asset_name).name != asset_name or asset_name in {".", ".."}:
        raise ValueError(
            f"Registry entry '{model_id}' github_release.asset_name must be a file name."
        )


def _validate_github_release_url(url: str, *, model_id: str) -> None:
    """Validate that a release URL is an HTTPS GitHub URL."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        raise ValueError(
            f"Registry entry '{model_id}' github_release.url must be an https://github.com URL."
        )


def _cached_release_path_is_valid(path: Path, expected_sha256: str) -> bool:
    """Return whether a cached release asset can be reused."""
    if expected_sha256:
        observed = _sha256(path)
        if observed != expected_sha256:
            logger.warning(
                "Cached GitHub release model artifact checksum mismatch; redownloading: {}",
                path,
            )
            path.unlink()
            return False
    resolved_cached_path = path.resolve()
    if resolved_cached_path not in _LOGGED_CACHED_MODEL_ARTIFACTS:
        _LOGGED_CACHED_MODEL_ARTIFACTS.add(resolved_cached_path)
        logger.info("Using cached model artifact: {}", path)
    return True


def _stream_download_url(url: str, target_path: Path) -> None:
    """Stream a URL to a local target path."""
    with urlopen(url, timeout=60) as response, target_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _verify_download_checksum(
    path: Path,
    *,
    expected_sha256: str,
    model_id: str,
    asset_name: str,
) -> None:
    """Validate a downloaded artifact checksum."""
    if not expected_sha256:
        return
    observed = _sha256(path)
    if observed != expected_sha256:
        path.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for model '{model_id}' from GitHub release asset "
            f"{asset_name}: expected {expected_sha256}, observed {observed}."
        )


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

    file_name = entry.get("wandb_file", "model.zip")
    model_id = entry.get("model_id", "unknown-model")
    cache_root = Path(cache_dir) if cache_dir is not None else Path("output/model_cache")
    cache_root = cache_root / model_id
    cache_root.mkdir(parents=True, exist_ok=True)

    cached_path = cache_root / file_name
    if cached_path.exists():
        resolved_cached_path = cached_path.resolve()
        if resolved_cached_path not in _LOGGED_CACHED_MODEL_ARTIFACTS:
            _LOGGED_CACHED_MODEL_ARTIFACTS.add(resolved_cached_path)
            logger.info("Using cached model artifact: {}", cached_path)
        return cached_path

    artifact_path = entry.get("wandb_artifact_path")
    if artifact_path:
        logger.info("Downloading model artifact {} from W&B artifact {}", file_name, artifact_path)
        api = wandb.Api()
        artifact = api.artifact(str(artifact_path))
        artifact.download(root=str(cache_root))
        downloaded_artifact_path = cache_root / file_name
        if downloaded_artifact_path.exists():
            return downloaded_artifact_path
        raise FileNotFoundError(
            f"W&B artifact '{artifact_path}' did not contain expected file '{file_name}'."
        )

    run_path = entry.get("wandb_run_path")
    if not run_path:
        entity = entry.get("wandb_entity")
        project = entry.get("wandb_project")
        run_id = entry.get("wandb_run_id")
        if entity and project and run_id:
            run_path = f"{entity}/{project}/{run_id}"
        else:
            raise ValueError(
                "Registry entry missing wandb_artifact_path, wandb_run_path, "
                "or wandb_entity/project/run_id."
            )

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
