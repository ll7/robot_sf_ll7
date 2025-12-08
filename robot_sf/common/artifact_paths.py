"""Canonical artifact root helpers and tooling integration support.

This module centralizes knowledge about the repository's artifact layout while
preserving the existing override behavior relied upon by tests. All artifact
producers should consult these helpers to resolve paths, honor
``ROBOT_SF_ARTIFACT_ROOT`` overrides, and keep the repository root clean.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_OVERRIDE_ENV = "ROBOT_SF_ARTIFACT_ROOT"
_ARTIFACT_ROOT_NAME = "output"
_RUN_TRACKER_CATEGORY = "run-tracker"


@dataclass(frozen=True)
class ArtifactCategory:
    """Metadata describing an artifact subdirectory managed by the project."""

    name: str
    relative_path: Path
    description: str
    retention_hint: str
    producers: tuple[str, ...] = ()


def _canonical_repository_root() -> Path:
    """Return the repository root directory resolved from this file location.

    Returns:
        Path: Absolute path to the repository root.
    """
    return Path(__file__).resolve().parents[2]


def get_repository_root() -> Path:
    """Expose the repository root path for tooling consumers.

    Returns:
        Path: Absolute path to the repository root.
    """

    return _canonical_repository_root()


@lru_cache(maxsize=1)
def _default_artifact_root() -> Path:
    """Return the default artifact root directory under the repository root.

    Returns:
        Path: Absolute path to the canonical artifact root (`output/`).
    """
    return _canonical_repository_root() / _ARTIFACT_ROOT_NAME


DEFAULT_ARTIFACT_CATEGORIES: dict[str, ArtifactCategory] = {
    "coverage": ArtifactCategory(
        name="coverage",
        relative_path=Path("coverage"),
        description="Coverage reports (HTML, XML, JSON) produced by pytest runs.",
        retention_hint="keep-latest",
        producers=("uv run pytest tests",),
    ),
    "benchmarks": ArtifactCategory(
        name="benchmarks",
        relative_path=Path("benchmarks"),
        description="Benchmark summaries, JSONL outputs, and performance reports.",
        retention_hint="keep-latest",
        producers=("scripts/benchmark02.py", "scripts/validation/performance_smoke_test.py"),
    ),
    "recordings": ArtifactCategory(
        name="recordings",
        relative_path=Path("recordings"),
        description="Simulation recordings and rendered videos for demos.",
        retention_hint="long-lived",
        producers=("scripts/play_recordings.py", "examples/demo_*"),
    ),
    "wandb": ArtifactCategory(
        name="wandb",
        relative_path=Path("wandb"),
        description="Weights & Biases run logs for experiment tracking.",
        retention_hint="short-lived",
        producers=("training scripts",),
    ),
    "tmp": ArtifactCategory(
        name="tmp",
        relative_path=Path("tmp"),
        description="Short-lived scratch space for auxiliary tooling.",
        retention_hint="short-lived",
        producers=("misc tooling",),
    ),
    "telemetry": ArtifactCategory(
        name="telemetry",
        relative_path=Path("telemetry"),
        description="Live/replay telemetry streams, pane exports, and summary artifacts.",
        retention_hint="keep-latest",
        producers=("telemetry pane", "headless telemetry runs"),
    ),
    _RUN_TRACKER_CATEGORY: ArtifactCategory(
        name=_RUN_TRACKER_CATEGORY,
        relative_path=Path(_RUN_TRACKER_CATEGORY),
        description="Run-tracking manifests, telemetry snapshots, and performance reports.",
        retention_hint="keep-latest",
        producers=(
            "examples/advanced/16_imitation_learning_pipeline.py",
            "scripts/tools/run_tracker_cli.py",
        ),
    ),
}

ARTIFACT_CATEGORIES = DEFAULT_ARTIFACT_CATEGORIES

LEGACY_ARTIFACT_PATHS = (
    Path("results"),
    Path("recordings"),
    Path("wandb"),
    Path("htmlcov"),
    Path("tmp"),
    Path("benchmark_results.json"),
    Path("coverage.json"),
)

LEGACY_MIGRATION_TARGETS: dict[Path, Path] = {
    Path("results"): Path("benchmarks/results"),
    Path("recordings"): Path("recordings"),
    Path("wandb"): Path("wandb"),
    Path("htmlcov"): Path("coverage/htmlcov"),
    Path("tmp"): Path("tmp/legacy"),
    Path("benchmark_results.json"): Path("benchmarks/benchmark_results.json"),
    Path("coverage.json"): Path("coverage/coverage.json"),
}


def get_artifact_override_root() -> Path | None:
    """Return the artifact override root when configured via the environment.

    Returns:
        Path | None: Expanded and resolved override path, or None when unset.
    """

    override = os.environ.get(_OVERRIDE_ENV)
    if not override:
        return None
    return Path(override).expanduser().resolve()


def get_artifact_root() -> Path:
    """Return the canonical artifact root, respecting overrides when present.

    Returns:
        Path: Artifact root path (override when set, else default).
    """

    override = get_artifact_override_root()
    if override is not None:
        return override
    return _default_artifact_root()


def get_artifact_category(name: str) -> ArtifactCategory:
    """Look up artifact category metadata by name.

    Returns:
        ArtifactCategory: Metadata record for the requested category.
    """

    try:
        return DEFAULT_ARTIFACT_CATEGORIES[name]
    except KeyError as exc:  # pragma: no cover - defensive: surfaced via tests
        raise KeyError(f"Unknown artifact category: {name!r}") from exc


def get_artifact_category_path(name: str) -> Path:
    """Return the absolute path for the given artifact category.

    Returns:
        Path: Resolved absolute path to the category directory.
    """

    category = get_artifact_category(name)
    return (get_artifact_root() / category.relative_path).resolve()


def ensure_canonical_tree(
    root: Path | None = None,
    categories: Iterable[str] | None = None,
) -> Path:
    """Create the canonical artifact tree and return the root path.

    Returns:
        Path: Resolved root path where categories were ensured.
    """

    target_root = Path(root).expanduser().resolve() if root is not None else get_artifact_root()
    target_root.mkdir(parents=True, exist_ok=True)
    category_names = categories or DEFAULT_ARTIFACT_CATEGORIES.keys()
    for name in category_names:
        category = get_artifact_category(name)
        (target_root / category.relative_path).mkdir(parents=True, exist_ok=True)
    return target_root


def ensure_run_tracker_tree(
    run_id: str | None = None,
    base_root: Path | None = None,
) -> Path:
    """Ensure the run-tracker directory exists and optionally create a child run folder.

    Returns:
        Path: Path to the tracker root or the specific run directory.
    """

    target_root = ensure_canonical_tree(root=base_root, categories=(_RUN_TRACKER_CATEGORY,))
    tracker_category = get_artifact_category(_RUN_TRACKER_CATEGORY)
    tracker_root = (target_root / tracker_category.relative_path).resolve()
    tracker_root.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        return tracker_root
    run_dir = (tracker_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_legacy_artifact_paths(base_root: Path | None = None) -> list[Path]:
    """Return a list of legacy artifact paths that still exist under ``base_root``.

    Returns:
        list[Path]: Existing legacy paths detected under the given base root.
    """

    root = Path(base_root).resolve() if base_root is not None else _canonical_repository_root()
    results: list[Path] = []
    for relative in LEGACY_ARTIFACT_PATHS:
        candidate = (root / relative).resolve()
        if candidate.exists():
            results.append(candidate)
    return results


def get_legacy_migration_plan() -> dict[Path, Path]:
    """Return a copy of the legacy path migration mapping.

    Returns:
        dict[Path, Path]: Mapping from legacy path to its canonical destination.
    """

    return dict(LEGACY_MIGRATION_TARGETS)


def resolve_artifact_path(path: str | Path) -> Path:
    """Resolve ``path`` to its on-disk location, honoring overrides when set.

    Returns:
        Path: Absolute path after applying override and legacy rules.
    """

    candidate = Path(path)
    override_root = get_artifact_override_root()
    repo_root = _canonical_repository_root()
    artifact_root = get_artifact_root()

    if candidate.is_absolute():
        if override_root is None:
            return candidate
        try:
            relative = candidate.resolve().relative_to(repo_root)
        except ValueError:
            # Path outside repository – leave untouched.
            return candidate
        return (override_root / relative).resolve()

    if override_root is not None:
        return (artifact_root / candidate).resolve()

    if candidate in LEGACY_MIGRATION_TARGETS:
        return (artifact_root / LEGACY_MIGRATION_TARGETS[candidate]).resolve()

    parts = candidate.parts
    if parts and parts[0] in DEFAULT_ARTIFACT_CATEGORIES:
        return (artifact_root / candidate).resolve()

    return (repo_root / candidate).resolve()


def iter_artifact_categories() -> Iterable[ArtifactCategory]:
    """Iterate over known artifact categories.

    Returns:
        Iterable[ArtifactCategory]: View over category metadata objects.
    """

    return DEFAULT_ARTIFACT_CATEGORIES.values()


def get_expert_policy_dir() -> Path:
    """Return the directory that stores expert policy manifests and checkpoints.

    Returns:
        Path: Directory path ensured to exist.
    """

    base = get_artifact_category_path("benchmarks") / "expert_policies"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_expert_policy_manifest_path(policy_id: str, extension: str = ".json") -> Path:
    """Return the manifest path for a given expert policy identifier.

    Returns:
        Path: File path under the expert policy directory.
    """

    return get_expert_policy_dir() / f"{policy_id}{extension}"


def get_trajectory_dataset_dir() -> Path:
    """Return the directory that stores curated trajectory datasets.

    Returns:
        Path: Directory path ensured to exist.
    """

    base = get_artifact_category_path("benchmarks") / "expert_trajectories"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_trajectory_dataset_path(dataset_id: str, extension: str = ".npz") -> Path:
    """Return the storage path for a trajectory dataset identifier.

    Returns:
        Path: File path under the trajectory dataset directory.
    """

    return get_trajectory_dataset_dir() / f"{dataset_id}{extension}"


def get_imitation_report_dir() -> Path:
    """Return the directory for comparative imitation reports.

    Returns:
        Path: Directory path ensured to exist.
    """

    base = get_artifact_category_path("benchmarks") / "ppo_imitation"
    base.mkdir(parents=True, exist_ok=True)
    return base
