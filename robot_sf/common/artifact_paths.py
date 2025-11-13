"""Canonical artifact root helpers and tooling integration support.

This module centralizes knowledge about the repository's artifact layout while
preserving the existing override behaviour relied upon by tests. All artifact
producers should consult these helpers to resolve paths, honour
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


@dataclass(frozen=True)
class ArtifactCategory:
    """Metadata describing an artifact subdirectory managed by the project."""

    name: str
    relative_path: Path
    description: str
    retention_hint: str
    producers: tuple[str, ...] = ()


def _canonical_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_repository_root() -> Path:
    """Expose the repository root path for tooling consumers."""

    return _canonical_repository_root()


@lru_cache(maxsize=1)
def _default_artifact_root() -> Path:
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
    """Return the artifact override root when configured via the environment."""

    override = os.environ.get(_OVERRIDE_ENV)
    if not override:
        return None
    return Path(override).expanduser().resolve()


def get_artifact_root() -> Path:
    """Return the canonical artifact root, respecting overrides when present."""

    override = get_artifact_override_root()
    if override is not None:
        return override
    return _default_artifact_root()


def get_artifact_category(name: str) -> ArtifactCategory:
    """Look up artifact category metadata by name."""

    try:
        return DEFAULT_ARTIFACT_CATEGORIES[name]
    except KeyError as exc:  # pragma: no cover - defensive: surfaced via tests
        raise KeyError(f"Unknown artifact category: {name!r}") from exc


def get_artifact_category_path(name: str) -> Path:
    """Return the absolute path for the given artifact category."""

    category = get_artifact_category(name)
    return (get_artifact_root() / category.relative_path).resolve()


def ensure_canonical_tree(
    root: Path | None = None,
    categories: Iterable[str] | None = None,
) -> Path:
    """Create the canonical artifact tree and return the root path."""

    target_root = Path(root).expanduser().resolve() if root is not None else get_artifact_root()
    target_root.mkdir(parents=True, exist_ok=True)
    category_names = categories or DEFAULT_ARTIFACT_CATEGORIES.keys()
    for name in category_names:
        category = get_artifact_category(name)
        (target_root / category.relative_path).mkdir(parents=True, exist_ok=True)
    return target_root


def find_legacy_artifact_paths(base_root: Path | None = None) -> list[Path]:
    """Return a list of legacy artifact paths that still exist under ``base_root``."""

    root = Path(base_root).resolve() if base_root is not None else _canonical_repository_root()
    results: list[Path] = []
    for relative in LEGACY_ARTIFACT_PATHS:
        candidate = (root / relative).resolve()
        if candidate.exists():
            results.append(candidate)
    return results


def get_legacy_migration_plan() -> dict[Path, Path]:
    """Return a copy of the legacy path migration mapping."""

    return dict(LEGACY_MIGRATION_TARGETS)


def resolve_artifact_path(path: str | Path) -> Path:
    """Resolve ``path`` to its on-disk location, honouring overrides when set."""

    candidate = Path(path)
    override_root = get_artifact_override_root()

    if candidate.is_absolute():
        if override_root is None:
            return candidate.resolve()
        try:
            relative = candidate.resolve().relative_to(_canonical_repository_root())
        except ValueError:
            # Path outside repository – leave untouched.
            return candidate.resolve()
        return (override_root / relative).resolve()

    if override_root is not None:
        return (override_root / candidate).resolve()

    return (_canonical_repository_root() / candidate).resolve()


def iter_artifact_categories() -> Iterable[ArtifactCategory]:
    """Iterate over known artifact categories."""

    return DEFAULT_ARTIFACT_CATEGORIES.values()
