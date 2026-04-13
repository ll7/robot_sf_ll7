"""Multi-scenario map-cache profiling for SAC training runs.

This module measures map-definition cache behaviour across a full scenario set,
separating cold (first-load) from warm (cached) env-creation cost and reporting
cache hit/miss/eviction rates.  Use it to confirm that the LRU cache is large
enough for the active scenario set and to quantify the per-episode cost of
scenario switching.

Typical usage::

    DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \\
      uv run python -m robot_sf.benchmark.perf_scenario_cache \\
        --scenario-config configs/scenarios/classic_interactions.yaml \\
        --output-json output/benchmarks/perf/scenario_cache_profile.json \\
        --output-markdown output/benchmarks/perf/scenario_cache_profile.md
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.common.artifact_paths import ensure_canonical_tree, get_artifact_category_path
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    map_cache_info,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(slots=True)
class ScenarioTiming:
    """Per-scenario config-build timing."""

    scenario_id: str
    config_build_sec: float
    map_file: str | None


@dataclass(slots=True)
class CacheProfileReport:
    """Cache profiling report for a scenario set."""

    scenario_config: str
    total_scenarios: int
    unique_maps: int
    cache_maxsize: int
    cache_hits: int
    cache_misses: int
    cache_currsize: int
    evictions_estimated: int
    timings: list[ScenarioTiming]

    @property
    def hit_rate(self) -> float:
        """Cache hit rate between 0.0 and 1.0."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly mapping.

        Returns:
            dict[str, Any]: JSON-serialisable cache profile payload.
        """
        return {
            "schema_version": "1.0",
            "suite": "sac-scenario-cache-profile-v1",
            "scenario_config": self.scenario_config,
            "total_scenarios": self.total_scenarios,
            "unique_maps": self.unique_maps,
            "cache_maxsize": self.cache_maxsize,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_currsize": self.cache_currsize,
            "evictions_estimated": self.evictions_estimated,
            "hit_rate": round(self.hit_rate, 4),
            "timings": [
                {
                    "scenario_id": t.scenario_id,
                    "config_build_sec": round(t.config_build_sec, 6),
                    "map_file": t.map_file,
                }
                for t in self.timings
            ],
        }


def profile_scenario_cache(
    scenario_config: Path,
    *,
    repetitions: int = 3,
) -> CacheProfileReport:
    """Profile map-cache behaviour across all scenarios in a config file.

    Loads every scenario from ``scenario_config`` in round-robin order for
    ``repetitions`` passes, measuring ``build_robot_config_from_scenario`` wall
    time per call and capturing LRU cache statistics before and after.

    Args:
        scenario_config: Path to the scenario YAML.
        repetitions: Number of full passes over all scenarios.

    Returns:
        CacheProfileReport: Cache hit/miss statistics and per-scenario timing.
    """
    scenarios = load_scenarios(scenario_config)
    if not scenarios:
        raise ValueError(f"No scenarios found in {scenario_config}")

    unique_maps: set[str] = set()
    for sc in scenarios:
        mf = sc.get("map_file")
        if mf:
            unique_maps.add(str(mf))

    # Reset module-level cache for a clean measurement.
    from robot_sf.training.scenario_loader import _load_map_definition  # noqa: PLC0415

    _load_map_definition.cache_clear()

    before = map_cache_info()
    timings: list[ScenarioTiming] = []

    for _ in range(repetitions):
        for sc in scenarios:
            sc_id = str(sc.get("name") or sc.get("scenario_id") or "unknown")
            t0 = time.perf_counter()
            build_robot_config_from_scenario(sc, scenario_path=scenario_config)
            elapsed = time.perf_counter() - t0
            timings.append(
                ScenarioTiming(
                    scenario_id=sc_id,
                    config_build_sec=elapsed,
                    map_file=sc.get("map_file"),
                )
            )

    after = map_cache_info()
    new_hits = after["hits"] - before["hits"]
    new_misses = after["misses"] - before["misses"]

    # LRU evictions = total loads that hit a full cache.
    # misses = first loads + eviction-triggered reloads; currsize caps at maxsize.
    evictions_estimated = max(0, new_misses - after["currsize"])

    return CacheProfileReport(
        scenario_config=str(scenario_config),
        total_scenarios=len(scenarios),
        unique_maps=len(unique_maps),
        cache_maxsize=after["maxsize"],
        cache_hits=new_hits,
        cache_misses=new_misses,
        cache_currsize=after["currsize"],
        evictions_estimated=evictions_estimated,
        timings=timings,
    )


def render_markdown(report: CacheProfileReport) -> str:
    """Render a human-readable Markdown summary of the cache profile report.

    Args:
        report: Cache profile report to render.

    Returns:
        str: Markdown report.
    """
    lines = [
        "# SAC Multi-Scenario Map-Cache Profile",
        "",
        f"- Scenario config: `{report.scenario_config}`",
        f"- Total scenarios profiled: `{report.total_scenarios}`",
        f"- Unique map files: `{report.unique_maps}`",
        f"- Cache maxsize: `{report.cache_maxsize}`",
        f"- Cache hits: `{report.cache_hits}`",
        f"- Cache misses: `{report.cache_misses}`",
        f"- Cache currsize: `{report.cache_currsize}`",
        f"- Evictions (estimated): `{report.evictions_estimated}`",
        f"- Hit rate: `{report.hit_rate:.1%}`",
        "",
    ]

    if report.unique_maps > report.cache_maxsize:
        lines.append(
            f"> **Warning**: {report.unique_maps} unique maps exceed cache maxsize "
            f"{report.cache_maxsize}; cache eviction is unavoidable."
        )
        lines.append("")
    elif report.evictions_estimated > 0:
        lines.append(
            f"> **Warning**: {report.evictions_estimated} estimated cache evictions "
            "despite sufficient cache capacity — check for path normalisation issues."
        )
        lines.append("")
    else:
        lines.append("> **OK**: all unique maps fit in the cache; zero evictions.")
        lines.append("")

    lines.extend(
        [
            "## Per-Scenario Config-Build Timing (first pass only)",
            "",
            "| Scenario | config_build_sec | map_file |",
            "| --- | ---: | --- |",
        ]
    )
    first_pass = report.timings[: report.total_scenarios]
    for t in first_pass:
        map_label = Path(t.map_file).name if t.map_file else "—"
        lines.append(f"| {t.scenario_id} | {t.config_build_sec:.4f} | {map_label} |")

    if len(report.timings) > report.total_scenarios:
        warm_timings = report.timings[report.total_scenarios :]
        if warm_timings:
            avg_warm = sum(t.config_build_sec for t in warm_timings) / len(warm_timings)
            lines.extend(
                [
                    "",
                    f"Average config-build time (warm passes): **{avg_warm * 1000:.2f} ms**",
                ]
            )

    return "\n".join(lines)


def _positive_int(value: str) -> int:
    """Argparse validator for positive integers.

    Returns:
        int: Parsed positive integer value.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be > 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=Path("configs/scenarios/classic_interactions.yaml"),
        help="Scenario YAML to profile.",
    )
    parser.add_argument(
        "--repetitions",
        type=_positive_int,
        default=3,
        help="Number of full passes over all scenarios.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=get_artifact_category_path("benchmarks") / "perf/scenario_cache_profile.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=get_artifact_category_path("benchmarks") / "perf/scenario_cache_profile.md",
        help="Markdown output path.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the multi-scenario map-cache profiler.

    Args:
        argv: Optional CLI argument overrides.

    Returns:
        int: Exit code (0 = ok, 1 = cache eviction detected).
    """
    ensure_canonical_tree(categories=("benchmarks",))
    args = parse_args(argv)

    logger.info("Profiling scenario cache: {}", args.scenario_config)
    report = profile_scenario_cache(args.scenario_config, repetitions=args.repetitions)

    payload = report.to_dict()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    markdown = render_markdown(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")

    logger.info(
        "Cache profile: hits={} misses={} evictions={} hit_rate={:.1%}",
        report.cache_hits,
        report.cache_misses,
        report.evictions_estimated,
        report.hit_rate,
    )
    logger.info("JSON: {}", args.output_json)
    logger.info("Markdown: {}", args.output_markdown)

    if report.evictions_estimated > 0:
        logger.warning(
            "{} cache evictions detected — consider increasing _load_map_definition maxsize",
            report.evictions_estimated,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
