"""Baseline environment factory creation timing script"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics as stats
import time
from datetime import (
    UTC,  # type: ignore[attr-defined]
    datetime,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from robot_sf.gym_env.environment_factory import (
        make_image_robot_env,
        make_robot_env,
    )
    from robot_sf.gym_env.unified_config import ImageRobotConfig, RobotSimulationConfig
except Exception as e:  # pragma: no cover - baseline script bootstrap
    raise SystemExit(f"Failed to import environment factories: {e}") from e


@runtime_checkable
class _EnvLike(Protocol):  # minimal protocol to appease static checks
    """EnvLike class."""

    def reset(self) -> Any:
        """Reset the environment."""
        ...

    def close(self) -> None:
        """Close the environment."""
        ...


def _time_creation(fn: Callable[[], _EnvLike], iterations: int) -> dict[str, float]:
    """Time creation.

    Args:
        fn: Auto-generated placeholder description.
        iterations: Auto-generated placeholder description.

    Returns:
        dict[str, float]: Auto-generated placeholder description.
    """
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        env = fn()
        # Minimal interaction to ensure full construction side effects executed
        with contextlib.suppress(Exception):
            env.reset()
        # Best effort close to free resources
        with contextlib.suppress(Exception):
            env.close()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)
    if not times:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "std_ms": 0.0, "raw": []}
    return {
        "mean_ms": stats.fmean(times),
        "p95_ms": _percentile(times, 95),
        "std_ms": stats.pstdev(times) if len(times) > 1 else 0.0,
        "raw": times,
    }


def _percentile(values: list[float], p: float) -> float:
    """Percentile.

    Args:
        values: Auto-generated placeholder description.
        p: Auto-generated placeholder description.

    Returns:
        float: Auto-generated placeholder description.
    """
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def main() -> None:
    """Main.

    Returns:
        None: Auto-generated placeholder description.
    """
    parser = argparse.ArgumentParser(description="Baseline factory creation timing")
    parser.add_argument("--iterations", type=int, default=30, help="Creations per factory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/factory_perf_baseline.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    iterations = args.iterations
    results = {}

    logger.info("Timing make_robot_env iterations={}", iterations)
    results["make_robot_env"] = _time_creation(
        lambda: make_robot_env(config=RobotSimulationConfig(), debug=False),
        iterations,
    )

    logger.info("Timing make_image_robot_env iterations={}", iterations)
    results["make_image_robot_env"] = _time_creation(
        lambda: make_image_robot_env(config=ImageRobotConfig(), debug=False),
        iterations,
    )

    # Pedestrian env requires a robot_model; skip unless trivial to supply (future enhancement)
    # Placeholder: could load a lightweight mock or stub policy; omitted for neutral baseline.

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "iterations": iterations,
        "results": results,
        "notes": "Creation timing only; excludes step performance.",
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Baseline creation timing written to {}", out_path)


if __name__ == "__main__":  # pragma: no cover
    main()
