"""Contract tests for the social-navigation benchmark runner."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

_ORCHESTRATOR_MODULE = "robot_sf.benchmark.full_classic.orchestrator"
_previous_orchestrator = sys.modules.get(_ORCHESTRATOR_MODULE)
_orchestrator_stub = ModuleType(_ORCHESTRATOR_MODULE)
_orchestrator_stub.run_full_benchmark = lambda _config: None  # type: ignore[attr-defined]
sys.modules[_ORCHESTRATOR_MODULE] = _orchestrator_stub
try:
    from scripts import run_social_navigation_benchmark as benchmark
finally:
    if _previous_orchestrator is None:
        sys.modules.pop(_ORCHESTRATOR_MODULE, None)
    else:
        sys.modules[_ORCHESTRATOR_MODULE] = _previous_orchestrator


def test_compute_aggregates_passes_expected_algorithms(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner forwards expected algorithms to current aggregate implementations."""

    captured: dict[str, object] = {}

    def current_aggregator(
        *,
        expected_algorithms: set[str],
        **kwargs: object,
    ) -> dict[str, object]:
        captured.update(kwargs)
        captured["expected_algorithms"] = expected_algorithms
        return {"_meta": {"missing_algorithms": []}}

    monkeypatch.setattr(benchmark, "compute_aggregates_with_ci", current_aggregator)

    result = benchmark._compute_aggregates_payload(
        [],
        expected_algorithms={"sf", "ppo"},
    )

    assert result == {"_meta": {"missing_algorithms": []}}
    assert captured["expected_algorithms"] == {"sf", "ppo"}


def test_compute_aggregates_omits_unsupported_expected_algorithms_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A directly inspectable legacy callable is invoked without the optional keyword."""

    calls: list[dict[str, object]] = []

    def legacy_aggregator(
        records: list[dict[str, object]],
        *,
        group_by: str,
        bootstrap_samples: int,
        bootstrap_confidence: float,
    ) -> dict[str, object]:
        calls.append(
            {
                "records": records,
                "group_by": group_by,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_confidence": bootstrap_confidence,
            }
        )
        return {"legacy": True}

    monkeypatch.setattr(benchmark, "compute_aggregates_with_ci", legacy_aggregator)

    result = benchmark._compute_aggregates_payload(
        [],
        expected_algorithms={"sf"},
    )

    assert result == {"legacy": True}
    assert len(calls) == 1
    assert "expected_algorithms" not in calls[0]


def test_compute_aggregates_passes_expected_algorithms_to_kwargs_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A keyword-capable current callable receives the completeness roster."""

    forwarded: list[dict[str, object]] = []

    def forwarding_wrapper(**kwargs: object) -> dict[str, object]:
        forwarded.append(kwargs)
        return {"current": True}

    monkeypatch.setattr(benchmark, "compute_aggregates_with_ci", forwarding_wrapper)

    result = benchmark._compute_aggregates_payload([], expected_algorithms={"sf"})

    assert result["current"] is True
    assert len(forwarded) == 1
    assert forwarded[0]["expected_algorithms"] == {"sf"}


def test_compute_aggregates_fails_closed_for_uninspectable_aggregator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opaque callable must not get an ambiguous keyword-retry fallback."""

    calls = 0

    class OpaqueAggregator:
        @property
        def __signature__(self) -> object:
            raise ValueError("opaque signature")

        def __call__(self, **_: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            raise TypeError("got an unexpected keyword argument 'expected_algorithms'")

    monkeypatch.setattr(benchmark, "compute_aggregates_with_ci", OpaqueAggregator())

    with pytest.raises(TypeError, match="inspectable signature"):
        benchmark._compute_aggregates_payload([], expected_algorithms={"sf"})

    assert calls == 0


def test_compute_aggregates_reraises_internal_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An internal aggregation TypeError must not silently disable completeness checks."""

    calls = 0

    def broken_aggregator(
        **_: object,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise TypeError("malformed episode payload")

    monkeypatch.setattr(benchmark, "compute_aggregates_with_ci", broken_aggregator)

    with pytest.raises(TypeError, match="malformed episode payload"):
        benchmark._compute_aggregates_payload([], expected_algorithms={"sf"})

    assert calls == 1


def test_compute_aggregates_reraises_canonical_internal_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching nested error from a keyword-capable callable must not trigger a retry."""

    calls = 0

    def broken_aggregator(
        **_: object,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise TypeError("got an unexpected keyword argument 'expected_algorithms'")

    monkeypatch.setattr(benchmark, "compute_aggregates_with_ci", broken_aggregator)

    with pytest.raises(TypeError, match="unexpected keyword argument 'expected_algorithms'"):
        benchmark._compute_aggregates_payload([], expected_algorithms={"sf"})

    assert calls == 1


def test_aggregate_validation_preserves_missing_algorithm_failure_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing algorithms survive the runner's aggregation-to-validation boundary."""

    real_aggregator = benchmark.compute_aggregates_with_ci

    def fast_current_aggregator(
        records: list[dict[str, object]],
        *,
        group_by: str,
        bootstrap_samples: int,
        bootstrap_confidence: float,
        expected_algorithms: set[str],
    ) -> dict[str, object]:
        return real_aggregator(
            records,
            group_by=group_by,
            bootstrap_samples=0,
            bootstrap_confidence=bootstrap_confidence,
            expected_algorithms=expected_algorithms,
            return_ci=False,
        )

    monkeypatch.setattr(benchmark, "compute_aggregates_with_ci", fast_current_aggregator)
    episodes_dir = tmp_path / "sf" / "episodes"
    episodes_dir.mkdir(parents=True)
    (episodes_dir / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_id": "sf-1",
                "scenario_id": "scenario-1",
                "scenario_params": {"algo": "sf"},
                "metrics": {"success_rate": 1.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    aggregation = benchmark.aggregate_all_results(
        [{"algo": "sf", "output_dir": str(tmp_path / "sf"), "success": True}],
        str(tmp_path),
        expected_algorithms={"sf", "ppo"},
    )
    validation = benchmark.validate_benchmark_results(
        {
            "output_root": str(tmp_path),
            "baselines": aggregation["baselines"],
            "total_episodes": aggregation["total_episodes"],
            "meta": aggregation["meta"],
        }
    )

    assert aggregation["success"] is True
    assert aggregation["meta"]["missing_algorithms"] == ["ppo"]
    assert validation["checks"]["expected_algorithms_present"] is False
    assert validation["missing_algorithms"] == ["ppo"]
