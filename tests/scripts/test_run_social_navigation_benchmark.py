"""Contract tests for the social-navigation benchmark runner."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

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

    def current_aggregator(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"_meta": {"missing_algorithms": []}}

    monkeypatch.setattr(benchmark, "compute_aggregates_with_ci", current_aggregator)

    result = benchmark._compute_aggregates_payload(
        [],
        expected_algorithms={"sf", "ppo"},
    )

    assert result == {"_meta": {"missing_algorithms": []}}
    assert captured["expected_algorithms"] == {"sf", "ppo"}


def test_compute_aggregates_preserves_legacy_keyword_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a legacy callable that rejects the keyword may use the compatibility retry."""

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


def test_compute_aggregates_reraises_internal_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An internal aggregation TypeError must not silently disable completeness checks."""

    calls = 0

    def broken_aggregator(**_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise TypeError("malformed episode payload")

    monkeypatch.setattr(benchmark, "compute_aggregates_with_ci", broken_aggregator)

    with pytest.raises(TypeError, match="malformed episode payload"):
        benchmark._compute_aggregates_payload([], expected_algorithms={"sf"})

    assert calls == 1
