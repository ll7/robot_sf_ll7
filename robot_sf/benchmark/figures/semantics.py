"""Versioned display semantics for benchmark figures and tables.

The registry maps raw metric and planner identifiers to stable labels and units.
Unknown identifiers are never learned silently: strict callers fail, while agent
workflows can emit a reviewable proposal without changing repository state.
"""

# Registry validation intentionally concentrates the complete fail-closed contract.
# ruff: noqa: DOC201

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

SCHEMA = "robot-sf-figure-semantics.v1"
DEFAULT_REGISTRY = Path(__file__).with_name("figure_semantics.v1.json")
_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}")
_LANGUAGES = ("en", "de")


def _object(path: Path) -> dict[str, Any]:
    """Read strict JSON and reject duplicate keys and non-finite constants."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ValueError(f"non-finite JSON literal: {value}")

    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid,
    )
    if not isinstance(value, dict):
        raise ValueError("figure semantics registry must be a JSON object")
    return value


def _identifier(value: Any, field: str) -> str:
    """Validate one stable raw or canonical identifier."""
    if not isinstance(value, str) or _KEY.fullmatch(value) is None:
        raise ValueError(f"{field} must be a stable identifier")
    return value


def _labels(value: Any, field: str) -> dict[str, str]:
    """Validate complete English/German display labels."""
    if not isinstance(value, dict) or set(value) != set(_LANGUAGES):
        raise ValueError(f"{field} must define exactly en and de")
    result: dict[str, str] = {}
    for language in _LANGUAGES:
        label = value.get(language)
        if (
            not isinstance(label, str)
            or not label.strip()
            or len(label) > 160
            or any(ord(char) < 32 for char in label)
        ):
            raise ValueError(f"{field}.{language} must be printable nonempty text")
        result[language] = label.strip()
    return result


def _aliases(value: Any, field: str) -> tuple[str, ...]:
    """Validate an ordered alias list."""
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    aliases = tuple(_identifier(alias, field) for alias in value)
    if len(aliases) != len(set(aliases)):
        raise ValueError(f"{field} contains duplicate aliases")
    return aliases


@dataclass(frozen=True)
class MetricSemantics:
    """Display and ordering contract for one canonical metric."""

    key: str
    aliases: tuple[str, ...]
    labels: dict[str, str]
    short_labels: dict[str, str]
    unit: str
    number_format: str
    direction: Literal["higher", "lower", "context"]
    scale: Literal["linear", "log"]
    order: int

    @classmethod
    def from_payload(cls, key: str, value: Any) -> MetricSemantics:
        """Validate one metric registry row without accepting unknown fields."""
        key = _identifier(key, "metric key")
        if not isinstance(value, dict):
            raise ValueError(f"metric {key} must be an object")
        expected = {
            "aliases",
            "labels",
            "short_labels",
            "unit",
            "number_format",
            "direction",
            "scale",
            "order",
        }
        if set(value) != expected:
            raise ValueError(f"metric {key} has missing or unknown fields")
        unit = value["unit"]
        if not isinstance(unit, str) or len(unit) > 32 or any(ord(char) < 32 for char in unit):
            raise ValueError(f"metric {key} unit must be short printable text")
        number_format = value["number_format"]
        if not isinstance(number_format, str) or len(number_format) > 20:
            raise ValueError(f"metric {key} number_format is invalid")
        try:
            format(1.25, number_format)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"metric {key} number_format is invalid") from exc
        direction = value["direction"]
        if direction not in {"higher", "lower", "context"}:
            raise ValueError(f"metric {key} direction is invalid")
        scale = value["scale"]
        if scale not in {"linear", "log"}:
            raise ValueError(f"metric {key} scale is invalid")
        order = value["order"]
        if type(order) is not int or not 0 <= order <= 10000:
            raise ValueError(f"metric {key} order must be an integer in [0, 10000]")
        return cls(
            key=key,
            aliases=_aliases(value["aliases"], f"metric {key} aliases"),
            labels=_labels(value["labels"], f"metric {key} labels"),
            short_labels=_labels(value["short_labels"], f"metric {key} short_labels"),
            unit=unit,
            number_format=number_format,
            direction=direction,
            scale=scale,
            order=order,
        )


@dataclass(frozen=True)
class PlannerSemantics:
    """Display order and non-color distinction for one planner family."""

    key: str
    aliases: tuple[str, ...]
    labels: dict[str, str]
    order: int
    marker: str
    line_style: str

    @classmethod
    def from_payload(cls, key: str, value: Any) -> PlannerSemantics:
        """Validate one planner registry row."""
        key = _identifier(key, "planner key")
        if not isinstance(value, dict):
            raise ValueError(f"planner {key} must be an object")
        expected = {"aliases", "labels", "order", "marker", "line_style"}
        if set(value) != expected:
            raise ValueError(f"planner {key} has missing or unknown fields")
        order = value["order"]
        if type(order) is not int or not 0 <= order <= 10000:
            raise ValueError(f"planner {key} order must be an integer in [0, 10000]")
        marker = value["marker"]
        line_style = value["line_style"]
        if not isinstance(marker, str) or not marker or len(marker) > 12:
            raise ValueError(f"planner {key} marker is invalid")
        if line_style not in {"-", "--", "-.", ":"}:
            raise ValueError(f"planner {key} line_style is invalid")
        return cls(
            key=key,
            aliases=_aliases(value["aliases"], f"planner {key} aliases"),
            labels=_labels(value["labels"], f"planner {key} labels"),
            order=order,
            marker=marker,
            line_style=line_style,
        )


@dataclass(frozen=True)
class SemanticsRegistry:
    """Strict canonical metric/planner registry with alias resolution."""

    metrics: dict[str, MetricSemantics]
    planners: dict[str, PlannerSemantics]
    metric_aliases: dict[str, str]
    planner_aliases: dict[str, str]

    @classmethod
    def from_file(cls, path: Path = DEFAULT_REGISTRY) -> SemanticsRegistry:
        """Load and validate the complete registry from one versioned JSON file."""
        payload = _object(path)
        if set(payload) != {"schema_version", "metrics", "planners"}:
            raise ValueError("figure semantics registry has missing or unknown top-level fields")
        if payload["schema_version"] != SCHEMA:
            raise ValueError(f"figure semantics schema_version must be {SCHEMA}")
        raw_metrics = payload["metrics"]
        raw_planners = payload["planners"]
        if not isinstance(raw_metrics, dict) or not raw_metrics:
            raise ValueError("figure semantics metrics must be a nonempty object")
        if not isinstance(raw_planners, dict) or not raw_planners:
            raise ValueError("figure semantics planners must be a nonempty object")
        metrics = {
            key: MetricSemantics.from_payload(key, value) for key, value in raw_metrics.items()
        }
        planners = {
            key: PlannerSemantics.from_payload(key, value) for key, value in raw_planners.items()
        }
        metric_aliases = cls._alias_map(metrics, "metric")
        planner_aliases = cls._alias_map(planners, "planner")
        return cls(metrics, planners, metric_aliases, planner_aliases)

    @staticmethod
    def _alias_map(rows: dict[str, Any], kind: str) -> dict[str, str]:
        """Build a collision-free canonical/alias lookup."""
        result: dict[str, str] = {}
        for key, row in rows.items():
            for alias in (key, *row.aliases):
                if alias in result:
                    raise ValueError(
                        f"{kind} alias {alias!r} maps to both {result[alias]!r} and {key!r}"
                    )
                result[alias] = key
        return result

    def payload(self) -> dict[str, Any]:
        """Return the canonical portable registry payload."""
        metrics: dict[str, Any] = {}
        for key, row in self.metrics.items():
            value = asdict(row)
            value.pop("key")
            value["aliases"] = list(row.aliases)
            metrics[key] = value
        planners: dict[str, Any] = {}
        for key, row in self.planners.items():
            value = asdict(row)
            value.pop("key")
            value["aliases"] = list(row.aliases)
            planners[key] = value
        return {"schema_version": SCHEMA, "metrics": metrics, "planners": planners}

    def canonical_json(self) -> str:
        """Serialize the validated registry deterministically."""
        return json.dumps(self.payload(), sort_keys=True, indent=2, allow_nan=False) + "\n"

    def sha256(self) -> str:
        """Hash normalized semantics rather than source formatting."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def metric(self, key: str) -> MetricSemantics:
        """Resolve one metric key or alias, failing when it is not registered."""
        normalized = key.strip() if isinstance(key, str) else ""
        canonical = self.metric_aliases.get(normalized)
        if canonical is None:
            raise KeyError(f"unmapped metric: {normalized or '<empty>'}")
        return self.metrics[canonical]

    def planner(self, key: str) -> PlannerSemantics:
        """Resolve one planner key or alias, failing when it is not registered."""
        normalized = key.strip() if isinstance(key, str) else ""
        canonical = self.planner_aliases.get(normalized)
        if canonical is None:
            raise KeyError(f"unmapped planner: {normalized or '<empty>'}")
        return self.planners[canonical]

    def metric_label(
        self,
        key: str,
        *,
        language: Literal["en", "de"] = "en",
        short: bool = False,
        include_unit: bool = True,
        aggregation: str | None = None,
        strict: bool = True,
        legacy_humanize: bool = False,
    ) -> str:
        """Render one label; compatibility fallback is explicit and opt-in."""
        if language not in _LANGUAGES:
            raise ValueError("language must be en or de")
        normalized = key.strip() if isinstance(key, str) else ""
        try:
            row = self.metric(normalized)
        except KeyError:
            if strict:
                raise
            label = (
                normalized.replace("_", " ").strip().title()
                if normalized and legacy_humanize
                else f"[unmapped: {normalized or '<empty>'}]"
            )
            if not normalized and legacy_humanize:
                label = "Metric"
            parts = [label]
        else:
            labels = row.short_labels if short else row.labels
            parts = [labels[language]]
            if include_unit and row.unit:
                parts.append(f"({row.unit})")
        if aggregation and aggregation.strip():
            parts.append(f"({aggregation.strip()})")
        return " ".join(parts)

    def planner_label(
        self,
        key: str,
        *,
        language: Literal["en", "de"] = "en",
        strict: bool = True,
    ) -> str:
        """Render a planner label or an explicit unmapped marker."""
        if language not in _LANGUAGES:
            raise ValueError("language must be en or de")
        try:
            return self.planner(key).labels[language]
        except KeyError:
            if strict:
                raise
            normalized = key.strip() if isinstance(key, str) else ""
            return f"[unmapped planner: {normalized or '<empty>'}]"

    def suggest_metric(
        self,
        key: str,
        *,
        unit: str = "",
        contexts: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Return a review proposal; never mutate the registry or infer scientific meaning."""
        key = _identifier(key.strip(), "metric key")
        if key in self.metric_aliases:
            raise ValueError(f"metric is already mapped to {self.metric_aliases[key]}")
        if not isinstance(unit, str) or len(unit) > 32:
            raise ValueError("unit must be short text")
        if any(not isinstance(item, str) or not item.strip() for item in contexts):
            raise ValueError("contexts must be nonempty strings")
        humanized = key.replace("_", " ").strip().title()
        return {
            "proposal_only": True,
            "canonical_key": key,
            "registry_schema": SCHEMA,
            "candidate": {
                "aliases": [],
                "labels": {"en": humanized, "de": "REVIEW REQUIRED"},
                "short_labels": {"en": humanized, "de": "REVIEW REQUIRED"},
                "unit": unit,
                "number_format": ".3f",
                "direction": "context",
                "scale": "linear",
                "order": 9999,
            },
            "contexts": list(contexts),
            "required_review": [
                "confirm canonical source variable and aliases",
                "confirm English and German display labels",
                "confirm unit, number format, direction, scale, and ordering",
                "commit the accepted registry diff before publication use",
            ],
        }


@lru_cache(maxsize=1)
def default_registry() -> SemanticsRegistry:
    """Return the validated packaged registry."""
    return SemanticsRegistry.from_file(DEFAULT_REGISTRY)


def metric_label(
    metric_key: str,
    *,
    language: Literal["en", "de"] = "en",
    short: bool = False,
    include_unit: bool = True,
    aggregation: str | None = None,
    strict: bool = True,
) -> str:
    """Resolve one metric through the packaged registry."""
    return default_registry().metric_label(
        metric_key,
        language=language,
        short=short,
        include_unit=include_unit,
        aggregation=aggregation,
        strict=strict,
    )


def planner_label(
    planner_key: str,
    *,
    language: Literal["en", "de"] = "en",
    strict: bool = True,
) -> str:
    """Resolve one planner through the packaged registry."""
    return default_registry().planner_label(planner_key, language=language, strict=strict)


def main(argv: list[str] | None = None) -> int:
    """Validate, query, or draft a review-only registry proposal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--metric")
    action.add_argument("--planner")
    action.add_argument("--suggest-metric")
    parser.add_argument("--language", choices=_LANGUAGES, default="en")
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--unit", default="")
    parser.add_argument("--context", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        registry = SemanticsRegistry.from_file(args.registry)
        if args.metric:
            payload: Any = {
                "canonical_key": registry.metric(args.metric).key,
                "label": registry.metric_label(
                    args.metric, language=args.language, short=args.short
                ),
                "semantics": asdict(registry.metric(args.metric)),
            }
        elif args.planner:
            payload = {
                "canonical_key": registry.planner(args.planner).key,
                "label": registry.planner_label(args.planner, language=args.language),
                "semantics": asdict(registry.planner(args.planner)),
            }
        elif args.suggest_metric:
            payload = registry.suggest_metric(
                args.suggest_metric,
                unit=args.unit,
                contexts=tuple(args.context),
            )
        else:
            payload = {
                "schema_version": SCHEMA,
                "sha256": registry.sha256(),
                "metric_count": len(registry.metrics),
                "planner_count": len(registry.planners),
            }
    except (KeyError, OSError, ValueError) as exc:
        parser.exit(2, f"figure semantics: {exc}\n")
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
