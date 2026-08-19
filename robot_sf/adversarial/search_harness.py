"""Preparation-only search contracts for adversarial scenario exploration.

This module freezes the finite, typed input surface needed before an adversarial search can be
authorized. It composes the existing adversarial candidate and materialization seams, produces
equal-budget random and Halton quasi-random proposal baselines, and records pre-simulation
rejections. It intentionally has no simulator, optimizer, campaign, or result-ranking path.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias

import yaml

from robot_sf.adversarial.bundle import (
    build_candidate_payload,
    validate_template_pedestrian_binding,
)
from robot_sf.adversarial.config import CandidateSpec, Pose2D
from robot_sf.adversarial.materialize import ImmutableScenarioOverlay

SEARCH_HARNESS_SCHEMA_VERSION = "adversarial_search_harness.v1"
PREPARATION_SCHEMA_VERSION = "adversarial_search_preparation.v1"
BASELINE_NAMES: tuple[str, ...] = ("random", "quasi_random")
CLAIM_BOUNDARY = (
    "diagnostic-only preparation: typed proposal generation, immutable overlays, and "
    "pre-simulation rejection accounting; no simulator or benchmark evidence"
)

Number: TypeAlias = int | float  # noqa: UP040 - repository supports Python 3.11
VariableKind: TypeAlias = Literal["continuous", "integer"]  # noqa: UP040
ObjectiveDirection: TypeAlias = Literal["maximize", "minimize"]  # noqa: UP040
CandidateSeedMode: TypeAlias = Literal["index_derived"]  # noqa: UP040


def _require_text(value: object, name: str) -> str:
    """Return a non-empty trimmed string or raise a contract error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_number(value: object, name: str) -> float:
    """Return a finite number while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _non_negative_int(value: object, name: str) -> int:
    """Return a non-negative integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    """Return a positive integer while rejecting booleans."""
    parsed = _non_negative_int(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _canonical_json(payload: object) -> bytes:
    """Encode a JSON-compatible payload deterministically."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: object) -> str:
    """Return the SHA-256 digest of a deterministic JSON payload."""
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class FiniteBounds:
    """Inclusive finite bounds for one scalar search variable."""

    min: float
    max: float
    kind: VariableKind = "continuous"

    def __post_init__(self) -> None:
        """Validate bounds and integer alignment."""
        lower = _finite_number(self.min, "bounds.min")
        upper = _finite_number(self.max, "bounds.max")
        if lower > upper:
            raise ValueError("bounds.min must be <= bounds.max")
        if self.kind not in {"continuous", "integer"}:
            raise ValueError("bounds.kind must be continuous or integer")
        if self.kind == "integer" and (not lower.is_integer() or not upper.is_integer()):
            raise ValueError("integer bounds must have integral min and max")
        object.__setattr__(self, "min", lower)
        object.__setattr__(self, "max", upper)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        name: str,
        kind: str = "continuous",
    ) -> FiniteBounds:
        """Parse one bounds mapping using the repository's inclusive ``min``/``max`` vocabulary."""
        if not isinstance(payload, Mapping):
            raise ValueError(f"{name}.bounds must be a mapping")
        raw_min = payload.get("min", payload.get("minimum"))
        raw_max = payload.get("max", payload.get("maximum"))
        if raw_min is None or raw_max is None:
            raise ValueError(f"{name}.bounds must define min and max")
        return cls(float(raw_min), float(raw_max), kind=kind)  # type: ignore[arg-type]

    def value_at(self, coordinate: float) -> Number:
        """Map a unit-interval coordinate into this inclusive finite interval."""
        unit = _finite_number(coordinate, "unit coordinate")
        if not 0.0 <= unit <= 1.0:
            raise ValueError("unit coordinate must be between 0 and 1")
        if self.kind == "integer":
            low = int(self.min)
            high = int(self.max)
            return low + min(high - low, math.floor(unit * (high - low + 1)))
        return float(self.min + unit * (self.max - self.min))

    def contains(self, value: object) -> bool:
        """Return whether a value is finite, typed, and within the inclusive bounds."""
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        number = float(value)
        if not math.isfinite(number) or not self.min <= number <= self.max:
            return False
        return self.kind != "integer" or number.is_integer()

    def to_dict(self) -> dict[str, Number | str]:
        """Return a JSON-safe bounds mapping."""
        if self.kind == "integer":
            return {"kind": self.kind, "max": int(self.max), "min": int(self.min)}
        return {"kind": self.kind, "max": float(self.max), "min": float(self.min)}


@dataclass(frozen=True, slots=True)
class SearchVariable:
    """Named, unit-annotated variable in a finite search space."""

    name: str
    unit: str
    bounds: FiniteBounds

    def __post_init__(self) -> None:
        """Validate the variable identity and unit."""
        _require_text(self.name, "variable.name")
        _require_text(self.unit, f"variable {self.name!r}.unit")
        if not isinstance(self.bounds, FiniteBounds):
            raise TypeError("variable.bounds must be FiniteBounds")

    @classmethod
    def from_mapping(cls, name: str, payload: Mapping[str, Any]) -> SearchVariable:
        """Parse a variable mapping with nested or inline bounds."""
        variable_name = _require_text(name, "variable name")
        if not isinstance(payload, Mapping):
            raise ValueError(f"variables.{variable_name} must be a mapping")
        unit = _require_text(payload.get("unit"), f"variables.{variable_name}.unit")
        raw_bounds = payload.get("bounds", payload)
        raw_kind = payload.get("kind", payload.get("dtype"))
        if raw_kind is None and isinstance(raw_bounds, Mapping):
            raw_kind = raw_bounds.get("kind")
        kind = str(raw_kind or "continuous").strip().lower()
        return cls(
            name=variable_name,
            unit=unit,
            bounds=FiniteBounds.from_mapping(
                raw_bounds, name=f"variables.{variable_name}", kind=kind
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe variable mapping."""
        return {"bounds": self.bounds.to_dict(), "name": self.name, "unit": self.unit}


_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Name,
    ast.Load,
    ast.Constant,
)


def _constraint_names(tree: ast.AST) -> tuple[str, ...]:
    """Return referenced variable names from a restricted constraint expression."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(
                f"constraint expression uses unsupported syntax: {type(node).__name__}"
            )
        if isinstance(node, ast.Name):
            names.add(node.id)
        if isinstance(node, ast.Constant) and not (
            isinstance(node.value, (int, float, bool)) and not isinstance(node.value, complex)
        ):
            raise ValueError("constraint constants must be numeric or boolean")
    return tuple(sorted(names))


def _evaluate_constraint_node(  # noqa: C901, PLR0912
    node: ast.AST, values: Mapping[str, Number]
) -> Number | bool:
    """Evaluate one previously validated arithmetic/boolean AST node."""
    if isinstance(node, ast.Expression):
        return _evaluate_constraint_node(node.body, values)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_constraint_node(node.operand, values)
        return +value if isinstance(node.op, ast.UAdd) else -value  # type: ignore[operator]
    if isinstance(node, ast.BinOp):
        left = _evaluate_constraint_node(node.left, values)
        right = _evaluate_constraint_node(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right  # type: ignore[operator]
        if isinstance(node.op, ast.Sub):
            return left - right  # type: ignore[operator]
        if isinstance(node.op, ast.Mult):
            return left * right  # type: ignore[operator]
        if isinstance(node.op, ast.Div):
            return left / right  # type: ignore[operator]
        if isinstance(node.op, ast.Mod):
            return left % right  # type: ignore[operator]
    if isinstance(node, ast.BoolOp):
        results = [_evaluate_constraint_node(value, values) for value in node.values]
        return (
            all(bool(value) for value in results)
            if isinstance(node.op, ast.And)
            else any(bool(value) for value in results)
        )
    if isinstance(node, ast.Compare):
        left: Number | bool = _evaluate_constraint_node(node.left, values)
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = _evaluate_constraint_node(comparator, values)
            if isinstance(operator, ast.Eq):
                passed = left == right
            elif isinstance(operator, ast.NotEq):
                passed = left != right
            elif isinstance(operator, ast.Lt):
                passed = left < right  # type: ignore[operator]
            elif isinstance(operator, ast.LtE):
                passed = left <= right  # type: ignore[operator]
            elif isinstance(operator, ast.Gt):
                passed = left > right  # type: ignore[operator]
            elif isinstance(operator, ast.GtE):
                passed = left >= right  # type: ignore[operator]
            else:  # pragma: no cover - AST validation rejects this branch
                raise ValueError("unsupported comparison operator")
            if not passed:
                return False
            left = right
        return True
    raise ValueError(f"unsupported constraint node: {type(node).__name__}")


@dataclass(frozen=True, slots=True)
class CrossVariableConstraint:
    """Named, safely evaluated cross-variable predicate."""

    name: str
    expression: str

    def __post_init__(self) -> None:
        """Parse and validate the restricted expression grammar."""
        _require_text(self.name, "constraint.name")
        expression = _require_text(self.expression, f"constraint {self.name!r}.expression")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"constraint {self.name!r}.expression is invalid") from exc
        names = _constraint_names(tree)
        if not names:
            raise ValueError(f"constraint {self.name!r}.expression must reference variables")

    @property
    def referenced_variables(self) -> tuple[str, ...]:
        """Return sorted variable names referenced by the predicate."""
        tree = ast.parse(self.expression, mode="eval")
        return _constraint_names(tree)

    def satisfied(self, values: Mapping[str, Number]) -> bool:
        """Evaluate the predicate, treating arithmetic failures as rejection."""
        try:
            result = _evaluate_constraint_node(ast.parse(self.expression, mode="eval"), values)
        except (KeyError, OverflowError, TypeError, ValueError, ZeroDivisionError):
            return False
        return isinstance(result, bool) and result

    def rejection_reason(self) -> str:
        """Return the stable rejection code for this constraint."""
        return f"constraint:{self.name}:unsatisfied"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, index: int) -> CrossVariableConstraint:
        """Parse one strict named expression predicate."""
        if not isinstance(payload, Mapping):
            raise ValueError(f"constraints[{index}] must be a mapping")
        return cls(
            name=_require_text(payload.get("name"), f"constraints[{index}].name"),
            expression=_require_text(payload.get("expression"), f"constraints[{index}].expression"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe constraint mapping."""
        return {
            "expression": self.expression,
            "name": self.name,
            "variables": list(self.referenced_variables),
        }


@dataclass(frozen=True, slots=True)
class ObjectiveComponent:
    """One named component of the declared objective vector."""

    name: str
    direction: ObjectiveDirection
    unit: str

    def __post_init__(self) -> None:
        """Validate objective identity, direction, and unit."""
        _require_text(self.name, "objective component name")
        _require_text(self.unit, f"objective component {self.name!r}.unit")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("objective component direction must be maximize or minimize")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, index: int) -> ObjectiveComponent:
        """Parse one objective component."""
        if not isinstance(payload, Mapping):
            raise ValueError(f"objective_vector.components[{index}] must be a mapping")
        return cls(
            name=_require_text(payload.get("name"), f"objective_vector.components[{index}].name"),
            direction=str(payload.get("direction", "maximize")).strip().lower(),  # type: ignore[arg-type]
            unit=_require_text(payload.get("unit"), f"objective_vector.components[{index}].unit"),
        )

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe objective component mapping."""
        return {"direction": self.direction, "name": self.name, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class ObjectiveVector:
    """Ordered objective components without choosing an optimizer or scalarization."""

    components: tuple[ObjectiveComponent, ...]

    def __post_init__(self) -> None:
        """Require a non-empty vector with unique component names."""
        if not self.components:
            raise ValueError("objective_vector.components must be non-empty")
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("objective_vector component names must be unique")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ObjectiveVector:
        """Parse the declared objective vector without ranking observed outcomes."""
        components = payload.get("components") if isinstance(payload, Mapping) else None
        if not isinstance(components, list) or not components:
            raise ValueError("objective_vector.components must be a non-empty list")
        return cls(
            tuple(
                ObjectiveComponent.from_mapping(item, index=index)
                for index, item in enumerate(components)
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe objective vector mapping."""
        return {"components": [component.to_dict() for component in self.components]}


@dataclass(frozen=True, slots=True)
class SeedPolicy:
    """Search and held-out replay seed policy for preparation artifacts."""

    search_seed: int
    held_out_replay_seeds: tuple[int, ...] = ()
    candidate_seed_mode: CandidateSeedMode = "index_derived"

    def __post_init__(self) -> None:
        """Validate seed separation and deterministic derivation mode."""
        _non_negative_int(self.search_seed, "seed_policy.search_seed")
        if self.candidate_seed_mode != "index_derived":
            raise ValueError("seed_policy.candidate_seed_mode must be index_derived")
        replay = tuple(
            _non_negative_int(seed, "held-out replay seed") for seed in self.held_out_replay_seeds
        )
        if len(replay) != len(set(replay)):
            raise ValueError("seed_policy.held_out_replay_seeds must be unique")
        if self.search_seed in replay:
            raise ValueError("search seed must not overlap held-out replay seeds")
        object.__setattr__(self, "held_out_replay_seeds", tuple(sorted(replay)))

    def candidate_seed(self, candidate_index: int) -> int:
        """Derive a stable candidate seed disjoint from held-out replay seeds."""
        index = _non_negative_int(candidate_index, "candidate_index")
        raw = f"{self.search_seed}:candidate:{index}".encode()
        candidate_seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31 - 1)
        while candidate_seed in self.held_out_replay_seeds:
            candidate_seed = (candidate_seed + 1) % (2**31 - 1)
        return candidate_seed

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SeedPolicy:
        """Parse a seed policy mapping."""
        if not isinstance(payload, Mapping):
            raise ValueError("seed_policy must be a mapping")
        raw_replay = payload.get("held_out_replay_seeds", payload.get("replay_seeds", ()))
        if not isinstance(raw_replay, Sequence) or isinstance(raw_replay, str | bytes):
            raise ValueError("seed_policy.held_out_replay_seeds must be a list")
        mode = str(payload.get("candidate_seed_mode", "index_derived")).strip().lower()
        if mode in {"derived", "index-derived"}:
            mode = "index_derived"
        return cls(
            search_seed=_non_negative_int(payload.get("search_seed"), "seed_policy.search_seed"),
            held_out_replay_seeds=tuple(
                _non_negative_int(seed, "held-out replay seed") for seed in raw_replay
            ),
            candidate_seed_mode=mode,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe seed policy mapping."""
        return {
            "candidate_seed_mode": self.candidate_seed_mode,
            "held_out_replay_seeds": list(self.held_out_replay_seeds),
            "search_seed": self.search_seed,
        }


@dataclass(frozen=True, slots=True)
class RolloutBudget:
    """Finite preparation budget shared by every baseline arm."""

    candidate_budget: int
    rollouts_per_candidate: int = 1
    max_steps: int = 1

    def __post_init__(self) -> None:
        """Validate candidate, rollout, and step budgets."""
        _positive_int(self.candidate_budget, "rollout_budget.candidate_budget")
        _positive_int(self.rollouts_per_candidate, "rollout_budget.rollouts_per_candidate")
        _positive_int(self.max_steps, "rollout_budget.max_steps")

    @property
    def total_rollouts(self) -> int:
        """Return the declared total rollout count for one baseline arm."""
        return self.candidate_budget * self.rollouts_per_candidate

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RolloutBudget:
        """Parse a rollout budget mapping with a candidate-count alias."""
        if not isinstance(payload, Mapping):
            raise ValueError("rollout_budget must be a mapping")
        candidate_budget = payload.get(
            "candidate_budget", payload.get("candidates", payload.get("max_candidates"))
        )
        return cls(
            candidate_budget=_positive_int(candidate_budget, "rollout_budget.candidate_budget"),
            rollouts_per_candidate=_positive_int(
                payload.get("rollouts_per_candidate", 1),
                "rollout_budget.rollouts_per_candidate",
            ),
            max_steps=_positive_int(
                payload.get("max_steps", payload.get("horizon_steps", 1)),
                "rollout_budget.max_steps",
            ),
        )

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-safe rollout budget mapping."""
        return {
            "candidate_budget": self.candidate_budget,
            "max_steps": self.max_steps,
            "rollouts_per_candidate": self.rollouts_per_candidate,
        }


@dataclass(frozen=True, slots=True)
class FiniteSearchSpaceManifest:
    """Typed finite search-space, objective, seed, and rollout contract."""

    name: str
    variables: tuple[SearchVariable, ...]
    constraints: tuple[CrossVariableConstraint, ...]
    objective_vector: ObjectiveVector
    seed_policy: SeedPolicy
    rollout_budget: RolloutBudget
    description: str = ""
    source_scenario: str | None = None
    schema_version: str = SEARCH_HARNESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate names, schema identity, and cross-variable references."""
        _require_text(self.name, "manifest.name")
        if self.schema_version != SEARCH_HARNESS_SCHEMA_VERSION:
            raise ValueError(f"manifest.schema_version must be {SEARCH_HARNESS_SCHEMA_VERSION}")
        if not self.variables:
            raise ValueError("manifest.variables must be non-empty")
        variable_names = [variable.name for variable in self.variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("manifest variable names must be unique")
        if self.source_scenario is not None:
            _require_text(self.source_scenario, "manifest.source_scenario")
        constraint_names = [constraint.name for constraint in self.constraints]
        if len(constraint_names) != len(set(constraint_names)):
            raise ValueError("manifest constraint names must be unique")
        known_names = set(variable_names)
        for constraint in self.constraints:
            unknown = set(constraint.referenced_variables) - known_names
            if unknown:
                raise ValueError(
                    f"constraint {constraint.name!r} references unknown variables: "
                    + ", ".join(sorted(unknown))
                )

    @classmethod
    def from_file(cls, path: str | Path) -> FiniteSearchSpaceManifest:
        """Load a YAML search-harness manifest."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"search-harness manifest must be a mapping: {path}")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FiniteSearchSpaceManifest:
        """Parse the strict manifest shape used by the preparation-only fixture."""
        if not isinstance(payload, Mapping):
            raise ValueError("search-harness manifest must be a mapping")
        variables_raw = payload.get("variables")
        variables: list[SearchVariable] = []
        if isinstance(variables_raw, Mapping):
            variables = [
                SearchVariable.from_mapping(name, variable_payload)
                for name, variable_payload in variables_raw.items()
            ]
        elif isinstance(variables_raw, list):
            for index, variable_payload in enumerate(variables_raw):
                if not isinstance(variable_payload, Mapping):
                    raise ValueError(f"variables[{index}] must be a mapping")
                variables.append(
                    SearchVariable.from_mapping(
                        _require_text(variable_payload.get("name"), f"variables[{index}].name"),
                        variable_payload,
                    )
                )
        else:
            raise ValueError("manifest.variables must be a mapping or list")

        constraints_raw = payload.get("constraints", [])
        if not isinstance(constraints_raw, list):
            raise ValueError("manifest.constraints must be a list")
        objective_raw = payload.get("objective_vector", payload.get("objective"))
        if not isinstance(objective_raw, Mapping):
            raise ValueError("manifest.objective_vector must be a mapping")
        return cls(
            schema_version=str(payload.get("schema_version", SEARCH_HARNESS_SCHEMA_VERSION)),
            name=_require_text(payload.get("name"), "manifest.name"),
            description=str(payload.get("description", "")).strip(),
            source_scenario=(
                _require_text(payload["source_scenario"], "manifest.source_scenario")
                if payload.get("source_scenario") is not None
                else None
            ),
            variables=tuple(variables),
            constraints=tuple(
                CrossVariableConstraint.from_mapping(item, index=index)
                for index, item in enumerate(constraints_raw)
            ),
            objective_vector=ObjectiveVector.from_mapping(objective_raw),
            seed_policy=SeedPolicy.from_mapping(payload.get("seed_policy", {})),
            rollout_budget=RolloutBudget.from_mapping(payload.get("rollout_budget", {})),
        )

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Return variables in declared order, which is the sampler dimension order."""
        return tuple(variable.name for variable in self.variables)

    def validate_values(self, values: Mapping[str, Number]) -> tuple[str, ...]:
        """Return stable bound and cross-variable rejection codes."""
        reasons: list[str] = []
        declared = set(self.variable_names)
        for name in self.variable_names:
            if name not in values:
                reasons.append(f"variable:{name}:missing")
                continue
            if not self._variable(name).bounds.contains(values[name]):
                reasons.append(f"variable:{name}:outside_bounds")
        for extra_name in sorted(set(values) - declared):
            reasons.append(f"variable:{extra_name}:undeclared")
        if not reasons:
            for constraint in self.constraints:
                if not constraint.satisfied(values):
                    reasons.append(constraint.rejection_reason())
        return tuple(reasons)

    def build_candidate(
        self,
        *,
        baseline: str,
        index: int,
        unit_coordinates: Sequence[float],
    ) -> SearchCandidate:
        """Map one unit-cube point to a typed immutable candidate."""
        baseline_name = _require_text(baseline, "baseline")
        if len(unit_coordinates) != len(self.variables):
            raise ValueError("unit coordinate dimension does not match manifest variables")
        coordinates = tuple(_finite_number(value, "unit coordinate") for value in unit_coordinates)
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError("unit coordinates must be between 0 and 1")
        values = tuple(
            (variable.name, variable.bounds.value_at(coordinate))
            for variable, coordinate in zip(self.variables, coordinates, strict=True)
        )
        candidate_index = _non_negative_int(index, "candidate index")
        return SearchCandidate(
            baseline=baseline_name,
            candidate_id=f"{baseline_name}:{candidate_index:04d}",
            candidate_index=candidate_index,
            values=values,
            unit_coordinates=coordinates,
            manifest_digest=self.digest,
            search_seed=self.seed_policy.search_seed,
            candidate_seed=self.seed_policy.candidate_seed(candidate_index),
        )

    def _variable(self, name: str) -> SearchVariable:
        """Return a declared variable by name."""
        return next(variable for variable in self.variables if variable.name == name)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON/YAML manifest mapping."""
        payload: dict[str, Any] = {
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "description": self.description,
            "name": self.name,
            "objective_vector": self.objective_vector.to_dict(),
            "rollout_budget": self.rollout_budget.to_dict(),
            "schema_version": self.schema_version,
            "seed_policy": self.seed_policy.to_dict(),
            # A list is used for canonical serialization so declared dimension order survives
            # JSON object-key sorting and remains part of the manifest digest.
            "variables": [variable.to_dict() for variable in self.variables],
        }
        if self.source_scenario is not None:
            payload["source_scenario"] = self.source_scenario
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        """Return deterministic manifest JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            separators=(",", ":") if indent is None else (",", ": "),
            allow_nan=False,
        )

    @property
    def digest(self) -> str:
        """Return the canonical manifest digest used in candidate provenance."""
        return _sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """Immutable typed proposal with sampler and seed provenance."""

    baseline: str
    candidate_id: str
    candidate_index: int
    values: tuple[tuple[str, Number], ...]
    unit_coordinates: tuple[float, ...]
    manifest_digest: str
    search_seed: int
    candidate_seed: int

    def __post_init__(self) -> None:
        """Validate candidate identity, coordinates, and scalar values."""
        _require_text(self.baseline, "candidate.baseline")
        _require_text(self.candidate_id, "candidate.candidate_id")
        _non_negative_int(self.candidate_index, "candidate.candidate_index")
        _require_text(self.manifest_digest, "candidate.manifest_digest")
        _non_negative_int(self.search_seed, "candidate.search_seed")
        _non_negative_int(self.candidate_seed, "candidate.candidate_seed")
        if not self.values:
            raise ValueError("candidate.values must be non-empty")
        names = [name for name, _value in self.values]
        if len(names) != len(set(names)):
            raise ValueError("candidate value names must be unique")
        if len(self.values) != len(self.unit_coordinates):
            raise ValueError("candidate values and unit coordinates must have equal dimensions")
        for name, value in self.values:
            _require_text(name, "candidate value name")
            _finite_number(value, f"candidate.values.{name}")
        for coordinate in self.unit_coordinates:
            number = _finite_number(coordinate, "candidate unit coordinate")
            if not 0.0 <= number <= 1.0:
                raise ValueError("candidate unit coordinates must be between 0 and 1")

    @property
    def value_map(self) -> Mapping[str, Number]:
        """Return an immutable candidate-value mapping."""
        return MappingProxyType(dict(self.values))

    def to_dict(self) -> dict[str, Any]:
        """Return candidate values and deterministic provenance."""
        return {
            "baseline": self.baseline,
            "candidate_id": self.candidate_id,
            "candidate_index": self.candidate_index,
            "candidate_seed": self.candidate_seed,
            "manifest_digest": self.manifest_digest,
            "search_seed": self.search_seed,
            "unit_coordinates": list(self.unit_coordinates),
            "values": dict(self.values),
        }


class CandidateOverlayAdapter(Protocol):
    """Pure adapter seam from a prepared candidate to an immutable scenario overlay."""

    adapter_id: str

    def validate(
        self,
        source_scenario: Mapping[str, Any],
        candidate: SearchCandidate,
    ) -> Sequence[str]:
        """Return adapter-level pre-simulation rejection codes."""

    def materialize(
        self,
        source_scenario: Mapping[str, Any],
        candidate: SearchCandidate,
    ) -> ImmutableScenarioOverlay:
        """Materialize one candidate without simulation or file I/O."""


class BaselineSampler(Protocol):
    """Unit-cube proposal seam shared by every preparation baseline."""

    baseline_id: str

    def unit_coordinates(self, index: int, budget: int, dimensions: int) -> tuple[float, ...]:
        """Return one deterministic unit-cube point."""


def _baseline_seed(seed: int, baseline: str, index: int) -> int:
    """Derive a stable per-proposal random seed without process-global state."""
    raw = f"{seed}:{baseline}:{index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class RandomBaseline:
    """Dependency-free uniform random proposal baseline."""

    seed: int
    baseline_id: str = "random"

    def unit_coordinates(self, index: int, budget: int, dimensions: int) -> tuple[float, ...]:
        """Return one reproducible pseudo-random point."""
        _validate_sampler_request(index, budget, dimensions)
        rng = random.Random(_baseline_seed(self.seed, self.baseline_id, index))
        return tuple(rng.random() for _ in range(dimensions))


def _first_primes(count: int) -> tuple[int, ...]:
    """Return the first ``count`` prime bases for the Halton sequence."""
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % divisor for divisor in range(2, int(math.sqrt(candidate)) + 1)):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _van_der_corput(index: int, base: int) -> float:
    """Return one radical-inverse value in the open unit interval."""
    value = 0.0
    denominator = 1.0
    remainder = index
    while remainder:
        remainder, digit = divmod(remainder, base)
        denominator *= base
        value += digit / denominator
    return value


@dataclass(frozen=True, slots=True)
class QuasiRandomBaseline:
    """Dependency-free Halton low-discrepancy proposal baseline."""

    seed: int
    baseline_id: str = "quasi_random"

    def unit_coordinates(self, index: int, budget: int, dimensions: int) -> tuple[float, ...]:
        """Return one deterministic Halton point with a seed-derived skip."""
        _validate_sampler_request(index, budget, dimensions)
        skip = _baseline_seed(self.seed, self.baseline_id, 0) % 10_000
        point_index = skip + index + 1
        return tuple(_van_der_corput(point_index, base) for base in _first_primes(dimensions))


def _validate_sampler_request(index: int, budget: int, dimensions: int) -> None:
    """Validate one baseline proposal request."""
    _non_negative_int(index, "sampler index")
    _positive_int(budget, "sampler budget")
    _positive_int(dimensions, "sampler dimensions")
    if index >= budget:
        raise ValueError("sampler index must be smaller than budget")


def build_baseline(name: str, *, seed: int) -> BaselineSampler:
    """Build a named preparation baseline; adaptive optimizers are intentionally absent."""
    key = _require_text(name, "baseline").lower().replace("-", "_")
    if key == "random":
        return RandomBaseline(seed=_non_negative_int(seed, "baseline seed"))
    if key in {"quasi_random", "halton"}:
        return QuasiRandomBaseline(seed=_non_negative_int(seed, "baseline seed"))
    raise ValueError("baseline must be one of: random, quasi_random")


@dataclass(frozen=True, slots=True)
class RejectionRecord:
    """Deterministic pre-simulation rejection ledger entry."""

    candidate_id: str
    candidate_values: Mapping[str, Number]
    stage: Literal["manifest", "adapter"]
    reasons: tuple[str, ...]
    simulation_executed: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate rejection identity and freeze nested details."""
        _require_text(self.candidate_id, "rejection.candidate_id")
        if self.stage not in {"manifest", "adapter"}:
            raise ValueError("rejection.stage must be manifest or adapter")
        if not self.reasons:
            raise ValueError("rejection.reasons must be non-empty")
        if self.simulation_executed:
            raise ValueError("pre-simulation rejection cannot record simulation execution")
        object.__setattr__(self, "candidate_values", MappingProxyType(dict(self.candidate_values)))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe rejection entry."""
        return {
            "candidate_id": self.candidate_id,
            "candidate_values": dict(self.candidate_values),
            "details": dict(self.details),
            "reasons": list(self.reasons),
            "simulation_executed": False,
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class PreparedCandidate:
    """One feasible overlay or one pre-simulation rejection."""

    candidate: SearchCandidate
    overlay: ImmutableScenarioOverlay | None = None
    rejection: RejectionRecord | None = None

    def __post_init__(self) -> None:
        """Require exactly one preparation outcome."""
        if (self.overlay is None) == (self.rejection is None):
            raise ValueError("prepared candidate requires exactly one overlay or rejection")
        if (
            self.rejection is not None
            and self.rejection.candidate_id != self.candidate.candidate_id
        ):
            raise ValueError("rejection candidate_id does not match candidate")

    @property
    def feasible(self) -> bool:
        """Return whether the candidate reached the overlay seam."""
        return self.overlay is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe preparation record."""
        return {
            "candidate": self.candidate.to_dict(),
            "feasible": self.feasible,
            "overlay": self.overlay.to_dict() if self.overlay is not None else None,
            "rejection": self.rejection.to_dict() if self.rejection is not None else None,
        }


@dataclass(frozen=True, slots=True)
class BaselinePreparation:
    """Preparation result for one equal-budget baseline arm."""

    manifest: FiniteSearchSpaceManifest
    baseline: str
    source_digest: str
    candidates: tuple[PreparedCandidate, ...]

    def __post_init__(self) -> None:
        """Validate equal-budget candidate accounting and freeze result rows."""
        if self.baseline not in BASELINE_NAMES:
            raise ValueError("preparation baseline must be random or quasi_random")
        if len(self.candidates) != self.manifest.rollout_budget.candidate_budget:
            raise ValueError("preparation candidate count must equal the declared candidate budget")
        if tuple(row.candidate.baseline for row in self.candidates) != (self.baseline,) * len(
            self.candidates
        ):
            raise ValueError("preparation candidates must belong to the declared baseline")

    @property
    def prepared_count(self) -> int:
        """Return the count of candidates materialized as overlays."""
        return sum(candidate.feasible for candidate in self.candidates)

    @property
    def rejected_count(self) -> int:
        """Return the count of candidates rejected before simulation."""
        return len(self.candidates) - self.prepared_count

    @property
    def provenance(self) -> dict[str, Any]:
        """Return deterministic preparation provenance and claim boundary."""
        return {
            "baseline": self.baseline,
            "candidate_budget": self.manifest.rollout_budget.candidate_budget,
            "claim_boundary": CLAIM_BOUNDARY,
            "evidence_status": "diagnostic_only_preparation",
            "manifest_digest": self.manifest.digest,
            "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
            "search_seed": self.manifest.seed_policy.search_seed,
            "simulation_executed": False,
            "source_digest": self.source_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a stable preparation report with rejection ledger."""
        return {
            "baseline": self.baseline,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "manifest": self.manifest.to_dict(),
            "provenance": self.provenance,
            "summary": {
                "candidate_budget": len(self.candidates),
                "prepared_count": self.prepared_count,
                "rejected_count": self.rejected_count,
                "rollouts_per_candidate": self.manifest.rollout_budget.rollouts_per_candidate,
                "total_declared_rollouts": self.manifest.rollout_budget.total_rollouts,
            },
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Return deterministic JSON for a preparation artifact."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            separators=(",", ":") if indent is None else (",", ": "),
            allow_nan=False,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write a deterministic preparation artifact without running a campaign."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path


def _source_digest(source_scenario: Mapping[str, Any]) -> str:
    """Hash one source scenario through the immutable overlay serializer."""
    snapshot = ImmutableScenarioOverlay(
        source=source_scenario,
        patch={},
        candidate_id="source",
        adapter_id="source_snapshot.v1",
    )
    return snapshot.source_digest


def _adapter_rejection(
    candidate: SearchCandidate,
    *,
    stage: Literal["manifest", "adapter"],
    reasons: Sequence[str],
    details: Mapping[str, Any] | None = None,
) -> RejectionRecord:
    """Build one normalized rejection record."""
    normalized = tuple(str(reason).strip() for reason in reasons if str(reason).strip())
    return RejectionRecord(
        candidate_id=candidate.candidate_id,
        candidate_values=candidate.value_map,
        stage=stage,
        reasons=normalized,
        details=details or {},
    )


def prepare_baseline(
    manifest: FiniteSearchSpaceManifest,
    source_scenario: Mapping[str, Any],
    adapter: CandidateOverlayAdapter,
    *,
    baseline: str,
) -> BaselinePreparation:
    """Prepare one exact-budget baseline without invoking simulation.

    Manifest bounds and cross-variable constraints are checked before the adapter is called.
    Adapter validation/materialization is also pre-simulation and failures become explicit ledger
    rows. The returned overlays are immutable snapshots suitable for a later, separately approved
    execution layer.
    """
    baseline_name = _require_text(baseline, "baseline").lower().replace("-", "_")
    if baseline_name not in BASELINE_NAMES:
        raise ValueError("baseline must be one of: random, quasi_random")
    source_snapshot = ImmutableScenarioOverlay(
        source=source_scenario,
        patch={},
        candidate_id="source",
        adapter_id="source_snapshot.v1",
    ).source
    sampler = build_baseline(baseline_name, seed=manifest.seed_policy.search_seed)
    budget = manifest.rollout_budget.candidate_budget
    rows: list[PreparedCandidate] = []
    for index in range(budget):
        coordinates = sampler.unit_coordinates(index, budget, len(manifest.variables))
        candidate = manifest.build_candidate(
            baseline=baseline_name,
            index=index,
            unit_coordinates=coordinates,
        )
        manifest_reasons = manifest.validate_values(candidate.value_map)
        if manifest_reasons:
            rows.append(
                PreparedCandidate(
                    candidate=candidate,
                    rejection=_adapter_rejection(
                        candidate,
                        stage="manifest",
                        reasons=manifest_reasons,
                    ),
                )
            )
            continue
        try:
            adapter_reasons = tuple(adapter.validate(source_snapshot, candidate))
        except Exception as exc:  # noqa: BLE001 - adapter boundary records a stable rejection
            rows.append(
                PreparedCandidate(
                    candidate=candidate,
                    rejection=_adapter_rejection(
                        candidate,
                        stage="adapter",
                        reasons=(f"adapter_validation_error:{type(exc).__name__}",),
                    ),
                )
            )
            continue
        if adapter_reasons:
            rows.append(
                PreparedCandidate(
                    candidate=candidate,
                    rejection=_adapter_rejection(
                        candidate,
                        stage="adapter",
                        reasons=adapter_reasons,
                    ),
                )
            )
            continue
        try:
            overlay = adapter.materialize(source_snapshot, candidate)
            if not isinstance(overlay, ImmutableScenarioOverlay):
                raise TypeError("adapter must return ImmutableScenarioOverlay")
        except Exception as exc:  # noqa: BLE001 - adapter boundary records a stable rejection
            rows.append(
                PreparedCandidate(
                    candidate=candidate,
                    rejection=_adapter_rejection(
                        candidate,
                        stage="adapter",
                        reasons=(f"adapter_materialization_error:{type(exc).__name__}",),
                    ),
                )
            )
            continue
        rows.append(PreparedCandidate(candidate=candidate, overlay=overlay))
    return BaselinePreparation(
        manifest=manifest,
        baseline=baseline_name,
        source_digest=_source_digest(source_scenario),
        candidates=tuple(rows),
    )


def prepare_equal_budget_baselines(
    manifest: FiniteSearchSpaceManifest,
    source_scenario: Mapping[str, Any],
    adapter: CandidateOverlayAdapter,
    *,
    baselines: Sequence[str] = BASELINE_NAMES,
) -> dict[str, BaselinePreparation]:
    """Prepare random and quasi-random arms with the same declared candidate budget."""
    normalized = tuple(name.strip().lower().replace("-", "_") for name in baselines)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError("baseline names must be non-empty and unique")
    unknown = set(normalized) - set(BASELINE_NAMES)
    if unknown:
        raise ValueError("unsupported preparation baselines: " + ", ".join(sorted(unknown)))
    return {
        name: prepare_baseline(
            manifest,
            source_scenario,
            adapter,
            baseline=name,
        )
        for name in normalized
    }


class MappingOverlayAdapter:
    """Generic adapter that maps candidate variables to nested scenario paths."""

    def __init__(
        self,
        variable_paths: Mapping[str, str | Sequence[str]],
        *,
        adapter_id: str = "mapping_overlay.v1",
    ) -> None:
        """Initialize a frozen variable-to-path mapping."""
        self.adapter_id = _require_text(adapter_id, "adapter_id")
        normalized: dict[str, tuple[str, ...]] = {}
        for variable, raw_path in variable_paths.items():
            variable_name = _require_text(variable, "adapter variable")
            path = (
                tuple(raw_path)
                if not isinstance(raw_path, str)
                else tuple(part for part in raw_path.split(".") if part)
            )
            if not path or any(not str(part).strip() for part in path):
                raise ValueError(f"adapter path for {variable_name!r} must be non-empty")
            normalized[variable_name] = tuple(str(part) for part in path)
        self._variable_paths = MappingProxyType(normalized)

    def validate(
        self,
        source_scenario: Mapping[str, Any],
        candidate: SearchCandidate,
    ) -> Sequence[str]:
        """Reject candidates whose variables are not mapped to scenario fields."""
        del source_scenario
        return tuple(
            f"adapter:unmapped_variable:{name}"
            for name, _value in candidate.values
            if name not in self._variable_paths
        )

    def materialize(
        self,
        source_scenario: Mapping[str, Any],
        candidate: SearchCandidate,
    ) -> ImmutableScenarioOverlay:
        """Create an immutable nested patch with candidate provenance."""
        reasons = self.validate(source_scenario, candidate)
        if reasons:
            raise ValueError("; ".join(reasons))
        patch: dict[str, Any] = {}
        for name, value in candidate.values:
            current = patch
            path = self._variable_paths[name]
            for part in path[:-1]:
                current = current.setdefault(part, {})
            current[path[-1]] = value
        return ImmutableScenarioOverlay(
            source=source_scenario,
            patch=patch,
            candidate_id=candidate.candidate_id,
            adapter_id=self.adapter_id,
            provenance=candidate.to_dict(),
        )


_CANDIDATE_SPEC_VARIABLES = (
    "start_x",
    "start_y",
    "goal_x",
    "goal_y",
    "spawn_time_s",
    "pedestrian_speed_mps",
    "pedestrian_delay_s",
    "scenario_seed",
)


def _candidate_spec_from_search_candidate(candidate: SearchCandidate) -> CandidateSpec:
    """Convert the generic typed point into the existing CandidateSpec seam."""
    values = candidate.value_map
    missing = [name for name in _CANDIDATE_SPEC_VARIABLES if name not in values]
    if missing:
        raise ValueError("candidate is missing CandidateSpec variables: " + ", ".join(missing))
    scenario_seed = values["scenario_seed"]
    if isinstance(scenario_seed, bool) or not float(scenario_seed).is_integer():
        raise ValueError("candidate scenario_seed must be integral")
    return CandidateSpec(
        start=Pose2D(
            float(values["start_x"]),
            float(values["start_y"]),
            float(values.get("start_theta", 0.0)),
        ),
        goal=Pose2D(
            float(values["goal_x"]), float(values["goal_y"]), float(values.get("goal_theta", 0.0))
        ),
        spawn_time_s=float(values["spawn_time_s"]),
        pedestrian_speed_mps=float(values["pedestrian_speed_mps"]),
        pedestrian_delay_s=float(values["pedestrian_delay_s"]),
        scenario_seed=int(scenario_seed),
    )


class CandidateSpecOverlayAdapter:
    """Adapter that reuses ``bundle.build_candidate_payload`` for existing search seams."""

    adapter_id = "candidate_spec_overlay.v1"

    def __init__(
        self,
        *,
        pedestrian_id: str | None = None,
        pedestrian_route_mode: str = "candidate",
        route_file_name: str = "route_overrides.yaml",
    ) -> None:
        """Configure existing CandidateSpec route/materialization semantics."""
        self.pedestrian_id = pedestrian_id
        self.pedestrian_route_mode = pedestrian_route_mode
        self.route_file_name = route_file_name

    @staticmethod
    def _template_scenario(source_scenario: Mapping[str, Any]) -> Mapping[str, Any]:
        """Extract the first scenario from a matrix or accept a direct scenario mapping."""
        scenarios = source_scenario.get("scenarios")
        if scenarios is None:
            return source_scenario
        if (
            not isinstance(scenarios, Sequence)
            or isinstance(scenarios, str | bytes)
            or not scenarios
        ):
            raise ValueError("source scenario must contain a non-empty scenarios sequence")
        first = scenarios[0]
        if not isinstance(first, Mapping):
            raise ValueError("source scenario first entry must be a mapping")
        return first

    def validate(
        self,
        source_scenario: Mapping[str, Any],
        candidate: SearchCandidate,
    ) -> Sequence[str]:
        """Return adapter errors before the existing materializer is called."""
        missing = [name for name in _CANDIDATE_SPEC_VARIABLES if name not in candidate.value_map]
        if missing:
            return ("adapter:missing_candidate_spec_variables:" + ",".join(missing),)
        try:
            template = self._template_scenario(source_scenario)
            if self.pedestrian_id:
                binding_error = validate_template_pedestrian_binding(template, self.pedestrian_id)
                if binding_error is not None:
                    return (f"adapter:template_binding:{binding_error}",)
        except (TypeError, ValueError) as exc:
            return (f"adapter:source_template:{type(exc).__name__}",)
        return ()

    def materialize(
        self,
        source_scenario: Mapping[str, Any],
        candidate: SearchCandidate,
    ) -> ImmutableScenarioOverlay:
        """Build an immutable overlay through the existing pure bundle seam."""
        reasons = self.validate(source_scenario, candidate)
        if reasons:
            raise ValueError("; ".join(reasons))
        candidate_spec = _candidate_spec_from_search_candidate(candidate)
        specialized, route_payload = build_candidate_payload(
            candidate_spec,
            index=candidate.candidate_index,
            template_scenario=self._template_scenario(source_scenario),
            pedestrian_id=self.pedestrian_id,
            pedestrian_route_mode=self.pedestrian_route_mode,
            route_file_name=self.route_file_name,
        )
        if "scenarios" in source_scenario:
            patch: dict[str, Any] = {
                "route_overrides": route_payload,
                "scenarios": [specialized],
            }
        else:
            patch = {**specialized, "route_overrides": route_payload}
        return ImmutableScenarioOverlay(
            source=source_scenario,
            patch=patch,
            candidate_id=candidate.candidate_id,
            adapter_id=self.adapter_id,
            provenance={
                "candidate": candidate.to_dict(),
                "existing_materializer": "robot_sf.adversarial.bundle.build_candidate_payload",
                "route_file_name": self.route_file_name,
            },
        )


# Friendly aliases for callers that prefer the search-oriented names.
CandidatePoint = SearchCandidate
HaltonQuasiRandomBaseline = QuasiRandomBaseline
RandomSearchBaseline = RandomBaseline
QuasiRandomSearchBaseline = QuasiRandomBaseline


__all__ = [
    "BASELINE_NAMES",
    "CLAIM_BOUNDARY",
    "PREPARATION_SCHEMA_VERSION",
    "SEARCH_HARNESS_SCHEMA_VERSION",
    "BaselinePreparation",
    "CandidateOverlayAdapter",
    "CandidatePoint",
    "CandidateSpecOverlayAdapter",
    "CrossVariableConstraint",
    "FiniteBounds",
    "FiniteSearchSpaceManifest",
    "HaltonQuasiRandomBaseline",
    "ImmutableScenarioOverlay",
    "MappingOverlayAdapter",
    "ObjectiveComponent",
    "ObjectiveVector",
    "PreparedCandidate",
    "QuasiRandomBaseline",
    "QuasiRandomSearchBaseline",
    "RandomBaseline",
    "RandomSearchBaseline",
    "RejectionRecord",
    "RolloutBudget",
    "SearchCandidate",
    "SearchVariable",
    "SeedPolicy",
    "build_baseline",
    "prepare_baseline",
    "prepare_equal_budget_baselines",
]
