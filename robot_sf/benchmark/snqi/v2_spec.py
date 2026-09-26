"""Strict, immutable SNQI-v2 specification and calibration provenance.

The Social Navigation Quality Index is a declared simulator aggregate, not a
validated measure of human comfort or safety. No calibration anchor has a default.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.robot_force_contract import declared_force_source_contract
from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

TERMS = ("S", "C", "T", "N", "F", "J", "K")
QUALITY_TERMS = TERMS[2:]
WEIGHTS = dict(zip(TERMS, (1.0, 2.0, 0.25, 0.25, 0.25, 0.10, 0.10), strict=True))
SIMULATED_FORCE = "robot_force_impulse_total"
PP_EQUIV_FORCE = "robot_force_pp_equiv_impulse_total"
SOURCES = dict(
    zip(
        TERMS,
        (
            "success",
            "total_collision_count",
            "time_to_goal_ideal_ratio",
            "near_misses",
            SIMULATED_FORCE,
            "jerk_mean",
            "curvature_mean",
        ),
        strict=True,
    )
)
# Known aliases/derivations share an ancestor and cannot occupy two score terms.
DERIVATIONS = {
    "comfort_exposure": ("force_exceed_events",),
    "near_misses": ("clearance_series",),
    "min_distance": ("clearance_series",),
    "human_discomfort_exposure_m_s": ("clearance_series",),
    "robot_force_impulse_per_exposed_ped": (SIMULATED_FORCE,),
}
FAMILY = {
    "version": "V2-F",
    "seed": 20260924,
    "draws": 2000,
    "dirichlet_alpha": [1, 1, 1, 1, 1],
    "quality_mass": 0.95,
    "minimum_weight": 0.02,
    "sampling": "reject_below_minimum",
    "grid": ["equal", "each_term_heavy", "leave_one_out"],
    "heavy_weight": 0.55,
    "heavy_other_weight": 0.10,
    "leave_one_out_basis": "declared_remaining_proportions",
    "relaxed_success_weights": [0.5],
    "relaxed_collision_weights": [0.5, 1.0],
}


def finite_nonnegative(value: Any, label: str) -> float:
    """Return a finite nonnegative numeric value, rejecting absent/invalid data."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"SNQI-v2 {label} must be finite and nonnegative")
    return float(value)


def validate_sources(sources: Mapping[str, str]) -> None:
    """Reject duplicate sources and known common derivation ancestors."""
    if set(sources) != set(TERMS):
        raise ValueError("SNQI-v2 source registry must contain exactly S,C,T,N,F,J,K")
    used: set[str] = set()
    for source in sources.values():
        lineage = {source}
        pending = [source]
        while pending:
            for parent in DERIVATIONS.get(pending.pop(), ()):
                if parent not in lineage:
                    lineage.add(parent)
                    pending.append(parent)
        if lineage & used:
            raise ValueError(f"SNQI-v2 duplicate/derived source: {source}")
        used.update(lineage)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject JSON duplicate keys instead of silently accepting a replacement.

    Returns:
        Validated result described above.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_v2_json(raw: str | bytes) -> Any:
    """Parse V2 source JSON without discarding duplicate keys at any nesting depth.

    Returns:
        The decoded JSON value after every object has passed unique-key validation.
    """
    return json.loads(raw, object_pairs_hook=_unique_object)


class _UniqueYamlLoader(yaml.SafeLoader):
    """Keep SafeLoader constructors while refusing ambiguous mapping keys."""

    def construct_mapping(self, node: yaml.nodes.MappingNode, deep: bool = False) -> dict:
        """Validate keys, including merge-expanded keys, before constructing a mapping.

        Returns:
            The unambiguous safe YAML mapping.
        """
        self.flatten_mapping(node)
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise ValueError(f"duplicate YAML key: {key}")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def parse_v2_yaml(raw: str | bytes) -> Any:
    """Load V2 family/acquisition YAML without last-wins mapping replacement.

    Returns:
        The decoded safe YAML value with unique keys at every nesting depth.
    """
    loader = _UniqueYamlLoader(raw)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


@dataclass(frozen=True)
class SnqiV2Spec:
    """A validated score specification; nested mappings are immutable snapshots."""

    weights: Mapping[str, float]
    upper_anchors: Mapping[str, float]
    force_source: str
    calibration_split_id: str
    calibration_seeds: tuple[int, ...]
    calibration_rho: float
    paths: Mapping[str, str]
    hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        """Enforce the complete score contract even for direct construction."""
        if set(self.weights) != set(TERMS):
            raise ValueError("SNQI-v2 weights must contain exactly S,C,T,N,F,J,K")
        weights = {
            key: finite_nonnegative(value, f"weight {key}") for key, value in self.weights.items()
        }
        quality = math.fsum(weights[key] for key in QUALITY_TERMS)
        if not weights["S"] > quality or not weights["C"] > weights["S"] + quality:
            raise ValueError("SNQI-v2 weights violate safety strata inequalities")
        if set(self.upper_anchors) != set(QUALITY_TERMS):
            raise ValueError("SNQI-v2 anchors must contain exactly T,N,F,J,K")
        anchors = {
            key: finite_nonnegative(value, f"anchor {key}")
            for key, value in self.upper_anchors.items()
        }
        if any(value <= 0 for value in anchors.values()):
            raise ValueError("SNQI-v2 upper anchors must be positive")
        if anchors["T"] != 3 or anchors["N"] != 0.25:
            raise ValueError("SNQI-v2 normative anchors require T=3 and N=0.25")
        if (
            isinstance(self.calibration_rho, bool)
            or not math.isfinite(self.calibration_rho)
            or abs(self.calibration_rho) > 1
        ):
            raise ValueError("SNQI-v2 calibration rho must be finite in [-1,1]")
        expected = PP_EQUIV_FORCE if abs(self.calibration_rho) >= 0.90 else SIMULATED_FORCE
        if self.force_source != expected:
            raise ValueError("SNQI-v2 F source violates preregistered |rho| >= 0.90 decision")
        if self.calibration_seeds != (101, 102) or not self.calibration_split_id:
            raise ValueError("SNQI-v2 requires identified calibration seeds 101,102")
        for name, value in (
            ("weights", weights),
            ("upper_anchors", anchors),
            ("paths", self.paths),
            ("hashes", self.hashes),
        ):
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        validate_sources(self.sources)

    @property
    def sources(self) -> dict[str, str]:
        """Return a fresh term-to-source registry with the frozen F decision."""
        return {**SOURCES, "F": self.force_source}

    def provenance(self) -> dict[str, Any]:
        """Return campaign/manifest provenance with explicit versioned file hashes."""
        return {
            "snqi_v2_version": "SNQI-v2",
            "snqi_v2_calibration_split_id": self.calibration_split_id,
            "snqi_v2_force_source": self.force_source,
            "snqi_v2_force_source_contract": declared_force_source_contract(self.force_source),
            **{f"snqi_v2_{key}_path": _provenance_path(value) for key, value in self.paths.items()},
            **{f"snqi_v2_{key}_sha256": value for key, value in self.hashes.items()},
        }

    def validate_evaluation_seeds(self, seeds: Sequence[int]) -> None:
        """Fail closed on calibration/evaluation seed leakage."""
        if set(seeds) & set(self.calibration_seeds):
            raise ValueError("SNQI-v2 evaluation seeds overlap calibration split")


def load_snqi_v2_spec(weights_path: Path, anchors_path: Path, family_path: Path) -> SnqiV2Spec:
    """Load versioned assets; reject unfrozen calibration and malformed provenance.

    Returns:
        Validated result described above.
    """
    paths = {
        "weights": weights_path.resolve(),
        "anchors": anchors_path.resolve(),
        "family": family_path.resolve(),
    }
    raw = {key: path.read_bytes() for key, path in paths.items()}
    weights_doc = parse_v2_json(raw["weights"])
    anchors_doc = parse_v2_json(raw["anchors"])
    family_doc = parse_v2_yaml(raw["family"])
    if weights_doc.get("version") != "SNQI-v2.0" or family_doc != FAMILY:
        raise ValueError("SNQI-v2 unrecognized weights/family version or family contract")
    entries = weights_doc["weights"]
    if set(entries) != {f"w_{term}" for term in TERMS}:
        raise ValueError("SNQI-v2 requires exactly seven explicit weights including curvature")
    for entry in entries.values():
        if (
            set(entry) != {"value", "rationale"}
            or not isinstance(entry["rationale"], str)
            or not entry["rationale"].strip()
        ):
            raise ValueError("SNQI-v2 every weight needs value and nonempty rationale")
    if any(entries[f"w_{term}"]["value"] != WEIGHTS[term] for term in TERMS):
        raise ValueError("SNQI-v2.0 requires the exact declared weight values")
    if anchors_doc.get("version") != "SNQI-v2.0" or anchors_doc.get("status") != "frozen":
        raise ValueError("SNQI-v2 calibration anchors are not frozen")
    anchors = anchors_doc["anchors"]
    if set(anchors) != set(QUALITY_TERMS):
        raise ValueError("SNQI-v2 anchors must contain exactly T,N,F,J,K")
    if any(
        finite_nonnegative(entry.get("lower"), f"lower anchor {term}") != 0
        for term, entry in anchors.items()
    ):
        raise ValueError("SNQI-v2 lower anchors must be physical zero")
    _validate_calibration(anchors_doc)
    calibration = anchors_doc["calibration"]
    spec = SnqiV2Spec(
        weights={term: entries[f"w_{term}"]["value"] for term in TERMS},
        upper_anchors={term: entry["upper"] for term, entry in anchors.items()},
        force_source=anchors_doc["force_decision"]["source"],
        calibration_rho=anchors_doc["force_decision"]["spearman_rho_F_N"],
        calibration_split_id=calibration["split_id"],
        calibration_seeds=tuple(calibration["seeds"]),
        paths={key: str(path) for key, path in paths.items()},
        hashes={key: hashlib.sha256(value).hexdigest() for key, value in raw.items()},
    )
    return spec


def _validate_calibration(anchors_doc: dict[str, Any]) -> None:
    """Require complete nonfallback calibration provenance and declared p95 anchors."""
    calibration = anchors_doc["calibration"]
    if calibration.get("quantile_method") != "linear":
        raise ValueError("SNQI-v2 calibration quantile_method must be linear")
    _validate_calibration_grid(calibration)
    _validate_command_mode_census(calibration)
    _validate_frozen_custody(calibration)
    _validate_force_decision_contract(anchors_doc)
    if any(anchors_doc["anchors"][key].get("type") != "calibration_p95" for key in ("F", "J", "K")):
        raise ValueError("SNQI-v2 F/J/K must be calibration p95 anchors")


def _validate_calibration_grid(calibration: dict[str, Any]) -> None:
    """Check the declared dev split and its self-consistent grid identity."""
    arms = calibration.get("arms")
    scenarios = calibration.get("scenarios")
    seeds = calibration.get("seeds")
    if (
        not isinstance(arms, list)
        or len(arms) != 14
        or any(not isinstance(arm, str) or not arm or Path(arm).name != arm for arm in arms)
        or len(set(arms)) != 14
        or not isinstance(scenarios, list)
        or len(scenarios) != 48
        or any(not isinstance(scenario, str) or not scenario for scenario in scenarios)
        or len(set(scenarios)) != 48
        or seeds != [101, 102]
        or calibration.get("episode_count") != 1344
        or calibration.get("benchmark_execution") != "nonfallback"
    ):
        raise ValueError("SNQI-v2 calibration requires 14 arms x48 scenarios x2 nonfallback seeds")
    grid = sorted(product(arms, scenarios, seeds))
    grid_sha256 = hashlib.sha256(json.dumps(grid, separators=(",", ":")).encode()).hexdigest()
    if calibration.get("grid_sha256") != grid_sha256:
        raise ValueError("SNQI-v2 calibration grid_sha256 does not match its declared split")
    if calibration.get("split_id") != f"snqi-v2-dev101-102-{grid_sha256[:12]}":
        raise ValueError("SNQI-v2 calibration split_id does not match its declared grid")


def _validate_command_mode_census(calibration: dict[str, Any]) -> None:
    """Require a complete and bounded per-arm command-mode census."""
    arms = calibration["arms"]
    census = calibration.get("command_mode_counts")
    if not isinstance(census, dict) or set(census) != set(arms):
        raise ValueError("SNQI-v2 calibration requires every arm's command-mode census")
    for counts in census.values():
        if (
            not isinstance(counts, dict)
            or not counts
            or not set(counts) <= {"native", "adapter", "mixed"}
            or any(type(count) is not int or count <= 0 for count in counts.values())
            or sum(counts.values()) != 96
        ):
            raise ValueError("SNQI-v2 calibration command-mode census requires 96 rows per arm")


def _valid_digest(value: Any, length: int = 64) -> bool:
    """Return whether a value is a lowercase hexadecimal digest of the required size."""
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_frozen_custody(calibration: dict[str, Any]) -> None:
    """Require source maps and hashes produced only after the archive freeze checks."""
    if not calibration.get("run_id") or not _valid_digest(calibration.get("source_commit"), 40):
        raise ValueError("SNQI-v2 calibration run/source identity is incomplete")
    episode_files = calibration.get("episode_files_sha256")
    sidecars = calibration.get("producer_sidecars_sha256")
    arms = calibration["arms"]
    expected_episode_paths = {f"runs/{arm}__differential_drive/episodes.jsonl" for arm in arms}
    expected_sidecar_paths = {
        f"runs/{arm}__differential_drive/episodes.jsonl.provenance.json" for arm in arms
    }
    if (
        not isinstance(episode_files, dict)
        or set(episode_files) != expected_episode_paths
        or any(not _valid_digest(value) for value in episode_files.values())
        or not isinstance(sidecars, dict)
        or set(sidecars) != expected_sidecar_paths
        or any(not _valid_digest(value) for value in sidecars.values())
    ):
        raise ValueError(
            "SNQI-v2 frozen calibration requires every episode and producer-sidecar hash"
        )
    expected_episodes_sha256 = hashlib.sha256(
        json.dumps(episode_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if calibration.get("episodes_sha256") != expected_episodes_sha256:
        raise ValueError("SNQI-v2 calibration episodes_sha256 does not bind the episode-file map")
    for key in ("campaign_config_hash", "campaign_manifest_sha256"):
        if not _valid_digest(calibration.get(key)):
            raise ValueError(f"SNQI-v2 frozen calibration requires {key}")
    if calibration.get("episodes_hash_rule") != (
        "sha256(sorted compact JSON relative-path-to-file-sha256 map)"
    ):
        raise ValueError("SNQI-v2 frozen calibration has an unsupported episode hash rule")


def _validate_force_decision_contract(anchors_doc: dict[str, Any]) -> None:
    """Require the pre-registered threshold, complete coverage and source contract."""
    force_decision = anchors_doc.get("force_decision")
    if not isinstance(force_decision, dict):
        raise ValueError("SNQI-v2 calibration force decision is missing")
    rho = force_decision.get("spearman_rho_F_N")
    if (
        isinstance(rho, bool)
        or not isinstance(rho, (int, float))
        or not math.isfinite(rho)
        or abs(rho) > 1
    ):
        raise ValueError("SNQI-v2 calibration rho must be finite in [-1,1]")
    if (
        force_decision.get("threshold_absolute_rho") != 0.90
        or force_decision.get("selected_source_coverage") != 1344
        or force_decision.get("N") != "clip(near_misses/steps/0.25)"
    ):
        raise ValueError("SNQI-v2 force decision threshold, N term or source coverage is invalid")
    source = force_decision.get("source")
    if source not in {SIMULATED_FORCE, PP_EQUIV_FORCE}:
        raise ValueError("SNQI-v2 calibration force source is unsupported")
    expected_source = PP_EQUIV_FORCE if abs(rho) >= 0.90 else SIMULATED_FORCE
    if source != expected_source:
        raise ValueError("SNQI-v2 force source violates the preregistered rho threshold")
    if force_decision.get("selected_source_contract") != declared_force_source_contract(source):
        raise ValueError(
            "SNQI-v2 force anchor must bind its recorded producer and reference contract"
        )


def _provenance_path(value: str) -> str:
    """Return repository-relative asset locations for portable public provenance."""
    path = Path(value)
    try:
        return str(path.relative_to(get_repository_root()))
    except ValueError:
        return str(path)
