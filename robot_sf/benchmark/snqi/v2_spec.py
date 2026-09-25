"""Strict, immutable SNQI-v2 specification and calibration provenance.

The Social Navigation Quality Index is a declared simulator aggregate, not a
validated measure of human comfort or safety. No calibration anchor has a default.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

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
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
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
        if not math.isfinite(self.calibration_rho) or abs(self.calibration_rho) > 1:
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
    weights_doc = json.loads(raw["weights"], object_pairs_hook=_unique_object)
    anchors_doc = json.loads(raw["anchors"], object_pairs_hook=_unique_object)
    family_doc = yaml.safe_load(raw["family"])
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
    if any(entry.get("lower") != 0 for entry in anchors.values()):
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
    """Require complete native calibration provenance and declared p95 anchors."""
    calibration = anchors_doc["calibration"]
    if (
        calibration.get("episode_count") != 1344
        or len(calibration["arms"]) != 14
        or len(set(calibration["arms"])) != 14
        or len(calibration["scenarios"]) != 48
        or len(set(calibration["scenarios"])) != 48
        or calibration.get("execution_mode") != "native"
    ):
        raise ValueError("SNQI-v2 calibration must cover 14 arms x48 scenarios x2 seeds natively")
    for key, length in (("episodes_sha256", 64), ("source_commit", 40)):
        value = calibration.get(key, "")
        if len(value) != length or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"SNQI-v2 invalid calibration {key}")
    if not calibration.get("run_id"):
        raise ValueError("SNQI-v2 calibration run_id is required")
    if any(anchors_doc["anchors"][key].get("type") != "calibration_p95" for key in ("F", "J", "K")):
        raise ValueError("SNQI-v2 F/J/K must be calibration p95 anchors")


def _provenance_path(value: str) -> str:
    """Return repository-relative asset locations for portable public provenance."""
    path = Path(value)
    try:
        return str(path.relative_to(get_repository_root()))
    except ValueError:
        return str(path)
