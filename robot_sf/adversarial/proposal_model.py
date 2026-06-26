"""Learned proposal and ranking models over failure archive metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robot_sf.adversarial.archive import failure_archive_feature_rows
from robot_sf.adversarial.certification import CertificationStatus, certify_candidate
from robot_sf.adversarial.config import CandidateSpec, SearchSpaceConfig
from robot_sf.adversarial.scenario_manifest import (
    AdversarialScenarioManifest,
    GeneratorInfo,
    SourceLineage,
    build_manifest,
)


class FailureArchiveProposalModel:
    """A deterministic ranking and proposal model over failure archive metadata."""

    def __init__(
        self,
        archive_path_or_data: str | Path | dict[str, Any] | None = None,
        search_space: SearchSpaceConfig | None = None,
    ) -> None:
        """Initialize the FailureArchiveProposalModel.

        Args:
            archive_path_or_data: Filepath or parsed dictionary of archive entries.
            search_space: Search space bounds for normalizing distance metrics.
        """
        self.archive_path_or_data = archive_path_or_data
        self.search_space = search_space
        self.entries: list[dict[str, Any]] = []
        self.state = "active"

        # Load archive data
        if not archive_path_or_data:
            self.state = "blocked"
            return

        try:
            if isinstance(archive_path_or_data, (str, Path)):
                path = Path(archive_path_or_data)
                if not path.exists() or path.stat().st_size == 0:
                    self.state = "blocked"
                    return
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = archive_path_or_data

            if not isinstance(data, dict) or "entries" not in data:
                self.state = "blocked"
                return

            raw_entries = data.get("entries", [])
            if not isinstance(raw_entries, list):
                self.state = "blocked"
                return
            if not raw_entries:
                self.state = "blocked"
                return
            if any(
                not isinstance(entry, dict) or not isinstance(entry.get("candidate", {}), dict)
                for entry in raw_entries
            ):
                self.state = "blocked"
                return
            self.entries = raw_entries
        except (ValueError, TypeError, json.JSONDecodeError, OSError):
            self.state = "blocked"

    def get_tabular_view(self) -> list[dict[str, Any]]:
        """Build a tabular feature view from archive entries."""
        return failure_archive_feature_rows(
            {"schema_version": "adversarial_failure_archive.v1", "entries": self.entries}
        )

    def _get_candidate_value(self, cand_dict: dict[str, Any], name: str) -> float | None:
        """Helper to safely extract a scalar feature from candidate dict."""
        if name.startswith("start_") or name.startswith("goal_"):
            parts = name.split("_")
            pose = cand_dict.get(parts[0], {})
            return pose.get(parts[1]) if isinstance(pose, dict) else None
        return cand_dict.get(name)

    def _get_feature_scale(self, name: str) -> float:
        """Helper to get normalization scale for a feature."""
        if self.search_space is not None:
            range_cfg = getattr(self.search_space, name, None)
            if range_cfg is not None and hasattr(range_cfg, "max") and hasattr(range_cfg, "min"):
                span = range_cfg.max - range_cfg.min
                if span > 0.0:
                    return span

        # Calculate scale from entries
        vals = []
        for entry in self.entries:
            val = self._get_candidate_value(entry.get("candidate", {}), name)
            if val is not None:
                vals.append(float(val))
        if vals:
            span = max(vals) - min(vals)
            if span > 0.0:
                return span
        return 1.0

    def _distance(self, c1: CandidateSpec, c2_dict: dict[str, Any]) -> float:
        """Calculate a normalized L1 distance between CandidateSpec and an archive candidate dict."""
        c2_start = c2_dict.get("start", {})
        features = {
            "start_x": (c1.start.x, c2_start.get("x")),
            "start_y": (c1.start.y, c2_start.get("y")),
            "goal_x": (c1.goal.x, c2_dict.get("goal", {}).get("x")),
            "goal_y": (c1.goal.y, c2_dict.get("goal", {}).get("y")),
            "spawn_time_s": (c1.spawn_time_s, c2_dict.get("spawn_time_s")),
            "pedestrian_speed_mps": (c1.pedestrian_speed_mps, c2_dict.get("pedestrian_speed_mps")),
            "pedestrian_delay_s": (c1.pedestrian_delay_s, c2_dict.get("pedestrian_delay_s")),
        }

        total_dist = 0.0
        for name, (v1, v2) in features.items():
            if v1 is not None and v2 is not None:
                scale = self._get_feature_scale(name)
                total_dist += abs(float(v1) - float(v2)) / scale

        return total_dist

    def score_candidate(
        self, candidate: CandidateSpec, strategy: str = "nearest_neighbor"
    ) -> float:
        """Calculate a ranking score for a candidate based on the archive."""
        if not self.entries:
            return 0.0

        distances = []
        for entry in self.entries:
            c2_dict = entry.get("candidate", {})
            d = self._distance(candidate, c2_dict)
            distances.append((d, entry))

        if not distances:
            return 0.0

        if strategy == "nearest_neighbor":
            min_dist = min(d for d, _ in distances)
            return -min_dist

        elif strategy == "objective_weighted":
            epsilon = 0.1
            total_score = 0.0
            for d, entry in distances:
                obj = entry.get("objective_value")
                if obj is None:
                    obj = 0.0
                total_score += float(obj) / (d + epsilon)
            return total_score

        else:
            min_dist = min(d for d, _ in distances)
            return -min_dist

    def rank_candidates(
        self,
        candidates: list[CandidateSpec],
        strategy: str = "nearest_neighbor",
    ) -> list[tuple[CandidateSpec, float]]:
        """Rank candidates using the specified strategy.

        Returns:
            A list of tuples (candidate, score) sorted by score descending.
        """
        if not candidates:
            return []

        if self.state == "blocked" or not self.entries:
            return [(c, 0.0) for c in candidates]

        scored_candidates = []
        for candidate in candidates:
            score = self.score_candidate(candidate, strategy)
            scored_candidates.append((candidate, score))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        return scored_candidates

    def emit_manifest(
        self,
        candidate: CandidateSpec,
        *,
        source: SourceLineage | None = None,
        generator_seed: int = 0,
        candidate_index: int = 0,
    ) -> AdversarialScenarioManifest:
        """Emit an AdversarialScenarioManifest for a given candidate."""
        gen_info = GeneratorInfo(
            family="learned_proposal_model",
            generator_id="FailureArchiveProposalModel",
            seed=generator_seed,
            candidate_index=candidate_index,
        )
        return build_manifest(
            candidate,
            source=source,
            generator=gen_info,
            search_space=self.search_space,
        )

    def certify_candidate(
        self,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        require_certification: bool = False,
    ) -> CertificationStatus:
        """Certify a candidate using existing certification helpers."""
        return certify_candidate(
            candidate,
            scenario_yaml_path=scenario_yaml_path,
            require_certification=require_certification,
        )
