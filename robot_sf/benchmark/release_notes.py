"""Fail-closed 0.0.8 disclosure admission over versioned notes and retained bytes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

NOTES_PATH = "docs/release/0.0.8/release_notes.md"
DECISIONS_PATH = "docs/release/0.0.8/decisions.md"
CHECKER_PATH = "robot_sf/benchmark/release_notes.py"
RECEIPT_NAME = "release_notes_gate.v1.json"
SCHEMA = "release-notes-gate.v1"

# Complete paragraphs in their named sections, never keyword presence.
REQUIREMENTS = {
    "D-039": (
        "Scripted pedestrian speed",
        [
            "The declared scripted pedestrian speed is the desired speed. Maximum speed = 1.3 x "
            "desired speed."
        ],
    ),
    "D-040": (
        "Group forces",
        [
            "Group gaze and group repulsion differ from the published laws: gaze scales with "
            "inverse goal distance and ignores the field of view, and repulsion weakens as "
            "members get closer. Groups form in 14-18 of 48 scenarios. The group laws are "
            "retained for 0.0.8 and will be fixed with recalibration in the next release."
        ],
    ),
    "D-044": (
        "Elevator geometry",
        [
            "The elevator scenario interior walls changed in #9972. The geometry differs between "
            "0.0.7 and 0.0.8."
        ],
    ),
    "D-053": (
        "Learned reference",
        [
            "The plain ppo arm replaces the published checkpoint with "
            "ppo_release_robot_b1002_last_20261001, a collision-prone learned reference. It is "
            "not a competitive baseline.",
            "Model asset SHA-256: "
            "764a7d88f5b608237641d973634899e05a67b25e65f8b1607cfca025459824bc. The policy was "
            "chosen by a pre-registered rule: most successes on 48 scenarios x dev seeds "
            "1001-1005, then fewer collisions within five successes of the maximum.",
        ],
    ),
    "D-055": (
        "Narrow passages",
        [
            "Flagged rows: classic_doorway_low, classic_doorway_medium, classic_doorway_high, "
            "francis2023_narrow_hallway, francis2023_narrow_doorway (2.0 m probe, disclosure "
            "only), francis2023_narrow_doorway_width_2p20, "
            "francis2023_narrow_doorway_width_2p80, francis2023_narrow_doorway_width_3p60.",
            "Rows in the classic doorway scenarios, narrow hallway and three-width doorway slice "
            "should be read as door or hallway passage with pedestrians held back by the wall "
            "force, not as doorway or hallway counterflow. Draw no doorway- or hallway-specific "
            "planner claims, doorway rankings or door-width effects from these rows. The "
            "obstacle force will be refitted in the next release.",
            "Crowd-model limitation in narrow passages (issue #10061). The pedestrian model uses "
            "the released pysocialforce obstacle potential (factor 10, offset −0.57 m). It "
            "pushes a pedestrian back from a wall edge with more force than the pedestrian's own "
            "walking force (at 0.65 m/s) up to about 2 m before an opening. As a result, "
            "simulated pedestrians do not walk through doorways of 3.6 m or narrower toward the "
            "robot. A lone pedestrian stops 2.3 m (2.0 m door), 2.25 m (2.2 m), 2.0 m (2.8 m) or "
            "0.4 m (3.6 m, creeping) in front of the door and never passes within the episode. "
            "In the classic doorway scenarios (3.3 m door), under half of the approaching "
            "pedestrians cross, against about two thirds without this effect; the rest stall or "
            "mill 1–2.4 m in front of the door. In robot episodes, pedestrians from the far side "
            "almost never reach the door (0 of 187 at high density). The robot therefore meets "
            "pedestrians beside or beyond the door, not in it. In the 4 m narrow hallway, the "
            "same wall force keeps a pedestrian about 1.2 m from each wall, so it cannot get "
            "past the robot, which makes that scenario harder than intended. Rows for "
            "classic_doorway_{low,medium,high}, francis2023_narrow_hallway and the three-width "
            'doorway slice should be read as "door or hallway passage with pedestrians held back '
            'by the wall force", not as doorway or hallway counterflow. Do not draw doorway- or '
            "hallway-specific planner comparisons from them. In a diagnostic re-run on "
            "development seeds with the obstacle force scaled to 0.3×: - classic_doorway_high "
            "robot success fell from 24/30 to 7/30 and classic_doorway_medium from 24/30 to "
            "12/30 (6 planners × 5 seeds), and planner ordering changed within those scenarios; "
            "- hallway success rose from 9/30 to 15/30; - overall success over all 48 scenarios "
            "changed by at most 2.5 points for goal, ORCA and social_force, and their ranking "
            "was unchanged. The 2.0 m infeasibility probe is unaffected in outcome. The "
            "obstacle-force calibration will be revised in 0.0.9.",
        ],
    ),
    "D-056": (
        "Upstream queue",
        [
            "The three-width doorway slice stays in 0.0.8. Pedestrians queue in front of "
            "openings up to about 3 m wide; robot results there are measured against that queue. "
            "The effect depends on geometry: pedestrians already inside a straight corridor keep "
            "walking. Next-release acceptance requires a lone pedestrian to pass every opening "
            "of 1.0 m and wider without stopping and pedestrians to flow through all three slice "
            "widths on dev seeds; then the slice is re-run."
        ],
    ),
    "D-057": (
        "Crowd speed and body size",
        [
            "Crowd desired speed and hard cap are both 0.65 m/s, with no spread and no headroom "
            "above the target. Crowd pedestrians walk at one shared 0.65 m/s, about half the "
            "measured adult free speed of 1.29-1.34 m/s; results describe slow, uniform crowds.",
            "Pedestrians are rigid discs of radius 0.40 m (0.35 m in the force kernel), or "
            "0.7-0.8 m wide. Speed, radius and wall law are coupled and remain unchanged for "
            "0.0.8; the literature-based speed model and a smaller radius go to the next release "
            "through joint calibration.",
        ],
    ),
    "D-058": (
        "Pedestrian robot response",
        [
            "Simulated pedestrians slow down near the robot but do not steer around it, and "
            "collisions are not attributed.",
            "The force is radial and has lateral components off-axis, but lacks an anticipatory "
            "side-selection mechanism. The current activation centre distance is 2.0 + 0.35 + "
            "1.0 = 3.35 m, including force radius and robot radius, rather than a 2.0 m centre "
            "cutoff.",
        ],
    ),
    "D-059": (
        "Robot motion",
        [
            "The 0.0.8 release robot uses differential drive and cannot reverse: allow_backwards "
            "is false. Reverse remains off in 0.0.8."
        ],
    ),
    "D-060": (
        "Occupancy grid",
        [
            "The occupancy-grid rasteriser tests cell corners instead of cell centres. "
            "Pedestrians appear about half a cell (+0.08 to +0.14 m) towards +x/+y. PPO and "
            "guarded PPO read this shifted grid; the retrained plain ppo was also trained on it. "
            "This known limitation stays in 0.0.8 and will be fixed together with retraining in "
            "the next release to avoid a distribution shift."
        ],
    ),
    "D-061": (
        "Collision geometry and TTC",
        [
            "Wall and agent collisions in release rows come from the simulator footprint check; "
            "the cross-check fails closed on disagreement.",
            "Time to collision is centre-based, ignores both radii and overstates the time by "
            "about 0.7 s at 2 m/s. It appears only in metrics.time_to_collision_min and "
            "time_to_collision_min_mean in campaign_summary.json, not in SNQI, tables or "
            "comparators.",
        ],
    ),
    "D-062": (
        "Cross-release interpretation",
        [
            "0.0.7 and 0.0.8 are different benchmarks (#9856, #9762, #9725, #10026): seeds, "
            "worlds and planner contracts differ. Compare distributions only, not paired by "
            "seed. Gains in 0.0.8 come mainly from new algorithm configs that fix 0.0.7 plant "
            "mismatches and must not be attributed to planners alone.",
            "The 0.0.7 predictive_mppi baseline is invalid. The #9764 planner-side kernel had no "
            "effect. Robot-radius alignment is incomplete (#4856). H400 budgets are fair.",
        ],
    ),
    "D-063": (
        "Small-crowd groups",
        [
            "The groups setting remains the fraction of pedestrians in groups. Integer "
            "allocation truncates the realised fraction in small crowds: 0.17 / 0.36 / 0.44 in "
            "the cases measured. The formula stays unchanged for 0.0.8; exact allocation is "
            "deferred to #10040."
        ],
    ),
    "D-065": (
        "Non-release sampler test",
        [
            "The 400-step safety ruling applies only to the non-release "
            "classic_interactions_francis2023_goal_zone_entry_v1.yaml / "
            "socnav_sampling_bounded_v2.yaml test cell, not to release results. That test "
            "asserts zero wall and total collisions and records goal arrival without requiring "
            "it. It stops if more than 50 consecutive steps have |v| < 0.05 m/s or the last 150 "
            "steps make no net progress. It makes no release claim."
        ],
    ),
    "D-075": (
        "Doorway slice admission",
        [
            "The benchmark-doorway-width-slice.v1 dataset has widths 2.2 / 2.8 / 3.6 m, the same "
            "14-arm roster and sealed seeds, 1,260 cells, H400 and its own denominator. Main "
            "acceptance remains 20,160 cells across 48 scenarios. The slice does not relax main "
            "acceptance."
        ],
    ),
    "D-076": (
        "Versioned release notes",
        [
            "The versioned 0.0.8 release notes live at docs/release/0.0.8/release_notes.md. The "
            "short GitHub release body must link to these versioned notes. Decisions are stated "
            "by content in these notes."
        ],
    ),
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_notes(data: bytes, *, phase: str) -> list[str]:
    """Require each adopted statement in its section.

    Returns:
        Admitted decision identifiers.
    """
    if phase not in {"mint", "publication"}:
        raise ValueError("release notes phase must be mint or publication")
    text = data.decode("utf-8")
    if phase == "publication" and re.search(
        r"PLACEHOLDER|\bTODO\b|\bTBD\b|\{\{|<(?![!/?])[^<>\n]+>", text, re.IGNORECASE
    ):
        raise ValueError("release notes publication refuses placeholders")
    sections: dict[str, list[str]] = {}
    heading = None
    for block in text.strip().split("\n\n"):
        if block.startswith("## ") and "\n" not in block:
            heading = block[3:]
            if heading in sections:
                raise ValueError("release notes duplicate section: " + heading)
            sections[heading] = []
        elif heading is not None:
            sections[heading].append(" ".join(block.split()))
    for decision, (section, statements) in REQUIREMENTS.items():
        for statement in statements:
            if statement not in sections.get(section, []):
                raise ValueError(
                    f"release notes {decision}: missing required statement in {section}"
                )
    return list(REQUIREMENTS)


def notes_gate(
    repository_root: Path,
    *,
    source_commit: str,
    phase: str = "mint",
    receipt: dict[str, Any] | None = None,
    notes_path: Path | None = None,
) -> dict[str, Any]:
    """Admit actual notes; publication requires the exact mint receipt and digest.

    Source inputs are checked by the existing exact-source route. This receipt
    retains their digests separately so resolved campaign identities do not change.

    Returns:
        Passing receipt binding notes, source and disclosure requirements.
    """
    root = repository_root.resolve()
    path = notes_path or root / NOTES_PATH
    data = path.read_bytes()
    if receipt is not None:
        if not isinstance(receipt, dict):
            raise ValueError("release notes receipt must be an object")
        if (
            receipt.get("schema_version") != SCHEMA
            or receipt.get("status") != "pass"
            or receipt.get("source_commit") != source_commit
            or receipt.get("notes_path") != NOTES_PATH
            or receipt.get("decisions") != list(REQUIREMENTS)
        ):
            raise ValueError("release notes receipt identity mismatch")
        if receipt.get("notes_sha256") != _digest(data):
            raise ValueError("release notes digest mismatch against mint receipt")
        for key, relative in (
            ("checker_sha256", CHECKER_PATH),
            ("decisions_sha256", DECISIONS_PATH),
        ):
            if receipt.get(key) != _digest((root / relative).read_bytes()):
                raise ValueError("release notes source digest mismatch: " + key)
    elif phase == "publication":
        raise ValueError("release notes publication requires mint receipt")
    decisions = check_notes(data, phase=phase)
    return {
        "schema_version": SCHEMA,
        "status": "pass",
        "source_commit": source_commit,
        "notes_path": NOTES_PATH,
        "notes_sha256": _digest(data),
        "checker_sha256": _digest((root / CHECKER_PATH).read_bytes()),
        "decisions_sha256": _digest((root / DECISIONS_PATH).read_bytes()),
        "decisions": decisions,
    }


def gate_manifest(manifest: Any, repository_root: Path) -> dict[str, Any] | None:
    """Recheck the retained mint disclosure receipt before production preflight.

    Returns:
        Passing receipt for 0.0.8 production, otherwise None.
    """
    if getattr(manifest, "release_kind", None) == "development_rehearsal":
        return None
    campaign = Path(getattr(manifest, "canonical_campaign_config_path", "")).name
    if "0_0_8" not in campaign:
        return None
    identity = getattr(manifest, "resolved_identity_path", None)
    if identity is None:
        raise ValueError("release notes gate requires production mint receipt")
    receipt = json.loads((Path(identity).parent / RECEIPT_NAME).read_bytes())
    return notes_gate(repository_root, source_commit=manifest.source_sha, receipt=receipt)
