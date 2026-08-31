#!/usr/bin/env python3
"""Build and check evidence-bounded future-work bridge status cards (issue #8048).

Generates versioned status cards and a summary markdown report for four
dissertation future-work bridge directions:
1. carla_cross_simulator_bridge
2. route_choice_homotopy_observability
3. incident_to_scenario_provenance
4. amv_actuation_realism_bridge

Ensures strict separation between implemented engineering capabilities and
empirically verified scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_sf.evidence.writers import review_marker_json, write_json, write_text

SCHEMA = "future_work_bridge_status_card.v1"
SUMMARY_SCHEMA = "future_work_bridge_summary.v1"

DEFAULT_CARDS_DIR = Path("docs/context/evidence/future_work_cards")
DEFAULT_SUMMARY_FILE = Path("docs/context/evidence/future_work_bridge_status.md")

FORBIDDEN_TERMS_IN_SAFE_SENTENCE = (
    "validated transfer",
    "proven transfer",
    "human preference validated",
    "legally liable",
    "legal fault established",
    "physically realistic",
    "hardware validated",
    "benchmark ready",
)


@dataclass(frozen=True)
class BridgeCard:
    """Versioned bridge status card."""

    schema: str
    bridge_id: str
    title: str
    relationship_to_anchor: str
    implementation_status: str
    evidence_status: str
    strongest_evidence_surface: str
    owner_paths: list[str]
    linked_issues: list[int]
    implemented_now: list[str]
    verified_now: list[dict[str, str]]
    missing_proof: list[str]
    safe_sentence: str
    forbidden_inferences: list[str]
    next_decisive_experiment: dict[str, Any]
    admission_status: str
    source_digest: str
    card_digest: str = ""

    def compute_digest(self) -> str:
        """Compute deterministic SHA-256 digest over content excluding card_digest."""
        payload = asdict(self)
        payload["card_digest"] = ""
        serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def with_computed_digest(self) -> BridgeCard:
        """Return a copy with computed card_digest."""
        digest = self.compute_digest()
        d = asdict(self)
        d["card_digest"] = digest
        return BridgeCard(**d)


def _digest_text(text: str) -> str:
    """Return SHA-256 digest of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_bridge_cards() -> list[BridgeCard]:
    """Define the 4 canonical future work bridge status cards."""
    cards = [
        BridgeCard(
            schema=SCHEMA,
            bridge_id="carla_cross_simulator_bridge",
            title="CARLA Cross-Simulator Bridge",
            relationship_to_anchor="introduced_after_anchor",
            implementation_status="partial_prototype",
            evidence_status="diagnostic_only",
            strongest_evidence_surface="tests/carla_bridge/ and scripts/dev/check_carla_runtime.sh",
            owner_paths=[
                "robot_sf_carla_bridge/",
                "scripts/carla/",
                "docs/context/carla_replay_parity.md",
            ],
            linked_issues=[5700, 7000, 8048],
            implemented_now=[
                "Pinned CARLA client/server connector with versioned packaging.",
                "Standalone headless CARLA runtime checks and replay harness entry points.",
                "Diagnostic event logging and replay translation skeleton.",
            ],
            verified_now=[
                {
                    "item": "Client/server connection handshake passes on supported CARLA server instances.",
                    "tier": "Tier 1 (diagnostic)",
                    "status": "verified",
                },
                {
                    "item": "Bounded fixture replays translate basic waypoint and actor locations.",
                    "tier": "Tier 1 (diagnostic)",
                    "status": "verified",
                },
            ],
            missing_proof=[
                "Matched actor-complete cross-simulator scenario replay between Robot SF and CARLA.",
                "Coordinate, temporal, and action mapping formal equivalence proof.",
                "Cross-simulator metric semantic parity (TTC, comfort, SocialForce force distributions).",
                "Paired failure mode comparisons under native vs fallback execution.",
            ],
            safe_sentence=(
                "A pinned CARLA live-replay prototype exists and has demonstrated client/server "
                "connection plus bounded replay handling; matched actor-complete replay, metric "
                "parity, and cross-simulator validation remain unestablished."
            ),
            forbidden_inferences=[
                "Successful cross-simulator transfer.",
                "Validated simulation-to-real or simulation-to-CARLA policy equivalence.",
                "Benchmark-grade closed-loop evaluation on CARLA.",
            ],
            next_decisive_experiment={
                "title": "Paired Immutable Robot SF / CARLA Scenario Benchmark",
                "requirements": [
                    "One matched, immutable Robot SF/CARLA scenario family.",
                    "Actor-complete replay with declared coordinate/time/action mapping.",
                    "Identical planner input contract or explicit translation.",
                    "Metric semantic parity checks and native/fallback accounting.",
                    "Paired failure comparison with source and artifact custody.",
                ],
            },
            admission_status="diagnostic_only",
            source_digest=_digest_text("carla_cross_simulator_bridge_v1"),
        ),
        BridgeCard(
            schema=SCHEMA,
            bridge_id="route_choice_homotopy_observability",
            title="Route Choice and Homotopy Observability",
            relationship_to_anchor="introduced_after_anchor",
            implementation_status="synthetic_fixture_only",
            evidence_status="synthetic_fixture",
            strongest_evidence_surface="tests/planner/ and synthetic topological maps",
            owner_paths=[
                "robot_sf/planner/",
                "robot_sf/nav/",
                "scripts/tools/",
            ],
            linked_issues=[5308, 8048],
            implemented_now=[
                "Deterministic classification of route side (left/right passing) and topological homotopy consistency.",
                "Static topological feature extractors for planned trajectories on SVG and grid maps.",
            ],
            verified_now=[
                {
                    "item": "Synthetic geometric corridor and obstacle fixtures classify topological homotopy classes deterministically.",
                    "tier": "Tier 0 (synthetic)",
                    "status": "verified",
                },
            ],
            missing_proof=[
                "Human behavioral ground-truth or preference datasets validating route observability.",
                "Empirical proof that visible topological features improve human trajectory prediction or perceived social comfort.",
                "Controlled user studies or real-world pedestrian interaction logs.",
            ],
            safe_sentence=(
                "The repository can deterministically classify route side and homotopy consistency on synthetic fixtures; "
                "whether those observables improve human predictability or social acceptance remains unevaluated."
            ),
            forbidden_inferences=[
                "Human acceptance or preference validation.",
                "Demonstrated improvement in real-world human-robot interaction predictability.",
                "Validated pedestrian perception of robot intent.",
            ],
            next_decisive_experiment={
                "title": "Preregistered Route Observability and Human Response Study",
                "requirements": [
                    "Preregistered route conditions with planner-visible inputs.",
                    "Paired scenarios/seeds with route-side/homotopy outcomes separated from hard safety metrics.",
                    "Human-response data or clearly bounded simulator-only proxy interpretation.",
                    "Uncertainty and missingness quantification.",
                ],
            },
            admission_status="diagnostic_only",
            source_digest=_digest_text("route_choice_homotopy_observability_v1"),
        ),
        BridgeCard(
            schema=SCHEMA,
            bridge_id="incident_to_scenario_provenance",
            title="Incident-to-Scenario Provenance",
            relationship_to_anchor="introduced_after_anchor",
            implementation_status="schema_and_tooling_only",
            evidence_status="synthetic_fixture",
            strongest_evidence_surface="robot_sf/provenance/ and docs/context/evidence/",
            owner_paths=[
                "robot_sf/provenance/",
                "scripts/analysis/",
                "docs/context/evidence/",
            ],
            linked_issues=[7900, 8048],
            implemented_now=[
                "Fail-closed schema distinguishing source facts, extracted hypotheses, simulator assumptions, and replay identity.",
                "Deterministic checksum-covered scenario generation from structured incident descriptors.",
            ],
            verified_now=[
                {
                    "item": "Synthetic incident fixtures pass schema validation, digestion, and deterministic scenario generation.",
                    "tier": "Tier 0 (synthetic)",
                    "status": "verified",
                },
            ],
            missing_proof=[
                "Ingestion and validation of real-world public transportation or robot collision incident reports.",
                "Human-audited extraction accuracy and representativeness bounds.",
                "Empirical validation that reconstructed scenarios faithfully represent real incidents.",
            ],
            safe_sentence=(
                "A fail-closed provenance contract can distinguish source facts, extracted hypotheses, "
                "simulator assumptions, and replay identity for a synthetic incident fixture; "
                "real-report validity and representativeness remain future work."
            ),
            forbidden_inferences=[
                "Legal fault or liability determination.",
                "Real-world incident distribution representativeness.",
                "Automatic admission of reconstructed incidents into official benchmark leaderboards.",
            ],
            next_decisive_experiment={
                "title": "Rights-Cleared Real Incident Ingestion and Verification Protocol",
                "requirements": [
                    "Rights-cleared incident sources with human-verified extraction.",
                    "Immutable source digests with admissibility and rejection rules.",
                    "Deterministic mapping and replay with representativeness boundaries.",
                    "Separate author domain admission before benchmark use.",
                ],
            },
            admission_status="diagnostic_only",
            source_digest=_digest_text("incident_to_scenario_provenance_v1"),
        ),
        BridgeCard(
            schema=SCHEMA,
            bridge_id="amv_actuation_realism_bridge",
            title="AMV Actuation Realism Bridge",
            relationship_to_anchor="present_at_anchor",
            implementation_status="proxy_baseline_only",
            evidence_status="unsupported_proxy",
            strongest_evidence_surface="configs/algos/ and kinematic simulator parameters",
            owner_paths=[
                "robot_sf/sim/",
                "configs/algos/",
                "configs/training/",
            ],
            linked_issues=[2227, 8048],
            implemented_now=[
                "Bounded 2D unicycle and differential-drive kinematic models with acceleration, velocity, and jerk limits.",
                "Literature-backed longitudinal e-scooter proxy acceleration and deceleration profiles.",
            ],
            verified_now=[
                {
                    "item": "Simulator enforces kinematic bounds and clipping on synthetic trajectory integration.",
                    "tier": "Tier 0 (synthetic)",
                    "status": "verified",
                },
            ],
            missing_proof=[
                "Physical vehicle platform system identification (measured command-to-motion latency, motor response curves).",
                "Rotational dynamics, tire slip, terrain-dependent friction, and non-holonomic yaw inertia.",
                "Closed-loop sim-to-real trajectory tracking validation on a physical AMV.",
            ],
            safe_sentence=(
                "Public longitudinal e-scooter evidence provides a bounded proxy-source basis, "
                "while platform-specific yaw, latency, dynamics, and physical calibration remain absent."
            ),
            forbidden_inferences=[
                "Physical AMV validation or sim-to-real transfer.",
                "Accurate physical dynamics beyond 2D kinematic approximations.",
                "Vehicle-specific safety guarantees on real hardware.",
            ],
            next_decisive_experiment={
                "title": "Hardware AMV Actuation Identification and Step-Response Benchmark",
                "requirements": [
                    "Named physical platform with measured command-to-motion response.",
                    "Longitudinal and rotational dynamics, latency, and update rate measurements.",
                    "Braking and stopping behavior, geometry, and mass identification protocol.",
                    "Held-out validation with uncertainty mapped into simulator runtime parameters.",
                ],
            },
            admission_status="diagnostic_only",
            source_digest=_digest_text("amv_actuation_realism_bridge_v1"),
        ),
    ]
    return [card.with_computed_digest() for card in cards]


def validate_safe_sentence(sentence: str) -> None:
    """Validate that safe sentence does not make unverified claims."""
    lowered = sentence.lower()
    for forbidden in FORBIDDEN_TERMS_IN_SAFE_SENTENCE:
        if forbidden in lowered:
            raise ValueError(f"Unsafe claim detected in safe_sentence: '{forbidden}'")


def generate_summary_markdown(cards: list[BridgeCard]) -> str:
    """Generate the summary Markdown report."""
    lines = [
        "# Future-Work Bridge Status Summary",
        "",
        "<!-- schema: future_work_bridge_summary.v1 -->",
        "",
        "This summary documents the current engineering capability and empirical evidence distance "
        "for four future-work bridge directions in Robot SF. Implementation progress reduces engineering "
        "distance but does not itself close the dissertation empirical evidence gap.",
        "",
        "## Bridge Status Matrix",
        "",
        "| Bridge | Implemented Surface | Evidence Status | Missing Decisive Proof | Strongest Safe Interpretation |",
        "| --- | --- | --- | --- | --- |",
    ]

    for card in cards:
        validate_safe_sentence(card.safe_sentence)
        bridge_link = f"[`{card.bridge_id}`](future_work_cards/{card.bridge_id}.v1.json)"
        impl = "<br>".join(f"• {item}" for item in card.implemented_now[:2])
        ev = f"`{card.evidence_status}` ({card.admission_status})"
        missing = "<br>".join(f"• {item}" for item in card.missing_proof[:2])
        safe = card.safe_sentence
        lines.append(f"| {bridge_link} | {impl} | {ev} | {missing} | {safe} |")

    lines.extend(
        [
            "",
            "## Claim Boundary & Caution",
            "",
            "> [!IMPORTANT]",
            "> None of the future-work bridges documented here have established physical transfer, human preference ",
            "> validation, legal fault attribution, or unconstrained benchmark admission. ",
            "> All evidence is currently diagnostic-only, proxy-based, or synthetic-fixture-only.",
            "",
            "## Versioned Card Manifest",
            "",
        ]
    )

    for card in cards:
        lines.append(
            f"- **{card.title}** (`{card.bridge_id}`): "
            f"card digest `{card.card_digest[:16]}...`, "
            f"relationship: `{card.relationship_to_anchor}`"
        )

    lines.append("")
    return "\n".join(lines)


def build_all(cards_dir: Path, summary_file: Path) -> dict[str, Any]:
    """Generate cards and summary report using standard evidence writers."""
    cards_dir.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    cards = get_bridge_cards()
    written_cards: list[str] = []

    for card in cards:
        validate_safe_sentence(card.safe_sentence)
        card_path = cards_dir / f"{card.bridge_id}.v1.json"
        card_data = asdict(card)
        write_json(card_path, card_data)
        written_cards.append(str(card_path))

    summary_md = generate_summary_markdown(cards)
    write_text(summary_file, summary_md, issue_ref="#8048")

    return {
        "schema": SUMMARY_SCHEMA,
        "cards_written": written_cards,
        "summary_file": str(summary_file),
        "card_count": len(cards),
    }


def check_all(cards_dir: Path, summary_file: Path) -> bool:
    """Check whether generated artifacts match current codebase."""
    if not summary_file.exists():
        return False

    expected_cards = get_bridge_cards()
    for card in expected_cards:
        validate_safe_sentence(card.safe_sentence)
        card_path = cards_dir / f"{card.bridge_id}.v1.json"
        if not card_path.exists():
            return False
        try:
            disk_data = json.loads(card_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        expected_dict = {"review_marker": review_marker_json(), **asdict(card)}
        if disk_data != expected_dict:
            return False

    summary_disk = summary_file.read_text(encoding="utf-8")
    expected_summary = generate_summary_markdown(expected_cards)
    if not summary_disk.endswith(expected_summary):
        return False

    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-dir", type=Path, default=DEFAULT_CARDS_DIR)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument(
        "--check", action="store_true", help="Check if generated files are up to date"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)

    if args.check:
        ok = check_all(args.cards_dir, args.summary_file)
        if args.json:
            print(json.dumps({"schema": SUMMARY_SCHEMA, "ok": ok, "mode": "check"}))
        elif not ok:
            print("Drift detected in future-work bridge status cards or summary.")
            return 1
        return 0 if ok else 1

    result = build_all(args.cards_dir, args.summary_file)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"Generated {result['card_count']} bridge status cards and summary at {result['summary_file']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
