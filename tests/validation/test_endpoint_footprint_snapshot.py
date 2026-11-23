"""Keep the released-input 0.1.0 full-footprint evidence current."""

import json
from collections import Counter
from pathlib import Path

from scripts.validation.check_scenario_archetype_geometry import inspect_release_zones

SNAPSHOT = Path("docs/scenario_hygiene/endpoint_footprint_0_1_0.json")


def _findings(policy):
    return [
        {"matrix": row["matrix"], "scenario": row["scenario"], "zone": row["zone"], **hit}
        for row in inspect_release_zones(endpoint_policy=policy)
        for hit in row["intersections"]
    ]


def _identity(hit):
    return tuple(hit[key] for key in ("matrix", "scenario", "zone", "actor_kind", "actor"))


def test_endpoint_footprint_snapshot_matches_live_release_audit():
    """Real loader/auditor must reproduce identities, radii, geometry and fingerprints."""
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    legacy = _findings("pedestrian_radius_v1")
    footprint = _findings("robot_pedestrian_radii_v2")
    legacy_ids = {_identity(hit) for hit in legacy}
    additional = [hit for hit in footprint if _identity(hit) not in legacy_ids]
    assert legacy_ids <= {_identity(hit) for hit in footprint}
    assert len({_identity(hit) for hit in additional}) == len(additional)
    actual = {
        "legacy_count": len(legacy),
        "footprint_count": len(footprint),
        "additional_count": len(additional),
        "per_scenario": dict(Counter(hit["scenario"] for hit in additional)),
        "additional": sorted(additional, key=_identity),
    }
    expected["additional"] = sorted(expected["additional"], key=_identity)
    # JSON represents the loader's coordinate tuples as arrays.
    assert json.loads(json.dumps(actual)) == {key: expected[key] for key in actual}, (
        "Endpoint footprint evidence drift; regenerate and review dispositions"
    )
