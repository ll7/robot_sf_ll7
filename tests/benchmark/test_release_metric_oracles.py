"""Small manifest-bound release-drift alarms; never run release episodes here.

The synthetic numbers come from tests at historical F2 source 66f402ba, not acquisition
receipts. They keep a few absolute checks alongside paired tests, which cannot
notice shared drift. Rebaseline only at a release and record the moving commit.
The 0.0.8 F3 freeze is 373dbfde4f39667cf9e8732dabe7df5118bdeab1.
Any manifest byte change intentionally invalidates every sentinel association.
Existing release evidence and historical scoring/replay tests remain untouched.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark import metrics

ROOT = Path(__file__).resolve().parents[2]
ORACLE = json.loads(
    (ROOT / "tests/benchmark/fixtures/release_0_0_8_metric_sentinels.json").read_text()
)


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda case: case["name"])
def test_release_pinned_synthetic_metric(case):
    """Check the preserved freeze-source value with an explicit release tolerance."""
    manifest_bytes = (ROOT / ORACLE["manifest_path"]).read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == ORACLE["manifest_sha256"]
    pos = np.asarray(case["robot_pos"], dtype=float)
    data = metrics.EpisodeData(
        robot_pos=pos,
        robot_vel=np.asarray(case.get("robot_vel", np.zeros_like(pos)), dtype=float),
        robot_acc=np.asarray(case.get("robot_acc", np.zeros_like(pos)), dtype=float),
        peds_pos=np.zeros((len(pos), 0, 2)),
        ped_forces=np.zeros((len(pos), 0, 2)),
        goal=pos[-1],
        dt=case.get("dt", 0.5),
    )
    metric = getattr(metrics, case["metric"])
    kwargs = (
        {"metric_schema_version": ORACLE["metric_schema_version"]}
        if case["metric"] == "curvature_mean"
        else {}
    )
    assert metric(data, **kwargs) == pytest.approx(case["expected"], **ORACLE["tolerance"])
