"""Prepare the original 72 dev slots with explicit law and diagnostic controls."""

import json
import os
import subprocess
import sys
from pathlib import Path

source = Path(__file__).resolve().parents[3]
root = Path(os.environ["WALL_CONTACT_ARTIFACT_ROOT"])
root.mkdir(parents=True, exist_ok=True)
slots = json.loads(Path(__file__).with_name("slots.json").read_text())
sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
laws = {
    "legacy": "legacy_shifted_gradient_v1",
    "range_only": "body_edge_exponential_v3_range_only",
    "physical_margin": "body_edge_exponential_v3_physical_margin",
    "contact_stiff": "body_edge_exponential_v3_contact_stiff",
    "multi_segment": "body_edge_exponential_v3_multi_segment",
    "wall10x": "body_edge_exponential_v3_range_only",
    "small_dt": "body_edge_exponential_v3_range_only",
    "no_social": "body_edge_exponential_v3_range_only",
    "physical_margin10x": "body_edge_exponential_v3_physical_margin",
    "physical_margin10x_small_dt": "body_edge_exponential_v3_physical_margin",
}
controls = {
    "wall10x": {"wall_scale": 10.0},
    "small_dt": {"dt_s": 0.01},
    "no_social": {"disable_social": True, "disable_groups": True},
    "physical_margin10x": {"wall_scale": 10.0},
    "physical_margin10x_small_dt": {"wall_scale": 10.0, "dt_s": 0.01},
}
plan = {
    **slots,
    "python": sys.executable,
    "laws": laws,
    "sources": dict.fromkeys(laws, sha),
    "roots": dict.fromkeys(laws, str(source)),
    "controls": controls,
}
(root / "measurement-manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
print(json.dumps({"source_sha": sha, "tasks": len(plan["tasks"]), "laws": list(laws)}))
