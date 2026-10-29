"""Upload a raw run directory once to W&B Artifacts (ll7/robot_sf) and verify the remote entry table.

Credentials come from the user's netrc inside the wandb client; nothing is printed.
usage: wandb_upload.py <run_dir> <artifact_name> <metadata_json>
"""

import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import wandb

run_dir = Path(sys.argv[1]).resolve()
name = sys.argv[2]
meta = json.loads(Path(sys.argv[3]).read_text())
os.environ.setdefault("WANDB_SILENT", "true")
run = wandb.init(entity="ll7", project="robot_sf", job_type="campaign-preservation", name=name,
                 config=meta, dir="/tmp")
art = wandb.Artifact(name=name, type="campaign", metadata=meta)
art.add_dir(str(run_dir))
logged = run.log_artifact(art)
logged.wait()
uri = f"wandb://ll7/robot_sf/{name}:{logged.version}"
run.finish()

api = wandb.Api()
remote = api.artifact(f"ll7/robot_sf/{name}:{logged.version}")
entries = remote.manifest.entries
local = {p.relative_to(run_dir).as_posix(): p for p in run_dir.rglob("*") if p.is_file()}
bad = []
for rel, p in local.items():
    e = entries.get(rel)
    md5 = base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode()
    if e is None or int(e.size) != p.stat().st_size or e.digest != md5:
        bad.append(rel)
result = {"uri": uri, "local_files": len(local), "remote_entries": len(entries),
          "mismatches": bad[:10], "mismatch_count": len(bad),
          "method": "remote entry table re-fetched; paths, sizes and MD5 compared; no full remote download"}
print(json.dumps(result, indent=2))
