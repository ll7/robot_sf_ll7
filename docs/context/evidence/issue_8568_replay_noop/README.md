<!-- AI-GENERATED (issue8654 source intake) - NEEDS-REVIEW -->

# Source intake: corridor-medium, seed 115

The verified September release does **not** retain a source-complete restart
pair for this case. All 14 planner rows disable both step and decision trace
recording. They retain episode identities, aggregate metrics and sparse event
evidence, not complete hidden state. This is diagnostic-only intake for #8654,
not a replay result, new acquisition, planner comparison or completion of #8568.

The selected case is `classic_head_on_corridor_medium`, seed 115. Selection remains
unresolved: no planner passes source completeness, and native compatibility was
not executed. The first canonical arm is `prediction_planner`, episode
`classic_head_on_corridor_medium--115--01769919710c38b8`, line455 of its episode
file. No earlier-source nomination outcome is used as September evidence.

## Evidence and limitations

[source_manifest.json](source_manifest.json) records the ordered 14-arm intake,
channel dispositions, source Git identities and compatibility caveats.
[verification.json](verification.json) records archive safety, exact inspected
member hashes and full file inventory. [SHA256SUMS](SHA256SUMS) binds these two
files and this README with repository-root-relative paths.

The archive SHA-256 and 54,222,465-byte size matched detached publication custody;
all 121 listed member checksums passed. The archive contains 147 members/124 regular
files (715,090,822 uncompressed bytes). No bundled Python/shell file was executed.
Selected rows and provenance sidecars were inspected, not merely filenames.
The first row has `provenance.artifact_uri`, `map_digest`, `scenario_digest` and
`planner_commit` set to null. Its observed checkpoint digest is retained as a row
claim, not proof that checkpoint bytes have been staged or independently verified.

The resolved release binds `paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml`,
not the contextual `..._2026_08.yaml`. Their raw hashes differ legitimately; at
scientific source their scientific fields and planner order agree, while name,
release tag and DOI differ. These raw hashes are also distinct from short runtime
configuration hashes. Source-map/lock hashes do not fill missing captured runtime
state. An episode ID is not unique across planner arms: use the composite identity
in the verification receipt.

`event_ledger`, `termination_reason`, `metrics` and diagnostic command counts are
present. They do not provide full actor states, action sequences, RNG (random-number
generator) state, planner memory, observation history, or metric accumulators.
No absence claim extends beyond the verified release payload and its source
bindings. Unsupported state in the unmerged #8620 prototype remains unsupported;
this intake does not validate or modify the active replay stack.

## Reproduce intake verification

Run from the repository root with the recorded scientific Git object available.
The repository shared-environment wrapper supplies Python dependencies without a
simulation. Require approximately1GB free temporary storage. Do not execute any
code from the downloaded archive. The commands below inspect existing artifacts;
they do not run the benchmark.

```bash
set -euo pipefail
export INTAKE8654_DIR=$(mktemp -d /tmp/issue8654-verify.XXXXXX)
export INTAKE8654_URL=https://github.com/ll7/robot_sf_ll7/releases/download/paper-matrix-v2-h600-s30-2026-09-59577bad289dd692ba3580e1600c4a649ae27880-erratum.1
printf 'Task-owned verification directory: %s\n' "$INTAKE8654_DIR"
curl --fail --location "$INTAKE8654_URL/september-erratum-derived-build-v2_publication_bundle.tar.gz" --output "$INTAKE8654_DIR/archive.tar.gz"
for name in checksums.sha256 publication_manifest.json publication_custody.json; do
  curl --fail --location "$INTAKE8654_URL/$name" --output "$INTAKE8654_DIR/$name" || exit 1
done
scripts/dev/run_worktree_shared_venv.sh -- python - <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import yaml

root = Path("docs/context/evidence/issue_8568_replay_noop")
manifest = json.loads((root / "source_manifest.json").read_text())
receipt = json.loads((root / "verification.json").read_text())
stage = Path(os.environ["INTAKE8654_DIR"])
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
release = manifest["release"]
assert (stage / "archive.tar.gz").stat().st_size == release["archive_size_bytes"]
for name, field in [("archive.tar.gz", "archive_sha256"),
                    ("checksums.sha256", "checksums_sha256"),
                    ("publication_manifest.json", "publication_manifest_sha256"),
                    ("publication_custody.json", "publication_custody_sha256")]:
    assert digest(stage / name) == release[field], name
custody = json.loads((stage / "publication_custody.json").read_text())
assert custody["archive"]["sha256"] == release["archive_sha256"]
assert custody["source_execution_commit"] == manifest["scientific_source_commit"]
with tarfile.open(stage / "archive.tar.gz", "r:gz") as archive:
    members = archive.getmembers()
    assert len(members) == receipt["archive_members"]
    assert sum(m.size for m in members) == receipt["uncompressed_bytes"]
    assert len(members) <= 1000 and sum(m.size for m in members) < 1024**3
    for member in members:
        path = PurePosixPath(member.name)
        assert not path.is_absolute() and ".." not in path.parts
        assert member.isfile() or member.isdir()
    archive.extractall(stage / "extracted", filter="data")
bundle = stage / "extracted/september-erratum-derived-build-v2_publication_bundle"
assert digest(bundle / "checksums.sha256") == release["checksums_sha256"]
for line in (bundle / "checksums.sha256").read_text().splitlines():
    expected, relative = line.split(None, 1)
    assert digest(bundle / relative) == expected, relative
assert sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()) == receipt["archive_inventory"]
for relative, expected in receipt["inspected_member_sha256"].items():
    assert digest(bundle / relative) == expected, relative
src = manifest["scientific_source_commit"]
def source(path):
    return subprocess.check_output(["git", "show", src + ":" + path])
for entry in manifest["source_objects"]:
    assert hashlib.sha256(source(entry["path"])).hexdigest() == entry["raw_sha256"]
template = yaml.safe_load(source("configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"))
context = yaml.safe_load(source("configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml"))
assert {k for k in template if template[k] != context[k]} == {"name", "release_tag", "doi"}
assert [p["key"] for p in template["planners"]] == [p["planner"] for p in manifest["planner_evaluation"]]
for expected in manifest["planner_evaluation"]:
    matches = []
    with (bundle / expected["source_path"]).open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            row = json.loads(raw)
            if row.get("scenario_id") == "classic_head_on_corridor_medium" and row.get("seed") == 115:
                matches.append((number, raw, row))
    assert len(matches) == 1
    number, raw, row = matches[0]
    assert number == expected["line"]
    assert hashlib.sha256(raw).hexdigest() == expected["row_bytes_sha256"]
    assert row["episode_id"] == expected["episode_id"] and row["git_hash"] == src
    assert row["config_hash"] == expected["row_config_hash"]
    assert row["scenario_params"]["record_simulation_step_trace"] is False
    assert row["scenario_params"]["record_planner_decision_trace"] is False
    assert row["provenance"]["artifact_uri"] is None
    assert "metrics" in row and "event_ledger" in row and "termination_reason" in row
    for pointer, expected_digest in expected["checkpoint_identity_fields"].items():
        value = row
        for token in pointer.lstrip("/").split("/"):
            value = value[token.replace("~1", "/").replace("~0", "~")]
        assert value == expected_digest, pointer
    config = expected["planner_config_source"]
    if config["path"] is not None:
        assert hashlib.sha256(source(config["path"])).hexdigest() == config["config_sha256"]
print("Verified custody, archive/members, source objects and 14 ordered source-incomplete rows; no replay performed.")
PY
sha256sum -c docs/context/evidence/issue_8568_replay_noop/SHA256SUMS
```

Use ordinary Python without optimization: the recipe assertions are verification
checks. Repeat with a fresh download or an independently checksum-verified cache;
the same raw rows and canonical dispositions must match. Independently inspect
the first row and sidecar with `jq`, including `/event_ledger/collision_events`,
`/metrics`, `/scenario_params`, `/provenance` and the complete raw-artifact list.
Inspect all archived filenames and selected row keys before accepting missing-state
dispositions; the recipe's false trace flags alone are not a universal absence proof.

After preserving the verification output, remove only the printed task-owned
temporary directory through the repository's safe cleanup procedure. Temporary
paths/access times are operational metadata, not canonical identity.

## Provenance and next action

Source citation: Luttkus, Lennart; Tröster, Marco, September 2026 benchmark-data
erratum, version DOI 10.5281/zenodo.22265925. The bundled rights statement declares
GPL-3.0-only for this dataset and excludes unlisted external checkpoint/dataset
rights. Only compact source facts are retained here; raw rows, weights and archives
remain outside Git. No credentials or private operational logs are included.

The parent may use this packet to plan its separately authorized diagnostic
acquisition after snapshot compatibility is proven. It must preserve new run
identities, historical-fidelity unavailability and the three-case bound. This
child neither authorizes that execution nor closes the parent. Fallback, degraded
or omitted-state paths are not successful benchmark evidence.
