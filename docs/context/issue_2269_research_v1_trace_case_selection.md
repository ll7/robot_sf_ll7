# Issue #2269 Research-v1 Trace Case Selection

Issue: [#2269](https://github.com/ll7/robot_sf_ll7/issues/2269)
Parent issue: [#2159](https://github.com/ll7/robot_sf_ll7/issues/2159)
Date: 2026-06-05
Status: analysis-only case selection; no rendering or pack assembly.

## Goal

Select three to five durable trace-review cases for the Issue #2159 research-v1 failure-case pack.
The selection should identify source evidence, rationale, required render inputs, blockers, and
claim boundaries before any trace-viewer pack is built.

## Result

Five cases are selected in the manifest:

1. AMMV Social Force head-on corridor mechanism activation.
2. AMV cross-trap hazard/ODD coverage slice.
3. Head-on corridor pedestrian-route-offset trace response.
4. Leave-group pedestrian-speed outcome flip.
5. Intersection-wait `+0.5 m/s` speed-grid phase response.

The first two are research-v1 AMV-specific and have durable summary/coverage evidence, but still
need renderable trace inputs. The last three already have compact closest-approach trace-slice
evidence and can feed first render/review children without inventing data. None of the cases are
benchmark-strength or paper-facing evidence.

## Manifest

- [evidence/issue_2269_research_v1_trace_case_selection_2026-06-05/case_selection_manifest.yaml](evidence/issue_2269_research_v1_trace_case_selection_2026-06-05/case_selection_manifest.yaml)
- [evidence/issue_2269_research_v1_trace_case_selection_2026-06-05/README.md](evidence/issue_2269_research_v1_trace_case_selection_2026-06-05/README.md)

## Evidence Boundary

This work moves `research-v1.amv.failure_case_review` from speculative proposal to diagnostic case
selection only. It does not provide the trace-viewer pack, qualitative rendered panels, or
paper-ready failure analysis required for any stronger claim.

The selected cases are intended to unlock follow-up children:

- first render/review child from an already trace-sliced case;
- AMV-specific trace export child for the AMMV and cross-trap cases;
- later pack assembly after at least one rendered case is reviewed.

## Validation

```bash
rtk python - <<'PY'
from pathlib import Path
import yaml
p = Path("docs/context/evidence/issue_2269_research_v1_trace_case_selection_2026-06-05/case_selection_manifest.yaml")
data = yaml.safe_load(p.read_text())
assert len(data["cases"]) == 5
assert sum(1 for case in data["cases"] if case["durable_case_ready"]) >= 3
for case in data["cases"]:
    for path in case["source_evidence"]:
        assert Path(path).exists(), path
PY
rtk bash scripts/dev/check_docs_proof_consistency_diff.sh
rtk git diff --check
```
