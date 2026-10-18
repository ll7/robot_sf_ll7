# AUD2 test value and red/green proof

AI-GENERATED NEEDS-REVIEW. Offline diagnostic tooling only.
Comparator: reviewed PR head `2e0d42e4505cce1be7c2ba5180b650fe42424eb9`.

Every new node was run against this revision with the final test files:

```bash
cd /home/luttkule/aud2-before
PYTHONPATH=/home/luttkule/aud2-before /home/luttkule/aud-work/.venv/bin/pytest \
  /home/luttkule/aud-work/tests/analysis_workbench/test_aud2_refutations.py \
  /home/luttkule/aud-work/tests/analysis_workbench/test_aud2_detectors.py -q --no-cov
```

**28 failed**, all intended assertions: four reviewer Mapping nodes report clear;
12 physical NaN/+Inf/-Inf nodes report the generic row reason instead of the
measurement error; four physical boolean nodes report unavailable; one unnamed
Mapping node reports clear; two 1%-shift nodes report clear; five new-channel
nodes fail the explicit public-default-scan assertion that the independent
channel is present. No import, fixture or collection failures occurred in this
final proof. [Full red output](red.txt), [green output](green.txt).

The four reviewer nodes retain their names, input, public scan call and assertions
from `rv10-evidence/test_refutations.py`; only formatting/docstring were added.
New-channel tests exercise both the missing-default-registration regression and
specific measured values/positive/negative controls after registration.

| Behavior protected | Credible regression | Nearest coverage gap | Production seam |
| --- | --- | --- | --- |
| Mapping physical scalars return measurement errors (4 imported nodes) | Skip every Mapping in metrics | `test_aud_9952` covers valid structured published rows but no physical object corruption | None; public scan |
| NaN and both infinities return physical measurement errors (12 nodes) | Generic admission hides the physical error, or non-finite scalars get skipped | Prior admission tests assert error status, not the physical reason/path | None; public detect |
| Physical booleans are malformed (4 nodes) | Treat every boolean as an outcome diagnostic | Prior extreme test only covers legitimate outcome booleans | None |
| Only named structured records bypass scalar validation (1 node) | Restore Mapping-by-type exemption | Published fixtures have legitimate named records only | None |
| 24 success / 6 timeout subgroup has an independent signal (1 node) | Remove default incidence channel, condition denominator on outcome, or require external planners | `test_outlier_cohorts_match_terminal_outcome` protects benign matching but skips adverse subgroup sensitivity | None; public scan and literal 6/30 oracle |
| Uniform planner failures retain same-scenario rates and absolute backstop (1 node) | Match planner config across planners, mix scenarios, remove absolute backstop, or remove registration | Old common-mode requires absent initial-state/config compatibility and multiple planners | None; literal 1 versus cross-planner median 0 |
| Uniform metric shift has external planner-median signal (1 node) | Remove channel or restore within-planner/outcome normalization | Existing seed/multivariate tests cannot see a planner-wide shift | None; public scan, literal medians 0.4 vs [0.2, 0.2] |
| Numerical floors suppress micro-noise but preserve 1% shifts (2 nodes) | Restore 0.05 relative floor or zero-spread shortcut | Old noise test uses median zero and misses nonzero-center suppression | None; hand-selected 10.1 and 10.000001 versus 10 |
| Timeout horizon and termination consistency survive small cohorts (1 node) | Remove channel, excuse early timeout, accept steps beyond horizon, or disregard reason/outcome conflicts | Goal detector needs absent traces and optional horizon channel did not exist | None; counts and terminal states are explicit |
| Recorded unresolved candidate exposes both configured limits (1 node) | Ignore simulator limit, normalize timeout away, or remove channel | Prior sample documents this candidate without an independent flag | None; exact SHA-pinned public raw line |

Changed legacy expectations preserve their previous contracts:

- Registry length 14 -> 17 and scheduled attempts 70 -> 85: the three new default
  channels are independently tested through the public scan; the existing closed,
  typed and sorted registry/accounting assertions remain.
- Hand-derived robust score uses denominator `1.4826 * 1e-5` for the peer center 1,
  instead of `1.4826 * 0.05`. Restoring the old broad floor fails both this literal
  score and the independent nonzero-center sensitivity tests.
- The release-gate report digest changes only because it binds engine v1.2
  provenance. The collision gate remains opt-in and its explicit opt-in assertions
  remain. Before/after full release-row gate counts are compared separately; the
  golden is not used as evidence for new detector sensitivity.

No planner steps, test-only production seams, probability calibration or causal
planner conclusions are involved. Missing external controls are tested as
unavailable, and source non-finite inventory admission remains fail-closed.
