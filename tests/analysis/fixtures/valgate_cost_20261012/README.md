# Validation cost diagnostics

Tooling measurements only. These files establish no planner quality, baseline
admission, receipt completion or publication claim.

`bounded_profile_summary.json` records one 300-second, current-main MPPI sweep
probe with authored H400, full traces, dev seeds 1001/1002 and one worker.
The cap interrupted it after one written row: the sweep is incomplete. Detailed
function counts/timings are preserved in `bounded_profile.json.gz`; private
path prefixes were replaced with relative source labels. Cumulative profiler
times overlap and profiling itself adds overhead; they are not an uninstrumented
end-to-end sweep timing.

`timings.json` records three before/after receipt-loader measurements on the
same synthetic 30-slot scale fixture. It clones the archived baseline goal
trace (203 steps) across dev seeds 1001–1030, replacing inert private path
strings. No episodes were executed to produce this scale fixture. It is **not**
a behaviour receipt dataset and has no authoritative execution metadata.

`raw.jsonl.gz` and `summaries.jsonl.gz` preserve the exact synthetic loader
inputs. `loaded.json.gz` preserves the serialized loader output; both revisions
produced those exact bytes. Hashes in the manifests identify the named compressed
files; byte sizes describe their decompressed content.

Reproduce from the repository root (fresh output directory required):

```bash
UV_NO_SYNC=1 uv run python scripts/validation/profile_receipt_loading.py \
  --reference 33a4735900153e742828227cc2171436ea09cd90 \
  --output-dir ../receipt-loading-profile --repeats 3
```

The reference and patched loader execute on identical inputs in one process;
imports, generation and output serialization are outside the timed region.
Filesystem caches are warm after the first read; no persistent application cache
is used. Runtime and memory limits for the full receipt matrix remain unmeasured.
