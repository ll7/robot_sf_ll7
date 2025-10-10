# Quickstart: Accelerated Reproducibility Test

Validate deterministic episode sequencing with minimal runtime.

## One-Liner (Python)
```python
from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark
from pathlib import Path
import time

cfg = {  # pseudo-config illustrating minimal fields
    "seeds": [12345, 12346],
    "workers": 1,
    "smoke": True,
    "enable_videos": False,
    "enable_plots": False,
}

out1 = Path("/tmp/repro_fast/run1"); out1.mkdir(parents=True, exist_ok=True)
start = time.time()
res1 = run_full_benchmark(output_dir=out1, **cfg)
res2 = run_full_benchmark(output_dir=Path("/tmp/repro_fast/run2"), **cfg)
assert res1.episode_ids == res2.episode_ids
print("OK deterministic in", time.time() - start, "sec")
```

## Shell Smoke Variant
```bash
python - <<'PY'
from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark
from pathlib import Path
cfg = {"seeds": [111,112], "workers": 1, "smoke": True, "enable_videos": False, "enable_plots": False}
res1 = run_full_benchmark(output_dir=Path("/tmp/repro_fast/run1"), **cfg)
res2 = run_full_benchmark(output_dir=Path("/tmp/repro_fast/run2"), **cfg)
assert res1.episode_ids == res2.episode_ids
print("Repro OK")
PY
```

## Expectations
- Two tiny runs finish in <2s local.
- Episode IDs identical.
- Scenario matrix hash stable (inspect manifest if needed).

## Troubleshooting
| Symptom | Action |
|---------|--------|
| Slow (>4s) | Ensure videos/plots disabled; verify only 2 seeds |
| IDs differ | Check seed list and scenario matrix modifications |
| Zero episodes | Verify seeds list not empty |
