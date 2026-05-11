# Benchmark Static Dashboard

`scripts/tools/generate_benchmark_dashboard.py` creates a self-contained static HTML dashboard from
a camera-ready benchmark campaign bundle. The first supported bundle contract is a campaign root
containing `reports/campaign_summary.json`.

Run:

```bash
uv run python scripts/tools/generate_benchmark_dashboard.py \
  --bundle-root output/benchmarks/camera_ready/<campaign_id> \
  --out output/benchmarks/dashboards/<campaign_id> \
  --title "<campaign_id>"
```

The output directory contains:

- `index.html` summary page,
- `planners/*.html` per-planner pages,
- `assets/dashboard.css` local styling,
- `data/dashboard_data.json` normalized dashboard data,
- `downloads/` with compact copied report files when present,
- `dashboard_manifest.json` listing generated files and the entry point.

The dashboard has no external CSS, JavaScript, CDN, or backend dependency. It can be opened from the
filesystem or served by simple static hosting. It is intended for qualitative inspection and sharing,
not as a replacement for raw benchmark artifacts or publication bundles.

The generator fails closed when the bundle does not contain the supported campaign summary. Large raw
episode files and videos are not copied by default; keep those in the source bundle or publication
archive.
