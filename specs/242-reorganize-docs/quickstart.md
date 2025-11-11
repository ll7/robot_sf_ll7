# Quickstart: Navigating the Documentation

This feature reorganizes and clarifies the documentation layout. Use this as a fast entry point.

## Where to Start
- Central Index: [docs/README.md](../../docs/README.md) (lists all major guides by category)
- Development Workflow: [docs/dev_guide.md](../../docs/dev_guide.md)
- Environment Usage: [docs/ENVIRONMENT.md](../../docs/ENVIRONMENT.md)
- Benchmarking & Metrics: [docs/benchmark.md](../../docs/benchmark.md) (and related `benchmark_full_classic.md` / visuals) + SNQI tooling under [docs/snqi-weight-tools/](../../docs/snqi-weight-tools/)
- Simulation UI: [docs/SIM_VIEW.md](../../docs/SIM_VIEW.md)
- Refactoring & Architecture Notes: [docs/refactoring/](../../docs/refactoring/)

## Local Viewing
```bash
open docs/README.md  # macOS quick open in default viewer
```
Or inside VS Code use the Explorer or Quick Open (Cmd+P) and type part of a filename.

## Contribution Checklist (Docs)
1. Add/update the page under an appropriate directory.
2. Ensure the page has an H1 and a short purpose line.
3. Link the page from `docs/README.md` in the correct category.
4. If the page introduces a new public surface (factory, metric, baseline), cross-link it from relevant existing guides.
5. Run a manual link scan (search for filename changes) if you moved/renamed files.

## Planned Follow-Up (Out of Scope Here)
- Optional CI link checker integration.
- Consolidation of overlapping benchmark summary pages.

## Success Criteria
- All renamed or key pages are listed in the central index.
- No orphaned docs (each reachable within 2 clicks from index).
