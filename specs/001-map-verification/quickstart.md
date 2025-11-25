# Quickstart — Map Verification Workflow

1. **Install deps**
   ```bash
   uv sync && source .venv/bin/activate
   ```
2. **Run full map audit locally**
   ```bash
   uv run python scripts/validation/verify_maps.py --scope all --mode local --output output/validation/map_verification.json
   ```
3. **Targeted check (changed files only)**
   ```bash
   uv run python scripts/validation/verify_maps.py --scope changed --mode local
   ```
4. **CI-equivalent run**
   ```bash
   uv run python scripts/validation/verify_maps.py --scope ci --mode ci --seed 123
   ```
5. **Inspect manifest + logs**
   - JSON summary: `output/validation/map_verification.json`
   - Loguru logs: `output/run-tracker/logs/*.log` when tracker enabled
6. **Fix issues**
   ```bash
   uv run python scripts/validation/verify_maps.py --scope broken_map.svg --mode local --fix
   ```
7. **Update docs**
   - Add verifier mention to `docs/SVG_MAP_EDITOR.md`
   - Link quickstart from `docs/README.md`
