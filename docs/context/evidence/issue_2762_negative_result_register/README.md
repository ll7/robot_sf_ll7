# Negative Result Register Evidence

Machine-readable companion to
[docs/context/negative_result_register.md](../../negative_result_register.md).

## Files

- `register.json`: Structured entries for all seeded negative, diagnostic-only, failed, and
  revise-classified findings.

## Schema

`negative_result_register.v1` (see `register.json` for field definitions).

## Purpose

Planning aid for research workflow: ensures negative and diagnostic results remain visible and are
not re-run, promoted, or cited as benchmark evidence.

## Validation

```bash
uv run pytest tests/docs/test_negative_result_register.py -q
```
