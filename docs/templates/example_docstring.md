# Example Docstring Template

Use this template to author the module-level docstring for every active script in
`examples/`. The first line must match the `summary` field recorded in
`examples/examples_manifest.yaml` so that automated validation passes.

```python
"""<Purpose sentence copied verbatim into the manifest summary.>

Usage:
    uv run python examples/<relative/path/to_script.py> [--optional-flags]

Prerequisites:
    - List required assets, models, or environment variables.
    - Use "None" when no external artifacts are needed.

Expected Output:
    - Describe the primary side-effects (plots, JSONL files, console logs, etc.).
    - Include default output paths if artifacts are written to disk.

Limitations:
    - Call out constraints (e.g., "Requires GPU", "Interactive window opens").
    - If none apply, state "None".

References:
    - Link to relevant documentation sections (e.g., docs/dev_guide.md#section).
    - Omit the block entirely if no references exist.
"""
```

## Authoring Guidelines

- Keep the *Purpose* sentence under 100 characters so it renders well in tables.
- Provide `uv run python ...` commands to align with repository tooling.
- Use consistent bullet formatting; avoid nested lists inside the docstring.
- When referencing files, prefer paths relative to the repository root.
- Update both the docstring and the manifest entry together to maintain parity.
