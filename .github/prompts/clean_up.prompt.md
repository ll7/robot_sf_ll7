# Clean up the current codebase

Fix any upcoming smaller issues, lint errors, or formatting problems. If bigger problems arise, respond with a detialed problem description. Do these fixes according to the project's coding standards and best practices (`./docs/dev_guide.md`).

## Lint and format code
`uv run ruff check --fix . && uv run ruff format . && uv run ruff check .`

## Tests
`uv run pytest tests`

## Type check
`uvx ty check . --exit-zero`

