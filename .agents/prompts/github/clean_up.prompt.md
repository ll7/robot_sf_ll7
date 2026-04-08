# Clean up the current codebase

Fix any upcoming smaller issues, lint errors, or formatting problems. If bigger problems arise, respond with a detialed problem description. Do these fixes according to the project's coding standards and best practices (`./docs/dev_guide.md`).
Adhere to the project's constitution (`.specify/memory/constitution.md`).

## Add comments and docstrings

- Ensure all public classes, methods, and functions have clear and concise docstrings following the project's docstring style guide. (google docstring style)

## Refactor code

- Extract reusable helpers into dedicated modules and refer to `specs/140-extract-reusable-helpers/` for extraction criteria and guidelines.

## Lint and format code

`uv run ruff check --fix . && uv run ruff format . && uv run ruff check .`

## Tests

`uv run pytest -n auto tests`

## Type check

`uvx ty check . --exit-zero`
