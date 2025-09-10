# Copilot Instructions

You are assisting in a development workflow that uses GitHub Issues and follows software engineering best practices.

This project standardizes on uv for Python environment/dependency management and Ruff for linting/formatting. Use the provided VS Code tasks wherever possible.

## General Behavior

### Code Quality Standards

- Always follow the latest coding standards and best practices for the language being used
- Use clear, descriptive variable and function names that express intent
- Write code that is easy to read, understand, and maintain
- Ensure that all code is well-documented with meaningful comments and docstrings
- Follow the existing code style and patterns in the project
- Write comprehensive unit tests for new features and bug fixes
  - tests should be placed in the `tests/` directory or in the `test_pygame/` directory for tests that need a display output.
- Perform code reviews and ensure changes meet quality standards
- Use the linting task and the test task to ensure code quality before committing changes

### Always Ask Clarifying Questions (with options)

- Before implementing, ask targeted clarifying questions to confirm requirements
  - Prefer multiple-choice options when possible to speed decisions
  - Group questions by theme (scope, interfaces, data, UX, performance)
  - If answers are unknown, propose sensible defaults and confirm
- Keep questions concise and actionable; avoid blocking if not essential

### Version Control & Collaboration

- Use version control best practices with meaningful, descriptive commit messages
- When making changes, ensure backward compatibility unless explicitly specified otherwise
- Always check for existing issues, discussions, or similar work before starting new tasks
- Keep branches up-to-date with the main branch to avoid merge conflicts
- Use pull requests for code reviews and team discussions before merging
- Always run the full test suite before merging changes to ensure system stability

### Problem-Solving Approach

- Break down complex problems into smaller, manageable tasks
- Research existing solutions and patterns before implementing new approaches
- Consider the impact of changes on the entire system, not just the immediate problem
- Document architectural decisions and trade-offs made during implementation
- Think about edge cases, error handling, and potential failure modes

## Tooling and Tasks (uv, Ruff, pytest, VS Code tasks)

Use these tools and pre-defined tasks consistently:

- Environment & dependencies: uv
  - Install/resolve: Run the VS Code task "Install Dependencies" (uv sync)
  - Execute: Use `uv run <cmd>` for any Python-based commands (tests, linters)
  - Adding deps: Prefer `uv add <package>` (or edit `pyproject.toml` and run sync)
- Lint & format: Ruff
  - Run the VS Code task "Ruff: Format and Fix" before commits and PRs
  - Keep the codebase ruff-clean; fix or document rule exceptions with comments
- Tests: pytest
  - Run the VS Code task "Run Tests" for the default test suite
  - Use "Run Tests (Show All Warnings)" when diagnosing issues
  - Use "Run Tests (GUI)" for tests requiring a display (e.g., pygame)
- Code quality checks
  - Use the VS Code task "Check Code Quality" (ruff + pylint errors-only)
- Documentation/diagrams
  - Use the VS Code task "Generate UML" when updating class diagrams

Quality gates to run locally before pushing:
- Install Dependencies → Ruff: Format and Fix → Check Code Quality → Run Tests

## Documentation Standards

### Technical Documentation

- Create comprehensive documentation for all significant changes and new features
- Save documentation files in the `docs/` directory using a clear folder structure
- Each major feature or issue should have its own subfolder named in kebab-case
  - Format: `docs/42-fix-button-alignment/` or `docs/feature-name/`
- Use descriptive README.md files as the main documentation entry point for each folder

### Documentation Content Requirements

Documentation should include:
- **Problem Statement**: Clear description of the issue being addressed
- **Solution Overview**: High-level approach and architectural decisions
- **Implementation Details**: Code examples, API changes, and technical specifics
- **Impact Analysis**: What systems/users are affected and how
- **Testing Strategy**: How the changes were validated
- **Future Considerations**: Potential improvements or known limitations
- **Related Links**: References to GitHub issues, pull requests, or external resources

### Documentation Best Practices

- Use proper markdown formatting with clear headings and structure
- Include code examples with syntax highlighting
- Add diagrams or screenshots when they improve understanding
- Write for future developers who may be unfamiliar with the context
- Keep documentation up-to-date as code evolves
- Use consistent formatting and follow markdown linting standards

## Project-Specific Guidelines

### Robot SF Development

- This project focuses on robotic simulation and reinforcement learning
- Pay special attention to data integrity in simulation states and analysis
- Ensure consistency between simulation data generation and analysis pipelines
- Consider the impact on research workflows and data analysis tools
- Maintain compatibility with the fast-pysf reference implementation when applicable
- Test changes thoroughly as they may affect both simulation behavior and research results

## Design Doc First (required for non-trivial work)

Before significant changes (new features, refactors, cross-cutting fixes), create a short design doc in `docs/dev/issues/<topic>/README.md` or under an issue-specific folder as per Documentation Standards.

Recommended template:
- Title and context (link issue/PR)
- Problem statement and goals; explicit non-goals
- Constraints/assumptions
- Options considered (trade-offs, risks)
- Chosen approach (diagram if helpful)
- Data shapes/APIs/contracts (inputs/outputs, error modes)
- Test plan (unit/integration, fixtures, GUI tests if needed)
- Rollout plan and migration/back-compat notes
- Metrics/observability (how we’ll measure success)
- Open questions and follow-ups

## Examples

### Branch Naming

```
feature/42-fix-button-alignment
bugfix/89-memory-leak-in-simulator
enhancement/156-improve-lidar-performance
```

### Commit Messages

```
fix: resolve 2x speed multiplier in VisualizableSimState (#42)
feat: add new lidar sensor configuration options (#156)
docs: update installation guide with GPU setup instructions
test: add comprehensive integration tests for pedestrian simulation
```

### Documentation Structure

```
docs/42-fix-button-alignment/
├── README.md              # Main documentation
├── before-after-comparison.md
└── implementation-notes.md
```

### Version Control & Collaboration
- Use version control best practices with meaningful, descriptive commit messages
- When making changes, ensure backward compatibility unless explicitly specified otherwise
- Always check for existing issues, discussions, or similar work before starting new tasks
- Use issue numbers in commit messages to link changes to specific GitHub issues
  - Format: `fix: resolve button alignment issue (#42)`
- Create feature branches named after the issue number and title in kebab-case
  - Format: `feature/42-fix-button-alignment` or `bugfix/123-memory-leak-fix`
- Keep branches up-to-date with the main branch to avoid merge conflicts
- Use pull requests for code reviews and team discussions before merging
- Always run the full test suite before merging changes to ensure system stability

### Problem-Solving Approach
- Break down complex problems into smaller, manageable tasks
- Research existing solutions and patterns before implementing new approaches
- Consider the impact of changes on the entire system, not just the immediate problem
- Document architectural decisions and trade-offs made during implementation
- Think about edge cases, error handling, and potential failure modes

## Testing Policy

- Always add tests when fixing bugs or adding features
  - Unit tests belong in `tests/`; GUI-dependent tests in `test_pygame/`
  - Add regression tests to prevent reintroducing fixed bugs
- Ensure tests pass locally via VS Code tasks:
  - Default: "Run Tests"
  - With warnings and short tracebacks: "Run Tests (Show All Warnings)"
  - GUI-specific: "Run Tests (GUI)"
- For performance-sensitive code, add simple benchmarks or timing asserts where appropriate (and document variability)
- Keep tests deterministic; seed RNGs; use fixtures; avoid network and time flakiness

## Helpful Definitions

- uv: Fast Python package/dependency manager and runner. We use `uv sync` for lockfile/env sync and `uv run` to execute tools.
- Ruff: Python linter and formatter. Run via the "Ruff: Format and Fix" task.
- pytest: Testing framework. Run via the "Run Tests" tasks.
- VS Code tasks: Predefined commands in this repo to standardize workflows (install, lint, test, diagram).
- Quality gates: The minimal sequence of checks to pass before pushing: install → lint/format → quality check → tests.

## Markdown Documentation

- Use markdown files for documentation.
- Save documentation files in the `docs/` directory.
- Each issue has its own subfolder under `docs/`, named after the issue number and title in kebab-case.
  - Example: `docs/123-add-login-form/`
- The documentation should include:
  - A summary of the issue and its resolution
  - Code examples where applicable
  - Links to related issues or pull requests
  - Any relevant diagrams or images
- Use headings, lists, and code blocks to format the documentation for clarity.
- Write markdownlint to ensure the documentation is well-formatted and adheres to markdown standards.

## Example

If we are working on issue #42 titled “Fix button alignment”, then save the file to:

`docs/42-fix-button-alignment/README.md`

## Workflow Checklist (TL;DR)

1) Clarify requirements (ask concise, optioned questions)
2) Draft design doc under `docs/` (link issue, add test plan)
3) Implement with small, reviewed commits
4) Add/extend tests in `tests/` or `test_pygame/`
5) Run quality gates via tasks: Install Dependencies → Ruff: Format and Fix → Check Code Quality → Run Tests
6) Update docs and diagrams; run "Generate UML" if classes changed
7) Open PR with summary, risks, and links to docs/tests

