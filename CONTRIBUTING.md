# Contributing to robot_sf_ll7

Thank you for your interest in contributing. This document gives the public contribution path for
Robot SF. For deeper development details, use [`docs/dev_guide.md`](docs/dev_guide.md); for
agent-assisted work, use [`AGENTS.md`](AGENTS.md).

## What Kind of Contributions Are Welcome?

### 1. Bug Reports and Issue Tracking
- **Code bugs**: Problems running the simulator, incorrect behavior
- **Documentation issues**: Missing information, unclear instructions, broken examples
- **Dependency problems**: Installation failures, version conflicts
- **Test failures**: Tests that don't pass on your platform

**To report**: Open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (OS, Python version, hardware)

### 2. Feature Requests
- **New planners**: Additional navigation algorithms to benchmark
- **New scenarios**: Interaction patterns to test
- **New environment features**: Obstacles, dynamic elements, sensors
- **Performance improvements**: Optimizations for faster simulation

**To request**: Open an issue with:
- What feature would be useful
- Why it's needed (use case or research motivation)
- Rough idea of implementation (if you have one)

### 3. Code Contributions
- **Bug fixes**: Corrections to existing code
- **New planners**: New navigation algorithm implementations
- **Feature additions**: New environment features or sensors
- **Performance improvements**: Optimizations
- **Test improvements**: Better coverage or faster tests

**To contribute code**: See "Contributing Code" section below

### 4. Documentation Improvements
- **README enhancements**: Clearer instructions, better examples
- **Code examples**: New examples in `examples/`
- **Architecture documentation**: How systems work
- **API documentation**: Clearer docstrings and type hints

**To contribute docs**:
1. Make improvements to Markdown or docstrings
2. Verify examples run correctly
3. Submit PR with clear explanation

### 5. Research and Benchmarking
- **New benchmark configurations**: Additional scenario families
- **Comparison studies**: Comparing planners on different metrics
- **Platform validation**: Testing on new hardware/OS combinations
- **Performance characterization**: Measuring and documenting performance

**To contribute research**:
1. Use robot_sf_ll7 for your work
2. Document your methodology
3. Contribute benchmark configurations if they're reusable
4. Consider submitting results if notable

---

## Contributing Code

### Step 1: Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/ll7/robot_sf_ll7.git
cd robot_sf_ll7

# Install development dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Verify installation
scripts/dev/check_runtime_requirements.sh
```

### Step 2: Check Agent Workflow Requirements

Before starting work, review [`AGENTS.md`](AGENTS.md) for:
- Automated code validation requirements
- Pre-commit hook behavior
- PR readiness criteria
- How to work with agent-driven workflows

This ensures your contribution follows the project's automated review standards.

### Step 3: Create a Feature Branch

```bash
# Create a descriptive branch name
git checkout -b feature/add-your-feature
# or
git checkout -b fix/fix-issue-description
```

### Step 4: Make Your Changes

#### Code Style
- Follow existing code patterns
- Use type hints for new functions
- Write docstrings (one-line minimum, multi-line for complex functions)
- Keep functions focused and testable

#### For New Planners

Planner contributions must use an existing integration surface unless there is a clear reason to
add a new one. Start with [`docs/contributing_planner.md`](docs/contributing_planner.md). Most
map-benchmark adapters live under `robot_sf/planner/`, declare metadata in
`robot_sf/benchmark/algorithm_metadata.py`, and use config-first settings under `configs/algos/`
or `configs/policy_search/candidates/`.

New planners should fail closed when required dependencies, checkpoints, or scenario inputs are not
available. Do not silently fall back to a different planner and report it as benchmark evidence.

#### For New Environment Features
- Place in appropriate module (e.g., `robot_sf/sensor/` for sensors)
- Implement required interfaces
- Add tests in `tests/`
- Update examples to demonstrate usage

### Step 5: Write or Update Tests

```bash
# Run tests locally
scripts/dev/run_tests_parallel.sh

# Run specific test
uv run pytest tests/test_your_feature.py -v

# Check coverage for a focused change when needed
uv run pytest --cov=robot_sf tests/test_your_feature.py
```

New features should include:
- Unit tests for core logic
- Integration tests if relevant
- Example or documentation test

### Step 6: Validate Your Changes

```bash
# Format and lint fixes
scripts/dev/ruff_fix_format.sh

# Final PR readiness check
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

### External review routing

CodeRabbit reviews pull requests that change simulator code, tests, scripts, or GitHub Actions.
Documentation, context, and evidence-registration-only pull requests are excluded by default so
limited external-review capacity remains available for higher-risk changes. Add the `review-bot`
label to explicitly request CodeRabbit review for an excluded pull request. The repository applies
the `review-bot-auto` label to eligible code-bearing pull requests; do not add or remove that
managed label manually.

Gemini Code Assist has no equivalent repository-level path-and-label trigger, so this routing does
not change Gemini's automatic-review behavior. Treat Gemini feedback as optional and keep the
repository's human and gate-review requirements unchanged.

### Step 7: Commit Your Work

```bash
# Stage your changes
git add -A

# Commit with clear message
git commit -m "Brief description of what changed and why"

# Example messages:
# "Add grid-based planner implementation"
# "Fix race condition in sensor simulation"
# "Improve benchmark configuration loading"
```

### Step 8: Push and Create a Pull Request

```bash
git push origin feature/add-your-feature
```

Then open a PR on GitHub with:
- Clear title: What is changing
- Description: Why it's needed, any design decisions
- Testing notes: How you verified it works
- Related issues: Link to any GitHub issues

---

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style conventions
- [ ] All tests pass: `scripts/dev/run_tests_parallel.sh`
- [ ] Pre-commit checks pass: `scripts/dev/ruff_fix_format.sh`
- [ ] PR readiness verified: `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`
- [ ] Docstrings and comments are clear
- [ ] Acronyms and project terms are expanded on first use or linked to [`glossary.md`](docs/glossary.md); user-facing changes lead with a plain-language summary (see the `## Clarity` rule in [`maintainer_values.md`](docs/maintainer_values.md#clarity))
- [ ] Examples work (if relevant)

### In the PR Description

- **What**: Clear description of changes
- **Why**: Motivation and use case
- **How**: High-level explanation of approach
- **Testing**: How you verified it works
- **Breaking changes**: If any, clearly note them

### Review Process

- We'll review your PR and may ask questions or suggest changes
- Respond to feedback and iterate
- Once approved, a maintainer will merge your PR

---

## Benchmarking Contributions

If you're extending robot_sf_ll7 for research, consider contributing:

### New Scenario Configurations

```yaml
# In configs/scenarios/<scenario_name>.yaml
scenarios:
  scenario_name:
    description: "Description of interaction pattern"
    episodes: 100
    # ... scenario configuration
```

### New Planners for Benchmark

Planner benchmark routing is not automatic. Add the planner or adapter, metadata, readiness
classification, and a config-first path as described in
[`docs/contributing_planner.md`](docs/contributing_planner.md). For a map-runner smoke path, use
the `robot_sf_bench` entry point with explicit scenario, algorithm, and algorithm config:

```bash
uv run robot_sf_bench run \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --algo your_planner \
  --algo-config configs/algos/<planner_name>.yaml \
  --out output/benchmarks/your_planner_smoke/episodes.jsonl \
  --repeats 1 \
  --horizon 300 \
  --workers 1 \
  --no-video
```

Smoke proof shows that the wiring runs. It is not, by itself, benchmark-strength evidence.

### Research Results

If you use robot_sf_ll7 for research:
1. Document your methodology
2. Share your benchmark configurations
3. Consider contributing results/data
4. Cite robot_sf_ll7 (see [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md))

---

## Documentation Contributions

### Improving Examples

Examples go in `examples/` and are referenced in [`examples/README.md`](examples/README.md).

When adding an example:
1. Make it runnable: `uv run python examples/your_example.py`
2. Add comments explaining key steps
3. Document in `examples/README.md`
4. Test that it runs without errors

### Improving Docstrings

Project uses Google-style docstrings:

```python
def compute_distance(self, position_a, position_b):
    """Compute Euclidean distance between two positions.

    Args:
        position_a: First position [x, y].
        position_b: Second position [x, y].

    Returns:
        Euclidean distance as float.

    Raises:
        ValueError: If positions are not 2D.
    """
```

---

## Contribution Ideas

Not sure what to contribute? Here are some ideas:

### Easy
- [ ] Improve documentation clarity
- [ ] Add type hints to untyped functions
- [ ] Report bugs with clear reproduction steps
- [ ] Add unit tests for edge cases

### Medium
- [ ] Add a new example in `examples/`
- [ ] Implement a simple planner
- [ ] Add a new scenario configuration
- [ ] Improve error messages

### Hard
- [ ] Add new environment features (sensors, dynamics)
- [ ] Optimize performance bottlenecks
- [ ] Implement advanced planner algorithm
- [ ] Add cross-platform compatibility

---

## Questions? Need Help?

### Documentation
- [README.md](README.md) for project overview
- [`docs/README.md`](docs/README.md) for architecture and documentation index
- [`docs/dev_guide.md`](docs/dev_guide.md) for development guidance
- [`AGENTS.md`](AGENTS.md) for automated workflow

### Issues
- [GitHub Issues](https://github.com/ll7/robot_sf_ll7/issues) for discussions
- [GitHub Discussions](https://github.com/ll7/robot_sf_ll7/discussions) for questions

### Dissertation Context
- This repository is the Robot SF evidence base for dissertation work on AMV/AMMV safety
  validation.
- Use the public release DOI from [`README.md`](README.md) and the release tag or commit specified
  by the relevant artifact.
- Benchmark semantics and caveats are indexed from [`docs/README.md`](docs/README.md).

---

## Code of Conduct

We're committed to maintaining a welcoming and respectful community. Please:
- Be respectful and professional
- Value diverse perspectives
- Accept and give constructive feedback gracefully
- Help others succeed
- Report inappropriate behavior

---

## License

By contributing to robot_sf_ll7, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).

---

## Recognition

Contributors are recognized through:
- Commit history in Git
- GitHub contributor statistics
- [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for significant contributions

Thank you for making robot_sf_ll7 better!

---

**Last updated**: 2026-06-20
**Status**: Ready for community contributions
