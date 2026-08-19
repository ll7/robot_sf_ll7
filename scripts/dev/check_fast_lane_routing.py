"""Audit changed source modules for nearby tests excluded from fast PR shards."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA = "fast_lane_routing.v1"
SOURCE_ROOT = "robot_sf/"
TEST_ROOT = "tests/"

_CONTRACT_HINTS = (
    "atlas",
    "contract",
    "evidence",
    "harness",
    "manifest",
    "materializ",
    "provenance",
    "schema",
    "serializ",
)
_SLOW_PATH_HINTS = (
    "tests/campaign/",
    "tests/classic_interactions/",
    "tests/perf/",
    "tests/pygame/",
    "tests/scenario_certification/",
    "tests/scenarios/",
    "tests/visuals/",
)
_EXPLICIT_LANE_RE = re.compile(r"(?:robot[-_]sf[-_]test[-_]lane|fast[-_]lane)\s*:\s*(\w+)")
_SLOW_RUNTIME_RE = re.compile(
    r"\b(?:Simulator|RobotEnv|PyGame|PyVista)\s*\(|"
    r"\bsimulator\.(?:execute|reset|run|step)\s*\(|"
    r"\b(?:make_env|render_video|run_episode)\s*\(|\bslurm\b"
)


@dataclass(frozen=True)
class FastLanePolicy:
    """The subset of pytest auto-marking policy needed by this audit."""

    fast_files: frozenset[str]
    fast_path_fragments: tuple[str, ...]
    fast_file_prefixes: tuple[str, ...]
    slow_file_overrides: frozenset[str]

    def is_fast(self, test_path: str) -> bool:
        """Return whether ``tests/conftest.py`` keeps this path out of ``slow``."""

        normalized = test_path.replace("\\", "/")
        filename = Path(normalized).name
        if filename in self.slow_file_overrides:
            return False
        if filename in self.fast_files:
            return True
        if any(fragment in normalized for fragment in self.fast_path_fragments):
            return True
        return any(filename.startswith(prefix) for prefix in self.fast_file_prefixes)


@dataclass(frozen=True)
class RoutingObservation:
    """One changed source/test pairing classified by the routing audit."""

    source_module: str
    test_path: str
    classification: str
    policy_state: str
    rationale: str
    suggested_registration: str | None = None

    @property
    def needs_attention(self) -> bool:
        """Return whether the pairing needs a fast-lane registration or decision."""

        return self.policy_state in {"missing-fast-registration", "needs-classification"}


def _literal_assignments(source: str) -> dict[str, object]:
    """Read simple module-level literal assignments without importing pytest config."""

    tree = ast.parse(source)
    assignments: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                try:
                    assignments[target.id] = ast.literal_eval(value)
                except (ValueError, TypeError):
                    continue
    return assignments


def load_fast_lane_policy(source: str) -> FastLanePolicy:
    """Load the existing ``tests/conftest.py`` routing policy as data."""

    assignments = _literal_assignments(source)
    return FastLanePolicy(
        fast_files=frozenset(str(value) for value in assignments.get("_FAST_FILES", set())),
        fast_path_fragments=tuple(
            str(value) for value in assignments.get("_FAST_PATH_FRAGMENTS", ())
        ),
        fast_file_prefixes=tuple(
            str(value) for value in assignments.get("_FAST_FILE_PREFIXES", ())
        ),
        slow_file_overrides=frozenset(
            str(value) for value in assignments.get("_SLOW_FILE_OVERRIDES", set())
        ),
    )


def _has_slow_marker(test_path: str, source: str) -> bool:
    """Return whether a test explicitly or structurally belongs in the slow lane."""

    normalized = test_path.replace("\\", "/").lower()
    if any(fragment in normalized for fragment in _SLOW_PATH_HINTS):
        return True
    if "pytest.mark.slow" in source or "pytestmark = pytest.mark.slow" in source:
        return True
    explicit_lane = _EXPLICIT_LANE_RE.search(source.lower())
    if explicit_lane and explicit_lane.group(1) in {"slow", "simulation", "campaign"}:
        return True
    return bool(_SLOW_RUNTIME_RE.search(source))


def _is_contract_candidate(test_path: str, source: str) -> bool:
    """Return whether a test looks like deterministic schema/contract coverage."""

    haystack = f"{test_path.replace('\\', '/').lower()}\n{source.lower()}"
    explicit_lane = _EXPLICIT_LANE_RE.search(source.lower())
    if explicit_lane:
        return explicit_lane.group(1) in {
            "fast",
            "contract",
            "fast-contract",
        }
    return any(hint in haystack for hint in _CONTRACT_HINTS)


def classify_test(test_path: str, source: str) -> tuple[str, str]:
    """Classify a nearby test and return ``(classification, rationale)``."""

    if _has_slow_marker(test_path, source):
        return (
            "slow-simulation-or-campaign",
            "explicit slow marker or simulation/campaign path/runtime signal",
        )
    if _is_contract_candidate(test_path, source):
        return (
            "fast-contract-candidate",
            "deterministic schema, manifest, materialization, provenance, or harness signal",
        )
    return (
        "needs-explicit-classification",
        "nearby test is not clearly a contract test or an explicitly slow simulation/campaign test",
    )


def _suggest_registration(test_path: str) -> str:
    """Return the smallest human-actionable fast-lane registration hint."""

    filename = Path(test_path).name
    return (
        f"Add '{filename}' to tests/conftest.py:_FAST_FILES, or add a narrowly scoped "
        "_FAST_PATH_FRAGMENTS entry if the whole directory is contract-only."
    )


def _nearby_test_paths(source_module: str, test_paths: Sequence[str]) -> list[str]:
    """Find exact basename-matched tests for one source module."""

    stem = Path(source_module).stem.lower()
    expected_names = {f"test_{stem}.py", f"{stem}_test.py"}
    return sorted(
        path
        for path in test_paths
        if Path(path).name.lower() in expected_names
        and path.replace("\\", "/").startswith(TEST_ROOT)
    )


def audit_changed_modules(
    changed_modules: Sequence[str],
    test_contents: Mapping[str, str],
    policy: FastLanePolicy,
) -> tuple[RoutingObservation, ...]:
    """Classify changed source modules against tracked nearby test contents."""

    observations: list[RoutingObservation] = []
    for source_module in sorted(changed_modules):
        if not source_module.startswith(SOURCE_ROOT) or not source_module.endswith(".py"):
            continue
        if Path(source_module).name == "__init__.py":
            continue
        for test_path in _nearby_test_paths(source_module, tuple(test_contents)):
            classification, rationale = classify_test(test_path, test_contents[test_path])
            if classification == "slow-simulation-or-campaign":
                policy_state = "slow-by-policy"
                suggestion = None
            elif policy.is_fast(test_path):
                policy_state = "registered-fast"
                suggestion = None
            else:
                policy_state = (
                    "missing-fast-registration"
                    if classification == "fast-contract-candidate"
                    else "needs-classification"
                )
                suggestion = _suggest_registration(test_path)
            observations.append(
                RoutingObservation(
                    source_module=source_module,
                    test_path=test_path,
                    classification=classification,
                    policy_state=policy_state,
                    rationale=rationale,
                    suggested_registration=suggestion,
                )
            )
    return tuple(observations)


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _repo_root(path: Path) -> Path:
    return Path(_git(path, "rev-parse", "--show-toplevel").strip())


def _changed_modules(repo_root: Path, base_sha: str, head_sha: str) -> list[str]:
    output = _git(
        repo_root,
        "diff",
        "--name-only",
        "--diff-filter=ACMRTUXB",
        f"{base_sha}..{head_sha}",
        "--",
        SOURCE_ROOT,
    )
    return [path for path in output.splitlines() if path.endswith(".py")]


def _tracked_tests(repo_root: Path, head_sha: str) -> list[str]:
    output = _git(repo_root, "ls-tree", "-r", "--name-only", head_sha, "--", TEST_ROOT)
    return [path for path in output.splitlines() if path.endswith(".py")]


def _test_contents(repo_root: Path, head_sha: str, test_paths: Sequence[str]) -> dict[str, str]:
    return {path: _git(repo_root, "show", f"{head_sha}:{path}") for path in test_paths}


def build_report(repo_root: Path, base_sha: str, head_sha: str) -> dict[str, object]:
    """Build the machine-readable fast-lane routing report."""

    changed_modules = _changed_modules(repo_root, base_sha, head_sha)
    tracked_tests = _tracked_tests(repo_root, head_sha)
    candidate_tests = sorted(
        {
            test_path
            for source_module in changed_modules
            for test_path in _nearby_test_paths(source_module, tracked_tests)
        }
    )
    test_contents = _test_contents(repo_root, head_sha, candidate_tests)
    policy_source = _git(repo_root, "show", f"{head_sha}:tests/conftest.py")
    policy = load_fast_lane_policy(policy_source)
    observations = audit_changed_modules(changed_modules, test_contents, policy)
    findings = [asdict(observation) for observation in observations if observation.needs_attention]
    return {
        "schema": SCHEMA,
        "base_sha": _git(repo_root, "rev-parse", base_sha).strip(),
        "head_sha": _git(repo_root, "rev-parse", head_sha).strip(),
        "changed_modules": changed_modules,
        "observations": [asdict(observation) for observation in observations],
        "findings": findings,
        "passed": not findings,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find changed modules whose nearby contract tests are excluded from fast shards."
    )
    parser.add_argument("--base-sha", default="origin/main", help="Comparison base revision.")
    parser.add_argument("--head-sha", default="HEAD", help="Evaluated head revision.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository checkout to inspect (default: current checkout).",
    )
    parser.add_argument("--json", action="store_true", help="Emit fast_lane_routing.v1 JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fast-lane routing audit."""

    args = _parse_args(argv)
    try:
        repo_root = _repo_root(args.repo_root)
        report = build_report(repo_root, args.base_sha, args.head_sha)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        if args.json:
            print(json.dumps({"schema": SCHEMA, "passed": False, "error": str(exc)}, indent=2))
        else:
            print(f"fast-lane routing audit failed closed: {exc}")
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "fast-lane routing audit: "
            f"changed_modules={len(report['changed_modules'])}, "
            f"observations={len(report['observations'])}, "
            f"findings={len(report['findings'])}"
        )
        for observation in report["observations"]:
            print(
                f"[{observation['policy_state']}] {observation['source_module']} -> "
                f"{observation['test_path']} ({observation['rationale']})"
            )
            if observation["suggested_registration"]:
                print(f"  remediation: {observation['suggested_registration']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
