#!/usr/bin/env python3
"""Diagnose native-extension crashes in parallel pytest-xdist workers.

Local PR readiness (``scripts/dev/pr_ready_check.sh``) runs the suite with
pytest-xdist. On some Python/native-extension/concurrency combinations a
worker hits a segfault or abort inside a compiled extension (NumPy, SciPy,
Numba, Torch, VTK, ...). xdist then reports broken pipes and the controller
exits non-zero, so the crash masquerades as an ordinary test failure and
forces repeated manual reruns (issue #5633).

This module turns that into an actionable, fail-closed diagnostic: it
classifies the crash signature from captured pytest output, snapshots the
runtime (Python version and the loaded native extensions), and recommends a
safe concurrency setting. It never marks an incomplete run as success; the
calling shell wrapper stays fail-closed and only an explicit opt-in serial
rerun separates an environment crash from real failures.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import platform
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# Compiled Python/native extensions known to segfault under parallel pytest
# workers. Versions are fingerprinted lazily so this diagnostic runs even when
# an extension is missing.
NATIVE_EXTENSION_MODULES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "numba",
    "torch",
    "vtk",
    "pandas",
    "pyarrow",
    "duckdb",
    "matplotlib",
    "cv2",
    "shapely",
)


# Ordered crash signatures: name -> regex patterns. Order matters only for
# stable rendering; every matching class is reported.
CRASH_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "segfault",
        (
            r"Segmentation fault",
            r"Segmentation fault \(core dumped\)",
            r"SIGSEGV",
        ),
    ),
    (
        "abort",
        (
            r"Aborted",
            r"SIGABRT",
            r"Fatal Python error: Aborted",
        ),
    ),
    (
        "bus-error",
        (
            r" Bus error",
            r"SIGBUS",
        ),
    ),
    (
        "illegal-instruction",
        (
            r"Illegal instruction",
            r"SIGILL",
        ),
    ),
    (
        "xdist-worker-crash",
        (
            r"worker .*? crashed",
            r"INTERNALERROR>",
            r"Gateway Process Error",
            r"crash in worker",
        ),
    ),
    (
        "broken-pipe",
        (
            r"BrokenPipeError",
            r"broken pipe",
            r"Connection broken",
            r"ConnectionResetError",
        ),
    ),
)

# Crash classes that indicate an environment/native-extension failure rather
# than an ordinary test assertion failure.
ENVIRONMENT_CRASH_CLASSES: frozenset[str] = frozenset(
    {
        "segfault",
        "abort",
        "bus-error",
        "illegal-instruction",
        "xdist-worker-crash",
    }
)


@dataclass(frozen=True)
class NativeExtensionInfo:
    """A single detected compiled native extension and its version."""

    name: str
    version: str | None
    compiled: bool


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Minimal runtime fingerprint for crash reproduction."""

    python_version: str
    platform: str
    machine: str
    logical_cpus: int
    extensions: tuple[NativeExtensionInfo, ...] = ()

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the snapshot."""
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "machine": self.machine,
            "logical_cpus": self.logical_cpus,
            "extensions": [
                {
                    "name": ext.name,
                    "version": ext.version,
                    "compiled": ext.compiled,
                }
                for ext in self.extensions
            ],
        }


@dataclass(frozen=True)
class Diagnostic:
    """Classified crash diagnostic with a rendered fail-closed message."""

    crash_classes: tuple[str, ...]
    requested_workers: str
    dist_mode: str
    runtime: RuntimeSnapshot
    raw_excerpts: tuple[str, ...] = ()
    recommendation: str | None = None
    serialized_ok: bool | None = None

    @property
    def is_environment_crash(self) -> bool:
        """True when at least one environment/runtime crash class matched."""
        return bool(ENVIRONMENT_CRASH_CLASSES & set(self.crash_classes))

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the diagnostic."""
        return {
            "crash_classes": list(self.crash_classes),
            "is_environment_crash": self.is_environment_crash,
            "requested_workers": self.requested_workers,
            "dist_mode": self.dist_mode,
            "recommendation": self.recommendation,
            "serialized_ok": self.serialized_ok,
            "raw_excerpts": list(self.raw_excerpts),
            "runtime": self.runtime.to_dict(),
        }


def classify_crash(log_text: str) -> tuple[str, ...]:
    """Return the sorted, ordered tuple of crash class names present in ``log_text``.

    Order follows ``CRASH_SIGNATURES`` so rendering is stable.
    """
    if not log_text:
        return ()
    matches: list[str] = []
    for name, patterns in CRASH_SIGNATURES:
        if any(re.search(pattern, log_text) for pattern in patterns):
            matches.append(name)
    return tuple(matches)


def first_matching_lines(log_text: str, max_lines: int = 5) -> tuple[str, ...]:
    """Return a few representative lines that carry a crash signature.

    Helps a maintainer jump straight to the failing worker's import/test
    boundary without re-reading the whole captured log.
    """
    if not log_text:
        return ()
    compiled_patterns = [
        re.compile(pattern) for _, patterns in CRASH_SIGNATURES for pattern in patterns
    ]
    excerpts: list[str] = []
    for line in log_text.splitlines():
        if any(p.search(line) for p in compiled_patterns):
            excerpts.append(line.strip())
            if len(excerpts) >= max_lines:
                break
    return tuple(excerpts)


def _extension_version(name: str) -> str | None:
    """Best-effort version lookup for an installed distribution."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# A known compiled submodule per package, used to confirm the distribution
# actually ships a compiled extension. Top-level ``__init__.py`` is always
# pure-Python, so probing it would wrongly report "pure-python".
_COMPILED_SUBMODULE_HINT: dict[str, str] = {
    "numpy": "numpy._core._multiarray_umath",
    "scipy": "scipy._lib._ccallback_c",
    "numba": "numba._dispatcher",
    "torch": "torch._C",
    "vtk": "vtkmodules.vtkCommonCore",
    "pandas": "pandas._libs.lib",
    "pyarrow": "pyarrow.lib",
    "duckdb": "duckdb.duckdb",
    "matplotlib": "matplotlib._image",
    "cv2": "cv2.cv2",
    "shapely": "shapely.lib",
}


def _is_compiled_extension(name: str) -> bool:
    """Return True when the distribution ships a compiled extension.

    The top-level package ``__init__.py`` is always pure-Python, so we probe a
    known compiled submodule (or the package itself as a fallback) for a ``.so``
    / ``.pyd`` loadable file.
    """
    probe = _COMPILED_SUBMODULE_HINT.get(name, name)
    spec = importlib.util.find_spec(probe)
    if spec is None or spec.origin is None:
        return False
    return spec.origin.endswith((".so", ".pyd", ".pyd1", ".pyd2"))


def snapshot_runtime(
    *,
    extension_names: Iterable[str] = NATIVE_EXTENSION_MODULES,
    cpu_count: int | None = None,
) -> RuntimeSnapshot:
    """Capture a minimal runtime fingerprint for crash reproduction.

    Missing extensions are simply omitted; the snapshot never raises for an
    absent module so the diagnostic can run on a dependency-minimal host.
    """
    extensions: list[NativeExtensionInfo] = []
    for name in extension_names:
        if importlib.util.find_spec(name) is not None:
            extensions.append(
                NativeExtensionInfo(
                    name=name,
                    version=_extension_version(name),
                    compiled=_is_compiled_extension(name),
                )
            )
    return RuntimeSnapshot(
        python_version=platform.python_version(),
        platform=platform.system(),
        machine=platform.machine(),
        logical_cpus=int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)),
        extensions=tuple(extensions),
    )


def recommend_safe_concurrency(crash_classes: Iterable[str]) -> str | None:
    """Return a safe-concurrency recommendation, or None for ordinary failures."""
    classes = set(crash_classes)
    if classes & ENVIRONMENT_CRASH_CLASSES:
        return "serial"
    if "broken-pipe" in classes:
        return "reduce-or-serial"
    return None


def build_diagnostic(
    *,
    log_text: str,
    requested_workers: str = "auto",
    dist_mode: str = "load",
    runtime: RuntimeSnapshot | None = None,
    serialized_ok: bool | None = None,
) -> Diagnostic:
    """Classify ``log_text`` and assemble a fail-closed diagnostic."""
    crash_classes = classify_crash(log_text)
    runtime = runtime or snapshot_runtime()
    recommendation = recommend_safe_concurrency(crash_classes)
    return Diagnostic(
        crash_classes=crash_classes,
        requested_workers=requested_workers,
        dist_mode=dist_mode,
        runtime=runtime,
        raw_excerpts=first_matching_lines(log_text),
        recommendation=recommendation,
        serialized_ok=serialized_ok,
    )


def render_diagnostic(diag: Diagnostic) -> str:
    """Render the diagnostic as a human-readable, fail-closed message."""
    lines: list[str] = []
    lines.append("[pr_ready_check] pytest-xdist crash diagnostic (issue #5633)")
    if diag.crash_classes:
        lines.append("Detected crash signature(s): " + ", ".join(diag.crash_classes))
    else:
        lines.append("No native-extension/xdist crash signature detected in captured output.")
    lines.append("")
    lines.append("Runtime fingerprint:")
    lines.append(
        f"  python: {diag.runtime.python_version} ({diag.runtime.platform} {diag.runtime.machine})"
    )
    lines.append(f"  logical CPUs: {diag.runtime.logical_cpus}")
    if diag.runtime.extensions:
        ext_str = ", ".join(f"{ext.name} {ext.version or '?'}" for ext in diag.runtime.extensions)
        lines.append(f"  native extensions: {ext_str}")
    else:
        lines.append("  native extensions: none detected")
    lines.append(f"Requested concurrency: -n {diag.requested_workers} (dist={diag.dist_mode})")
    if diag.raw_excerpts:
        lines.append("")
        lines.append("First matching log lines:")
        for excerpt in diag.raw_excerpts:
            lines.append(f"  | {excerpt}")
    lines.append("")

    if diag.is_environment_crash:
        lines.append(
            "This is an environment/native-extension failure under parallel pytest "
            "workers, not an ordinary test assertion failure."
        )
        lines.append("It must not be read as a passed gate or as skipped validation (fail-closed).")
        if diag.serialized_ok is True:
            lines.append(
                "Serial rerun passed: the local full suite completed; the parallel "
                "crash was environment-only. Record this as a degraded/local caveat, "
                "not as benchmark evidence."
            )
        elif diag.serialized_ok is False:
            lines.append(
                "Serial rerun still failed with ordinary assertion errors: those are "
                "real failures and must be fixed before the gate can pass."
            )
        else:
            lines.append(
                "Recommended next step: separate the environment crash from real "
                "failures by re-running serially:"
            )
            if diag.recommendation == "serial":
                lines.append("  PYTEST_NUM_WORKERS=1 scripts/dev/run_tests_parallel.sh ...")
            else:
                lines.append(
                    "  PYTEST_NUM_WORKERS=1 (or a smaller worker count) "
                    "scripts/dev/run_tests_parallel.sh ..."
                )
            lines.append(
                "If serial passes, the local full suite completed (env-only crash); if "
                "it fails with assertion errors, those are real and must be fixed."
            )
    elif diag.crash_classes:
        lines.append(
            "Captured output shows a non-environment crash class; investigate the "
            "failing test rather than the runtime/concurrency combination."
        )
    else:
        lines.append(
            "No crash signature was found; if the gate reported failure it is an "
            "ordinary test/collection failure, not the parallel-worker crash tracked "
            "here."
        )
    return "\n".join(lines)


def _read_log_source(args: argparse.Namespace) -> str:
    """Read the log text from --log-file, --log-stdin, or --log-text."""
    if args.log_file:
        with open(args.log_file, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    if args.log_stdin:
        return sys.stdin.read()
    return args.log_text or ""


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the xdist crash diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--log-file", help="Path to a captured pytest log file.")
    source.add_argument(
        "--log-stdin",
        action="store_true",
        help="Read the captured pytest log from stdin.",
    )
    source.add_argument(
        "--log-text",
        help="Inline captured pytest log text.",
    )
    parser.add_argument(
        "--requested-workers",
        default="auto",
        help="Worker spec that was requested (e.g. 'auto', '8', '1').",
    )
    parser.add_argument(
        "--dist-mode",
        default="load",
        help="pytest-xdist distribution mode (load, worksteal, ...).",
    )
    parser.add_argument(
        "--serialized-ok",
        choices=("true", "false"),
        help="Whether an opt-in serial rerun passed (drives the verdict text).",
    )
    parser.add_argument(
        "--show-runtime",
        action="store_true",
        help="Always print the runtime fingerprint even when no crash is detected.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the diagnostic as JSON instead of rendered Markdown.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 on successful classification (a reporter)."""
    args = _build_parser().parse_args(argv)
    log_text = _read_log_source(args)
    serialized_ok: bool | None = None
    if args.serialized_ok == "true":
        serialized_ok = True
    elif args.serialized_ok == "false":
        serialized_ok = False

    # Only fingerprint the runtime when a crash was detected or the caller asked
    # for it; the import probes are cheap but unnecessary on a clean run.
    crash_classes = classify_crash(log_text)
    runtime = None
    if crash_classes or args.show_runtime:
        runtime = snapshot_runtime()

    diag = build_diagnostic(
        log_text=log_text,
        requested_workers=args.requested_workers,
        dist_mode=args.dist_mode,
        runtime=runtime,
        serialized_ok=serialized_ok,
    )

    if args.json:
        print(__import__("json").dumps(diag.to_dict(), indent=2))
    else:
        print(render_diagnostic(diag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
