#!/usr/bin/env python3
"""Generate and source-check the installed entry-point reference.

Usage:
    uv run python scripts/dev/generate_cli_reference.py [--check]

Reads every ``[project.scripts]`` entry from ``pyproject.toml`` in deterministic
order, runs an OS-isolated bounded ``--help`` smoke for each callable, and renders
``docs/cli_reference.md`` from ``docs/cli_reference_meta.yaml`` plus live parser
help. Entry-point imports, signatures, and parser introspection all stay inside
the bounded child. With ``--check``, exits 1 when the committed reference differs
from the deterministic render instead of writing it.

The ``--help`` smoke requires Linux Landlock ABI 4+ and seccomp on a supported
architecture. The child gets read-only source/runtime roots, a task-owned writable
temporary root, a fixed environment without inherited credential variables, and
denied network, process, thread, namespace, and scheduler escape syscalls.
Unsupported hosts fail closed; this is an OS-enforced boundary, not a Python-level
sandbox. The child retains the invoking Unix identity and host resource limits,
so this is not a general untrusted-code sandbox.
"""

from __future__ import annotations

import argparse
import ctypes
import difflib
import errno
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import yaml

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
PYPROJECT_REL = Path("pyproject.toml")
META_REL = Path("docs/cli_reference_meta.yaml")
OUTPUT_REL = Path("docs/cli_reference.md")

ALLOWED_PROFILES = ("core", "benchmark", "carla")
HELP_TIMEOUT_S = 15
HELP_COLUMNS = 80
PROBE_FLAG = "--_probe-help"
PROBE_RESULT_PREFIX = "__robot_sf_cli_reference_probe__:"

PARSER_GETTERS = ("get_parser", "_build_parser", "_configure_parser", "build_parser")

SUBCOMMAND_BRACE_RE = re.compile(r"\{([A-Za-z0-9_][A-Za-z0-9_\-, ]*)\}")

PROBE_RUNTIME_DIRS = ("home", "tmp", "cache", "config", "data", "state", "runtime")
PROBE_READ_ONLY_ENV = {
    "COLUMNS": str(HELP_COLUMNS),
    "LINES": "24",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    "PYTHONHASHSEED": "0",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONSAFEPATH": "1",
    "PYTHONUNBUFFERED": "1",
    "TERM": "dumb",
    "TZ": "UTC",
    "MPLBACKEND": "Agg",
    "MPLCONFIGDIR": "{task_root}/config/matplotlib",
    "IMAGEIO_FFMPEG_EXE": "/usr/bin/ffmpeg",
    "SDL_VIDEODRIVER": "dummy",
    "SDL_AUDIODRIVER": "dummy",
    "QT_QPA_PLATFORM": "offscreen",
    "PYGAME_HIDE_SUPPORT_PROMPT": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "WANDB_DISABLED": "true",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "GIT_CONFIG_NOSYSTEM": "1",
}

LANDLOCK_CREATE_RULESET_SYSCALL = 444
LANDLOCK_ADD_RULE_SYSCALL = 445
LANDLOCK_RESTRICT_SELF_SYSCALL = 446
LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_MINIMUM_ABI = 4
LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
LANDLOCK_ACCESS_FS_REFER = 1 << 13
LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14
LANDLOCK_ACCESS_FS_ALL = (1 << 15) - 1
LANDLOCK_ACCESS_NET_BIND_TCP = 1 << 0
LANDLOCK_ACCESS_NET_CONNECT_TCP = 1 << 1
LANDLOCK_ACCESS_NET_ALL = LANDLOCK_ACCESS_NET_BIND_TCP | LANDLOCK_ACCESS_NET_CONNECT_TCP
LANDLOCK_ACCESS_FS_READ_EXECUTE = (
    LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR
)

PR_SET_NO_NEW_PRIVS = 38
SECCOMP_SYSCALL = 317
SECCOMP_SET_MODE_FILTER = 1
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
BPF_LD_W_ABS = 0x20
BPF_JMP_JEQ_K = 0x15
BPF_RET_K = 0x06
SECCOMP_ARCH_X86_64 = 0xC000003E
SECCOMP_ARCH_AARCH64 = 0xC00000B7

_BLOCKED_SYSCALLS = {
    # Numbers from Linux arch/x86/entry/syscalls/syscall_64.tbl. Keep this
    # deny list limited to escape and externally visible side-effect paths;
    # Landlock handles filesystem policy.
    "x86_64": (
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        56,
        57,
        58,
        59,
        62,
        141,
        101,
        203,
        206,
        207,
        208,
        209,
        210,
        142,
        144,
        155,
        161,
        163,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        175,
        176,
        179,
        200,
        234,
        246,
        248,
        249,
        250,
        272,
        251,
        256,
        279,
        288,
        298,
        299,
        300,
        301,
        302,
        303,
        304,
        307,
        308,
        310,
        311,
        313,
        314,
        320,
        321,
        322,
        323,
        317,
        324,
        424,
        425,
        426,
        427,
        428,
        429,
        430,
        431,
        432,
        433,
        434,
        435,
        438,
        440,
        448,
    ),
    # Numbers from Linux include/uapi/asm-generic/unistd.h, used by aarch64.
    "aarch64": (
        0,
        2,
        3,
        39,
        40,
        41,
        51,
        58,
        60,
        89,
        95,
        97,
        104,
        105,
        106,
        107,
        117,
        118,
        119,
        129,
        130,
        131,
        122,
        140,
        142,
        161,
        162,
        198,
        199,
        200,
        201,
        202,
        203,
        204,
        205,
        206,
        207,
        208,
        209,
        210,
        211,
        212,
        217,
        218,
        219,
        220,
        221,
        224,
        225,
        241,
        240,
        242,
        243,
        260,
        261,
        262,
        263,
        264,
        265,
        266,
        267,
        268,
        269,
        270,
        271,
        272,
        273,
        274,
        280,
        281,
        282,
        277,
        279,
        425,
        426,
        427,
        424,
        434,
        435,
        438,
        440,
        448,
    ),
}

_SECCOMP_ARCHITECTURES = {
    "x86_64": SECCOMP_ARCH_X86_64,
    "amd64": SECCOMP_ARCH_X86_64,
    "aarch64": SECCOMP_ARCH_AARCH64,
    "arm64": SECCOMP_ARCH_AARCH64,
}


class CliReferenceError(ValueError):
    """Raised when metadata or project script inventory is malformed."""


class ProbeIsolationError(RuntimeError):
    """Raised when the OS cannot enforce the CLI probe boundary."""


class _LandlockRulesetAttr(ctypes.Structure):
    """Stable prefix of Linux's extensible Landlock ruleset structure."""

    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
    ]


class _LandlockPathBeneathAttr(ctypes.Structure):
    """Packed path rule structure required by the Landlock user-space API."""

    _pack_ = 1
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


class _SockFilter(ctypes.Structure):
    """One classic-BPF instruction for the child seccomp policy."""

    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    """Classic-BPF program descriptor accepted by the seccomp syscall."""

    _fields_ = [
        ("length", ctypes.c_uint16),
        ("filter", ctypes.POINTER(_SockFilter)),
    ]


@dataclass
class ProbeResult:
    """OS-isolated ``--help`` outcome for one console script."""

    script: str
    spec: str
    import_ok: bool = False
    import_error: str = ""
    help_ok: bool = False
    help_error: str = ""
    isolation_error: str = ""
    help_text: str = ""
    synopsis: str = ""
    subcommands: list[str] = field(default_factory=list)
    subcommand_help: dict[str, str] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Return the compact help status label."""
        if self.help_ok:
            return "available"
        return f"unavailable: {self.help_error or self.import_error or self.isolation_error or 'unknown'}"


def load_project_scripts(pyproject_path: Path) -> dict[str, str]:
    """Load ``[project.scripts]`` entries in deterministic sorted order.

    Args:
        pyproject_path: Path to ``pyproject.toml``.

    Returns:
        Mapping of script name to ``module:attr`` spec, sorted by script name.

    Raises:
        CliReferenceError: When the file or the scripts table is missing.
    """
    if not pyproject_path.is_file():
        raise CliReferenceError(f"pyproject.toml not found: {pyproject_path}")
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    try:
        raw = data["project"]["scripts"]
    except KeyError as exc:
        raise CliReferenceError(f"[project.scripts] missing in {pyproject_path}") from exc
    if not isinstance(raw, dict) or not raw:
        raise CliReferenceError(f"[project.scripts] must be a non-empty mapping: {pyproject_path}")
    scripts: dict[str, str] = {}
    for name in sorted(raw):
        spec = raw[name]
        if not isinstance(spec, str) or ":" not in spec or not spec.strip():
            raise CliReferenceError(f"script {name!r}: spec must look like 'module:attr'")
        scripts[name] = spec.strip()
    return scripts


def load_metadata(meta_path: Path) -> dict:
    """Load the compact CLI reference metadata file.

    Args:
        meta_path: Path to ``docs/cli_reference_meta.yaml``.

    Returns:
        Raw parsed YAML mapping.

    Raises:
        CliReferenceError: When the file is missing or malformed.
    """
    if not meta_path.is_file():
        raise CliReferenceError(f"metadata file not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise CliReferenceError(f"metadata YAML parse error: {exc}") from exc
    if not isinstance(raw, dict):
        raise CliReferenceError("metadata root must be a mapping")
    if raw.get("version") != 1:
        raise CliReferenceError(f"unsupported metadata version: {raw.get('version')!r}")
    entries = raw.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise CliReferenceError("'entries' must be a non-empty mapping")
    return raw


def _require_non_empty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CliReferenceError(f"{label} must be a non-empty string")
    return " ".join(value.split())


def _normalize_subcommands(
    name: str, raw_subs: object, repo_root: Path, errors: list[str]
) -> dict[str, str] | None:
    """Normalize one entry's subcommand guides or record errors."""
    if not isinstance(raw_subs, dict):
        errors.append(f"entry {name!r}: 'subcommands' must be a mapping")
        return None
    subcommands: dict[str, str] = {}
    ok = True
    for sub, sub_guide in raw_subs.items():
        if isinstance(sub_guide, dict):
            sub_guide = sub_guide.get("guide")
        if not isinstance(sub_guide, str) or not sub_guide.strip():
            errors.append(f"entry {name!r} subcommand {sub!r}: missing 'guide'")
            ok = False
            continue
        cleaned = sub_guide.strip()
        if not (repo_root / cleaned).is_file():
            errors.append(f"entry {name!r} subcommand {sub!r}: guide missing: {cleaned}")
            ok = False
            continue
        subcommands[sub] = cleaned
    return subcommands if ok else None


def _normalize_single_entry(
    name: str, entry: object, scripts: dict[str, str], repo_root: Path, errors: list[str]
) -> dict | None:
    """Normalize one metadata entry or record fail-closed errors."""
    if name not in scripts:
        errors.append(f"stale documented entry not in [project.scripts]: {name!r}")
        return None
    if not isinstance(entry, dict):
        errors.append(f"entry {name!r}: must be a mapping")
        return None
    try:
        purpose = _require_non_empty(entry.get("purpose"), f"entry {name!r} 'purpose'")
        profile = _require_non_empty(entry.get("profile"), f"entry {name!r} 'profile'")
        availability = _require_non_empty(
            entry.get("availability"), f"entry {name!r} 'availability'"
        )
        guide = _require_non_empty(entry.get("guide"), f"entry {name!r} 'guide'")
    except CliReferenceError as exc:
        errors.append(str(exc))
        return None
    if profile not in ALLOWED_PROFILES:
        errors.append(
            f"entry {name!r}: unknown profile {profile!r} (allowed: {', '.join(ALLOWED_PROFILES)})"
        )
        return None
    if not (repo_root / guide).is_file():
        errors.append(f"entry {name!r}: guide does not exist: {guide}")
        return None
    subcommands = _normalize_subcommands(
        name, entry.get("subcommands", {}) or {}, repo_root, errors
    )
    if subcommands is None:
        return None
    return {
        "purpose": purpose,
        "profile": profile,
        "availability": availability,
        "guide": guide,
        "subcommands": subcommands,
    }


def validate_metadata_entries(
    scripts: dict[str, str], raw: dict, repo_root: Path
) -> tuple[dict[str, dict], list[str]]:
    """Validate metadata against the declared scripts and existing guides.

    Args:
        scripts: Sorted script inventory from ``pyproject.toml``.
        raw: Raw metadata mapping from :func:`load_metadata`.
        repo_root: Repository root for guide existence checks.

    Returns:
        Tuple of normalized entries and fail-closed error strings.
    """
    errors: list[str] = []
    entries: dict[str, dict] = {}
    raw_entries = raw.get("entries", {})
    for name, entry in raw_entries.items():
        normalized = _normalize_single_entry(name, entry, scripts, repo_root, errors)
        if normalized is not None:
            entries[name] = normalized
    for name in sorted(scripts):
        if name not in raw_entries:
            errors.append(f"missing documentation owner for declared script: {name!r}")
    return entries, errors


def _probe_machine() -> str:
    """Normalize the Linux machine name used by the syscall policy."""
    machine = platform.machine().lower()
    aliases = {"amd64": "x86_64", "arm64": "aarch64"}
    return aliases.get(machine, machine)


def _probe_libc() -> tuple[ctypes.CDLL, str]:
    """Load libc only on Linux architectures with a known syscall policy."""
    if sys.platform != "linux":
        raise ProbeIsolationError(
            "secure CLI probe isolation is unavailable: Linux Landlock and seccomp are required"
        )
    machine = _probe_machine()
    if machine not in _BLOCKED_SYSCALLS or machine not in _SECCOMP_ARCHITECTURES:
        raise ProbeIsolationError(
            f"secure CLI probe isolation is unavailable: unsupported Linux architecture {machine!r}"
        )
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as exc:
        raise ProbeIsolationError(f"secure CLI probe isolation cannot load libc: {exc}") from exc
    libc.syscall.restype = ctypes.c_long
    libc.prctl.restype = ctypes.c_int
    return libc, machine


def _raise_probe_errno(action: str) -> None:
    """Raise a bounded, fail-closed diagnostic for a failed isolation syscall."""
    error_number = ctypes.get_errno()
    detail = os.strerror(error_number) if error_number else "unknown error"
    raise ProbeIsolationError(f"secure CLI probe isolation {action} failed: {detail}")


def _probe_landlock_abi(libc: ctypes.CDLL) -> None:
    """Require the Landlock ABI features used by the filesystem policy."""
    ctypes.set_errno(0)
    result = libc.syscall(
        LANDLOCK_CREATE_RULESET_SYSCALL,
        ctypes.c_void_p(),
        ctypes.c_size_t(0),
        ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION),
    )
    if result < 0:
        _raise_probe_errno("Landlock capability detection")
    abi = int(result)
    if abi < LANDLOCK_MINIMUM_ABI:
        raise ProbeIsolationError(
            "secure CLI probe isolation is unavailable: Linux Landlock ABI "
            f"{abi} is older than required ABI {LANDLOCK_MINIMUM_ABI}"
        )


def _close_probe_fds() -> None:
    """Close inherited descriptors so pre-opened handles cannot bypass Landlock."""
    try:
        descriptor_names = os.listdir("/proc/self/fd")
    except OSError as exc:
        raise ProbeIsolationError(
            f"secure CLI probe isolation cannot verify inherited file descriptors: {exc}"
        ) from exc
    for descriptor_name in descriptor_names:
        try:
            descriptor = int(descriptor_name)
        except ValueError:
            continue
        if descriptor <= 2:
            continue
        try:
            os.close(descriptor)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise ProbeIsolationError(
                    "secure CLI probe isolation could not close inherited file descriptor "
                    f"{descriptor}: {exc}"
                ) from exc


def _add_probe_landlock_path_rule(
    libc: ctypes.CDLL, ruleset_fd: int, path: Path, allowed_access: int
) -> None:
    """Add a Landlock rule for a trusted read root or the task-owned root."""
    try:
        path_fd = os.open(str(path), os.O_PATH | os.O_CLOEXEC)
    except (AttributeError, OSError) as exc:
        raise ProbeIsolationError(
            f"secure CLI probe isolation cannot open policy path {path}: {exc}"
        ) from exc
    try:
        rule = _LandlockPathBeneathAttr(allowed_access=allowed_access, parent_fd=path_fd)
        ctypes.set_errno(0)
        result = libc.syscall(
            LANDLOCK_ADD_RULE_SYSCALL,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(rule),
            ctypes.c_uint(0),
        )
        if result < 0:
            _raise_probe_errno(f"adding Landlock path rule for {path}")
    finally:
        try:
            os.close(path_fd)
        except OSError as exc:
            raise ProbeIsolationError(
                f"secure CLI probe isolation could not close policy descriptor for {path}: {exc}"
            ) from exc


def _required_probe_path(path: Path, label: str) -> Path:
    """Resolve a required sandbox path without accepting a missing or non-directory root."""
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ProbeIsolationError(
            f"secure CLI probe isolation cannot resolve {label}: {exc}"
        ) from exc
    if not resolved.is_dir():
        raise ProbeIsolationError(f"secure CLI probe isolation requires {label} to be a directory")
    return resolved


def _probe_read_roots(repo_root: Path) -> tuple[Path, ...]:
    """Return the narrow read/execute roots needed by the already-started interpreter."""
    candidates: list[Path] = [
        repo_root,
        Path(sys.executable).resolve().parent,
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(sys.exec_prefix),
        Path(sys.base_exec_prefix),
        Path("/lib"),
        Path("/lib64"),
        Path("/usr/lib"),
        Path("/usr/lib64"),
        Path("/usr/local/lib"),
        Path("/dev"),
    ]
    for value in (
        sysconfig.get_path("stdlib"),
        sysconfig.get_path("platstdlib"),
        sysconfig.get_path("purelib"),
        sysconfig.get_path("platlib"),
        sysconfig.get_config_var("LIBDIR"),
    ):
        if value:
            candidates.append(Path(value))
    for value in sys.path:
        if value and Path(value).is_absolute():
            candidates.append(Path(value))

    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved in seen:
            continue
        if not resolved.is_dir() and not resolved.is_file():
            continue
        seen.add(resolved)
        roots.append(resolved)
    return tuple(roots)


def _install_probe_landlock(libc: ctypes.CDLL, repo_root: Path, task_root: Path) -> None:
    """Allow source/runtime reads and task-root writes while denying other filesystem access."""
    repo_root = _required_probe_path(repo_root, "repository root")
    task_root = _required_probe_path(task_root, "task root")
    read_roots = _probe_read_roots(repo_root)
    if repo_root not in read_roots:
        raise ProbeIsolationError("secure CLI probe isolation could not authorize repository reads")

    ruleset = _LandlockRulesetAttr(
        handled_access_fs=LANDLOCK_ACCESS_FS_ALL,
        handled_access_net=LANDLOCK_ACCESS_NET_ALL,
    )
    ctypes.set_errno(0)
    ruleset_fd = libc.syscall(
        LANDLOCK_CREATE_RULESET_SYSCALL,
        ctypes.byref(ruleset),
        ctypes.sizeof(ruleset),
        ctypes.c_uint(0),
    )
    if ruleset_fd < 0:
        _raise_probe_errno("creating Landlock ruleset")
    try:
        for read_root in read_roots:
            allowed_access = (
                LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_FILE
                if read_root == Path("/dev")
                else LANDLOCK_ACCESS_FS_READ_EXECUTE
            )
            _add_probe_landlock_path_rule(libc, int(ruleset_fd), read_root, allowed_access)
        _add_probe_landlock_path_rule(libc, int(ruleset_fd), task_root, LANDLOCK_ACCESS_FS_ALL)
        ctypes.set_errno(0)
        result = libc.syscall(
            LANDLOCK_RESTRICT_SELF_SYSCALL,
            int(ruleset_fd),
            ctypes.c_uint(0),
        )
        if result < 0:
            _raise_probe_errno("restricting the child process")
    finally:
        try:
            os.close(int(ruleset_fd))
        except OSError as exc:
            raise ProbeIsolationError(
                f"secure CLI probe isolation could not close Landlock ruleset: {exc}"
            ) from exc


def _probe_seccomp_instructions(machine: str) -> list[_SockFilter]:
    """Build a deny filter for network, process, namespace, and scheduler syscalls."""
    instructions = [
        _SockFilter(BPF_LD_W_ABS, 0, 0, 4),
        _SockFilter(BPF_JMP_JEQ_K, 1, 0, _SECCOMP_ARCHITECTURES[machine]),
        _SockFilter(BPF_RET_K, 0, 0, SECCOMP_RET_KILL_PROCESS),
        _SockFilter(BPF_LD_W_ABS, 0, 0, 0),
    ]
    for syscall_number in _BLOCKED_SYSCALLS[machine]:
        instructions.extend(
            (
                _SockFilter(BPF_JMP_JEQ_K, 0, 1, syscall_number),
                _SockFilter(BPF_RET_K, 0, 0, SECCOMP_RET_ERRNO | errno.EPERM),
            )
        )
    instructions.append(_SockFilter(BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW))
    return instructions


def _install_probe_seccomp(libc: ctypes.CDLL, machine: str) -> None:
    """Deny target-created network sockets, processes, threads, and escape syscalls."""
    instructions = _probe_seccomp_instructions(machine)
    instruction_array = (_SockFilter * len(instructions))(*instructions)
    program = _SockFprog(
        length=len(instruction_array),
        filter=ctypes.cast(instruction_array, ctypes.POINTER(_SockFilter)),
    )
    ctypes.set_errno(0)
    result = libc.syscall(
        SECCOMP_SYSCALL,
        ctypes.c_uint(SECCOMP_SET_MODE_FILTER),
        ctypes.c_uint(0),
        ctypes.byref(program),
    )
    if result < 0:
        _raise_probe_errno("installing seccomp policy")


def _install_probe_isolation(repo_root: Path, task_root: Path) -> None:
    """Install all OS restrictions before importing or introspecting the target module."""
    libc, machine = _probe_libc()
    try:
        resolved_task_root = _required_probe_path(task_root, "task root")
        os.chdir(resolved_task_root)
    except OSError as exc:
        raise ProbeIsolationError(
            f"secure CLI probe isolation cannot enter task root: {exc}"
        ) from exc
    _probe_landlock_abi(libc)
    _close_probe_fds()
    ctypes.set_errno(0)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        _raise_probe_errno("setting no-new-privileges")
    _install_probe_landlock(libc, repo_root, resolved_task_root)
    _install_probe_seccomp(libc, machine)


def _parser_subcommands(parser: object) -> list[str]:
    """Extract sorted subcommand names from a live argparse parser in the child."""
    subcommands: set[str] = set()
    actions = getattr(parser, "_actions", [])
    for action in actions:
        choices = getattr(action, "choices", None)
        action_type = type(action).__name__
        if action_type == "_SubParsersAction" and isinstance(choices, dict):
            for key in choices:
                if isinstance(key, str) and key.strip():
                    subcommands.add(key.strip())
    return sorted(subcommands)


def _parser_subcommand_help(parser: object) -> dict[str, str]:
    """Extract one-line subcommand help from a live parser in the child."""
    helps: dict[str, str] = {}
    for action in getattr(parser, "_actions", []):
        if type(action).__name__ != "_SubParsersAction":
            continue
        choices = getattr(action, "choices", {}) or {}
        for key, subparser in choices.items():
            if not isinstance(key, str):
                continue
            text = str(getattr(subparser, "description", "") or getattr(action, "help", ""))
            help_text = ""
            for sub_action in getattr(action, "_choices_actions", []):
                if getattr(sub_action, "dest", None) == key or sub_action.metavar == key:
                    help_text = str(getattr(sub_action, "help", "") or "")
                    break
            candidate = help_text or text
            helps[key] = " ".join(candidate.split())[:200] or "-"
    return helps


def _child_parser_info(module: object) -> tuple[str, list[str], dict[str, str]]:
    """Return parser metadata; callers run this only inside the bounded child."""
    for getter in PARSER_GETTERS:
        try:
            factory = getattr(module, getter, None)
            if not callable(factory):
                continue
            parser = factory()
            description = str(getattr(parser, "description", "") or "").strip()
            description = " ".join(description.split())
            return description, _parser_subcommands(parser), _parser_subcommand_help(parser)
        except Exception:  # noqa: BLE001 - arbitrary parser factories stay child-bounded
            continue
    return "", [], {}


def extract_subcommands_from_help(help_text: str) -> list[str]:
    """Extract candidate subcommand names from ``--help`` usage braces."""
    found: set[str] = set()
    for match in SUBCOMMAND_BRACE_RE.finditer(help_text):
        for token in match.group(1).split(","):
            token = token.strip()
            if token and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]*", token):
                found.add(token)
    return sorted(found)


def _skip_usage_block(lines: list[str]) -> int:
    """Return the index just after the usage block and blank lines."""
    idx = 0
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("usage:"):
            idx = i + 1
            while idx < len(lines) and lines[idx].strip():
                if not lines[idx].startswith((" ", "\t")) and idx > i + 2:
                    break
                idx += 1
            break
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return idx


def _collect_description_paragraph(lines: list[str], idx: int) -> tuple[list[str], int]:
    """Collect wrapped description lines until a blank line or section header."""
    paragraph: list[str] = []
    section_headers = ("positional arguments:", "options:", "optional arguments:", "arguments:")
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped or stripped.lower() in section_headers:
            break
        paragraph.append(stripped)
        idx += 1
    return paragraph, idx


def extract_synopsis(help_text: str, fallback_description: str = "") -> str:
    """Extract the one-line help synopsis after the usage block."""
    lines = help_text.splitlines()
    idx = _skip_usage_block(lines)
    paragraph, _ = _collect_description_paragraph(lines, idx)
    if paragraph:
        return " ".join(" ".join(paragraph).split())[:500]
    if fallback_description:
        return fallback_description[:500]
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.lower().startswith("usage:"):
            return " ".join(stripped.split())[:500]
    return ""


def _child_exit_code(value: object) -> int:
    """Normalize a console callable return value to a process exit code."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return 1


def _child_import_target(module_name: str, attr: str) -> tuple[object | None, object | None, str]:
    """Import one entry point in the child and return a fail-closed error string."""
    import importlib

    try:
        module = importlib.import_module(module_name)
        return module, getattr(module, attr), ""
    except Exception as exc:  # noqa: BLE001 - arbitrary import failures stay child-bounded
        return None, None, f"{type(exc).__name__}: {exc}".strip()[:300]


def _child_takes_argv(target: object) -> bool:
    """Determine the callable convention inside the bounded child."""
    import inspect

    try:
        return len(inspect.signature(target).parameters) > 0
    except Exception:  # noqa: BLE001 - custom signatures stay child-bounded
        return True


def _child_invoke(
    script: str, target: object, stdout: StringIO, stderr: StringIO
) -> tuple[object | None, str]:
    """Invoke a help callable in the child and return its value or error."""
    try:
        if _child_takes_argv(target):
            sys.argv = [script]
            return target(["--help"]), ""
        sys.argv = [script, "--help"]
        return target(), ""
    except SystemExit as exc:
        return exc.code, ""
    except Exception as exc:  # noqa: BLE001 - arbitrary help failures stay fail-closed
        return None, f"{type(exc).__name__}: {exc}".strip()[:300]


def _child_help_payload(
    script: str,
    module: object,
    target: object,
    stdout: StringIO,
    stderr: StringIO,
) -> dict[str, object]:
    """Build the successful or fail-closed help payload inside the child."""
    returned, invocation_error = _child_invoke(script, target, stdout, stderr)
    if invocation_error:
        return {"help_error": invocation_error}
    exit_code = _child_exit_code(returned)
    if exit_code != 0:
        detail = ""
        if not isinstance(returned, (type(None), int, bool)):
            detail = str(returned).strip()
        if not detail:
            output = (stderr.getvalue() or stdout.getvalue()).strip().splitlines()
            detail = " ".join(output[0].split()) if output else "exit nonzero"
        return {"help_error": f"--help exit {exit_code}: {detail}"[:300]}

    help_text = stdout.getvalue()
    normalized = "\n".join(line.rstrip() for line in help_text.splitlines()).strip() + "\n"
    if not normalized.strip():
        return {"help_error": "--help produced no output"}
    description, subcommands, subcommand_help = _child_parser_info(module)
    return {
        "help_ok": True,
        "help_text": normalized,
        "description": description,
        "subcommands": subcommands,
        "subcommand_help": subcommand_help,
    }


def _run_probe_child(
    script: str, module_name: str, attr: str, repo_root: Path, task_root: Path
) -> int:
    """Import, introspect, and invoke one entry point inside the OS-bounded child."""
    import contextlib

    payload: dict[str, object] = {
        "import_ok": False,
        "import_error": "",
        "isolation_error": "",
        "help_ok": False,
        "help_error": "",
        "help_text": "",
        "description": "",
        "subcommands": [],
        "subcommand_help": {},
    }
    stdout = StringIO()
    stderr = StringIO()
    try:
        _install_probe_isolation(repo_root, task_root)
    except ProbeIsolationError as exc:
        payload["isolation_error"] = str(exc)[:300]
    else:
        # Keep libraries that open ``os.devnull`` during import inside the
        # task-owned root. This compatibility redirect is not the security
        # boundary; Landlock still denies every other filesystem write.
        os.devnull = str(task_root / "devnull")
        sys.path.insert(0, str(repo_root))
        # Keep import-time argv conservative. The exact callable invocation argv
        # is selected after the bounded child has inspected the target signature.
        sys.argv = [script]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            module, target, import_error = _child_import_target(module_name, attr)
            if import_error:
                payload["import_error"] = import_error
            else:
                payload["import_ok"] = True
                payload.update(_child_help_payload(script, module, target, stdout, stderr))

    sys.__stdout__.write(PROBE_RESULT_PREFIX + json.dumps(payload, sort_keys=True) + "\n")
    return 0


def _run_probe_subprocess(
    command: list[str], task_root: Path, env: dict[str, str], timeout_s: int
) -> tuple[subprocess.CompletedProcess[str] | None, str]:
    """Run one bounded child process and return a launch/timeout error separately."""
    try:
        return (
            subprocess.run(
                command,
                cwd=str(task_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                close_fds=True,
                stdin=subprocess.DEVNULL,
            ),
            "",
        )
    except subprocess.TimeoutExpired:
        return None, f"--help timed out after {timeout_s}s"
    except (OSError, ValueError) as exc:  # subprocess launch failures only
        return None, f"--help launch failed: {exc}".strip()[:300]


def _decode_probe_payload(
    completed: subprocess.CompletedProcess[str],
) -> tuple[dict[str, object] | None, str]:
    """Decode the structured result emitted by the bounded child."""
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        first = " ".join(detail[0].split()) if detail else "exit nonzero"
        return None, f"--help exit {completed.returncode}: {first}"[:300]
    payload_lines = [
        line[len(PROBE_RESULT_PREFIX) :]
        for line in (completed.stdout or "").splitlines()
        if line.startswith(PROBE_RESULT_PREFIX)
    ]
    if len(payload_lines) != 1:
        return None, "--help probe returned malformed child output"
    try:
        payload = json.loads(payload_lines[0])
    except (json.JSONDecodeError, TypeError) as exc:
        return None, f"--help probe returned invalid child payload: {exc}"[:300]
    if not isinstance(payload, dict):
        return None, "--help probe returned a non-object child payload"
    return payload, ""


def _apply_probe_payload(result: ProbeResult, payload: dict[str, object]) -> None:
    """Apply a decoded child result to its parent-side probe record."""
    isolation_error = payload.get("isolation_error")
    if isolation_error:
        result.isolation_error = str(isolation_error)[:300]
        return
    if not payload.get("import_ok"):
        result.import_error = str(payload.get("import_error") or "unknown import failure")[:300]
        return
    result.import_ok = True
    if not payload.get("help_ok"):
        result.help_error = str(payload.get("help_error") or "unknown help failure")[:300]
        return
    help_text = payload.get("help_text")
    if not isinstance(help_text, str) or not help_text.strip():
        result.help_error = "--help probe returned empty child help text"
        return
    result.help_ok = True
    result.help_text = help_text
    live_description = str(payload.get("description") or "")
    result.synopsis = extract_synopsis(help_text, live_description)
    raw_subcommands = payload.get("subcommands")
    live_subs = (
        sorted({value for value in raw_subcommands if isinstance(value, str) and value.strip()})
        if isinstance(raw_subcommands, list)
        else []
    )
    # Prefer parser subcommands when available; help-text braces also match
    # option choices, so text extraction is only a fallback.
    result.subcommands = live_subs or extract_subcommands_from_help(help_text)
    raw_subcommand_help = payload.get("subcommand_help")
    if isinstance(raw_subcommand_help, dict):
        result.subcommand_help = {
            key: value
            for key, value in raw_subcommand_help.items()
            if isinstance(key, str) and isinstance(value, str)
        }


def probe_help(
    script: str,
    spec: str,
    repo_root: Path,
    timeout_s: int = HELP_TIMEOUT_S,
) -> ProbeResult:
    """Run an OS-isolated, bounded ``--help`` smoke for one console script.

    The child owns entry-point imports, signature detection, parser construction,
    and subcommand introspection. Linux Landlock and seccomp enforce read-only
    source/runtime roots, a task-owned writable root, no inherited credential
    environment or file descriptors, and denied network/process/thread escape
    syscalls before target import. Hosts without that OS support fail closed;
    this is not a Python-level sandbox.

    Args:
        script: Console script name (used as ``sys.argv[0]`` for stable help).
        spec: ``module:attr`` callable spec from ``pyproject.toml``.
        repo_root: Repository root (read-only import and source root).
        timeout_s: Subprocess timeout in seconds.

    Returns:
        Populated :class:`ProbeResult` with normalized help text, synopsis,
        and subcommands when available.
    """
    result = ProbeResult(script=script, spec=spec)
    if ":" not in spec:
        result.import_error = f"malformed spec {spec!r}"
        return result
    module_name, attr = spec.split(":", 1)
    module_name, attr = module_name.strip(), attr.strip()
    try:
        resolved_repo_root = repo_root.resolve(strict=True)
    except OSError as exc:
        result.isolation_error = (
            f"secure CLI probe isolation cannot resolve repository root: {exc}"[:300]
        )
        return result
    if not resolved_repo_root.is_dir():
        result.isolation_error = (
            "secure CLI probe isolation requires repository root to be a directory"
        )
        return result

    try:
        with tempfile.TemporaryDirectory(
            prefix="cli-reference-probe-", dir="/tmp"
        ) as temporary_root:
            task_root = Path(temporary_root)
            for relative_path in PROBE_RUNTIME_DIRS:
                (task_root / relative_path).mkdir(parents=True, exist_ok=True)
            (task_root / "config" / "matplotlib").mkdir(parents=True, exist_ok=True)
            task_null = task_root / "devnull"
            task_null.touch()
            env = {
                key: value.format(task_root=str(task_root))
                for key, value in PROBE_READ_ONLY_ENV.items()
            }
            env.update(
                {
                    "HOME": str(task_root / "home"),
                    "TMPDIR": str(task_root / "tmp"),
                    "TMP": str(task_root / "tmp"),
                    "TEMP": str(task_root / "tmp"),
                    "XDG_CACHE_HOME": str(task_root / "cache"),
                    "XDG_CONFIG_HOME": str(task_root / "config"),
                    "XDG_CONFIG_DIRS": str(task_root / "config"),
                    "XDG_DATA_HOME": str(task_root / "data"),
                    "XDG_DATA_DIRS": str(task_root / "data"),
                    "XDG_STATE_HOME": str(task_root / "state"),
                    "XDG_RUNTIME_DIR": str(task_root / "runtime"),
                }
            )
            command = [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                PROBE_FLAG,
                "--repo-root",
                str(resolved_repo_root),
                "--task-root",
                str(task_root),
                "--script",
                script,
                "--module",
                module_name,
                "--attr",
                attr,
            ]
            completed, process_error = _run_probe_subprocess(command, task_root, env, timeout_s)
            if process_error:
                result.help_error = process_error
                return result
            if completed is None:
                result.help_error = "--help probe did not return a child result"
                return result
            payload, payload_error = _decode_probe_payload(completed)
            if payload_error:
                result.help_error = payload_error
                return result
            if payload is None:
                result.help_error = "--help probe did not return a child payload"
                return result
            _apply_probe_payload(result, payload)
    except OSError as exc:
        result.help_ok = False
        result.isolation_error = f"secure CLI probe isolation cannot create task root: {exc}"[:300]
    return result


def _check_single_script(
    name: str, entries: dict[str, dict], probes: dict[str, ProbeResult]
) -> list[str]:
    """Validate one script's owner, import, help, and subcommand coverage."""
    if name not in entries:
        return [f"missing documentation owner for declared script: {name!r}"]
    probe = probes.get(name)
    if probe is None:
        return [f"missing --help probe for declared script: {name!r}"]
    profile = entries[name]["profile"]
    if not probe.import_ok:
        if profile == "carla":
            return []
        detail = probe.import_error or probe.isolation_error or "unknown import failure"
        return [f"callable cannot import for {name!r}: {detail}"]
    if not probe.help_ok:
        if profile == "carla":
            return []
        detail = probe.help_error or probe.isolation_error or "unknown help failure"
        return [f"--help failed for {name!r}: {detail}"]
    if name != "robot-sf":
        return []
    want = set(probe.subcommands)
    have = set(entries[name].get("subcommands", {}))
    sub_errors = [
        f"missing documentation owner for subcommand: 'robot-sf {s}'" for s in sorted(want - have)
    ]
    sub_errors += [
        f"stale documented subcommand not in live parser: 'robot-sf {s}'"
        for s in sorted(have - want)
    ]
    return sub_errors


def check_entry_points(
    scripts: dict[str, str],
    entries: dict[str, dict],
    probes: dict[str, ProbeResult],
) -> list[str]:
    """Fail-closed validation over scripts, metadata owners, and help smokes.

    Args:
        scripts: Sorted script inventory from ``pyproject.toml``.
        entries: Validated metadata entries keyed by script name.
        probes: Help probe results keyed by script name.

    Returns:
        Error strings; an empty list means the reference is complete.
    """
    errors: list[str] = []
    for name in sorted(scripts):
        errors.extend(_check_single_script(name, entries, probes))
    for name in sorted(entries):
        if name not in scripts:
            errors.append(f"stale documented entry not in [project.scripts]: {name!r}")
    return errors


def _md_link(guide: str) -> str:
    """Return a docs-relative markdown link for a repo-relative guide path."""
    target = guide
    if target.startswith("docs/"):
        target = target[len("docs/") :]
    return f"[{target}]({target})"


def render_markdown(
    scripts: dict[str, str],
    entries: dict[str, dict],
    probes: dict[str, ProbeResult],
) -> str:
    """Render the byte-stable CLI reference markdown.

    Args:
        scripts: Sorted script inventory from ``pyproject.toml``.
        entries: Validated metadata entries keyed by script name.
        probes: Help probe results keyed by script name.

    Returns:
        Deterministic markdown text ending with exactly one newline.
    """
    lines = [
        "# CLI Reference (Installed Entry Points)",
        "",
        "Plain-language summary: this is the complete list of installed commands,",
        "generated from `pyproject.toml` and live `--help` so the docs cannot drift",
        "from the packaged entry points. See the [Glossary](glossary.md) for",
        "acronyms and project terms.",
        "",
        "> Generated file — do not edit by hand. Regenerate with",
        "> `uv run python scripts/dev/generate_cli_reference.py`.",
        "> Sources: `pyproject.toml [project.scripts]` plus",
        "> `docs/cli_reference_meta.yaml` plus live `--help` under an OS-enforced",
        "> local sandbox (Linux Landlock + seccomp; unsupported hosts fail closed).",
        "",
        "## Overview",
        "",
        "| Command | Purpose | Profile | Availability | Guide | Help |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in sorted(scripts):
        meta = entries.get(name, {})
        probe = probes.get(name)
        purpose = meta.get("purpose", "-")
        profile = meta.get("profile", "-")
        availability = meta.get("availability", "-")
        guide = meta.get("guide", "-")
        guide_cell = _md_link(guide) if guide != "-" else "-"
        help_cell = probe.status if probe is not None else "unknown"
        # Keep table cells single-line for byte-stability.
        purpose_cell = " ".join(str(purpose).split())
        availability_cell = " ".join(str(availability).split())
        lines.append(
            f"| `{name}` | {purpose_cell} | {profile} | {availability_cell} "
            f"| {guide_cell} | {help_cell} |"
        )
    lines += ["", "## Commands", ""]
    for name in sorted(scripts):
        meta = entries.get(name, {})
        probe = probes.get(name)
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"- Callable: `{scripts[name]}`")
        lines.append(f"- Profile: `{meta.get('profile', '-')}`")
        lines.append(f"- Availability: {meta.get('availability', '-')}")
        guide = meta.get("guide", "-")
        lines.append(f"- Guide: {_md_link(guide) if guide != '-' else '-'}")
        if probe is not None:
            lines.append(f"- Help: {probe.status}")
            if probe.synopsis:
                lines.append(f"- Synopsis: {probe.synopsis}")
        lines.append("")
        subcommands = list(probe.subcommands) if probe is not None else []
        if name == "robot-sf" and subcommands:
            lines.append(
                "Nested `robot-sf` subcommands (summary only; see each task guide for flags):"
            )
            lines.append("")
            lines.append("| Subcommand | Help | Guide |")
            lines.append("| --- | --- | --- |")
            sub_help = probe.subcommand_help if probe is not None else {}
            sub_guides: dict[str, str] = meta.get("subcommands", {})
            for sub in sorted(subcommands):
                help_text = sub_help.get(sub, "-")
                guide_path = sub_guides.get(sub, meta.get("guide", "-"))
                guide_cell = _md_link(guide_path) if guide_path != "-" else "-"
                lines.append(f"| `{sub}` | {help_text} | {guide_cell} |")
            lines.append("")
        elif subcommands:
            entry_guide = meta.get("guide", "-")
            lines.append("Subcommands (summary only; see the task guide for flags):")
            lines.append("")
            lines.append("| Subcommand | Guide |")
            lines.append("| --- | --- |")
            for sub in sorted(subcommands):
                guide_cell = _md_link(entry_guide) if entry_guide != "-" else "-"
                lines.append(f"| `{sub}` | {guide_cell} |")
            lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("- Script order is sorted from `[project.scripts]` for determinism.")
    lines.append(
        "- `--help` runs in an OS-enforced sandbox (Linux Landlock + seccomp) with a "
        "fixed width, task-owned temporary root, and timeout."
    )
    lines.append(
        "- Dynamic import/help code gets read-only source/runtime access, writes only "
        "inside that temporary root, and receives no inherited environment or file-descriptor "
        "credentials."
    )
    lines.append(
        "- Network, process/thread, namespace, and scheduler escape syscalls are denied; "
        "unsupported hosts fail closed."
    )
    lines.append(
        "- This is not a general untrusted-code sandbox: the child retains the invoking "
        "Unix identity and host resource limits."
    )
    lines.append("- CI fails on drift: run the generator without `--check` to refresh this file.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Return the generator argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--pyproject", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--timeout",
        type=int,
        default=HELP_TIMEOUT_S,
        help="Per-command --help subprocess timeout in seconds",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the committed reference differs from the deterministic render",
    )
    return parser


def generate(
    repo_root: Path, pyproject: Path, meta_path: Path, timeout_s: int
) -> tuple[str, list[str], dict[str, str], dict[str, dict], dict[str, ProbeResult]]:
    """Load, probe, validate, and render the CLI reference.

    Args:
        repo_root: Repository root.
        pyproject: Path to ``pyproject.toml``.
        meta_path: Path to the compact metadata YAML.
        timeout_s: Per-command help timeout.

    Returns:
        Tuple of rendered markdown, error list, scripts, entries, and probes.
    """
    scripts = load_project_scripts(pyproject)
    raw = load_metadata(meta_path)
    entries, meta_errors = validate_metadata_entries(scripts, raw, repo_root)
    probes: dict[str, ProbeResult] = {}
    for name in sorted(scripts):
        probes[name] = probe_help(name, scripts[name], repo_root, timeout_s)
    errors = list(meta_errors)
    errors.extend(check_entry_points(scripts, entries, probes))
    rendered = render_markdown(scripts, entries, probes)
    return rendered, errors, scripts, entries, probes


def main(argv: list[str] | None = None) -> int:
    """Render or source-check ``docs/cli_reference.md``."""
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    pyproject = Path(args.pyproject).resolve() if args.pyproject else repo_root / PYPROJECT_REL
    meta_path = Path(args.meta).resolve() if args.meta else repo_root / META_REL
    output_path = Path(args.output).resolve() if args.output else repo_root / OUTPUT_REL
    try:
        rendered, errors, scripts, _entries, _probes = generate(
            repo_root, pyproject, meta_path, args.timeout
        )
    except CliReferenceError as exc:
        print(f"cli reference invalid: {exc}", file=sys.stderr)
        return 2
    if args.check:
        if errors:
            for error in errors:
                print(f"cli reference error: {error}")
            return 1
        try:
            current = output_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"cli reference error: cannot read {output_path}: {exc}")
            return 1
        if rendered == current:
            print(f"cli reference up to date: {len(scripts)} entry points")
            return 0
        diff = difflib.unified_diff(
            current.splitlines(),
            rendered.splitlines(),
            fromfile=str(output_path),
            tofile=f"{output_path} (rendered)",
            lineterm="",
        )
        snippet = "\n".join(list(diff)[:40])
        print(f"cli reference drift detected; first diff lines:\n{snippet}")
        return 1
    if errors:
        for error in errors:
            print(f"cli reference error: {error}")
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"rendered {output_path} ({len(scripts)} entry points)")
    return 0


def _run_probe_child_cli(argv: list[str]) -> int:
    """Parse the private child command used by :func:`probe_help`."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--attr", required=True)
    args = parser.parse_args(argv)
    return _run_probe_child(
        args.script, args.module, args.attr, args.repo_root.resolve(), args.task_root.resolve()
    )


def _dispatch(argv: list[str]) -> int:
    """Dispatch normal generator or the private bounded probe child."""
    if argv and argv[0] == PROBE_FLAG:
        return _run_probe_child_cli(argv[1:])
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(_dispatch(sys.argv[1:]))
