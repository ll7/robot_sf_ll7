"""Test-only CLI entry signal without replacing the real ``python -m`` command."""

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

# Preserve subsequent caller hooks (including startup-delay probes), without
# re-entering the seedguard bootstrap that already chained into this hook.
_here = str(Path(__file__).resolve().parent)
_spec = importlib.machinery.PathFinder.find_spec(
    "sitecustomize", sys.path[sys.path.index(_here) + 1 :]
)
if _spec is not None and _spec.loader is not None:
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)


def _entry_signal(frame, event, _arg):
    if (
        event == "call"
        and frame.f_code.co_name == "main"
        and frame.f_code.co_filename.endswith("/analysis_workbench/review_context.py")
    ):
        sys.setprofile(None)
        ready = Path(os.environ["ROBOT_SF_TEST_CLI_READY"])
        temporary = ready.with_suffix(".tmp")
        temporary.write_text("CLI_READY\n", encoding="utf-8")
        temporary.replace(ready)


if "ROBOT_SF_TEST_CLI_READY" in os.environ:
    sys.setprofile(_entry_signal)
