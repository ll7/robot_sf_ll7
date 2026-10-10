"""Run bounded local variant measurements with at most eight workers."""

import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

root = Path(os.environ["WALL_CONTACT_ARTIFACT_ROOT"])
plan = json.loads((root / "measurement-manifest.json").read_text())
versions = sys.argv[1:] or list(plan["laws"])
indices = [int(i) for i in os.environ.get("TASK_INDICES", "").split(",") if i] or list(
    range(len(plan["tasks"]))
)
max_workers = min(int(plan.get("workers", 8)), 8)


def run_slot(version: str, index: int) -> dict[str, object]:
    """Run one bounded episode in an isolated process and preserve its receipt."""
    source = Path(plan["roots"][version])
    expected = plan["sources"][version]
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual == expected, (version, actual, expected)
    assert not subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--name-only"], text=True
    ).strip(), "Measurement requires committed source bytes"
    slot = root / "measurements" / version / str(index)
    slot.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        SOURCE_ROOT=str(source),
        PYTHONPATH=os.pathsep.join((str(source), str(source / "fast-pysf"))),
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TORCH_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
        MPLBACKEND="Agg",
        NUMBA_CACHE_DIR=str(slot / "numba-cache"),
    )
    command = [
        "nice",
        "-n",
        "15",
        plan["python"],
        "-u",
        str(Path(__file__).with_name("measure_pedestrians.py")),
        version,
        str(index),
        str(slot),
    ]
    started = time.monotonic()
    start_utc = datetime.now(UTC).isoformat()
    timeout = False
    with (slot / "run.log").open("w") as log:
        child = subprocess.Popen(
            command,
            cwd=source,
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            returncode = child.wait(timeout=int(plan["episode_limit_s"]))
        except subprocess.TimeoutExpired:
            timeout = True
            child.send_signal(signal.SIGUSR1)
            time.sleep(1)
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            returncode = child.returncode
    measurement = (
        json.loads((slot / "measurement.json").read_text())
        if (slot / "measurement.json").exists()
        else None
    )
    receipt = {
        "version": version,
        "index": index,
        "task": plan["tasks"][index],
        "source_sha": expected,
        "law": plan["laws"][version],
        "node": os.uname().nodename,
        "start_utc": start_utc,
        "end_utc": datetime.now(UTC).isoformat(),
        "runtime_s": round(time.monotonic() - started, 3),
        "limit_s": int(plan["episode_limit_s"]),
        "returncode": returncode,
        "classification": (
            "TIMEOUT-STALL"
            if timeout
            else "EXECUTION-FAILED"
            if returncode
            else "MEASURED"
            if measurement is not None
            else "MISSING-MEASUREMENT"
        ),
        "measurement": measurement,
        "log_tail": re.sub(
            r"\b5[0-9]{4}\b",
            "<seed>",
            "\n".join((slot / "run.log").read_text(errors="replace").splitlines()[-25:]),
        ),
    }
    (slot / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


futures = []
with ThreadPoolExecutor(max_workers=max_workers) as pool:
    for version in versions:
        for index in indices:
            futures.append(pool.submit(run_slot, version, index))
    records = []
    for future in as_completed(futures):
        record = future.result()
        records.append(record)
        print(
            json.dumps(
                {
                    "version": record["version"],
                    "index": record["index"],
                    "classification": record["classification"],
                    "runtime_s": record["runtime_s"],
                }
            ),
            flush=True,
        )

records.sort(key=lambda row: (str(row["version"]), int(row["index"])))
(root / "measurement-records.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps({"records": len(records), "versions": versions, "workers": max_workers}))
