"""Synthetic authenticated read endpoint; production has no injectable scheduler seam."""

import json
import os
import sys

NODE = "auxme-imech142"
CPU = "AMD EPYC 9115 16-Core Processor"


def install_scheduler(root, monkeypatch):
    records = root / "scheduler.json"
    records.write_text(
        json.dumps(
            {
                job: {
                    "job_id": job,
                    "state": "COMPLETED",
                    "exit_code": "0:0",
                    "node": NODE,
                    "partition": "pro6000",
                }
                for job in ("123", "124", "456", "457", "12345")
            }
        )
    )
    tools = root / "scheduler-bin"
    tools.mkdir(exist_ok=True)
    ssh = tools / "ssh"
    ssh.write_text(f"""#!{sys.executable}
import json,shlex,sys
args=shlex.split(sys.argv[-1])
assert sys.argv[1:4] == ["-oBatchMode=yes", "-oConnectTimeout=10", "imech192"]
assert args[0] == "sacct"
jobs=next(x.split("=",1)[1] for x in args if x.startswith("--jobs="))
records=json.load(open({str(records)!r}))
for job in jobs.split(","):
    if job in records:
        r=records[job]
        print("|".join(r[k] for k in ("job_id","state","exit_code","node","partition")))
""")
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    return records
