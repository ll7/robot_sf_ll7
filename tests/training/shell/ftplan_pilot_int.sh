#!/usr/bin/env bash
# Shell-only regression: the actual pilot must reach custody after INT.
set -euo pipefail
pilot_script=${1:?pilot script path required}
trainer_mode=${2:?cooperative or ignore_term required}
[[ "$trainer_mode" == cooperative || "$trainer_mode" == ignore_term ]]
case_root=$(mktemp -d "${TMPDIR:-/tmp}/ftplan-int-XXXXXX")
pilot_pid=''
cleanup() {
    local rc=$?
    trap - EXIT
    if [[ -f "$case_root/trainer.pid" ]]; then
        read -r worker_pid < "$case_root/trainer.pid"
        kill -KILL -- "-$worker_pid" 2>/dev/null || true
        kill -KILL "$worker_pid" 2>/dev/null || true
    fi
    if [[ -n "$pilot_pid" ]]; then
        kill -KILL -- "-$pilot_pid" 2>/dev/null || true
        wait "$pilot_pid" 2>/dev/null || true
    fi
    rm -rf -- "$case_root"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
mkdir -p "$case_root/checkout/scripts/dev" "$case_root/bin"
cat > "$case_root/bin/git" <<'GIT'
#!/bin/sh
if [ "$1" = rev-parse ]; then printf '%s\n' "$FTPLAN_SOURCE_HEAD"; fi
GIT
chmod +x "$case_root/bin/git"
cat > "$case_root/checkout/scripts/dev/run_worktree_shared_venv.sh" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
    if [[ "$arg" == -m ]]; then
        # Bash asynchronous commands inherit ignored INT, matching the uv trainer chain.
        trap -p INT > "$TEST_CASE_ROOT/int-disposition"
        if [[ "$TEST_TRAINER_MODE" == ignore_term ]]; then trap '' TERM; fi
        printf '%s\n' "$$" > "$TEST_CASE_ROOT/trainer.pid"
        while :; do sleep 0.05; done
    fi
done
code=$(cat)
if [[ "$code" == *'import wandb'* ]]; then
    dest="$FTPLAN_HUB_ROOT/ppo-ftplan-20261012/init100001/segment1"
    mkdir -p "${dest%/*}"
    cp -a "$ROBOT_SF_ARTIFACT_ROOT" "$dest"
    printf '%s\n' "$FTPLAN_ORIGINAL_EXIT_CODE" > "$TEST_CASE_ROOT/custody-exit"
    printf '%s\n' "$FTPLAN_SEGMENT_STATUS" > "$TEST_CASE_ROOT/custody-status"
else
    printf '%s\n' partial-checkpoint > "$ROBOT_SF_ARTIFACT_ROOT/partial.zip"
fi
WRAPPER
chmod +x "$case_root/checkout/scripts/dev/run_worktree_shared_venv.sh"
export PATH="$case_root/bin:$PATH"
export FTPLAN_REPO_ROOT="$case_root/checkout" FTPLAN_SOURCE_HEAD=fixture-head
export FTPLAN_REPLICATION_MANIFEST=unused FTPLAN_SELECTION_RECEIPT=unused
export FTPLAN_ACTION_CONTRACT_RECEIPT=unused FTPLAN_HUB_ROOT="$case_root/hub"
export FTPLAN_SEGMENT=1 SLURM_ARRAY_TASK_ID=0
export TEST_CASE_ROOT="$case_root" TEST_TRAINER_MODE="$trainer_mode"
# Reset INT for the pilot shell; its background trainer still inherits ignored INT.
setsid env --default-signal=INT bash "$pilot_script" > "$case_root/pilot.log" 2>&1 &
pilot_pid=$!
deadline=$((SECONDS + 10))
while [[ ! -f "$case_root/trainer.pid" && "$SECONDS" -lt "$deadline" ]]; do sleep 0.05; done
[[ -f "$case_root/trainer.pid" ]] || { cat "$case_root/pilot.log"; echo 'Trainer never started' >&2; exit 1; }
[[ $(cat "$case_root/int-disposition") == *"'' SIGINT"* ]] || { echo 'Fixture did not inherit ignored INT' >&2; exit 1; }
kill -INT "$pilot_pid"
deadline=$((SECONDS + 14))
while [[ ! -f "$case_root/custody-exit" && "$SECONDS" -lt "$deadline" ]]; do sleep 0.05; done
[[ -f "$case_root/custody-exit" ]] || { echo 'INT did not reach custody within 14 seconds' >&2; exit 1; }
rc=0
wait "$pilot_pid" || rc=$?
[[ "$rc" -eq 130 && $(cat "$case_root/custody-exit") == 130 ]]
[[ $(cat "$case_root/custody-status") == incomplete ]]
[[ -f "$case_root/hub/ppo-ftplan-20261012/init100001/segment1/partial.zip" ]]
read -r worker_pid < "$case_root/trainer.pid"
if kill -0 "$worker_pid" 2>/dev/null; then
    echo 'Custody ran before the trainer was terminated and reaped' >&2
    exit 1
fi
