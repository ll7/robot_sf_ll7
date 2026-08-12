#!/usr/bin/env bash
set -euo pipefail

umask 077

readonly required_files=(
  result.json
  RESULT.md
  diffstat.txt
  validation.json
)
readonly legacy_required_files=(
  result.json
  status.txt
  diffstat.txt
  changed_files.txt
)

usage() {
  cat <<'USAGE'
Usage:
  scripts/review-agent-run.sh --latest [--include-logs]
  scripts/review-agent-run.sh --run-dir <path> [--include-logs]

Review a delegated worker run from compact artifacts and write a private
candidate lesson note under the repository common Git-dir inbox. Worker logs
are never read unless --include-logs is provided.
USAGE
}

run_dir=""
use_latest="0"
include_logs="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "--run-dir requires a path" >&2
        usage >&2
        exit 2
      fi
      run_dir="$2"
      shift 2
      ;;
    --latest)
      use_latest="1"
      shift
      ;;
    --include-logs)
      include_logs="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$use_latest" == "1" && -n "$run_dir" ]]; then
  echo "Use only one of --latest or --run-dir" >&2
  exit 2
fi

if [[ "$use_latest" == "0" && -z "$run_dir" ]]; then
  usage >&2
  exit 2
fi

if ! common_git_dir="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" ||
  [[ -z "$common_git_dir" || ! -d "$common_git_dir" ]]; then
  echo "Unable to resolve the repository common Git directory" >&2
  exit 1
fi

run_root="$common_git_dir/codex-agent-runs"
common_note_root="$run_root/notes/inbox"

run_has_artifact_marker() {
  local candidate_dir="$1"
  local required_file

  for required_file in "${required_files[@]}" "${legacy_required_files[@]}"; do
    if [[ -f "$candidate_dir/$required_file" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "$use_latest" == "1" ]]; then
  latest_candidate=""
  if [[ -d "$run_root" ]]; then
    while IFS= read -r candidate_run_dir; do
      candidate_name="$(basename "$candidate_run_dir")"
      case "$candidate_name" in
        active|notes)
          continue
          ;;
      esac
      if run_has_artifact_marker "$candidate_run_dir"; then
        latest_candidate="$candidate_run_dir"
        break
      fi
    done < <(find "$run_root" -mindepth 1 -maxdepth 1 -type d -print | LC_ALL=C sort -r)
  fi

  if [[ -n "$latest_candidate" ]]; then
    # Inspect the newest artifact-bearing run, even when it is incomplete.
    # Skipping it in favor of an older complete run hides the tooling failure.
    run_dir="$latest_candidate"
  else
    echo "result=no-run" >&2
    echo "No worker run directory found under $run_root" >&2
    exit 1
  fi
fi

if [[ "$run_dir" != /* ]]; then
  run_dir="$PWD/$run_dir"
fi
if [[ -d "$run_dir" ]]; then
  run_dir="$(cd -- "$run_dir" && pwd -P)"
fi

artifact_ref="external-run"
if [[ "$run_dir" == "$run_root"/* ]]; then
  artifact_ref=".git/codex-agent-runs/${run_dir#"$run_root"/}"
elif [[ -n "$run_dir" ]]; then
  artifact_ref="external/$(basename -- "$run_dir" | tr -c 'A-Za-z0-9._-' '-')"
fi

write_failure_note() {
  local outcome="$1"
  local reason="$2"
  local missing_summary="$3"
  local invalid_summary="$4"
  local present_summary
  local timestamp
  local run_slug
  local note_path

  if [[ -d "$run_dir" ]]; then
    present_summary="$(
      find "$run_dir" -maxdepth 1 -type f -exec basename {} \; |
        LC_ALL=C sort |
        paste -sd ', ' -
    )"
  else
    present_summary=""
  fi
  if [[ -z "$present_summary" ]]; then
    present_summary="none"
  fi

  timestamp="$(date -u +%Y%m%dT%H%M%S%NZ)"
  run_slug="$(basename -- "$run_dir" | tr -c 'A-Za-z0-9._-' '-')"
  note_path="$common_note_root/${timestamp}-${run_slug}-agent-workflow-self-review.md"
  note_ref=".git/codex-agent-runs/notes/inbox/$(basename -- "$note_path")"
  mkdir -p "$common_note_root"

  {
    echo "---"
    echo "schema: agent_run_self_review.v1"
    echo "objective: compact delegated-agent artifact review"
    echo "outcome: incomplete"
    echo "run_id: $(basename -- "$run_dir")"
    echo "observation_class: tooling"
    echo "routing_signal: negative"
    echo "output_quality: incomplete_artifact"
    echo "confidence: medium"
    echo "---"
    echo
    echo "# Agent Workflow Self Review"
    echo
    echo "## Scope"
    echo
    echo "- task: inspect compact delegated-agent artifacts"
    echo "- allowed_edits: private candidate note only"
    echo "- artifact_ref: $artifact_ref"
    echo
    echo "## Outcome"
    echo
    echo "- outcome: incomplete"
    echo "- suspected_issue: tooling"
    echo "- observation_class: tooling"
    echo "- routing_signal: negative"
    echo "- output_quality: incomplete_artifact"
    echo "- candidate_lesson: Worker run cannot be accepted from compact artifacts; treat the result as low confidence and rely on local validation."
    echo "- routing_feedback: $reason"
    echo "- confidence: medium"
    echo "- needs_repetition: no"
    echo
    echo "## Validation"
    echo
    echo "- command: compact artifact contract"
    echo "  result: fail"
    echo "  note: required artifacts are missing or malformed"
    echo
    echo "## Blockers"
    echo
    echo "- $reason"
    echo
    echo "## Reusable Lessons"
    echo
    echo "- class: tooling"
    echo "  confidence: medium"
    echo "  lesson: Missing compact artifacts are negative reliability evidence until local validation supplies acceptance proof."
    echo "  promotion: inbox-only"
    echo
    echo "## Artifact Review"
    echo
    echo "- compact_artifacts_checked: true"
    echo "- local_validation_overrode_agent_summary: true"
    echo "- note_path: $note_ref"
    echo "- artifact_ref: $artifact_ref"
    echo "- missing_required_artifacts: ${missing_summary:-none}"
    echo "- malformed_json_artifacts: ${invalid_summary:-none}"
    echo "- present_artifacts: $present_summary"
    echo
    echo "## Next Action"
    echo
    echo "- Run local validation and review the direct diff before accepting or promoting any worker result."
    echo
    echo "## Compact Artifact Summary"
    echo
    echo "- missing_required_artifacts: ${missing_summary:-none}"
    echo "- malformed_json_artifacts: ${invalid_summary:-none}"
    echo "- present_artifacts: $present_summary"
    echo "- validation_evidence: not found"
  } > "$note_path"

  printf 'result=%s\nartifact_ref=%s\nnote=%s\n' "$outcome" "$artifact_ref" "$note_path"
}

if [[ ! -d "$run_dir" ]]; then
  echo "Run directory not found: $run_dir" >&2
  write_failure_note "incomplete-artifact" "The worker run directory is missing." "run-directory" ""
  exit 1
fi

artifact_contract="canonical"
canonical_complete="1"
for required_file in "${required_files[@]}"; do
  if [[ ! -f "$run_dir/$required_file" ]]; then
    canonical_complete="0"
    break
  fi
done
missing_files=()
if [[ "$canonical_complete" != "1" ]]; then
  legacy_complete="1"
  for required_file in "${legacy_required_files[@]}"; do
    if [[ ! -f "$run_dir/$required_file" ]]; then
      legacy_complete="0"
      break
    fi
  done
  if [[ "$legacy_complete" == "1" ]]; then
    artifact_contract="legacy"
  else
    for required_file in "${required_files[@]}"; do
      if [[ ! -f "$run_dir/$required_file" ]]; then
        missing_files+=("$required_file")
      fi
    done
  fi
fi
if [[ "${#missing_files[@]}" -gt 0 ]]; then
  missing_summary="$(printf '%s\n' "${missing_files[@]}" | paste -sd ', ' -)"
  echo "Missing required artifact(s): $missing_summary" >&2
  write_failure_note \
    "incomplete-artifact" \
    "Missing required compact artifact(s): $missing_summary" \
    "$missing_summary" \
    ""
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to validate worker JSON artifacts" >&2
  exit 1
fi

json_files=("$run_dir/result.json")
if [[ "$artifact_contract" == "canonical" ]]; then
  json_files+=("$run_dir/validation.json")
fi
if [[ -f "$run_dir/metrics.json" ]]; then
  json_files+=("$run_dir/metrics.json")
fi
invalid_json_files=()
for json_file in "${json_files[@]}"; do
  if ! jq -e 'type == "object"' "$json_file" >/dev/null 2>&1; then
    invalid_json_files+=("$(basename "$json_file")")
  fi
done
if [[ "${#invalid_json_files[@]}" -gt 0 ]]; then
  invalid_summary="$(printf '%s\n' "${invalid_json_files[@]}" | paste -sd ', ' -)"
  echo "Malformed JSON artifact(s): $invalid_summary" >&2
  write_failure_note \
    "malformed-json" \
    "Malformed or non-object JSON artifact(s): $invalid_summary" \
    "" \
    "$invalid_summary"
  exit 1
fi

json_value() {
  local expression="$1"
  local fallback="$2"
  local path="$3"

  jq -r --arg fallback "$fallback" "($expression) // \$fallback" "$path"
}

provider="$(json_value '.provider' 'unknown' "$run_dir/result.json")"
model="$(json_value '.model' 'unknown' "$run_dir/result.json")"
result_status="$(json_value '.status' 'unknown' "$run_dir/result.json")"
worker_status="$(json_value '.worker_status // .status' 'unknown' "$run_dir/result.json")"
task_class="$(json_value '.task_class // .mode' 'unknown' "$run_dir/result.json")"
route_score="$(json_value '.route.score // .score' 'unknown' "$run_dir/result.json")"
route_task_score="$(json_value '.route.task_score // .task_score' 'unknown' "$run_dir/result.json")"
route_reliability="$(json_value '.route.reliability // .reliability' 'unknown' "$run_dir/result.json")"
route_relative_cost="$(json_value '.route.relative_cost // .relative_cost' 'unknown' "$run_dir/result.json")"
metrics_path="$run_dir/metrics.json"
validation_status="unknown"
acceptance_status="unknown"
artifact_status="unknown"
failure_category="unknown"
if [[ "$artifact_contract" == "canonical" ]]; then
  validation_status="$(json_value '.status' 'unknown' "$run_dir/validation.json")"
  if [[ "$validation_status" == "unknown" ]]; then
    validation_command_count="$(json_value '.commands // [] | map(select((.result // .status) == \"pass\" or (.result // .status) == \"passed\")) | length' '0' "$run_dir/validation.json")"
    if [[ "$validation_command_count" -gt 0 ]]; then
      validation_status="evidence_present"
    else
      validation_status="not_run"
    fi
  fi
fi
if [[ -f "$metrics_path" ]]; then
  task_class="$(json_value '.task_class' "$task_class" "$metrics_path")"
  provider="$(json_value '.provider' "$provider" "$metrics_path")"
  model="$(json_value '.model' "$model" "$metrics_path")"
  validation_status="$(json_value '.validation_status' 'unknown' "$metrics_path")"
  acceptance_status="$(json_value '.acceptance_status' 'unknown' "$metrics_path")"
  artifact_status="$(json_value '.artifact_status' 'unknown' "$metrics_path")"
  failure_category="$(json_value '.failure_category' 'none' "$metrics_path")"
  route_score="$(json_value '.route.score' "$route_score" "$metrics_path")"
  route_task_score="$(json_value '.route.task_score' "$route_task_score" "$metrics_path")"
  route_reliability="$(json_value '.route.reliability' "$route_reliability" "$metrics_path")"
  route_relative_cost="$(json_value '.route.relative_cost' "$route_relative_cost" "$metrics_path")"
fi

normalized_status="$worker_status"
case "$normalized_status" in
  0|success|passed|complete|completed)
    normalized_status="complete"
    ;;
  failed|failure|error|nonzero)
    normalized_status="failed"
    ;;
esac

nonblank_count() {
  if [[ ! -f "$1" ]]; then
    printf 'unknown\n'
    return 0
  fi
  sed '/^[[:space:]]*$/d' "$1" | wc -l | tr -d ' '
}

changed_count="$(nonblank_count "$run_dir/changed_files.txt")"
if [[ "$changed_count" == "unknown" ]]; then
  changed_count="$(jq -r 'if (.fix_forward_changed_files? | type) == "array" then (.fix_forward_changed_files | length) elif (.changed_files? | type) == "array" then (.changed_files | length) elif (.changed_file_count? | type) == "number" then .changed_file_count else "unknown" end' "$run_dir/result.json")"
fi
baseline_changed_path="$run_dir/baseline_changed_files.txt"
baseline_status_path="$run_dir/baseline_status.txt"
new_changed_path="$run_dir/new_changed_files.txt"
baseline_changed_count="unknown"
baseline_status_count="unknown"
new_changed_count="unknown"
local_commits_path="$run_dir/local_commits.txt"
local_commit_count="unknown"
if [[ -f "$baseline_changed_path" ]]; then
  baseline_changed_count="$(nonblank_count "$baseline_changed_path")"
fi
if [[ -f "$baseline_status_path" ]]; then
  baseline_status_count="$(nonblank_count "$baseline_status_path")"
fi
if [[ -f "$new_changed_path" ]]; then
  new_changed_count="$(nonblank_count "$new_changed_path")"
fi
if [[ -f "$local_commits_path" ]]; then
  local_commit_count="$(nonblank_count "$local_commits_path")"
fi
status_count="$(nonblank_count "$run_dir/status.txt")"
diffstat_count="$(nonblank_count "$run_dir/diffstat.txt")"

stderr_path="$run_dir/worker.stderr.log"
stdout_path="$run_dir/worker.stdout.log"
stderr_size="not-read"
stdout_size="not-read"
if [[ "$include_logs" == "1" ]]; then
  stderr_size="missing"
  if [[ -f "$stderr_path" ]]; then
    stderr_size="$(wc -c < "$stderr_path" | tr -d ' ')"
  fi
  stdout_size="missing"
  if [[ -f "$stdout_path" ]]; then
    stdout_size="$(wc -c < "$stdout_path" | tr -d ' ')"
  fi
fi

validation_evidence="not found"
validation_pattern="validation run|validated|pytest|unittest|ruff|bash -n|git diff --check"
if grep -Eqi "$validation_pattern" \
  "$run_dir/RESULT.md" "$run_dir/validation.json" "$run_dir/status.txt" 2>/dev/null; then
  validation_evidence="present in compact artifacts"
elif [[ "$include_logs" == "1" ]] &&
  grep -Eqi "$validation_pattern" "$run_dir/worker.stdout.log" "$run_dir/worker.stderr.log" 2>/dev/null; then
  validation_evidence="present in logs"
fi
if [[ "$validation_status" == passed* || "$validation_status" == evidence_present ]]; then
  validation_evidence="present in validation.json"
fi

suspected_issue="none from compact artifacts"
candidate_lesson="No durable lesson yet; collect repeated evidence before changing system instructions."
recommended_target="none"
confidence="low"
needs_repetition="yes"
routing_signal="neutral"
output_quality="unknown"
score_adjustment_candidate="none"
routing_feedback="No routing score update from this single run; aggregate repeated medium/high confidence notes before changing model assessments."
review_outcome="useful"

if [[ "$normalized_status" == "failed" || "$normalized_status" == "unknown" ]]; then
  suspected_issue="tooling"
  candidate_lesson="Worker exited non-zero; inspect the failure mode before reusing this route or prompt shape."
  recommended_target="worker command guidance"
  confidence="medium"
  routing_signal="negative"
  output_quality="failed"
  score_adjustment_candidate="consider lowering reliability for this provider/model on repeated route-collapse, capacity, or artifact failures"
  routing_feedback="Do not lower scores from one failure alone unless it was high-cost or repeatable; record this as negative reliability evidence."
  review_outcome="failed"
elif [[ "$validation_evidence" == "not found" && "$validation_status" == "unknown" ]]; then
  suspected_issue="validation"
  candidate_lesson="Worker result lacks clear validation evidence; future prompts may need an explicit validation reporting requirement."
  recommended_target="prompt contract or delegation guidance"
  confidence="medium"
  routing_signal="weak_negative"
  output_quality="partial"
  score_adjustment_candidate="consider lowering reliability if sparse validation repeats for this provider/model/task class"
  routing_feedback="Missing validation is mainly reliability evidence, not task-fit evidence, unless local review shows the answer was wrong."
  review_outcome="mixed"
elif [[ "$new_changed_count" != "unknown" && "$new_changed_count" -gt 12 ]]; then
  suspected_issue="file-scope"
  candidate_lesson="Worker introduced many changed files beyond the pre-run baseline; review whether the prompt needs tighter ownership boundaries."
  recommended_target="prompt contract guidance"
  confidence="medium"
  routing_signal="weak_negative"
  output_quality="partial"
  score_adjustment_candidate="consider lowering reliability for broad edit tasks if scope drift repeats"
  routing_feedback="Scope drift is a task-specific signal only when it repeats for the same task class."
elif [[ "$new_changed_count" == "unknown" && "$changed_count" != "unknown" && "$changed_count" -gt 12 ]]; then
  suspected_issue="file-scope"
  candidate_lesson="Worker run predates baseline change tracking and shows many changed files; inspect attribution before promoting lessons."
  recommended_target="worker wrapper or prompt contract guidance"
  confidence="low"
  routing_signal="unknown"
  output_quality="unclear"
  score_adjustment_candidate="none until attribution is clean"
  routing_feedback="Attribution contamination should not update ranking without local diff proof."
elif [[ "$validation_evidence" != "not found" && "$normalized_status" == "complete" ]]; then
  routing_signal="positive"
  output_quality="useful"
  score_adjustment_candidate="consider raising reliability only after repeated accepted runs for this provider/model/task class"
  routing_feedback="A single useful run is positive evidence, but promotion should wait for repetition or high-impact validation."
fi

timestamp="$(date -u +%Y%m%dT%H%M%S%NZ)"
run_slug="$(basename -- "$run_dir" | tr -c 'A-Za-z0-9._-' '-')"
note_path="$common_note_root/${timestamp}-${run_slug}-agent-workflow-self-review.md"
note_ref=".git/codex-agent-runs/notes/inbox/$(basename -- "$note_path")"
mkdir -p "$common_note_root"

changed_files_summary="from result metadata"
if [[ -f "$run_dir/changed_files.txt" ]]; then
  changed_files_summary="$(awk 'NF { files = files ? files ", " $0 : $0 } END { print files }' "$run_dir/changed_files.txt")"
  if [[ -z "$changed_files_summary" ]]; then
    changed_files_summary="none"
  fi
fi
new_changed_files_summary="unknown"
if [[ -f "$new_changed_path" ]]; then
  new_changed_files_summary="$(awk 'NF { files = files ? files ", " $0 : $0 } END { print files }' "$new_changed_path")"
  if [[ -z "$new_changed_files_summary" ]]; then
    new_changed_files_summary="none"
  fi
fi
local_commits_summary="unknown"
if [[ -f "$local_commits_path" ]]; then
  local_commits_summary="$(awk 'NF { commits = commits ? commits "; " $0 : $0 } END { print commits }' "$local_commits_path")"
  if [[ -z "$local_commits_summary" ]]; then
    local_commits_summary="none"
  fi
fi

{
  echo "---"
  echo "schema: agent_run_self_review.v1"
  echo "objective: compact delegated-agent artifact review"
  echo "outcome: $review_outcome"
  echo "run_id: $(basename -- "$run_dir")"
  echo "provider: $provider"
  echo "model: $model"
  echo "task_class: $task_class"
  echo "observation_class: $suspected_issue"
  echo "routing_signal: $routing_signal"
  echo "output_quality: $output_quality"
  echo "confidence: $confidence"
  echo "---"
  echo
  echo "# Agent Workflow Self Review"
  echo
  echo "## Scope"
  echo
  echo "- task: inspect compact delegated-agent artifacts"
  echo "- allowed_edits: private candidate note only"
  echo "- artifact_ref: $artifact_ref"
  echo
  echo "## Outcome"
  echo
  echo "- outcome: $review_outcome"
  echo "- worker_status: $normalized_status"
  echo "- validation_status: $validation_status"
  echo "- acceptance_status: $acceptance_status"
  echo "- artifact_status: $artifact_status"
  echo "- failure_category: $failure_category"
  echo "- changed_files: $changed_files_summary"
  echo "- new_changed_files: $new_changed_files_summary"
  echo "- local_commits: $local_commits_summary"
  echo "- observation_class: $suspected_issue"
  echo "- routing_signal: $routing_signal"
  echo "- output_quality: $output_quality"
  echo "- route_score: $route_score"
  echo "- route_task_score: $route_task_score"
  echo "- route_reliability: $route_reliability"
  echo "- route_relative_cost: $route_relative_cost"
  echo "- score_adjustment_candidate: $score_adjustment_candidate"
  echo "- routing_feedback: $routing_feedback"
  echo
  echo "## Validation"
  echo
  echo "- command: compact artifact contract"
  echo "  result: pass"
  echo "  note: $artifact_contract compact artifact set is present"
  echo "- command: validation evidence scan"
  if [[ "$validation_evidence" == "not found" && "$validation_status" == "unknown" ]]; then
    echo "  result: fail"
  else
    echo "  result: pass"
  fi
  echo "  note: $validation_evidence"
  echo
  echo "## Blockers"
  echo
  echo "- none"
  echo
  echo "## Reusable Lessons"
  echo
  echo "- class: $suspected_issue"
  echo "  confidence: $confidence"
  echo "  lesson: $candidate_lesson"
  echo "  promotion: inbox-only"
  echo
  echo "## Artifact Review"
  echo
  echo "- compact_artifacts_checked: true"
  echo "- local_validation_overrode_agent_summary: false"
  echo "- note_path: $note_ref"
  echo "- artifact_ref: $artifact_ref"
  echo
  echo "## Compact Artifact Summary"
  echo
  echo "- status_entries: $status_count"
  echo "- baseline_status_entries: $baseline_status_count"
  echo "- diffstat_entries: $diffstat_count"
  echo "- changed_file_count: $changed_count"
  echo "- baseline_changed_file_count: $baseline_changed_count"
  echo "- new_changed_file_count: $new_changed_count"
  echo "- local_commit_count: $local_commit_count"
  echo "- stderr_size_bytes: $stderr_size"
  echo "- stdout_size_bytes: $stdout_size"
  echo "- validation_evidence: $validation_evidence"
  if [[ "$include_logs" == "1" ]]; then
    echo
    echo "## Log Snippets"
    echo
    echo "### stderr tail"
    echo
    echo '```text'
    if [[ -f "$stderr_path" ]]; then
      tail -n 40 "$stderr_path"
    else
      echo "missing"
    fi
    echo '```'
    echo
    echo "### stdout tail"
    echo
    echo '```text'
    if [[ -f "$stdout_path" ]]; then
      tail -n 40 "$stdout_path"
    else
      echo "missing"
    fi
    echo '```'
  fi
} > "$note_path"

printf 'result=reviewed\nartifact_ref=%s\nnote=%s\n' "$artifact_ref" "$note_path"
