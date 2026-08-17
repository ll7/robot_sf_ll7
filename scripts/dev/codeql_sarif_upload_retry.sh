#!/usr/bin/env bash
# Bound CodeQL SARIF upload recovery and preserve a fail-closed final verdict.
#
# The workflow keeps CodeQL analysis separate from SARIF upload.  The upload
# action is attempted at most twice; this helper supplies the bounded delay
# before the second attempt and records both attempt outcomes in the step
# summary.  A successful retry is upload recovery, not evidence that analysis
# succeeded: the finalizer still requires a successful analysis outcome.
#
# Usage:
#   scripts/dev/codeql_sarif_upload_retry.sh wait 2
#   scripts/dev/codeql_sarif_upload_retry.sh finalize

set -euo pipefail

command_name="${1:-}"

write_summary() {
  local line="$1"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    printf '%s\n' "$line" >>"$GITHUB_STEP_SUMMARY"
  fi
}

non_negative_integer_or_default() {
  local value="$1"
  local fallback="$2"
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$fallback"
  fi
}

wait_before_retry() {
  local retry_attempt="${1:-}"
  if [[ "$retry_attempt" != "2" ]]; then
    echo "codeql_sarif_upload_retry: only bounded retry attempt 2 is supported" >&2
    exit 2
  fi

  local backoff_base
  local backoff_cap
  backoff_base="$(non_negative_integer_or_default "${CODEQL_SARIF_UPLOAD_BACKOFF_BASE:-15}" 15)"
  backoff_cap="$(non_negative_integer_or_default "${CODEQL_SARIF_UPLOAD_BACKOFF_CAP:-60}" 60)"
  if [[ "$backoff_cap" -gt 60 ]]; then
    backoff_cap=60
  fi

  local delay="$backoff_base"
  if [[ "$delay" -gt "$backoff_cap" ]]; then
    delay="$backoff_cap"
  fi

  echo "codeql_sarif_upload_retry retry=1 attempt=${retry_attempt} backoff_seconds=${delay}"
  write_summary "- retry_scheduled: true (attempt ${retry_attempt}, backoff ${delay}s)"
  sleep "$delay"
}

valid_outcome() {
  case "$1" in
    success | failure | skipped | cancelled) return 0 ;;
    *) return 1 ;;
  esac
}

finalize_status() {
  local analysis_outcome="${CODEQL_ANALYSIS_OUTCOME:-}"
  local attempt_1_outcome="${CODEQL_SARIF_UPLOAD_ATTEMPT_1_OUTCOME:-skipped}"
  local attempt_2_outcome="${CODEQL_SARIF_UPLOAD_ATTEMPT_2_OUTCOME:-skipped}"

  for outcome in "$analysis_outcome" "$attempt_1_outcome" "$attempt_2_outcome"; do
    if ! valid_outcome "$outcome"; then
      echo "::error::codeql_sarif_upload_retry: unsupported step outcome '${outcome}'" >&2
      exit 1
    fi
  done

  local effective_status=""
  local effective_attempt="0"
  local retry_count="0"
  local status=0
  if [[ "$analysis_outcome" != "success" ]]; then
    effective_status="analysis_failed"
    echo "::error::codeql_sarif_upload_retry: CodeQL analysis outcome=${analysis_outcome}; no upload is admitted" >&2
    status=1
  elif [[ "$attempt_1_outcome" == "success" ]]; then
    effective_status="uploaded"
    effective_attempt="1"
  elif [[ "$attempt_2_outcome" == "success" ]]; then
    effective_status="uploaded_after_retry"
    effective_attempt="2"
    retry_count="1"
  else
    effective_status="upload_failed"
    echo "::error::codeql_sarif_upload_retry: SARIF upload failed on all bounded attempts" >&2
    status=1
  fi

  echo "codeql_sarif_upload_retry analysis=${analysis_outcome} attempt_1=${attempt_1_outcome} attempt_2=${attempt_2_outcome} effective_status=${effective_status} effective_attempt=${effective_attempt} retry_count=${retry_count}"
  write_summary "## CodeQL SARIF upload"
  write_summary "- analysis_outcome: ${analysis_outcome}"
  write_summary "- upload_attempt_1: ${attempt_1_outcome}"
  write_summary "- upload_attempt_2: ${attempt_2_outcome}"
  write_summary "- effective_status: ${effective_status}"
  write_summary "- effective_attempt: ${effective_attempt}"
  write_summary "- retry_count: ${retry_count}"
  return "$status"
}

case "$command_name" in
  wait)
    wait_before_retry "${2:-}"
    ;;
  finalize)
    finalize_status
    ;;
  *)
    echo "usage: $0 {wait <retry-attempt>|finalize}" >&2
    exit 2
    ;;
esac
