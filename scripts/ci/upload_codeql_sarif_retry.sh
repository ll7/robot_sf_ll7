#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  echo "usage: $0 <sarif-directory>" >&2
  exit 2
fi

sarif_directory=$1
if [[ ! -d "$sarif_directory" ]]; then
  echo "codeql_sarif_upload: SARIF directory does not exist: $sarif_directory" >&2
  exit 2
fi

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GH_TOKEN:?GH_TOKEN is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}"
: "${GITHUB_RUN_ATTEMPT:?GITHUB_RUN_ATTEMPT is required}"
: "${GITHUB_WORKFLOW:?GITHUB_WORKFLOW is required}"
: "${CODEQL_ANALYSIS_KEY:?CODEQL_ANALYSIS_KEY is required}"

max_attempts=${CODEQL_UPLOAD_MAX_ATTEMPTS:-3}
backoff_base=${CODEQL_UPLOAD_BACKOFF_BASE_SECONDS:-5}
backoff_cap=${CODEQL_UPLOAD_BACKOFF_CAP_SECONDS:-30}
if ! [[ "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "codeql_sarif_upload: CODEQL_UPLOAD_MAX_ATTEMPTS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$GITHUB_RUN_ID" =~ ^[1-9][0-9]*$ && "$GITHUB_RUN_ATTEMPT" =~ ^[1-9][0-9]*$ ]]; then
  echo "codeql_sarif_upload: workflow run identifiers must be positive integers" >&2
  exit 2
fi
if ! [[ "$backoff_base" =~ ^[0-9]+$ && "$backoff_cap" =~ ^[0-9]+$ ]]; then
  echo "codeql_sarif_upload: backoff settings must be non-negative integers" >&2
  exit 2
fi
if ! command -v gh >/dev/null 2>&1; then
  echo "codeql_sarif_upload: gh is required for SARIF upload" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "codeql_sarif_upload: jq is required for SARIF upload" >&2
  exit 2
fi

is_retryable_status() {
  case "$1" in
    429|500|502|503|504) return 0 ;;
    *) return 1 ;;
  esac
}

mapfile -t sarif_files < <(find "$sarif_directory" -type f -name '*.sarif' -print | sort)
if ((${#sarif_files[@]} == 0)); then
  echo "codeql_sarif_upload: no SARIF files found in $sarif_directory" >&2
  exit 1
fi

for sarif_file in "${sarif_files[@]}"; do
  encoded_sarif_file=$(mktemp)
  gzip -c "$sarif_file" | base64 -w0 >"$encoded_sarif_file"
  tool_names=$(jq -c '[.runs[]?.tool?.driver?.name // empty] | unique' "$sarif_file")
  checkout_uri="file://${GITHUB_WORKSPACE}"
  uploaded=0
  for ((attempt = 1; attempt <= max_attempts; attempt += 1)); do
    response_file=$(mktemp)
    if jq -n \
      --arg commit_oid "$GITHUB_SHA" \
      --arg ref "$GITHUB_REF" \
      --arg analysis_key "$CODEQL_ANALYSIS_KEY" \
      --arg analysis_name "$GITHUB_WORKFLOW" \
      --arg checkout_uri "$checkout_uri" \
      --argjson workflow_run_id "$GITHUB_RUN_ID" \
      --argjson workflow_run_attempt "$GITHUB_RUN_ATTEMPT" \
      --argjson tool_names "$tool_names" \
      --rawfile sarif "$encoded_sarif_file" \
      '{commit_oid: $commit_oid, ref: $ref, analysis_key: $analysis_key,
        analysis_name: $analysis_name, sarif: ($sarif | gsub("\\n"; "")),
        workflow_run_id: $workflow_run_id, workflow_run_attempt: $workflow_run_attempt,
        checkout_uri: $checkout_uri, tool_names: $tool_names}' \
      | gh api --method PUT "repos/${GITHUB_REPOSITORY}/code-scanning/analysis" --input - \
      >"$response_file" 2>&1; then
      rm -f "$response_file" "$encoded_sarif_file"
      printf 'codeql_sarif_upload file=%s attempt=%d/%d final_status=success\n' \
        "$sarif_file" "$attempt" "$max_attempts"
      uploaded=1
      break
    fi

    response=$(<"$response_file")
    rm -f "$response_file"
    status=$(printf '%s\n' "$response" | sed -nE 's/.*HTTP ([0-9]{3}).*/\1/p' | tail -n 1)
    status=${status:-unknown}
    if is_retryable_status "$status" && ((attempt < max_attempts)); then
      delay=$((backoff_base * (2 ** (attempt - 1))))
      if ((delay > backoff_cap)); then
        delay=$backoff_cap
      fi
      printf '::warning::codeql_sarif_upload file=%s status=%s retry_in=%ss attempt=%d/%d\n' \
        "$sarif_file" "$status" "$delay" "$attempt" "$max_attempts"
      sleep "$delay"
      continue
    fi

    if is_retryable_status "$status"; then
      printf '::error::codeql_sarif_upload file=%s status=%s final_status=failure retryable=true attempts=%d/%d\n' \
        "$sarif_file" "$status" "$attempt" "$max_attempts" >&2
    else
      printf '::error::codeql_sarif_upload file=%s status=%s final_status=failure retryable=false attempts=%d/%d\n' \
        "$sarif_file" "$status" "$attempt" "$max_attempts" >&2
    fi
    rm -f "$encoded_sarif_file"
    printf '%s\n' "$response" >&2
    exit 1
  done

  if ((uploaded == 0)); then
    echo "codeql_sarif_upload: upload loop ended without success" >&2
    exit 1
  fi
done
