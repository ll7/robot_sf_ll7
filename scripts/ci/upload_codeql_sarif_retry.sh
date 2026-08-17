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
: "${GITHUB_SHA:?GITHUB_SHA is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"

max_attempts=${CODEQL_UPLOAD_MAX_ATTEMPTS:-3}
backoff_base=${CODEQL_UPLOAD_BACKOFF_BASE_SECONDS:-5}
backoff_cap=${CODEQL_UPLOAD_BACKOFF_CAP_SECONDS:-30}
if ! [[ "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "codeql_sarif_upload: CODEQL_UPLOAD_MAX_ATTEMPTS must be a positive integer" >&2
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
  uploaded=0
  for ((attempt = 1; attempt <= max_attempts; attempt += 1)); do
    response_file=$(mktemp)
    if jq -n \
      --arg commit_sha "$GITHUB_SHA" \
      --arg ref "$GITHUB_REF" \
      --rawfile sarif "$sarif_file" \
      '{commit_sha: $commit_sha, ref: $ref, sarif: ($sarif | @base64)}' \
      | gh api --method POST "repos/${GITHUB_REPOSITORY}/code-scanning/sarifs" --input - \
      >"$response_file" 2>&1; then
      rm -f "$response_file"
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
    printf '%s\n' "$response" >&2
    exit 1
  done

  if ((uploaded == 0)); then
    echo "codeql_sarif_upload: upload loop ended without success" >&2
    exit 1
  fi
done
