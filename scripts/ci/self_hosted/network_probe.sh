#!/usr/bin/env bash
# Run inside the dedicated bridge before starting any runner supervisor.
set -euo pipefail

[[ $# -eq 2 ]] || exit 2
gateway="$1"
internal_address="$2"

# Confirm ICMP works in this probe before treating failed pings as isolation.
ping -n -c 1 -W 2 127.0.0.1 >/dev/null 2>&1 || {
  echo "Network probe cannot perform ICMP checks" >&2
  exit 1
}

if ping -n -c 1 -W 2 "$gateway" >/dev/null 2>&1; then
  echo "Host gateway is reachable from the runner network" >&2
  exit 1
fi
if ping -n -c 1 -W 2 "$internal_address" >/dev/null 2>&1; then
  echo "University address is reachable from the runner network" >&2
  exit 1
fi
for site in github.com pypi.org; do
  curl --fail --location --silent --show-error --connect-timeout 5 --max-time 15 \
    --output /dev/null "https://$site/" || {
      echo "Runner network cannot reach $site" >&2
      exit 1
    }
done
