#!/usr/bin/env bash
# Run a command in a Linux network namespace with no usable network interface.
#
# The unprivileged user-namespace path keeps local runs under the invoking user. GitHub-hosted
# runners normally require the sudo path because unprivileged user namespaces are disabled there.
# Both paths fail closed: the command is never run in the ambient network namespace.
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "network-deny sandbox requires Linux unshare --net" >&2
    exit 125
fi

if command -v unshare >/dev/null 2>&1 \
    && unshare --user --map-root-user --net -- true >/dev/null 2>&1; then
    exec unshare --user --map-root-user --net -- \
        /usr/bin/env UV_OFFLINE=1 UV_NO_SYNC=1 "$@"
fi

# GitHub-hosted Ubuntu runners provide passwordless sudo. Resolve the command through the caller's
# PATH without preserving any other environment variable into the privileged namespace.
if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    exec sudo -n /usr/bin/env PATH="$PATH" /usr/bin/unshare --net -- \
        /usr/bin/env UV_OFFLINE=1 UV_NO_SYNC=1 "$@"
fi

echo "network-deny sandbox unavailable: need unshare user namespace or passwordless sudo" >&2
exit 125
