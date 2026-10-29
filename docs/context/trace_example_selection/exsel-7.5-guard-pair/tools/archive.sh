#!/usr/bin/env bash
# Copy a raw run directory to the host archive with a sha256 MANIFEST and verify the copy.
# usage: archive.sh <run_dir> <archive_name>
set -euo pipefail
SRC=$(realpath "$1"); NAME=$2
DST=$HOME/archive/experiments/exscr2/$NAME
( cd "$SRC" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
mkdir -p "$DST"
cp -a "$SRC"/. "$DST"/
( cd "$DST" && sha256sum --quiet -c MANIFEST.sha256 && echo "ARCHIVE_VERIFIED $(wc -l < MANIFEST.sha256) files" )
sha256sum "$DST/MANIFEST.sha256" | cut -d' ' -f1
du -sh "$DST" | cut -f1
