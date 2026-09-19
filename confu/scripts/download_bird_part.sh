#!/bin/bash
# Resumable download of one Bird-MML file. Usage: download_bird_part.sh <filename>
set -u
FNAME="$1"
TARGET="data/bird_mml/$1"
URL="https://zenodo.org/api/records/18920487/files/$1/content"
until curl -sfL -C - -o "$TARGET" "$URL"; do sleep 3; done
echo "COMPLETE $FNAME $(stat -c%s "$TARGET")"
