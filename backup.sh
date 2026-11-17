#!/usr/bin/env bash
# Snapshot code, git history, protocol, results and trained runs (not data, which is regenerable)
# to the session file store, which survives a reset of the compute environment.
set -u
cd "$(dirname "$0")"
DEST=/mnt/user-data/outputs/.admod_backup
mkdir -p "$DEST"
tar czf /tmp/admod_snapshot.tgz --exclude='./data' --exclude='*/__pycache__' --exclude='./.pytest_cache' . \
  && cp /tmp/admod_snapshot.tgz "$DEST/admod_snapshot.tgz.tmp" && mv "$DEST/admod_snapshot.tgz.tmp" "$DEST/admod_snapshot.tgz" \
  && echo "backup $(date -Is) $(du -h /tmp/admod_snapshot.tgz | cut -f1)"
