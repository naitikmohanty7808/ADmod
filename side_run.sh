#!/bin/bash
# Run a CPU-heavy side job with training paused (avoids thread thrashing on 2 cores), then resume training.
trap 'pkill -CONT -f "admod.train"' EXIT
pkill -STOP -f "admod.train"
echo "paused training $(date -Is): $*" >> runs_ax/logs/pauses.log
"$@"
echo "resumed training $(date -Is)" >> runs_ax/logs/pauses.log
