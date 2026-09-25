#!/bin/bash
export PROTCOMPASS_ROOT="${PROTCOMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"  # release: repository root
# Wait for the running master runner, then re-run it so the stage-2 tasks are picked up.
cd "$PROTCOMPASS_ROOT/protcompass/extract"
while pgrep -f "xargs -a jobs.txt" > /dev/null; do sleep 60; done
echo "$(date +%T) stage-1 runner finished, starting stage 2" >> logs/run_all.log
bash run_all.sh 3 >> logs/run_all_stdout.log 2>&1
