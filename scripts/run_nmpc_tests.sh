#!/usr/bin/env bash
# CBF-NMPC runs: simulator obstacles with policy risks vs one uniform risk (the controller on its own), then the full
# pipeline (SAM 3 + tracker + student v2), in both scenarios. Usage: scripts/run_nmpc_tests.sh [gt|student|all]
set -uo pipefail
cd "$(dirname "$0")/.."
ISAAC=$HOME/miniforge3/envs/env_isaaclab/bin/python
LOG=data/risk_v1/logs
WHAT=${1:-all}
run() { $ISAAC scripts/run_cbf_nmpc.py --headless "$@" > "$LOG/nmpc_$(echo "$@" | tr ' -' '__').log" 2>&1; }
if [[ $WHAT == gt || $WHAT == all ]]; then
  for sc in person gripper_robot; do
    run --scenario $sc --risk policy & run --scenario $sc --risk uniform & wait
  done
fi
if [[ $WHAT == student || $WHAT == all ]]; then
  run --scenario person --risk student & run --scenario gripper_robot --risk student & wait
fi
grep -ah "^RESULT\|^min clearance\|Traceback\|Error:" $LOG/nmpc_*.log
echo "NMPC TESTS DONE"
