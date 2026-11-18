#!/usr/bin/env bash
# The full benchmark in tmux session "bench": two lanes of Isaac Sim processes side by side, a progress window, and the
# summary at the end (data/benchmark/summary.md).
#
#   scripts/run_benchmarks.sh            # start (runs already in data/benchmark/*.jsonl are skipped: safe to restart)
#   tmux attach -t bench                 # watch: window "progress" shows every benchmark and the time left
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${ISAACLAB_PYTHON:-$HOME/miniforge3/envs/env_isaaclab/bin/python}
SESSION=bench
LOGS=data/benchmark/logs
mkdir -p "$LOGS"
B="$PY scripts/run_benchmark.py --headless"
lane1="$B --family gate_context --episodes 30 --video-episodes 2 > $LOGS/gate_context.log 2>&1; \
       $B --family passby --episodes 36 --video-episodes 1 > $LOGS/passby.log 2>&1; \
       $B --family transfer --scenario person --episodes 15 --video-episodes 1 > $LOGS/transfer_person.log 2>&1"
lane2="$B --family gate_class --episodes 30 --video-episodes 2 > $LOGS/gate_class.log 2>&1; \
       $B --family transfer --scenario gripper_robot --episodes 15 --video-episodes 1 > $LOGS/transfer_gripper.log 2>&1"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session $SESSION exists: attach with tmux attach -t $SESSION"; exit 1; }
tmux new-session -d -s "$SESSION" -n progress "$PY scripts/pipeline_progress.py --root data/benchmark; exec bash"
tmux new-window -d -t "$SESSION" -n lane1 "$lane1; echo LANE1 DONE; exec bash"
tmux new-window -d -t "$SESSION" -n lane2 "$lane2; echo LANE2 DONE; exec bash"
tmux new-window -d -t "$SESSION" -n summary "while tmux list-windows -t $SESSION -F '#W' | grep -q lane && \
  ! (tmux capture-pane -pt $SESSION:lane1 | grep -q 'LANE1 DONE' && tmux capture-pane -pt $SESSION:lane2 | grep -q 'LANE2 DONE'); do sleep 60; done; \
  python3 scripts/summarize_benchmark.py gate_context gate_class passby transfer_person transfer_gripper_robot; exec bash"
echo "started: tmux attach -t $SESSION"
