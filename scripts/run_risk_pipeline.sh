#!/usr/bin/env bash
# Runs the risk dataset pipeline in a tmux session, with a live progress window.
#
#   scripts/run_risk_pipeline.sh generate   # frames for both scenarios (two Isaac Sim processes side by side)
#   scripts/run_risk_pipeline.sh review     # teacher labels for the review frames, then the review sheet
#   scripts/run_risk_pipeline.sh label      # teacher labels for every frame, then the review sheet again
#   scripts/run_risk_pipeline.sh all        # generate, then label everything
#
#   tmux attach -t risk     watch it: window "progress" shows what is done and how long the rest takes,
#                           the other windows show each stage's log (Ctrl-b n: next window, Ctrl-b d: detach)
#
# Every stage skips what is already there, so running a stage again continues it.
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION=${RISK_SESSION:-risk}
PY=${ISAACLAB_PYTHON:-$HOME/miniforge3/envs/env_isaaclab/bin/python}   # Isaac Lab's Python; the label stages only need numpy + OpenCV
ROOT=$($PY -c "import sys; sys.path.insert(0, '.'); from distillation import teacher as T; print(T.dataset_root(T.load_config()))")
mkdir -p "$ROOT/logs"
STAGE=${1:-all}

window() {  # window <name> <command>: run the command in its own tmux window and keep the window open afterwards
  tmux kill-window -t "$SESSION:$1" 2>/dev/null || true
  tmux new-window -d -t "$SESSION" -n "$1" "$2; echo; echo '[$1 finished: exit status '\$?']'; exec bash"
}

tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n progress "$PY scripts/pipeline_progress.py; exec bash"
tmux list-windows -t "$SESSION" -F '#W' | grep -qx progress || tmux new-window -d -t "$SESSION" -n progress "$PY scripts/pipeline_progress.py; exec bash"

if [[ $STAGE == generate || $STAGE == all ]]; then
  for scenario in person gripper_robot; do
    window "gen-$scenario" "$PY scripts/generate_risk_dataset.py --headless --scenario $scenario 2>&1 | tee '$ROOT/logs/generate_$scenario.log' | grep --line-buffered -aE '^LAYOUT|^GENERATION|^Traceback|^[A-Za-z]*Error:|^  File'"
  done
fi
SHEET="$PY scripts/build_risk_review.py"
if [[ $STAGE == review ]]; then
  window labels "$PY scripts/label_risk_dataset.py --subset review --wait 2>&1 | tee '$ROOT/logs/label_review.log' && $SHEET"
fi
if [[ $STAGE == label || $STAGE == all ]]; then
  # the review frames first, with their sheet, so they can be checked while the rest is labelled
  window labels "$PY scripts/label_risk_dataset.py --subset review --wait 2>&1 | tee '$ROOT/logs/label_review.log'; $SHEET; $PY scripts/label_risk_dataset.py 2>&1 | tee -a '$ROOT/logs/label_all.log'; $SHEET"
fi
echo "Started '$STAGE' in tmux session '$SESSION'. Watch with: tmux attach -t $SESSION"
