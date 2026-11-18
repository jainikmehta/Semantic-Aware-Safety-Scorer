#!/usr/bin/env bash
# Raw material for the proposal media, one Isaac Sim process at a time (scripts/record_media.py), then the composition.
#   scripts/run_media.sh            # in tmux session "media": window "progress" shows what is done and what is left
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${ISAACLAB_PYTHON:-$HOME/miniforge3/envs/env_isaaclab/bin/python}
LOGS=data/media_raw/logs
mkdir -p "$LOGS"
WIDE="--cam -5.8 -3.2 5.2 -0.3 0.0 0.0"
PARK="--robot -2.0 0.05 10"
run() {  # run <log name> <args...>
  local name=$1; shift
  echo "[$(date +%T)] $name"
  $PY scripts/record_media.py --headless "$@" > "$LOGS/$name.log" 2>&1
  echo "[$(date +%T)] $name: exit $?  $(grep -aE '^STILL|Traceback' "$LOGS/$name.log" | tr '\n' ' ' | cut -c1-400)"
}
while pgrep -f "record_medi[a].py --headless episode" > /dev/null; do sleep 10; done   # the episode first
run stills_person stills --scenario person --name stills_person $WIDE --layouts ps0016,ps0042,ps0081,px0016,ps0022,ps0061
run stills_person_carry stills --scenario person --name stills_person_carry $WIDE $PARK --layouts default --advance CARRY_TO_B2 --advance-extra 45
run stills_gripper stills --scenario gripper_robot --name stills_gripper $WIDE --layouts gs0029,gs0068,gx0032,gx0016,gs0048,gs0087
run stills_ood_person stills --scenario person --name stills_ood_person --env-config configs/env_config_ood_media.yaml $WIDE $PARK --layouts default --advance CARRY_TO_B2 --advance-extra 45
run stills_ood_gripper stills --scenario gripper_robot --name stills_ood_gripper --env-config configs/env_config_ood_media.yaml $WIDE $PARK --layouts default --advance CARRY_TO_B2 --advance-extra 30
echo "[$(date +%T)] OOD video (student v2)"
$PY scripts/record_risk_video.py --headless --scenario person --checkpoint checkpoints/student_v2/best --tag student_v2 \
    --env-config configs/env_config_ood.yaml > "$LOGS/ood_video_v2.log" 2>&1
echo "[$(date +%T)] OOD video: exit $?  $(grep -aE '^RECORDED|Traceback' "$LOGS/ood_video_v2.log")"
echo ALL DONE
