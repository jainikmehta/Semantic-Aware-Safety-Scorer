#!/usr/bin/env bash
# Student v2: extra scenes (glass or vase on the couch and armchair seats), teacher labels for them, retraining, evaluation
# of v1 and v2 on the same test splits, and the same videos as for v1. Runs in tmux session "risk", window "v2";
# progress in window "progress".
set -euo pipefail
cd "$(dirname "$0")/.."
ISAAC=$HOME/miniforge3/envs/env_isaaclab/bin/python
TRAIN="env -u PYTHONPATH HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false $HOME/miniforge3/envs/sass_train/bin/python"
LOG=data/risk_v1/logs
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

step "1/8 extra scenes for both scenarios"
$ISAAC scripts/generate_risk_dataset.py --headless --scenario person --kinds extra > $LOG/generate_extra_person.log 2>&1 &
$ISAAC scripts/generate_risk_dataset.py --headless --scenario gripper_robot --kinds extra > $LOG/generate_extra_gripper_robot.log 2>&1 &
wait
grep -a "GENERATION" $LOG/generate_extra_*.log

step "2/8 teacher (Qwen3-VL-32B) on the lab LLM service"
/opt/llm/service/llm start qwen
step "3/8 teacher labels for the new frames (rubric v2)"
$ISAAC scripts/label_risk_dataset.py --skip-labelled-in v1_hints 2>&1 | tee $LOG/label_v2.log | grep -aE "labelled|FAILED|DONE"
step "4/8 stop the LLM service again (frees the GPU for training)"
/opt/llm/service/llm stop

step "5/8 corrected labels and SAM 3 masks for the new frames"
$ISAAC scripts/correct_risk_labels.py
$ISAAC scripts/build_risk_review.py --labels v2_corrected | head -24
$TRAIN scripts/precompute_sam_masks.py 2>&1 | grep -aE "DONE|Error"

step "6/8 train student v2"
$TRAIN scripts/train_student.py 2>&1 | grep -av "Loading weights" | tee $LOG/train_student_v2.log | grep -aE "^step|DONE|trainable|Error"

step "7/8 evaluate v1 and v2 on the same test splits"
$TRAIN scripts/evaluate_student.py --checkpoint checkpoints/student_v1/best 2>&1 | grep -av "Loading weights" | tee $LOG/eval_student_v1_on_v2data.log | tail -32
$TRAIN scripts/evaluate_student.py --checkpoint checkpoints/student_v2/best 2>&1 | grep -av "Loading weights" | tee $LOG/eval_student_v2.log | tail -32

step "8/8 videos with student v2"
for run in "person default" "gripper_robot default" "person vase_on_ottoman"; do
  set -- $run
  $ISAAC scripts/record_risk_video.py --headless --scenario $1 --scene $2 --checkpoint checkpoints/student_v2/best --tag student_v2 \
    > $LOG/record_v2_$1_$2.log 2>&1 &
done
wait
grep -a -A16 "^RECORDED" $LOG/record_v2_*.log
step "ALL DONE"
