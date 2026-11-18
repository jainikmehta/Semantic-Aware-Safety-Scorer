#!/usr/bin/env bash
# Test the student models in a scene they never saw (configs/env_config_ood.yaml): other buckets, sofa, armchair, ottoman,
# floor, walls and sky. Frames from random layouts of that scene, scored against the policy ranges, the same scoring on
# the normal test frames for reference, then videos with v2 and v1. Runs in tmux session "risk", window "ood".
set -euo pipefail
cd "$(dirname "$0")/.."
ISAAC=$HOME/miniforge3/envs/env_isaaclab/bin/python
TRAIN="env -u PYTHONPATH HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false $HOME/miniforge3/envs/sass_train/bin/python"
LOG=data/risk_v1/logs
OOD=data/risk_ood
mkdir -p $OOD
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

step "1/4 frames of the unseen scene (48 random layouts x 12 views per scenario)"
for sc in person gripper_robot; do
  $ISAAC scripts/generate_risk_dataset.py --headless --scenario $sc --kinds static,extra --limit 48 --root $OOD \
    --env-config configs/env_config_ood.yaml > $LOG/generate_ood_$sc.log 2>&1 &
done
wait
grep -a "GENERATION" $LOG/generate_ood_*.log

step "2/4 v1 and v2 against the policy: unseen scene, then the normal test frames"
$TRAIN scripts/evaluate_vs_policy.py --root $OOD --name ood --checkpoints checkpoints/student_v1/best checkpoints/student_v2/best \
  2>&1 | grep -av "Loading weights" | tee $LOG/policy_eval_ood.log | tail -30
$TRAIN scripts/evaluate_vs_policy.py --root data/risk_v1 --name test --splits test test_heldout \
  --checkpoints checkpoints/student_v1/best checkpoints/student_v2/best 2>&1 | grep -av "Loading weights" | tee $LOG/policy_eval_test.log | tail -30

for tag in student_v2 student_v1; do
  step "videos in the unseen scene with $tag"
  for run in "person default" "gripper_robot default" "person vase_on_ottoman"; do
    set -- $run
    $ISAAC scripts/record_risk_video.py --headless --scenario $1 --scene $2 --env-config configs/env_config_ood.yaml \
      --checkpoint checkpoints/$tag/best --tag $tag > $LOG/record_ood_${tag}_$1_$2.log 2>&1 &
  done
  wait
  grep -a -A16 "^RECORDED" $LOG/record_ood_${tag}_*.log
done
step "ALL DONE"
