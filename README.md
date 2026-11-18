# Semantic Aware Safety Scorer

A robotics safety scoring project built on Isaac Sim 5.1 / Isaac Lab 2.3. It contains the simulation environment with two
scenarios, real-time obstacle segmentation with SAM 3 and the teacher-labelled risk dataset; the student model and the controller are not implemented yet.

## Repository structure

```
configs/env_config.yaml   Scenario, alley, robot, camera, objects, actors, SAM 3, lighting and render settings
envs/alley_env.py         AlleyEnv with the step(v_cmd, w_cmd) API
envs/glass_transfer_actor.py  State machine, routes and glass transfer, shared by both scenarios
envs/person_rig.py        Skinned character: walk cycle, reaching and holding the glass
envs/gripper_robot_rig.py Jackal + UR10e + Robotiq 2F-140: follows the state machine, IK on the glass
envs/ur_kinematics.py     UR10e forward / inverse kinematics and the 2F-140 linkage model
envs/scene_assets.py      Procedural meshes (bucket, glass, vase, walls) and USD helpers
perception/sam3_segmenter.py  SAM 3 client: obstacle masks, on-the-ground test from depth, drawing
perception/sam3_worker.py     SAM 3 inference process (its own Python with transformers >= 5)
assets/turtlebot4/        TurtleBot 4 URDF and meshes with licenses; usd/ is generated at first launch
scripts/test_env.py       Full run to the goal with checks and video recording
scripts/test_physics.py   Friction, robot bumps and the transfer in both placement modes (no video)
perception/scorer.py      SafetyScorer plug-in interface (see "Testing your own scorer")
examples/depth_scorer.py  Baseline scorer to copy
scripts/evaluate_scorer.py  Runs an episode with a scorer and writes a CSV next to ground truth
envs/layout.py            Random scene layouts for the risk dataset (objects, what stands on what, colours, lighting)
configs/risk_config.yaml  Risk rubric, teacher model, dataset sizes, review settings
distillation/teacher.py   Teacher: builds the request (camera image + one close-up per obstacle) and validates the answer
distillation/progress.py  Progress files that every dataset stage writes
scripts/generate_risk_dataset.py  Frames with exact object masks, depth and scene facts, per scenario
scripts/label_risk_dataset.py     Teacher labels for the frames (resumable)
scripts/build_risk_review.py      Label statistics and the review page
scripts/pipeline_progress.py      Live progress with time remaining
scripts/run_risk_pipeline.sh      Runs the stages in the tmux session `risk`
distillation/corrections.py   Corrected labels: reviewed bands per obstacle, reviewer values, misidentified ratings
distillation/student.py   Student: Qwen3-VL-4B + LoRA with one query token per mask and a score head; data and metrics
configs/student_config.yaml  Student model, training and gate settings
scripts/correct_risk_labels.py   Writes the corrected label set the student trains on
scripts/precompute_sam_masks.py  SAM 3 masks for every frame (the student trains on them)
scripts/train_student.py, scripts/evaluate_student.py  Train, then test and time the student
control/                  Placeholder
recordings/               Saved videos
```

## Testing your own scorer

1. Fork or clone this repo, and set up Isaac Sim 5.1 / Isaac Lab 2.3 in your own conda env (`env_isaaclab` below).
2. Copy `examples/depth_scorer.py`, subclass `perception.scorer.SafetyScorer`, and implement `score(obs)`. It returns a float in
   [0, 1] (1 = safe) or a `ScoreResult`. The scorer gets only the sensor observations, never the ground truth.
3. Run it:

```bash
$ISAACLAB_PATH/isaaclab.sh -p scripts/evaluate_scorer.py --scorer examples.my_scorer:MyScorer --headless [--scenario gripper_robot]
```

This writes `outputs/MyScorer_<scenario>_scores.csv` with the score per step beside the true distances to every object and the glass
eccentricity, so you can compare your scores against what actually happened.

## Running the tests

```bash
conda activate env_isaaclab
cd Semantic-Aware-Safety-Scorer
export ISAACLAB_PATH=/path/to/IsaacLab   # your Isaac Lab 2.3 checkout
$ISAACLAB_PATH/isaaclab.sh -p scripts/test_physics.py --headless [--scenario gripper_robot]
$ISAACLAB_PATH/isaaclab.sh -p scripts/test_env.py --headless                            # Person scenario, with SAM 3
$ISAACLAB_PATH/isaaclab.sh -p scripts/test_env.py --headless --scenario gripper_robot   # the robot moves the glass
$ISAACLAB_PATH/isaaclab.sh -p scripts/test_env.py --headless --no-sam3                  # without segmentation
```

Each process exits with status 0 only if every check passes. The first launch converts the robot and downloads the
materials, the sofa, the sky and the character from NVIDIA's asset server; they need network access or a filled local cache.

`recordings/alley_env_test_<scenario>.mp4` (H.264, 1920 x 1440, 30 fps) shows the overview with the robot state and the SAM 3
detections below it, and in the right column the robot's RGB image, its depth image and the SAM 3 masks, all at native
resolution.

`test_env.py` drives the robot to the goal (1050 steps at 0.2 m/s) and verifies:

- RGB and depth have the right shapes, and contain an actual scene (not uniform, not flat).
- The camera faces the robot's heading, with the image's down direction pointing at the floor.
- Projecting each object's ground truth with the intrinsics and extrinsics lands on a pixel with the right depth.
- With SAM 3: both buckets are found where the ground truth projects, as obstacles, and the glass on Bucket_1 is not called
  an obstacle.
- The centre pixel's depth equals the distance to the end wall until the actor starts moving, which also detects a frame of
  lag between image and state.
- The robot travels the commanded distance through physics and stays in its lane.
- Objects have distinct prim paths and the right semantic labels; Blue_Glass starts centred on Bucket_1.
- Blue_Glass goes Bucket_1 -> actor -> Bucket_2, with the hand or the gripper on it throughout, and ends at the configured
  eccentricity.
- The buckets, the vase and the couch stay where they are, and the robot never touches the actor.

`test_physics.py` verifies the lid friction, the bump behaviour below, and the transfer in both placement modes.

## Environment

- **Alley:** 8.0 m x 2.2 m with 1.2 m brick walls, concrete floor, sky lighting, RTX ray tracing at native resolution.
- **Robot:** TurtleBot 4 Standard from its URDF, with a physics-driven differential drive. It starts at (-3.5, 0, 0) with
  the goal at (3.5, 0, 0). Velocity commands are limited to the Create 3's speed and acceleration. Measured tracking:
  v within 0.1%, w within 1.5%.
- **Camera:** at the OAK-D Pro's position from the URDF, 69 deg horizontal field of view, 640 x 480, 30 Hz.

| Object | Prim path | Label | Position | Size |
|---|---|---|---|---|
| Bucket_1 | `/World/Objects/Bucket_1` | `bucket` | (-1.0, 0.35, 0) | r 0.20 m, h 0.40 m, lidded pail |
| Bucket_2 | `/World/Objects/Bucket_2` | `bucket` | (1.2, -0.42, 0) | same as Bucket_1 |
| Blue_Glass | `/World/Objects/Blue_Glass` | `blue_glass` | centred on top of Bucket_1 | r 0.05 m, h 0.12 m, empty, same glass as the vase |
| Glass_Vase | `/World/Objects/Glass_Vase` | `glass_vase` | (-0.8, -0.45, 0) | r 0.12 m, h 0.35 m |
| Soft_Couch | `/World/Objects/Soft_Couch` | `soft_couch` | (1.0, 0.65, 0) | 1.64 m x 0.70 m x 0.66 m, sofa asset, thin metal legs |
| Soft_Chair | `/World/Objects/Soft_Chair` | `soft_chair` | (-1.9, -0.7, 0) | 0.63 m x 0.78 m x 0.78 m, upholstered armchair asset, wooden legs |
| Ottoman | `/World/Objects/Ottoman` | `ottoman` | (-2.05, 0.7, 0) | 0.88 m x 0.55 m x 0.47 m, padded ottoman asset, thin legs |
| Bean_Bag | `/World/Objects/Bean_Bag` | `bean_bag` | (3.2, -0.66, 0) | r 0.42 m, h 0.48 m, procedural, no legs |
| Person | `/World/Objects/Person` | `person` | starts at (-0.2, 0.9, 0), at an alcove in the wall | cylinder collider r 0.25 m, h 1.75 m, skinned character |

Walls, floor, goal marker and robot carry the labels `wall`, `ground`, `goal` and `robot`.

## The Person

A kinematic body, driven by a deterministic state machine (`envs/glass_transfer_actor.py`), shared with the gripper robot:

| State | What happens |
|---|---|
| `IDLE` | Waits until the robot crosses x = -2.9 m |
| `APPROACH_B1` | Walks at 0.55 m/s to beside Bucket_1, leaving the robot's lane free |
| `PICK_GLASS` | Reaches for the glass and lifts it to z = 0.95 m. Parent becomes `Person` when the hand closes |
| `CARRY_TO_B2` | Walks across the alley to beside Bucket_2 |
| `PLACE_GLASS` | Lowers the glass onto the lid, releases it to physics, takes the hand back |
| `EXIT` | Walks around Bucket_2 to (3.8, 0.8, 0) and stops |

`placement_mode` in the config selects `center` (eccentricity 0) or `edge` (0.85, on the side of the robot's path). The target
is computed from the bucket's current pose.

Because it is kinematic, the Person pushes anything it touches with unlimited force. **At startup the routes are checked against
the objects, and the environment refuses to start if one passes too close.** If you move an object, you may have to change
`approach_path`, `carry_path` or `exit_path` in the config.

The character is animated by `envs/person_rig.py`, without the omni.anim.graph runtime: Isaac Sim's walk and idle clips are
retargeted to the character at startup, the walk cycle advances with the distance walked, and inverse kinematics bends the back
and knees and keeps the hand on the glass.

## Scenarios

`scenario` in the config (or `--scenario` on every script, or `AlleyEnv(scenario=...)`) chooses who moves the glass:

| Scenario | Actor (ground truth key) | Body |
|---|---|---|
| `person` (default) | `Person` | The skinned character above |
| `gripper_robot` | `Gripper_Robot` | Clearpath Jackal base, UR10e arm, Robotiq 2F-140 gripper |

Both use the same state machine, trigger, placement modes and checks, so their runs are directly comparable. Each actor has
its own section in the config (`human_actor`, `gripper_robot`) with its own speeds, timings and routes.

## The gripper robot

A mobile manipulator built from Isaac Sim's robot assets: a Clearpath Jackal (0.51 m x 0.43 m) with a UR10e on its top plate
and a Robotiq 2F-140 gripper. The 2F-140 is needed because the glass is 100 mm wide; the 2F-85 and the Franka hand open only
85 and 80 mm.

It starts in the alcove, drives to Bucket_1, reaches over the lid with a horizontal approach, closes the gripper on the glass,
lifts it and drives across the alley to Bucket_2. There it lowers the glass onto the lid, opens the gripper, retracts the
arm and leaves between the couch and Bucket_2 to park beside the goal, out of the robot's lane.

How it is animated (`envs/gripper_robot_rig.py`):

- Like the Person, it is a kinematic box collider (the Jackal's chassis, 0.30 m high) moved along the routes by the state
  machine. Routes are checked at startup against the objects with the real rectangle, and with its bounding circle
  (r 0.33 m) where it turns on the spot.
- The Jackal and the arm are articulations with their colliders off. Each physics step their roots are written to the
  collider's pose, the wheels turn with the distance driven, and the arm's joints come from inverse kinematics.
- `envs/ur_kinematics.py` has the UR10e's forward kinematics, read off the USD joints and matching the simulator to 0.01 mm,
  and a damped-least-squares IK warm-started from the last solution so that the arm stays on one branch.
- The 2F-140's parallel linkage is driven as multiples of `finger_joint`, calibrated in Isaac Sim (the loop closes to
  0.03 mm). The simulator's loop-closure joints are disabled. The calibration also gives the opening and the pad midpoint
  for each angle, so that the fingers close to 98 mm on the glass with the glass centred between the pads.
- `test_env.py` measures the simulated tool centre point against the glass while it is held (`hand_error`).

## SAM 3 obstacle segmentation

[SAM 3](https://ai.meta.com/research/sam3/) segments every instance of a concept named by a short text prompt. Every frame of
the robot's RGB camera goes through SAM 3 with the prompts in `sam3.prompts` ("bucket", "drinking glass", "glass vase",
"sofa", "armchair", "ottoman", "bean bag", "person", "robot"). The image encoder runs once per frame and is shared by all prompts, and the text features are
computed once at startup. Masks found by two prompts are merged, keeping the higher score.

Each mask is then back-projected with the depth image and the camera pose. An object is an **obstacle** when its lowest
visible point is below `sam3.ground.max_base_height` (0.35 m, the TurtleBot's height): some part of it is low enough to hit.
The glass standing on a bucket (base 0.40 m) or held at carrying height is detected but not counted; if it falls to the floor
it is counted. The video fills the obstacles' masks in colour and outlines the other detections in grey, with label, score and
distance.

Why not "touches the floor"? The soft furniture (couch on thin metal legs, armchair on wooden legs, ottoman on thin legs, bean
bag with none) was added to test whether SAM 3 leaves thin legs out of its masks. Scored against Isaac Sim's instance masks on
1200 frames of both scenarios: it mostly does not (95 % of the ottoman's leg pixels are in its mask). But with a 0.12 m
"touches the floor" test the couch passed in only 71 % of the frames it was detected, and in 86 % of its failures no part of the
couch near the floor was visible at all, behind other objects or below the image. The robot's height as the limit fixes that
and still excludes the glass:

| Object (frames detected) | legs / base in SAM's mask | lowest point < 0.12 m | < 0.35 m (used) |
|---|---|---|---|
| Soft_Couch (386) | 81 % | 71 % | 100 % |
| Ottoman (194) | 95 % | 94 % | 100 % |
| Soft_Chair (201) | 94 % | 100 % | 100 % |
| Bean_Bag (215) | 98 % | 100 % | 100 % |
| Bucket_2 (447) | 97 % | 95 % | 100 % |
| Person (289) | 95 % | 94 % | 97 % (only the upper body in view) |
| Gripper_Robot (278) | 96 % | 94 % | 94 % (only the arm in view) |
| Blue_Glass on a bucket / held (116 / 93) | - | 0 % | 0 % (correct) |

SAM 3 found the couch, armchair and ottoman in 81-99 % of the frames, but often under another prompt's name: the ottoman is
mostly called "sofa". The bean bag is found once it is close (52 % of the frames, from the far end of the alley).

SAM 3 runs in a worker process (`perception/sam3_worker.py`) with its own Python, because it needs transformers >= 5 and
Isaac Lab 2.3 ships transformers 4.x. Set `sam3.python` to an interpreter with torch and transformers >= 5.

Weights: `sam3.model` is a Hugging Face repo id or a local folder in the transformers format.

- With access to the gated `facebook/sam3` (accept the license, `hf auth login`), set it to `"facebook/sam3"`.
- Or convert Meta's original `sam3.pt` (image model + video tracker) into a local folder with transformers' official converter
  ([`convert_sam3_to_hf.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam3/convert_sam3_to_hf.py)).
  That converter expects the image model's keys under `sam3_model.`, while `sam3.pt` has them under `detector.`, so add
  `r"^detector\.": r""` as the first entry of its key mapping. The result must report no missing keys; the `tracker.*` and
  `sam2_convs` keys are unused by the image model. Converted this way from the official `sam3.pt`, the model gave detections
  identical to `facebook/sam3` (same labels, scores and boxes).

SAM 3.1 (`facebook/sam3.1`, checkpoint only, no transformers release) mainly speeds up multi-object video tracking, but it
also ships a fine-tuned detector. The same converter turns `sam3.1_multiplex.pt` into a `Sam3Model` folder: its only missing
weights are the fourth FPN level, which the detector does not use. Run it with
`scripts/test_env.py --sam3_model ~/models/sam3.1_hf`. Scored against Isaac Sim's instance masks on 420 robot-camera frames
of both scenarios (score threshold 0.4), the two are almost the same:

| | mean mask IoU | objects found (IoU >= 0.5) | found with the right label | false positives |
|---|---|---|---|---|
| SAM 3 | 0.857 | 88.9 % | 80.9 % | 4 |
| SAM 3.1 | 0.862 | 89.6 % | 80.9 % | 5 |

SAM 3.1 is a little better on the small transparent glass (IoU 0.44 -> 0.49 in the robot scenario, 0.52 -> 0.56 with the
Person) and a little worse on the couch (0.82 -> 0.81). Both rarely call the glass a "drinking glass" (mostly "glass vase" or
"bucket"); the on-the-ground test does not depend on the label. Speed is the same.

`sam3.compile: true` compiles the model with CUDA graphs: about 1.3x faster, at the cost of about a minute at start-up (the
model loads while the scene is built). The image encoder (about 19 ms) is shared; each prompt adds about 3 ms. On an RTX PRO
6000 Blackwell a frame takes about 38 ms with 6 prompts and 49 ms with the 9 configured (20 FPS); drop prompts you do not need
to get closer to the camera's 30 FPS.

```python
from perception.sam3_segmenter import Sam3Segmenter, draw_detections
segmenter = Sam3Segmenter(env.cfg["sam3"])     # loads the model in the worker
result = segmenter.segment(obs)                 # sync: this frame's masks
for d in result.obstacles:                      # detections low enough to hit
    print(d.label, d.score, d.distance, d.mask.shape)
```

`sam3.mode: async` (with `submit(obs)` and `latest()`) never blocks the simulation and returns the newest finished result;
the recordings use `sync`, so that every frame shows its own masks.

## Semantic risk dataset

Every obstacle gets a risk in [0, 1]: how bad it would be if the TurtleBot bumped into it, from the rubric in
`configs/risk_config.yaml` (soft furniture near 0.1, an empty bucket near 0.25, anything carrying a glass 0.6-0.85, a person
near 1). The large teacher model (Qwen3-VL-32B on the lab's LLM service) labels simulator frames with it; a small student
model will later be trained on those labels to give the scores in real time.

```bash
scripts/run_risk_pipeline.sh all     # frames for both scenarios, then teacher labels; in the tmux session "risk"
tmux attach -t risk                  # window "progress": bars, elapsed time and time left per stage
```

Stages, each resumable: `generate` (130 random layouts x 12 viewpoints and 6 transfer episodes per scenario, about 10 min),
`review` (64 frames chosen to cover every kind of obstacle and arrangement, labelled first, about 3 min), `label` (all frames,
about 2.5 h at 2-3 s per frame). Output in `data/risk_v1/`: `frames/<split>/<layout>/`, `labels/v1_hints/`, and
`review/v1_hints/review.html` (the review frames with their ratings; tick the ones you disagree with and copy the notes).

The split `test_heldout` holds arrangements no other split has (the glass or the vase on the ottoman), to test whether
the student learns "fragile thing on top" rather than "bucket with glass".

What the pilot (64 frames) showed about the teacher:

| Teacher input | Bucket with a glass rated in its band (0.60-0.85) | Glass on it seen |
|---|---|---|
| 640 x 480 image + close-ups | 27 % (mean 0.38) | 27 % |
| 1920 x 1440 image + close-ups | 29 % (mean 0.39) | 55 % |
| 640 x 480 + scene hints | 100 % (mean 0.75) | 100 % |

The thin, transparent glass on the lid is hard for the teacher to see in a robot-height view, so `teacher.hints` tells it
what the simulator knows stands on each obstacle or is held by it, when that thing is visible in the frame. This is
privileged information for the teacher only: the student sees only the image. Hints say nothing about risk, so the rating is
still the teacher's judgement of the rubric. Obstacles that are mostly hidden behind others (under 25 % visible, estimated
from their size and distance) are not rated, because the teacher could not tell what they were.

### Corrected labels

The review of the teacher's labels (2026-11-11) found the teacher rated another robot and a glass on the floor too low.
`corrections.bands` in `configs/risk_config.yaml` holds the reviewed range per kind of obstacle; `scripts/correct_risk_labels.py`
keeps a teacher rating inside its range, clips one outside it, replaces it with the typical rating when the teacher described
a different object (e.g. "person's legs" for a bucket behind a person: 221 cases), and applies the reviewer's own values
(`review/v1_hints/overrides.txt`). Of 9,589 ratings: 7,569 kept, 1,788 clipped, 221 replaced, 11 reviewer values. The
teacher's labels stay untouched; the result is `labels/v1_hints_corrected/`. Rubric v2 carries the review into future labels.

## Student risk model

Qwen3-VL-4B, run like a vision-language-action model with the action decoder replaced by a score head: the camera image and an
instruction go in, followed by one query token per SAM 3 obstacle mask (the mask-weighted average of the image tokens under
it, plus its box and size). The language model (LoRA, 42 M trainable parameters) lets every query attend to the whole image,
and the head reads each query's hidden state as a risk and four attributes. No text is generated.

```bash
PY="env -u PYTHONPATH $HOME/miniforge3/envs/sass_train/bin/python"   # conda env sass_train: a clone of risl with peft
$PY scripts/precompute_sam_masks.py    # 5 min
$PY scripts/train_student.py           # 4 epochs, 10 min
$PY scripts/evaluate_student.py        # writes checkpoints/student_v1/best/evaluation.json
```

Results of `checkpoints/student_v1/best` (SAM 3's masks; error against the corrected labels):

| Split | Obstacles | MAE | Within 0.10 | Pairs ranked right |
|---|---|---|---|---|
| test | 843 | 0.047 | 86 % | 99.8 % |
| test_heldout (glass or vase on the ottoman) | 480 | 0.137 | 79 % | 80.6 % |

One frame with 6 obstacles takes 29 ms (median, image preprocessing included), next to SAM 3's 49 ms. SAM 3 finds 92 % of
the labelled obstacles; the rest are not rated.

The held-out split fails: an ottoman with a glass or vase on it gets 0.10 (target 0.65-0.75), the same as an empty ottoman,
and the student never says it carries something fragile. In training only buckets ever carry anything, so the student
learned "bucket with a glass" rather than "anything with something fragile on top". The fix is training layouts where the
glass and the vase stand on the couch, the armchair and the bean bag too, with the ottoman still held out.

The Qwen3-VL patch embedding is a Conv3d that takes 5 s per image in bfloat16 on the RTX PRO 6000;
`student.linear_patch_embed` replaces it with the equivalent matrix product (identical to 3e-6).

## Friction and bumps

A bucket has two colliders, because a material applies to a whole collider: a grippy base (1.4 / 1.1) and a slippery lid
(0.06 / 0.02). When materials with different combine modes touch, PhysX uses the higher-priority mode
(average < min < multiply < max), which is what `physics_materials` in the config relies on.

The lid's dynamic friction of 0.02 is far lower than real plastic and glass (about like ice). That is deliberate: the bucket is
pushed from under the glass and the glass slips by up to v^2 / (2 mu g). In the head-on bump at 0.25 m/s the glass stayed on at 0.025
and fell at 0.02. A hand calculation suggests that an upright glass will not tip over at these speeds, whatever the friction; that
was not tested separately. The result depends on how long the robot pushes (0.8 s in the test):

| Glass | Bump | Result |
|---|---|---|
| Edge (0.85) | 0.25 - 0.35 m/s, head-on or glancing | Knocked off |
| Edge (0.85) | 0.20 m/s | Stays head-on, knocked off when glancing |
| Centre (0) | up to 0.35 m/s | Stays, sliding towards the edge (eccentricity 0.5 at 0.35 m/s head-on) |
| Centre (0) | 0.46 m/s | Knocked off head-on; stays at eccentricity 0.68 when glancing |

The bucket travels about as far as the robot pushes it (0.15 - 0.31 m head-on at 0.20 - 0.35 m/s in the test), because the robot's
drive easily overcomes the friction. A glass that slides off towards the robot can land on the robot's top plate instead of the
floor (z = 0.40 m); its parent is None either way.

## API

```python
env = AlleyEnv()              # after the Isaac Sim app is launched
obs = env.reset()
obs = env.step(v_cmd=0.2, w_cmd=0.0)   # advances 1/30 s
env.close()
```

| Key | Type | Content |
|---|---|---|
| `rgb` | `torch.uint8 (480, 640, 3)` | Colour image |
| `depth` | `torch.float32 (480, 640)` | `distance_to_image_plane` in metres, `inf` where nothing was hit (sky) |
| `camera_intrinsics` | `np.ndarray (3, 3)` | K |
| `camera_extrinsics` | `np.ndarray (4, 4)` | Camera to world, ROS optical frame (+Z forward, +X right, +Y down) |
| `robot_state` | `np.ndarray (5,)` | `[x, y, theta, v, w]` of `base_link`, from the simulator |
| `ground_truth_objects` | `dict` | Per object: `pose` (`position`, `orientation` as w, x, y, z), `velocity` [vx, vy], `shape_type` (`cylinder` or `box`), `size` [lx, ly, lz], `radius`, `height`, `parent`, `children`, `semantic_label`, `prim_path`. Blue_Glass also has `eccentricity_xy` |

Ground truth positions are object centres. `parent` and `children` describe the current state. Blue_Glass has parent `Bucket_1`
or `Bucket_2` while it stands on that bucket, `Person` while held, otherwise None. `eccentricity_xy` is
\|p_glass_xy - p_bucket_xy\| / R_bucket, and None unless the glass stands on a bucket. For Soft_Couch use `size` and the
orientation: its `radius` is only the bounding circle of its footprint.

## Modelling limits

- Objects are rigid bodies: the glass and the vase do not break, and the couch does not deform.
- Depth is ideal: it has no sensor noise and it reads the surface of the glass and the vase, whereas a real stereo camera often
  fails on glass.
- At eccentricity 0.85 the glass's outer edge overhangs the top of the bucket by about 2 cm; its centre of mass is still supported.
- The Person's collider is a cylinder, so it does not follow the arms or a bent back, and it reaches 5 cm into the end wall at
  (3.8, 0.8). Walls do not push a kinematic body, so this has no physical effect.
- While passing Bucket_2 on the wall side, the Person's collider reaches about 3 cm into the wall (the gap is 0.48 m and the
  collider 0.50 m wide). Widening the gap between the objects will change this route.
- The sprung wheel-drop joints are rigid and the wheel colliders are spheres. The caster is a fixed sphere with very low
  friction rather than a rolling ball.
- `AlleyEnv.close()` must not call `sim.stop()`: with RTX cameras alive that call never returned.
- The gripper robot collides only through its chassis box: the arm and the gripper pass through anything, as the Person's
  arms do. The glass is attached kinematically while gripped, not held by friction.
- SAM 3's obstacle test uses the lowest visible point, so an object seen only above the robot's height (a person's upper body
  close up, the robot's arm) is not counted, and one whose lower part is hidden behind a nearer one can be
  missed as an obstacle.
- The floor's collider must stay a plane: on a large box collider the wheels lost contact and one sank through the floor.
