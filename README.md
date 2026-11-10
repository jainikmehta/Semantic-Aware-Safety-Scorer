# Semantic Aware Safety Scorer

A robotics safety scoring project built on Isaac Sim 5.1 / Isaac Lab 2.3. So far it contains the simulation environment;
the vision model and the controller are not implemented yet.

## Repository structure

```
configs/env_config.yaml   Alley, robot, camera, object, lighting and render settings
envs/alley_env.py         AlleyEnv with the step(v_cmd, w_cmd) API
envs/human_glass_actor.py The Person's state machine, routes and glass transfer
envs/person_rig.py        Skinned character: walk cycle, reaching and holding the glass
envs/scene_assets.py      Procedural meshes (bucket, glass, vase, walls) and USD helpers
assets/turtlebot4/        TurtleBot 4 URDF and meshes with licenses; usd/ is generated at first launch
scripts/test_env.py       Full run to the goal with checks and video recording
scripts/test_physics.py   Friction, robot bumps and the transfer in both placement modes (no video)
perception/scorer.py      SafetyScorer plug-in interface (see "Testing your own scorer")
examples/depth_scorer.py  Baseline scorer to copy
scripts/evaluate_scorer.py  Runs an episode with a scorer and writes a CSV next to ground truth
control/ distillation/    Placeholders
recordings/               Saved videos
```

## Testing your own scorer

1. Fork or clone this repo, and set up Isaac Sim 5.1 / Isaac Lab 2.3 in your own conda env (`env_isaaclab` below).
2. Copy `examples/depth_scorer.py`, subclass `perception.scorer.SafetyScorer`, and implement `score(obs)`. It returns a float in
   [0, 1] (1 = safe) or a `ScoreResult`. The scorer gets only the sensor observations, never the ground truth.
3. Run it:

```bash
$ISAACLAB_PATH/isaaclab.sh -p scripts/evaluate_scorer.py --scorer examples.my_scorer:MyScorer --headless
```

This writes `outputs/MyScorer_scores.csv` with the score per step beside the true distances to every object and the glass
eccentricity, so you can compare your scores against what actually happened.

## Running the tests

```bash
conda activate env_isaaclab
cd Semantic-Aware-Safety-Scorer
export ISAACLAB_PATH=/path/to/IsaacLab   # your Isaac Lab 2.3 checkout
$ISAACLAB_PATH/isaaclab.sh -p scripts/test_physics.py --headless
$ISAACLAB_PATH/isaaclab.sh -p scripts/test_env.py --headless
```

Each process exits with status 0 only if every check passes. The first launch converts the robot and downloads the
materials, the sofa, the sky and the character from NVIDIA's asset server; they need network access or a filled local cache.

`recordings/alley_env_test.mp4` (H.264, 1920 x 1040, 30 fps) shows the overview, the robot's RGB image and its depth image
at native resolution, with the robot state below.

`test_env.py` drives the robot to the goal (1050 steps at 0.2 m/s) and verifies:

- RGB and depth have the right shapes, and contain an actual scene (not uniform, not flat).
- The camera faces the robot's heading, with the image's down direction pointing at the floor.
- Projecting each object's ground truth with the intrinsics and extrinsics lands on a pixel with the right depth.
- The centre pixel's depth equals the distance to the end wall until the Person starts moving, which also detects a frame of
  lag between image and state.
- The robot travels the commanded distance through physics and stays in its lane.
- Objects have distinct prim paths and the right semantic labels; Blue_Glass starts centred on Bucket_1.
- Blue_Glass goes Bucket_1 -> Person -> Bucket_2, with the hand on it throughout, and ends at the configured eccentricity.
- The buckets, the vase and the couch stay where they are, and the robot never touches the Person.

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
| Soft_Couch | `/World/Objects/Soft_Couch` | `soft_couch` | (1.0, 0.65, 0) | 1.64 m x 0.70 m x 0.66 m, sofa asset |
| Person | `/World/Objects/Person` | `person` | starts at (-0.2, 0.9, 0), at an alcove in the wall | cylinder collider r 0.25 m, h 1.75 m, skinned character |

Walls, floor, goal marker and robot carry the labels `wall`, `ground`, `goal` and `robot`.

## The Person

A kinematic body, driven by a deterministic state machine (`envs/human_glass_actor.py`):

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
- The floor's collider must stay a plane: on a large box collider the wheels lost contact and one sank through the floor.
