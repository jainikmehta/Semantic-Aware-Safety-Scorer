# Proposal media: context-aware safety in dynamic environments

All material comes from Isaac Sim 5.1 runs of our project. The risk scores shown are the actual outputs of our compact
vision-language safety scorer (VLSS: Qwen3-VL-4B + LoRA, one score per SAM 3 obstacle mask), computed live from the
TurtleBot 4's camera. The simulator only supplies the object names in the labels and the scene-graph relations.

## Figures (`figures/`)

| File | What it shows |
|---|---|
| `risk_shift.pdf` / `.png` | **The requested figure.** Initial, in-between and final setup of the mobile-manipulator scene: simulator view, robot camera with VLSS scores, scene graph, and the risk of each object over time. |
| `risk_shift_compact.pdf` / `.png` | Same without the scene-graph row, for tighter space. |
| `panels/` | The three stages as separate full-resolution images (simulator, robot camera + VLSS, scene graph), to re-arrange freely. |
| `panels/*_simulator_wide.png` | The same three moments seen from the viewpoint of the controller video (`videos/nmpc_semantic_vs_uniform.mp4`), which shows the whole alley: 1920 × 1200 each. Same run, same moments as the other panels. |
| `risk_shift_simulator_wide.png` | Those three wide views side by side with their (a)/(b)/(c) titles. |
| `scene_collage.pdf` / `.png` | Six other scenes, none of them in the training data: random test layouts, a held-out arrangement, a person carrying the glass, and a scene with new object models and materials. |
| `risk_colour_scale.png` | The colour scale (0 harmless, green, to 1 worst, red), if a figure needs it separately. |
| `nmpc_semantic_person_default_t18_top.png`, `nmpc_uniform_person_default_t18_top.png` | Top views of the risk-aware controller: obstacle circles coloured by risk, dashed risk margins, planned path. |

### Suggested captions

**risk_shift.** Context-dependent risk in a dynamic scene. A mobile manipulator (Clearpath Jackal with a UR10e arm) moves
a drinking glass from Bucket 1 to Bucket 2 while a TurtleBot 4 watches. The middle row shows the TurtleBot's camera, with
each SAM 3 obstacle mask coloured by its VLSS risk score.
(a) Bucket 1 holds the glass and is the riskiest object (0.73); Bucket 2 is empty (0.16).
(b) While the manipulator carries the glass, it becomes the riskiest object (0.87), and Bucket 1 drops to 0.17.
(c) With the glass on Bucket 2, Bucket 2 is rated 0.76 and Bucket 1 0.18.
The scene graph gives the support relation behind each change. The bottom plot shows each object's score over time: dots
are single frames, lines are the smoothed scores the controller uses. SAM 3 takes 48 ms and VLSS 35 ms per frame (median,
RTX PRO 6000).

**scene_collage.** VLSS scores in scenes it was not trained on. The same object gets a different score depending on what it
carries: a bucket with a vase on it is 0.80, while empty buckets are 0.16–0.20; a sofa with a vase on it is 0.81; an
ottoman with a vase on it is 0.84, an arrangement held out of training entirely. The bottom row is a scene with new
models of the pails, sofa and chairs, and new floor and wall materials; a pail model never seen in training is rated 0.78
when the glass stands on it, and 0.17–0.20 when empty. Over a full drive through this unseen scene
(`videos/vlss_unseen_scene.mp4`), VLSS's score is inside the reviewed policy range on 90–100 % of frames for every
object type.

## Videos (`videos/`)

| File | Length | What it shows |
|---|---|---|
| `risk_shift.mp4` | 24 s | The full manipulator episode behind the figure: simulator view, robot camera with live VLSS scores, live scene graph, risk timeline with a cursor. |
| `nmpc_semantic_vs_uniform.mp4` | 25 s | The risk-aware CBF-NMPC controller driving the TurtleBot through the alley twice, side by side. Left: each object gets its policy score. Right: every object gets 0.5. With semantic risk the robot keeps 0.11 m from the bucket holding the glass instead of 0.01 m, and 0.40 m from the person instead of 0.33 m. Neither run has a contact. |
| `vlss_unseen_scene.mp4` | 35 s | The TurtleBot drives through the unseen scene (new models and materials) while a person moves the glass. Panels: overview, camera, SAM 3 masks, VLSS scores, and a table comparing each score with the policy range. |

## Things to be aware of

- **Vase removed.** In the manipulator episode, the glass vase that normally stands on the floor was left out so the
  figure tells one story. With it, the vase on the floor would be the riskiest object throughout (about 0.95).
- **Manipulator scores high on its own.** The rubric rates any robot 0.60–0.80, so the manipulator stays around 0.67
  even without the glass. In (c), Bucket 2 with the glass (0.76) is still above it.
- **Bucket 2 flickers during placing (13.5–16 s).** While the gripper is still around the glass, the bucket is partly
  hidden, and its single-frame scores jump. The smoothing the controller uses damps this; the dots in the plot show it.
- **The controller video uses policy scores, not VLSS.** It uses the simulator's objects with scores from the reviewed
  policy, to test the controller on its own. The full on-robot pipeline (SAM 3 + tracker + VLSS driving the controller)
  runs, but it still stalls in one narrow gap, so it is not shown. See the section below.

## The reviewed policy, and what the controller video does and does not show

**The reviewed policy** is a table of allowed risk ranges for each kind of object, given what it carries
(`configs/risk_config.yaml`, `corrections.bands`):

| Object (and what it carries) | Allowed range |
|---|---|
| couch, armchair, ottoman, bean bag | 0.00 – 0.15 |
| bucket (empty) | 0.15 – 0.35 |
| gripper robot | 0.65 – 0.75 |
| gripper robot carrying a glass or vase | 0.80 – 0.90 |
| anything with a glass or vase on it (bucket, ottoman, …) | 0.60 – 0.85 |
| glass or vase on the floor | 0.80 – 0.90 |
| person (with or without the glass) | 0.90 – 1.00 |

It comes from the labelling rubric plus a human review of the teacher's labels. It is called "reviewed" because the
reviewer's corrections went into it: for example, the gripper robot at 0.65–0.75, and fragile items on the floor at
0.80–0.90. It has three uses:

1. Correcting the teacher (Qwen3-VL-32B) labels before VLSS is trained.
2. Checking VLSS: the "inside / OUTSIDE" marks in the videos, and the "inside the policy range on 90–100 % of frames"
   figure.
3. Supplying risk to the controller in the controller test.

**The controller video.** `scripts/run_cbf_nmpc.py` has three risk modes:

| Mode | Where the obstacles come from | Where the risk comes from |
|---|---|---|
| `policy` (left half of the video) | The simulator's exact object positions and sizes | The midpoint of the object's range above (e.g. bucket with a glass 0.725, person 0.95) |
| `uniform` (right half of the video) | The simulator | 0.5 for every object |
| `student` (the full robot pipeline) | SAM 3 masks + depth image + obstacle tracker | VLSS, live, per mask |

The video therefore answers one question: *given correct risk scores, does the controller use them correctly?* It does.
The TurtleBot keeps more distance from the bucket holding the glass (0.11 m instead of 0.01 m) and from the person
(0.40 m instead of 0.33 m), with no contacts. It does **not** show the full chain from camera to SAM 3 to VLSS to the
controller.

**Why the full pipeline (`student`) is not shown.** It runs, but in the person scenario the TurtleBot stalls in the narrow
gap between the ottoman and the armchair. It touches nothing; it just stops making progress. The likely cause is the
tracker: it builds the obstacle circles from depth points, and they come out larger than the objects' true footprints,
which closes the gap. Fixing this is an open item.

**How to describe it in the proposal.** The two halves are each demonstrated separately:

- VLSS produces scores that agree with the policy: the main figure, the collage and the unseen-scene video.
- The controller turns policy scores into suitable clearances: the controller video.

Stating that "VLSS drives the controller" would overclaim until the tracker stall is fixed.

## Regenerating

    scripts/run_media.sh                                           # Isaac Sim runs (tmux session "media"), raw data in data/media_raw/
    ~/miniforge3/envs/env_isaaclab/bin/python scripts/compose_media.py all   # figures and videos in Media/
