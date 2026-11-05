# Semantic Aware Safety Scorer

A modular robotics safety evaluation and scoring framework built with **Isaac Sim** and **Isaac Lab**.

---

## 📁 Repository Structure

```
Semantic Aware Safety Scorer/
├── configs/
│   ├── __init__.py
│   └── env_config.yaml         # Alley dimensions, robot parameters, camera, and object definitions
├── envs/
│   ├── __init__.py
│   └── alley_env.py            # Standalone AlleyEnv with TurtleBot 4 & semantic object attachments
├── perception/
│   └── __init__.py             # Vision & semantic perception modules (placeholders)
├── control/
│   └── __init__.py             # Robot controllers & safety scoring interfaces (placeholders)
├── distillation/
│   └── __init__.py             # Model distillation & safety scoring network (placeholders)
├── scripts/
│   ├── __init__.py
│   └── test_env.py             # Verification script: 200-step rollout, RGB-D verification & video recording
├── recordings/                 # Saved simulation videos (.mp4)
├── .gitignore
└── README.md
```

---

## 🏛️ Environment Details (`envs/alley_env.py`)

- **Corridor Geometry**: Narrow alley ($8.0\,\text{m} \times 2.2\,\text{m}$) enclosed by 4 boundary walls (1.2m height) and a flat ground plane.
- **Differential-Drive Robot**: TurtleBot 4 spawned at $(-3.5, 0.0, 0.0)$ heading toward goal at $(3.5, 0.0, 0.0)$.
- **Sensor Rig**: Front-mounted RGB-D camera streaming synchronized RGB ($640 \times 480$) and Depth (`distance_to_image_plane`) at 30 Hz.
- **Odometry**: Tracks robot state vector $[x, y, \theta, v, \omega]$.
- **Semantic Objects**:
  1. `Bucket_1`: Cylinder asset ($r = 0.20\,\text{m}, h = 0.40\,\text{m}$) at $(-1.0, 0.35, 0.0)$.
  2. `Bucket_2`: Identical cylinder ($r = 0.20\,\text{m}, h = 0.40\,\text{m}$) at $(1.2, -0.30, 0.0)$.
  3. `Blue_Glass`: Transparent cup ($r = 0.05\,\text{m}, h = 0.12\,\text{m}$) containing blue mesh, centered on top of `Bucket_1` ($z = 0.46\,\text{m}$). Parent: `Bucket_1`.
  4. `Glass_Vase`: Fragile glass cylinder asset ($r = 0.12\,\text{m}, h = 0.35\,\text{m}$) at $(-0.8, -0.45, 0.0)$.
  5. `Soft_Couch`: Soft rectangular/capsule sofa asset ($0.85\,\text{m} \times 0.50\,\text{m} \times 0.50\,\text{m}$) along the wall at $(1.0, 0.65, 0.0)$.

### Clean API: `step(v_cmd, w_cmd)`

Calling `env.step(v_cmd, w_cmd)` returns a dictionary:

```python
obs = {
    "rgb": torch.Tensor,                # Shape: [480, 640, 3] or [480, 640, 4]
    "depth": torch.Tensor,              # Shape: [480, 640] (distance_to_image_plane)
    "camera_intrinsics": np.ndarray,    # Shape: [3, 3]
    "camera_extrinsics": np.ndarray,    # Shape: [4, 4] (camera pose in world frame)
    "robot_state": np.ndarray,          # [x, y, theta, v, w]
    "ground_truth_objects": {           # 3D poses, bounding radii, parent-child relations
        "Bucket_1": {"pose": ..., "radius": 0.20, "parent": None, "children": ["Blue_Glass"], "semantic_label": "bucket"},
        "Bucket_2": {"pose": ..., "radius": 0.20, "parent": None, "children": [], "semantic_label": "bucket"},
        "Blue_Glass": {"pose": ..., "radius": 0.05, "parent": "Bucket_1", "children": [], "semantic_label": "blue_glass"},
        "Glass_Vase": {"pose": ..., "radius": 0.12, "parent": None, "children": [], "semantic_label": "glass_vase"},
        "Soft_Couch": {"pose": ..., "radius": 0.25, "parent": None, "children": [], "semantic_label": "soft_couch"},
    }
}
```

---

## 🚀 Running the Verification Script

### Running inside your tmux session with `risl` / Isaac Lab:

```bash
# 1. Activate conda environment
conda activate risl
# (or conda activate env_isaaclab depending on your Isaac Lab installation)

# 2. Navigate to project root
cd "/home/mehtaj3/Documents/scripts/Semantic Aware Safety Scorer"

# 3. Run test script using Isaac Lab launcher
/home/mehtaj3/Documents/scripts/IsaacLab/isaaclab.sh -p scripts/test_env.py --headless --steps 200

# To view the simulation live in GUI mode:
/home/mehtaj3/Documents/scripts/IsaacLab/isaaclab.sh -p scripts/test_env.py --steps 200
```

### Video Recording Output
The test script creates `recordings/alley_env_test.mp4` featuring:
- **Left Main Window**: Simulation overview (spectator camera overlooking the alley).
- **Right Panel (Top)**: Robot front RGB camera stream ($640 \times 480$).
- **Right Panel (Bottom)**: Robot front Depth stream (`distance_to_image_plane`) colorized with Turbo colormap.
- **Overlays**: Real-time HUD displaying robot odometry $[x, y, \theta, v, \omega]$ and tracked semantic objects.
