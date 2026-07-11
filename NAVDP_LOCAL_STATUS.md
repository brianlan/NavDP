# NavDP Local Status and Runbook

Last updated: 2026-07-11

This document records the current local setup for running the NavDP benchmark and exporting NavDP model subgraphs for NVIDIA Orin NX deployment.

## Current Status

- Main working repository: `/home/rlan/projects/NavDP`
- The benchmark and export work currently depends on this NavDP repository root.
- `code-internnav` was inspected in the old non-version-controlled workspace only to understand model/checkpoint differences. It is not used by the working benchmark run.
- Python environment used: `/ssd4/envs/navdp_py310/bin/python`
- PyTorch in that env: `2.7.1+cu128`
- GPU tested: `NVIDIA GeForce RTX 5090`
- IsaacSim/IsaacLab benchmark startup works with the current local patch in `eval_pointgoal_wheeled.py`.
- No packages were installed, upgraded, uninstalled, or modified in `/ssd4/envs/navdp_py310` during this setup.

## Jetson Orin NX Deployment Status

Updated on 2026-07-06:

- Target Jetson: `10.133.109.123`
- SSH access works from this workstation.
- Jetson TensorRT version: `10.3.0`
- Four NavDP point-goal TensorRT subgraphs were built and run on the Jetson:
  - `rgbd_encoder`
  - `pointgoal_encoder`
  - `pointgoal_denoiser`
  - `critic`
- The original denoiser ONNX needed a TensorRT compatibility patch because PyTorch exported fixed-shape `nn.MultiheadAttention` as `If` control flow with mismatched branch ranks. Patched ONNX:

```text
/home/rlan/projects/NavDP/exports/navdp_onnx/navdp_pointgoal_denoiser_b1_s16_trt.onnx
```

- Deterministic host-vs-Jetson parity passed with FP32 engines built using `--noTF32` at `atol=1e-3, rtol=1e-3`.
- FP16 engines also built and ran. They preserve the selected trajectory indices for the deterministic parity input but have larger intermediate tensor drift.
- The RGBD encoder has also been split for cached RGB history deployment:
  - `rgb_frame_encoder`: current RGB frame to 256 tokens
  - `depth_frame_encoder`: current depth to 256 tokens
  - `rgbd_fusion`: 8 cached RGB token blocks plus current depth tokens to `rgbd_embed`
- Cached PyTorch parity and cached TensorRT parity both pass `atol=1e-3, rtol=1e-3`, with exact selected trajectory indices.
- Cache availability is currently limited to the TensorRT parity harness through
  `tools/run_navdp_tensorrt_parity.py --cached-rgbd`. The benchmark NavDP Flask
  server and `NavDP_Agent` still use the monolithic 8-frame RGBD encoder; they do
  not yet expose a production cache toggle.
- When the production runtime is integrated, the intended behavior is a single
  `--rgb-token-cache`-style option: disabled uses the existing monolithic path;
  enabled stores seven previous `[B, 256, 384]` RGB token blocks, encodes only the
  newest RGB frame, encodes current depth, and fuses all eight RGB token blocks
  with the current depth tokens. The per-environment cache must reset on both
  navigator reset and environment reset to prevent cross-episode history leakage.
- Detailed notes, commands, and measured parity numbers are in:

```text
/home/rlan/projects/NavDP/NAVDP_JETSON_DEPLOY_EVAL.md
```

## Important Local Paths

### Repositories

- NavDP benchmark repo: `/home/rlan/projects/NavDP`
- InternNav repo: not part of this working root and not used for the benchmark.
- IsaacLab checkout: `/home/rlan/projects/NavDP/IsaacLab-v1.2.0`

### Model Checkpoint

- Standalone NavDP checkpoint: `/ssd4/models/NavDP/navdp-cross-modal.ckpt`

This checkpoint is compatible with `baselines/navdp`.

Validation summary:

- Loaded as a PyTorch `OrderedDict`.
- Compatible with `NavDP_Policy(image_size=224, memory_size=8, predict_size=24, temporal_depth=16, heads=8, token_dim=384)`.
- All expected model keys match shape.
- Four extra auxiliary keys are present and ignored by `strict=False`:
  - `pixel_aux_head.weight`
  - `pixel_aux_head.bias`
  - `image_aux_head.weight`
  - `image_aux_head.bias`

### Scene Assets

Benchmark scene assets are stored under:

```text
/home/rlan/projects/NavDP/assets/scenes
```

Downloaded/extracted Scene-N1 assets include:

```text
/home/rlan/projects/NavDP/assets/scenes/cluttered_hard
/home/rlan/projects/NavDP/assets/scenes/cluttered_easy
/home/rlan/projects/NavDP/assets/scenes/SkyTexture
/home/rlan/projects/NavDP/assets/scenes/Materials
```

For the validated smoke run, the actual selected scene was:

```text
/home/rlan/projects/NavDP/assets/scenes/cluttered_hard/hard_1/cluttered-1.usd
/home/rlan/projects/NavDP/assets/scenes/cluttered_hard/hard_1/pointgoal_start_goal_pairs.npy
```

The point-goal `.npy` has shape `(100, 5)` and each row is:

```text
[start_x, start_y, goal_x, goal_y, initial_yaw]
```

The RGB/depth inputs are generated live by IsaacSim from the USD scene and the Dingo robot camera. They are not loaded from a pre-rendered RGBD dataset.

### Generated Outputs

Smoke benchmark output:

```text
/home/rlan/projects/NavDP/pointgoal_navdp_cluttered_hard/hard_1/fps_0.mp4
/home/rlan/projects/NavDP/pointgoal_navdp_cluttered_hard/hard_1/metric.csv
```

Smoke benchmark metric:

```csv
success,spl,distance
1.0,0.9793099643128591,12.471094
```

ONNX exports:

```text
/home/rlan/projects/NavDP/exports/navdp_onnx/navdp_rgbd_encoder_b1_s16.onnx
/home/rlan/projects/NavDP/exports/navdp_onnx/navdp_pointgoal_denoiser_b1_s16.onnx
/home/rlan/projects/NavDP/exports/navdp_onnx/navdp_critic_b1_s16.onnx
```

## How To Run A PointGoal Benchmark

Use two terminals.

### Terminal 1: Start The NavDP Server

```bash
cd /home/rlan/projects/NavDP/baselines/navdp

CONDA_PREFIX=/ssd4/envs/navdp_py310 \
/ssd4/envs/navdp_py310/bin/python navdp_server.py \
  --port 8890 \
  --checkpoint /ssd4/models/NavDP/navdp-cross-modal.ckpt
```

The server binds to:

```text
http://127.0.0.1:8890
```

Useful smoke check:

```bash
curl -sS -X POST http://127.0.0.1:8890/navigator_reset \
  -H 'Content-Type: application/json' \
  -d '{"intrinsic":[[1,0,0],[0,1,0],[0,0,1]],"batch_size":1,"stop_threshold":-3}'
```

Expected response:

```json
{"algo":"navdp"}
```

### Terminal 2: Run One-Episode Evaluation

```bash
cd /home/rlan/projects/NavDP

TERM=xterm CONDA_PREFIX=/ssd4/envs/navdp_py310 \
/ssd4/envs/navdp_py310/bin/python eval_pointgoal_wheeled.py \
  --port 8890 \
  --scene_dir /home/rlan/projects/NavDP/assets/scenes/cluttered_hard \
  --scene_index 8 \
  --scene_scale 1.0 \
  --num_envs 1 \
  --num_episodes 1 \
  --speed 0.5
```

Notes:

- `CONDA_PREFIX=/ssd4/envs/navdp_py310` is important so IsaacLab uses the intended env.
- `TERM=xterm` avoids terminal capability warnings.
- `eval_pointgoal_wheeled.py` now filters `--scene_dir` to subdirectories that contain a valid `.usd` file and sorts them naturally by name. This avoids selecting stray non-scene folders such as an empty `scenes/` directory.
- For `cluttered_hard`, `--scene_index 8` now selects `hard_8`:

```text
0 hard_0
1 hard_1
2 hard_2
3 hard_3
4 hard_4
5 hard_5
6 hard_6
7 hard_7
8 hard_8
9 hard_9
```

For `cluttered_easy`, `--scene_index 8` selects `easy_8`.

Verify the scene index before larger runs:

```bash
CONDA_PREFIX=/ssd4/envs/navdp_py310 /ssd4/envs/navdp_py310/bin/python - <<'PY'
import os, re

def scene_sort_key(name):
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part for part in parts]

scene_dir = "/home/rlan/projects/NavDP/assets/scenes/cluttered_hard"
names = []
for name in os.listdir(scene_dir):
    path = os.path.join(scene_dir, name)
    if os.path.isdir(path) and any(f.endswith(".usd") and "noMDL" not in f for f in os.listdir(path)):
        names.append(name)
for i, name in enumerate(sorted(names, key=scene_sort_key)):
    print(i, name)
PY
```

## Where The Benchmark Input Comes From

The point-goal eval script does the following:

1. Selects a scene subdirectory from `--scene_dir` by filtering to directories that contain a valid `.usd`, sorting those names naturally, and indexing that filtered list with `--scene_index`.
2. Uses `utils_tasks.basic_utils.find_usd_path()` to find:
   - the `.usd` scene file
   - the task-specific `pointgoal_start_goal_pairs.npy`
3. Loads the USD into IsaacSim as the terrain.
4. Spawns the Dingo robot and front camera.
5. Resets the robot and goal using rows from the `.npy`.
6. Reads live IsaacLab observations:
   - `infos['observations']['rgb']`
   - `infos['observations']['depth']`
   - `infos['observations']['goal_pose']`
7. Sends RGB, depth, and point goal to the NavDP Flask server.
8. Receives predicted trajectories and critic scores.
9. Uses MPC to convert the selected trajectory into wheel commands.
10. Writes visualization video and metrics.

Relevant files:

```text
/home/rlan/projects/NavDP/eval_pointgoal_wheeled.py
/home/rlan/projects/NavDP/utils_tasks/basic_utils.py
/home/rlan/projects/NavDP/utils_tasks/client_utils.py
/home/rlan/projects/NavDP/configs/tasks/wheeled_task.py
/home/rlan/projects/NavDP/configs/robots/dingo_config.py
```

## Local Patch For IsaacSim Replicator

`eval_pointgoal_wheeled.py` currently preloads:

```python
import boto3
import botocore
import s3transfer
```

before:

```python
from omni.isaac.lab.app import AppLauncher
```

Reason: IsaacSim prepends bundled pip archives at runtime. Without this preload, Replicator camera initialization can mix an older bundled `boto3` with a different `botocore`, leading to camera initialization failure and the later symptom:

```text
AttributeError: 'Camera' object has no attribute '_is_outdated'
```

This patch fixed the benchmark startup on this machine.

## Model Structure In This Checkpoint

The standalone `navdp-cross-modal.ckpt` contains the weights needed by the benchmark model:

- `rgbd_encoder`: DepthAnythingV2-style RGB/depth visual encoders plus NavDP fusion transformer.
- point/image/pixel goal encoders for the cross-modal policy.
- diffusion denoiser/action model:
  - transformer decoder
  - `action_head`
- critic/ranking model:
  - transformer decoder path
  - `critic_head`

For point-goal benchmark inference, the main path is:

```text
RGB history + current depth + point goal
  -> rgbd_encoder
  -> point_encoder
  -> 10-step diffusion denoising
  -> critic scores
  -> select best trajectory
  -> MPC wheel commands
```

## How To Export ONNX

Exporter added:

```text
/home/rlan/projects/NavDP/tools/export_navdp_onnx.py
```

Run:

```bash
cd /home/rlan/projects/NavDP

CONDA_PREFIX=/ssd4/envs/navdp_py310 \
/ssd4/envs/navdp_py310/bin/python tools/export_navdp_onnx.py \
  --checkpoint /ssd4/models/NavDP/navdp-cross-modal.ckpt \
  --output-dir /home/rlan/projects/NavDP/exports/navdp_onnx \
  --batch-size 1 \
  --sample-num 16
```

Outputs:

```text
navdp_rgbd_encoder_b1_s16.onnx
navdp_pointgoal_denoiser_b1_s16.onnx
navdp_critic_b1_s16.onnx
```

The exporter also runs `onnx.checker.check_model()` on each exported file.

Important export detail:

- The RGBD encoder export requires `torch.backends.mha.set_fastpath_enabled(False)`.
- Without that, PyTorch exports `aten::_native_multi_head_attention`, which is unsupported by ONNX opset 17.

## Orin NX Deployment Notes

The practical Orin NX path is to convert the ONNX subgraphs to TensorRT engines on the Orin board, not to export the entire Flask server as one model.

On the Orin NX, convert with the TensorRT version installed on that board:

```bash
trtexec --onnx=navdp_rgbd_encoder_b1_s16.onnx \
  --saveEngine=navdp_rgbd_encoder_b1_s16_fp16.engine \
  --fp16

trtexec --onnx=navdp_pointgoal_denoiser_b1_s16.onnx \
  --saveEngine=navdp_pointgoal_denoiser_b1_s16_fp16.engine \
  --fp16

trtexec --onnx=navdp_critic_b1_s16.onnx \
  --saveEngine=navdp_critic_b1_s16_fp16.engine \
  --fp16
```

Host-side code on Orin still needs to implement:

- camera RGB/depth preprocessing
- NavDP memory queue
- point-goal clipping and embedding
- diffusion scheduler loop, currently 10 DDPM denoising steps
- trajectory `cumsum`
- critic ranking
- robot/controller integration

## Dependencies And Environment Policy

No dependency installation or mutation was performed during this setup.

The following command pattern was used for all Python execution:

```bash
CONDA_PREFIX=/ssd4/envs/navdp_py310 /ssd4/envs/navdp_py310/bin/python ...
```

If future package changes are needed, do not mutate the env silently. Create a requirements file or ask for manual installation.

## Network/Download Notes

When network access is slow or unreliable, use proxy:

```text
http://127.0.0.1:18080
```

Scene-N1 gated assets were downloaded using authenticated HTTP access and extracted locally. Do not store access tokens in this document.

## Known Caveats

- `scene_index` is based on naturally sorted valid scene directories in `eval_pointgoal_wheeled.py`. Verify mapping before long benchmark sweeps.
- The benchmark outputs can be large because each episode writes video.
- IsaacSim logs many warnings in headless mode, including display/window warnings. These did not prevent the validated smoke run.
- Default headless camera rendering is optimized for throughput, not smooth demo video. The loaded IsaacLab kit is `IsaacLab-v1.2.0/source/apps/isaaclab.python.headless.rendering.kit`, which sets `rtx.directLighting.sampledLighting.samplesPerPixel = 1` and disables several quality features. This can make RGB frames look speckled or bumpy.
- `cluttered_easy/easy_8` also uses the USD material `Fabric_Carpet_Long_Floor`, so its floor is intrinsically textured/wavy and is not visually comparable with the smooth glossy floor shown in `assets/images/demo_nogoal.gif`.
- The ONNX files are fixed-shape exports for batch size 1 and sample count 16. Export again if the deployment shape changes.
- TensorRT engine files should be built on the target Orin NX, because TensorRT engines are hardware/software-version specific.

## Optional Higher-Quality Render Check

`eval_pointgoal_wheeled.py` supports an opt-in render-quality override through `NAVDP_RENDER_QUALITY=high`. This does not change dependencies or the Python environment. It only appends Kit RTX settings before IsaacSim starts:

```bash
cd /home/rlan/projects/NavDP
NAVDP_RENDER_QUALITY=high TERM=xterm CONDA_PREFIX=/ssd4/envs/navdp_py310 \
  /ssd4/envs/navdp_py310/bin/python eval_pointgoal_wheeled.py \
  --port 8890 \
  --scene_dir /home/rlan/projects/NavDP/assets/scenes/cluttered_easy \
  --scene_index 8 \
  --scene_scale 1.0 \
  --num_envs 1 \
  --num_episodes 1 \
  --speed 0.5
```

Comparison files from the render investigation:

```text
/home/rlan/projects/NavDP/pointgoal_navdp_cluttered_easy/easy_8/fps_0_default_quality_before_render_check.mp4
/home/rlan/projects/NavDP/pointgoal_navdp_cluttered_easy/easy_8/fps_0_high_quality_render_check.mp4
```

The high-quality run reduced the RGB high-frequency edge/noise score on extracted frames from roughly 21-25 to roughly 16-18 in the left camera view, confirming that the default renderer settings contribute to the visual artifact. The remaining floor roughness is from the selected scene material itself.
