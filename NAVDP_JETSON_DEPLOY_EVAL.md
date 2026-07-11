# NavDP Jetson Orin NX Deployment Evaluation

Last updated: 2026-07-06

Target board: Jetson Orin NX 16G at `10.133.109.123`.

## Current Conclusion

The point-goal NavDP inference path is deployable to Orin NX as TensorRT subgraphs, but exact host-vs-Jetson result matching requires one extra runtime constraint: the diffusion random tensors must be explicit inputs or generated from a controlled, identical RNG stream.

The current PyTorch server is stochastic:

1. `predict_pointgoal_action()` samples the initial action tensor with `torch.randn`.
2. `DDPMScheduler.step()` adds fresh random variance noise for timesteps `9..1`.

Therefore, the same image/depth/point-goal does not uniquely define the output trajectory. A valid parity test must compare with fixed `initial_action` and fixed per-step `variance_noise`.

## Jetson Validation Results

Validated on `10.133.109.123`:

```text
host: nvidia-desktop
kernel: Linux 5.15.148-tegra aarch64
TensorRT: 10.3.0
CUDA device: Orin, compute capability 8.7
Python: 3.10.12
```

TensorRT engines were built on the Jetson under:

```text
~/navdp_deploy/engines
```

The original denoiser ONNX did not parse in TensorRT because PyTorch exported fixed-shape `nn.MultiheadAttention` as `If` control flow with mismatched branch ranks. The TensorRT-compatible denoiser is:

```text
exports/navdp_onnx/navdp_pointgoal_denoiser_b1_s16_trt.onnx
```

It was created with:

```bash
CONDA_PREFIX=/ssd4/envs/navdp_py310 \
/ssd4/envs/navdp_py310/bin/python tools/patch_navdp_denoiser_onnx_for_trt.py \
  --input exports/navdp_onnx/navdp_pointgoal_denoiser_b1_s16.onnx \
  --output exports/navdp_onnx/navdp_pointgoal_denoiser_b1_s16_trt.onnx
```

FP32 with TensorRT default TF32 enabled ran successfully but exceeded the strict `1e-3` parity threshold:

```text
max_abs roughly 0.002 to 0.004 on major tensors
```

FP32 with `--noTF32` passed `atol=1e-3, rtol=1e-3` for all compared tensors:

```text
rgbd_embed:          max_abs 3.86298e-04
goal_embed:          max_abs 0
noise_pred_steps:   max_abs 1.63361e-04
action_outputs:     max_abs 1.03250e-04
critic_values:      max_abs 5.42402e-05
all_trajectory:     max_abs 3.89405e-05
selected_trajectory:max_abs 3.89405e-05
selected_indices:   exact match
```

FP16 engines also built and ran. They were faster and preserved `selected_indices` for the deterministic test input, but intermediate diffusion tensors drifted more:

```text
rgbd_embed max_abs 0.0180581
noise_pred_steps max_abs 0.124982
action_outputs max_abs 0.126133
critic_values max_abs 0.00873709
selected_trajectory max_abs 0.00163215
selected_indices exact match
```

Measured with `tools/run_navdp_tensorrt_parity.py` over the deterministic synthetic bundle:

```text
FP32 --noTF32: ~2.25 s per full pipeline
FP16:          ~0.84 s per full pipeline
```

The measured "full pipeline" includes RGBD encoder, point-goal encoder, 10 denoiser engine calls plus DDPM host steps, critic, cumsum, and ranking. It also includes this Python harness's allocation/copy overhead, so it is a correctness-oriented number, not an optimized runtime-server latency.

## Cached RGB History Validation

Implemented a split RGBD path to support caching the previous RGB history tokens:

```text
rgb_frame_encoder:
  image [1, 224, 224, 3] -> rgb_tokens [1, 256, 384]

depth_frame_encoder:
  depth [1, 224, 224, 1] -> depth_tokens [1, 256, 384]

rgbd_fusion:
  rgb_tokens [1, 2048, 384]
  depth_tokens [1, 256, 384]
  -> rgbd_embed [1, 128, 384]
```

Relevant code:

```text
baselines/navdp/policy_backbone.py
tools/check_navdp_cached_rgbd_parity.py
tools/run_navdp_tensorrt_parity.py --cached-rgbd
```

### Current Toggle and Runtime Status

`--cached-rgbd` is available only in the TensorRT parity harness. It switches
between the monolithic `rgbd_encoder` engine and the split RGB-frame, depth, and
fusion engines.

The benchmark NavDP Flask server and `NavDP_Agent` do not yet use these split
engines and do not currently provide a production cache switch. The intended
runtime interface is one `--rgb-token-cache`-style option:

```text
off: monolithic 8-frame RGBD encoder
on:  RGB token FIFO + current RGB frame encoder + current depth encoder + fusion
```

With caching enabled, each environment owns a FIFO of seven prior RGB token
blocks, each shaped `[B, 256, 384]`. At each planning update, the runtime should:

1. encode only the newest `B=1` RGB image into one token block;
2. evict the oldest token block and append the newest one;
3. encode current depth;
4. concatenate the eight RGB token blocks to `[B, 2048, 384]` and run fusion.

During initial history fill, the cache must use token blocks from the same black
image padding used by the original agent. The cache must reset for both a global
navigator reset and an individual environment reset; otherwise prior-episode
visual history would leak into the next episode.

PyTorch cached RGBD path vs original monolithic PyTorch path passed `atol=1e-3, rtol=1e-3`:

```text
rgbd_embed:          max_abs 3.85821e-04
noise_pred_steps:   max_abs 1.65209e-04
action_outputs:     max_abs 1.04949e-04
critic_values:      max_abs 5.57899e-05
all_trajectory:     max_abs 3.97526e-05
selected_trajectory:max_abs 3.97526e-05
selected_indices:   exact match
```

The small nonzero difference comes from encoding eight RGB frames as one batch in the original path versus encoding frames one at a time in the cached path.

Split cached TensorRT FP32 `--noTF32` path vs host PyTorch monolithic bundle also passed `atol=1e-3, rtol=1e-3`:

```text
rgbd_embed:          max_abs 3.84986e-04
noise_pred_steps:   max_abs 1.63972e-04
action_outputs:     max_abs 1.03816e-04
critic_values:      max_abs 5.62668e-05
all_trajectory:     max_abs 3.93353e-05
selected_trajectory:max_abs 3.93353e-05
selected_indices:   exact match
```

The cached TensorRT correctness harness averaged about `2.14 s` over 5 full runs, but this timing still encodes all 8 RGB frames each run to populate the cache. In a real rolling-cache runtime, only the newest frame should call `rgb_frame_encoder`; the previous 7 frame-token blocks should be reused from the FIFO cache.

Measured on the Jetson with seven token blocks already cached, FP32 `--noTF32`:

```text
monolithic RGBD engine:       0.359557 s, 2.781 FPS
cached steady RGBD engines:   0.109529 s, 9.130 FPS
RGBD-only speedup:            3.28x

monolithic full pipeline:     1.845818 s, 0.542 FPS
cached steady full pipeline:  1.594555 s, 0.627 FPS
full-pipeline speedup:        1.16x
```

The full pipeline remains dominated by the ten denoiser calls and critic call,
so RGB caching materially improves the visual stage but does not by itself yield
a proportional end-to-end trajectory-rate gain.

## Exported Subgraphs

ONNX artifacts in `exports/navdp_onnx`:

```text
navdp_rgbd_encoder_b1_s16.onnx
navdp_pointgoal_encoder_b1_s16.onnx
navdp_pointgoal_denoiser_b1_s16.onnx
navdp_critic_b1_s16.onnx
```

The point-goal encoder export was added because the deployed point-goal path needs that linear layer too. The existing denoiser ONNX starts after goal embedding.

Fixed shapes:

```text
rgbd_encoder:
  images [1, 8, 224, 224, 3]
  depths [1, 224, 224, 1]
  rgbd_embed [1, 128, 384]

pointgoal_encoder:
  point_goal [1, 3]
  goal_embed [1, 1, 384]

pointgoal_denoiser:
  last_actions [16, 24, 3]
  timestep [1]
  goal_embed [16, 1, 384]
  rgbd_embed [16, 128, 384]
  noise_pred [16, 24, 3]

critic:
  trajectory [16, 24, 3]
  rgbd_embed [16, 128, 384]
  critic_values [16]
```

## Host-Side Runtime Still Needed

TensorRT engines cover neural network inference only. The Jetson host process still needs to implement:

1. RGB and depth preprocessing to the exact `224x224` BGR/float formats.
2. 8-frame RGB memory queue.
3. Point-goal clipping: `x in [0, 10]`, all coordinates clipped to `[-10, 10]`.
4. Goal/rgbd embedding repeat from batch `1` to sample count `16`.
5. 10 DDPM reverse steps using timesteps `[9,8,7,6,5,4,3,2,1,0]`.
6. Trajectory `cumsum(action / 4.0)`.
7. Short-trajectory masking for final length `< 0.5`.
8. Critic ranking and top trajectory selection.
9. Robot/controller integration.

## Parity Bundle

Generated deterministic test bundle:

```text
exports/navdp_jetson_parity/navdp_jetson_parity_bundle.npz
exports/navdp_jetson_parity/manifest.json
```

Create it with:

```bash
cd /home/rlan/projects/NavDP
CONDA_PREFIX=/ssd4/envs/navdp_py310 \
/ssd4/envs/navdp_py310/bin/python tools/make_navdp_jetson_parity_bundle.py \
  --checkpoint /ssd4/models/NavDP/navdp-cross-modal.ckpt \
  --output-dir /home/rlan/projects/NavDP/exports/navdp_jetson_parity \
  --batch-size 1 \
  --sample-num 16 \
  --seed 1234
```

For a real captured frame, pass an NPZ containing:

```text
images     float32 [1, 8, 224, 224, 3]
depths     float32 [1, 224, 224, 1]
point_goal float32 [1, 3]
```

Use `--input-npz <path>` to build expected outputs for that exact input.

## Jetson Build Commands

After copying `exports/navdp_onnx` to the Jetson, first build FP32 engines for parity:

```bash
trtexec --onnx=navdp_rgbd_encoder_b1_s16.onnx \
  --saveEngine=navdp_rgbd_encoder_b1_s16_fp32.engine

trtexec --onnx=navdp_pointgoal_encoder_b1_s16.onnx \
  --saveEngine=navdp_pointgoal_encoder_b1_s16_fp32.engine

trtexec --onnx=navdp_pointgoal_denoiser_b1_s16.onnx \
  --saveEngine=navdp_pointgoal_denoiser_b1_s16_fp32.engine

trtexec --onnx=navdp_critic_b1_s16.onnx \
  --saveEngine=navdp_critic_b1_s16_fp32.engine
```

For the denoiser, use the patched TensorRT-compatible ONNX:

```bash
trtexec --onnx=navdp_pointgoal_denoiser_b1_s16_trt.onnx \
  --saveEngine=navdp_pointgoal_denoiser_b1_s16_fp32_notf32.engine \
  --noTF32
```

For parity-accurate FP32 engines, add `--noTF32` to all four engine builds.

Then build FP16 engines for deployment performance:

```bash
trtexec --onnx=navdp_rgbd_encoder_b1_s16.onnx \
  --saveEngine=navdp_rgbd_encoder_b1_s16_fp16.engine \
  --fp16

trtexec --onnx=navdp_pointgoal_encoder_b1_s16.onnx \
  --saveEngine=navdp_pointgoal_encoder_b1_s16_fp16.engine \
  --fp16

trtexec --onnx=navdp_pointgoal_denoiser_b1_s16.onnx \
  --saveEngine=navdp_pointgoal_denoiser_b1_s16_fp16.engine \
  --fp16

trtexec --onnx=navdp_critic_b1_s16.onnx \
  --saveEngine=navdp_critic_b1_s16_fp16.engine \
  --fp16
```

Use FP32 tolerance for parity first. Only compare FP16 after FP32 passes, and expect larger numeric differences around transformer LayerNorm/attention blocks.

## Access Blocker

The Jetson responds to ping, but this workstation currently cannot SSH in:

```text
rlan@10.133.109.123: Permission denied (publickey,password).
```

Once SSH access is available, the next validation step is:

1. Check JetPack/TensorRT versions.
2. Copy ONNX and parity bundle.
3. Build FP32 engines.
4. Run the parity bundle through TensorRT and compare every saved subgraph output.
5. Build FP16 engines and measure error/latency.

Expected-vs-actual NPZ comparison helper:

```bash
CONDA_PREFIX=/ssd4/envs/navdp_py310 \
/ssd4/envs/navdp_py310/bin/python tools/compare_navdp_parity_outputs.py \
  --expected exports/navdp_jetson_parity/navdp_jetson_parity_bundle.npz \
  --actual <jetson-produced-output.npz> \
  --atol 1e-3 \
  --rtol 1e-3
```
