import argparse
import ctypes
import os
import time

import numpy as np
import tensorrt as trt


CUDA_SUCCESS = 0
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Run NavDP parity bundle through TensorRT engines.")
    parser.add_argument("--bundle", required=True, help="Input parity NPZ.")
    parser.add_argument("--engine-dir", required=True, help="Directory containing TensorRT engines.")
    parser.add_argument("--output", required=True, help="Output NPZ with actual TensorRT results.")
    parser.add_argument("--suffix", default="fp32", help="Engine suffix, usually fp32 or fp16.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of full pipeline runs for timing.")
    parser.add_argument("--cached-rgbd", action="store_true", help="Use split RGB-frame/depth/fusion engines.")
    return parser.parse_args()


class CudaRuntime:
    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def check(self, code, label):
        if code != CUDA_SUCCESS:
            msg = self.lib.cudaGetErrorString(code).decode("utf-8", errors="replace")
            raise RuntimeError(f"{label} failed: {code} {msg}")

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr

    def free(self, ptr):
        self.check(self.lib.cudaFree(ptr), "cudaFree")

    def stream(self):
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(ptr)), "cudaStreamCreate")
        return ptr

    def stream_synchronize(self, stream):
        self.check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def stream_destroy(self, stream):
        self.check(self.lib.cudaStreamDestroy(stream), "cudaStreamDestroy")

    def memcpy_htod_async(self, dst, src, nbytes, stream):
        self.check(
            self.lib.cudaMemcpyAsync(
                dst,
                ctypes.c_void_p(src.ctypes.data),
                nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            ),
            "cudaMemcpyAsync HtoD",
        )

    def memcpy_dtoh_async(self, dst, src, nbytes, stream):
        self.check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst.ctypes.data),
                src,
                nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            ),
            "cudaMemcpyAsync DtoH",
        )


class TrtEngine:
    def __init__(self, path, cuda):
        self.path = path
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {path}")
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            item = (name, dtype, shape)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(item)
            else:
                self.outputs.append(item)

    def run(self, **inputs):
        stream = self.cuda.stream()
        allocations = []
        outputs = {}
        try:
            for name, dtype, shape in self.inputs:
                if name not in inputs:
                    raise KeyError(f"{name} missing for {self.path}")
                array = np.ascontiguousarray(inputs[name].astype(dtype, copy=False))
                if tuple(array.shape) != tuple(shape):
                    raise ValueError(f"{name} expected shape {shape}, got {array.shape}")
                ptr = self.cuda.malloc(array.nbytes)
                allocations.append(ptr)
                self.cuda.memcpy_htod_async(ptr, array, array.nbytes, stream)
                self.context.set_tensor_address(name, ptr.value)

            for name, dtype, shape in self.outputs:
                array = np.empty(shape, dtype=dtype)
                ptr = self.cuda.malloc(array.nbytes)
                allocations.append(ptr)
                self.context.set_tensor_address(name, ptr.value)
                outputs[name] = (array, ptr)

            if not self.context.execute_async_v3(stream.value):
                raise RuntimeError(f"execute_async_v3 failed for {self.path}")

            for array, ptr in outputs.values():
                self.cuda.memcpy_dtoh_async(array, ptr, array.nbytes, stream)
            self.cuda.stream_synchronize(stream)
            return {name: array for name, (array, _) in outputs.items()}
        finally:
            for ptr in allocations:
                self.cuda.free(ptr)
            self.cuda.stream_destroy(stream)


def ddpm_step(noise_pred, timestep, sample, variance_noise, alphas_cumprod):
    prev_t = timestep - 1
    alpha_prod_t = alphas_cumprod[timestep]
    alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else np.float32(1.0)
    beta_prod_t = np.float32(1.0) - alpha_prod_t
    beta_prod_t_prev = np.float32(1.0) - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = np.float32(1.0) - current_alpha_t

    pred_original_sample = (sample - np.sqrt(beta_prod_t) * noise_pred) / np.sqrt(alpha_prod_t)
    pred_original_sample = np.clip(pred_original_sample, -1.0, 1.0)
    pred_original_sample_coeff = np.sqrt(alpha_prod_t_prev) * current_beta_t / beta_prod_t
    current_sample_coeff = np.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    if timestep > 0:
        variance = np.sqrt((beta_prod_t_prev / beta_prod_t) * current_beta_t) * variance_noise
        pred_prev_sample = pred_prev_sample + variance
    return pred_prev_sample.astype(np.float32)


def engine_path(engine_dir, stem, suffix):
    return os.path.join(engine_dir, f"{stem}_{suffix}.engine")


def run_pipeline(args, bundle, engines):
    if args.cached_rgbd:
        rgb_token_blocks = []
        for frame_index in range(bundle["images"].shape[1]):
            frame_tokens = engines["rgb_frame"].run(image=bundle["images"][:, frame_index])["rgb_tokens"].astype(np.float32)
            rgb_token_blocks.append(frame_tokens)
        rgb_tokens = np.concatenate(rgb_token_blocks, axis=1)
        depth_tokens = engines["depth_frame"].run(depth=bundle["depths"])["depth_tokens"].astype(np.float32)
        rgbd_embed = engines["rgbd_fusion"].run(rgb_tokens=rgb_tokens, depth_tokens=depth_tokens)["rgbd_embed"].astype(np.float32)
    else:
        rgb_tokens = None
        depth_tokens = None
        rgbd_embed = engines["rgbd"].run(images=bundle["images"], depths=bundle["depths"])["rgbd_embed"].astype(np.float32)
    goal_embed = engines["pointgoal"].run(point_goal=bundle["point_goal"])["goal_embed"].astype(np.float32)
    sample_num = bundle["initial_action"].shape[0] // bundle["point_goal"].shape[0]
    rgbd_embed_repeated = np.repeat(rgbd_embed, sample_num, axis=0)
    goal_embed_repeated = np.repeat(goal_embed, sample_num, axis=0)

    action = bundle["initial_action"].astype(np.float32).copy()
    noise_pred_steps = []
    action_outputs = []
    for step_index, timestep in enumerate(bundle["timesteps"].astype(np.int64).tolist()):
        timestep_array = np.array([float(timestep)], dtype=np.float32)
        noise_pred = engines["denoiser"].run(
            last_actions=action,
            timestep=timestep_array,
            goal_embed=goal_embed_repeated,
            rgbd_embed=rgbd_embed_repeated,
        )["noise_pred"].astype(np.float32)
        noise_pred_steps.append(noise_pred)
        action = ddpm_step(
            noise_pred,
            timestep,
            action,
            bundle["variance_noise"][step_index].astype(np.float32),
            bundle["alphas_cumprod"].astype(np.float32),
        )
        action_outputs.append(action)

    critic_flat = engines["critic"].run(trajectory=action, rgbd_embed=rgbd_embed_repeated)["critic_values"].astype(np.float32)
    batch_size = bundle["point_goal"].shape[0]
    critic_values = critic_flat.reshape(batch_size, sample_num)
    all_trajectory = np.cumsum(action / np.float32(4.0), axis=1).reshape(batch_size, sample_num, 24, 3)
    trajectory_length = np.linalg.norm(all_trajectory[:, :, -1, 0:2], axis=-1)
    all_trajectory[trajectory_length < 0.5] *= np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    selected_indices = np.argsort(-critic_values, axis=1)[:, 0:2]
    selected_trajectory = all_trajectory[np.arange(batch_size)[:, None], selected_indices]
    return {
        "rgbd_embed": rgbd_embed,
        "rgb_tokens": rgb_tokens if rgb_tokens is not None else np.empty((0,), dtype=np.float32),
        "depth_tokens": depth_tokens if depth_tokens is not None else np.empty((0,), dtype=np.float32),
        "goal_embed": goal_embed,
        "rgbd_embed_repeated": rgbd_embed_repeated,
        "goal_embed_repeated": goal_embed_repeated,
        "noise_pred_steps": np.stack(noise_pred_steps),
        "action_outputs": np.stack(action_outputs),
        "final_action": action,
        "critic_values": critic_values,
        "all_trajectory": all_trajectory,
        "selected_trajectory": selected_trajectory,
        "selected_indices": selected_indices,
    }


def main():
    args = parse_args()
    bundle = np.load(args.bundle)
    cuda = CudaRuntime()
    engines = {
        "pointgoal": TrtEngine(engine_path(args.engine_dir, "navdp_pointgoal_encoder_b1_s16", args.suffix), cuda),
        "denoiser": TrtEngine(engine_path(args.engine_dir, "navdp_pointgoal_denoiser_b1_s16", args.suffix), cuda),
        "critic": TrtEngine(engine_path(args.engine_dir, "navdp_critic_b1_s16", args.suffix), cuda),
    }
    if args.cached_rgbd:
        engines.update(
            {
                "rgb_frame": TrtEngine(engine_path(args.engine_dir, "navdp_rgb_frame_encoder_b1", args.suffix), cuda),
                "depth_frame": TrtEngine(engine_path(args.engine_dir, "navdp_depth_frame_encoder_b1", args.suffix), cuda),
                "rgbd_fusion": TrtEngine(engine_path(args.engine_dir, "navdp_rgbd_fusion_b1", args.suffix), cuda),
            }
        )
    else:
        engines["rgbd"] = TrtEngine(engine_path(args.engine_dir, "navdp_rgbd_encoder_b1_s16", args.suffix), cuda)

    result = None
    start = time.perf_counter()
    for _ in range(args.repeat):
        result = run_pipeline(args, bundle, engines)
    elapsed = time.perf_counter() - start
    result["elapsed_seconds"] = np.array([elapsed], dtype=np.float32)
    result["repeat"] = np.array([args.repeat], dtype=np.int64)
    np.savez(args.output, **result)
    print(f"wrote {args.output}")
    print(f"full pipeline average: {elapsed / args.repeat:.6f} seconds over {args.repeat} run(s)")


if __name__ == "__main__":
    main()
