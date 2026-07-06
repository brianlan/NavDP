import argparse
import json
import os
import sys

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create deterministic NavDP tensors for host-vs-Jetson parity checks."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to NavDP .ckpt file.")
    parser.add_argument("--output-dir", required=True, help="Directory for the parity bundle.")
    parser.add_argument(
        "--input-npz",
        help=(
            "Optional NPZ with images, depths, and point_goal arrays. "
            "If omitted, deterministic synthetic tensors are generated."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for synthetic inputs and explicit DDPM noise.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-num", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def numpy_to_torch(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def ddpm_step_with_explicit_noise(scheduler, model_output, timestep, sample, variance_noise):
    prev_t = scheduler.previous_timestep(timestep)

    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(sample.device)
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t].to(sample.device) if prev_t >= 0 else scheduler.one.to(sample.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    pred_original_sample = pred_original_sample.clamp(
        -scheduler.config.clip_sample_range,
        scheduler.config.clip_sample_range,
    )

    pred_original_sample_coeff = alpha_prod_t_prev.sqrt() * current_beta_t / beta_prod_t
    current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    if timestep > 0:
        variance = scheduler._get_variance(timestep).to(sample.device).sqrt() * variance_noise
        pred_prev_sample = pred_prev_sample + variance

    return pred_prev_sample


def load_or_make_inputs(args):
    rng = np.random.default_rng(args.seed)
    if args.input_npz:
        data = np.load(args.input_npz)
        images = data["images"].astype(np.float32)
        depths = data["depths"].astype(np.float32)
        point_goal = data["point_goal"].astype(np.float32)
    else:
        images = rng.random((args.batch_size, 8, 224, 224, 3), dtype=np.float32)
        depths = rng.random((args.batch_size, 224, 224, 1), dtype=np.float32)
        point_goal = np.array([[3.0, 1.25, 0.0]], dtype=np.float32)
        if args.batch_size != 1:
            point_goal = np.repeat(point_goal, args.batch_size, axis=0)

    if images.shape != (args.batch_size, 8, 224, 224, 3):
        raise ValueError(f"images must have shape {(args.batch_size, 8, 224, 224, 3)}, got {images.shape}")
    if depths.shape != (args.batch_size, 224, 224, 1):
        raise ValueError(f"depths must have shape {(args.batch_size, 224, 224, 1)}, got {depths.shape}")
    if point_goal.shape != (args.batch_size, 3):
        raise ValueError(f"point_goal must have shape {(args.batch_size, 3)}, got {point_goal.shape}")

    denoise_batch = args.batch_size * args.sample_num
    initial_action = rng.standard_normal((denoise_batch, 24, 3), dtype=np.float32)
    variance_noise = rng.standard_normal((10, denoise_batch, 24, 3), dtype=np.float32)
    return images, depths, point_goal, initial_action, variance_noise


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")

    torch.backends.mha.set_fastpath_enabled(False)
    navdp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "baselines", "navdp"))
    sys.path.insert(0, navdp_dir)
    from policy_network import NavDP_Policy

    images, depths, point_goal, initial_action, variance_noise = load_or_make_inputs(args)

    policy = NavDP_Policy(
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        device=args.device,
    )
    state = torch.load(args.checkpoint, map_location=args.device)
    policy.load_state_dict(state, strict=False)
    policy.to(args.device).eval()

    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    timesteps = [int(t.item()) for t in scheduler.timesteps]

    with torch.no_grad():
        tensor_images = numpy_to_torch(images, args.device)
        tensor_depths = numpy_to_torch(depths, args.device)
        tensor_point_goal = numpy_to_torch(point_goal, args.device)

        rgbd_embed = policy.rgbd_encoder(tensor_images, tensor_depths)
        goal_embed = policy.point_encoder(tensor_point_goal).unsqueeze(1)
        rgbd_embed_repeated = torch.repeat_interleave(rgbd_embed, args.sample_num, dim=0)
        goal_embed_repeated = torch.repeat_interleave(goal_embed, args.sample_num, dim=0)

        action = numpy_to_torch(initial_action, args.device)
        noise_pred_steps = []
        action_inputs = []
        action_outputs = []
        for step_index, timestep in enumerate(timesteps):
            action_inputs.append(action.detach().cpu().numpy())
            timestep_tensor = torch.tensor([float(timestep)], dtype=torch.float32, device=args.device)
            noise_pred = policy.predict_noise(action, timestep_tensor, goal_embed_repeated, rgbd_embed_repeated)
            noise_pred_steps.append(noise_pred.detach().cpu().numpy())
            action = ddpm_step_with_explicit_noise(
                scheduler,
                noise_pred,
                timestep,
                action,
                numpy_to_torch(variance_noise[step_index], args.device),
            )
            action_outputs.append(action.detach().cpu().numpy())

        critic_values = policy.predict_critic(action, rgbd_embed_repeated).reshape(args.batch_size, args.sample_num)
        all_trajectory = torch.cumsum(action / 4.0, dim=1).reshape(args.batch_size, args.sample_num, 24, 3)
        trajectory_length = all_trajectory[:, :, -1, 0:2].norm(dim=-1)
        all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor(
            [[[0, 0, 1.0]]],
            device=all_trajectory.device,
        )
        sorted_indices = (-critic_values).argsort(dim=1)
        batch_indices = torch.arange(args.batch_size, device=args.device).unsqueeze(1).expand(-1, 2)
        selected_trajectory = all_trajectory[batch_indices, sorted_indices[:, 0:2]]

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        os.path.join(args.output_dir, "navdp_jetson_parity_bundle.npz"),
        images=images,
        depths=depths,
        point_goal=point_goal,
        initial_action=initial_action,
        variance_noise=variance_noise,
        timesteps=np.array(timesteps, dtype=np.int64),
        alphas_cumprod=scheduler.alphas_cumprod.detach().cpu().numpy().astype(np.float32),
        rgbd_embed=rgbd_embed.detach().cpu().numpy(),
        rgbd_embed_repeated=rgbd_embed_repeated.detach().cpu().numpy(),
        goal_embed=goal_embed.detach().cpu().numpy(),
        goal_embed_repeated=goal_embed_repeated.detach().cpu().numpy(),
        action_inputs=np.stack(action_inputs),
        noise_pred_steps=np.stack(noise_pred_steps),
        action_outputs=np.stack(action_outputs),
        final_action=action.detach().cpu().numpy(),
        critic_values=critic_values.detach().cpu().numpy(),
        all_trajectory=all_trajectory.detach().cpu().numpy(),
        selected_trajectory=selected_trajectory.detach().cpu().numpy(),
        selected_indices=sorted_indices[:, 0:2].detach().cpu().numpy(),
    )
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "sample_num": args.sample_num,
                "timesteps": timesteps,
                "onnx_exports": {
                    "rgbd_encoder": "navdp_rgbd_encoder_b1_s16.onnx",
                    "pointgoal_encoder": "navdp_pointgoal_encoder_b1_s16.onnx",
                    "pointgoal_denoiser": "navdp_pointgoal_denoiser_b1_s16.onnx",
                    "critic": "navdp_critic_b1_s16.onnx",
                },
                "comparison_notes": [
                    "Compare TensorRT rgbd_encoder output with rgbd_embed.",
                    "Compare TensorRT pointgoal_encoder output with goal_embed.",
                    "For each timestep, feed action_inputs[i], timesteps[i], goal_embed_repeated, and rgbd_embed_repeated into the denoiser and compare with noise_pred_steps[i].",
                    "Apply the explicit DDPM step with variance_noise[i]; compare with action_outputs[i].",
                    "Feed final_action and rgbd_embed_repeated into critic; compare with critic_values reshaped to batch_size x sample_num.",
                ],
            },
            f,
            indent=2,
        )

    print(f"wrote {os.path.join(args.output_dir, 'navdp_jetson_parity_bundle.npz')}")
    print(f"wrote {os.path.join(args.output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
