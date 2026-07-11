import argparse
import os
import sys

import numpy as np
import torch

from make_navdp_jetson_parity_bundle import ddpm_step_with_explicit_noise, load_or_make_inputs, numpy_to_torch


def parse_args():
    parser = argparse.ArgumentParser(description="Check PyTorch parity for cached NavDP RGB token history.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-npz")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-num", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def compare(name, expected, actual, atol, rtol):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    diff = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(expected.astype(np.float32)), atol)
    max_rel = float((diff / denom).max()) if diff.size else 0.0
    ok = np.allclose(expected, actual, atol=atol, rtol=rtol)
    print(f"{'PASS' if ok else 'FAIL'} {name}: max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
    return ok


def cached_rgbd_embed(policy, images, depths, device):
    rgb_tokens = []
    for frame_index in range(images.shape[1]):
        frame = numpy_to_torch(images[:, frame_index], device)
        rgb_tokens.append(policy.rgbd_encoder.encode_rgb(frame))
    rgb_tokens = torch.cat(rgb_tokens, dim=1)
    depth_tokens = policy.rgbd_encoder.encode_depth(numpy_to_torch(depths, device))
    return policy.rgbd_encoder.fuse_tokens(rgb_tokens, depth_tokens)


def run_with_rgbd_embed(policy, rgbd_embed, point_goal, initial_action, variance_noise, sample_num, device):
    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    timesteps = [int(t.item()) for t in scheduler.timesteps]

    tensor_point_goal = numpy_to_torch(point_goal, device)
    goal_embed = policy.point_encoder(tensor_point_goal).unsqueeze(1)
    rgbd_embed_repeated = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
    goal_embed_repeated = torch.repeat_interleave(goal_embed, sample_num, dim=0)

    action = numpy_to_torch(initial_action, device)
    noise_pred_steps = []
    action_outputs = []
    for step_index, timestep in enumerate(timesteps):
        timestep_tensor = torch.tensor([float(timestep)], dtype=torch.float32, device=device)
        noise_pred = policy.predict_noise(action, timestep_tensor, goal_embed_repeated, rgbd_embed_repeated)
        noise_pred_steps.append(noise_pred.detach().cpu().numpy())
        action = ddpm_step_with_explicit_noise(
            scheduler,
            noise_pred,
            timestep,
            action,
            numpy_to_torch(variance_noise[step_index], device),
        )
        action_outputs.append(action.detach().cpu().numpy())

    critic_values = policy.predict_critic(action, rgbd_embed_repeated).reshape(point_goal.shape[0], sample_num)
    all_trajectory = torch.cumsum(action / 4.0, dim=1).reshape(point_goal.shape[0], sample_num, 24, 3)
    trajectory_length = all_trajectory[:, :, -1, 0:2].norm(dim=-1)
    all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor(
        [[[0, 0, 1.0]]],
        device=all_trajectory.device,
    )
    sorted_indices = (-critic_values).argsort(dim=1)
    batch_indices = torch.arange(point_goal.shape[0], device=device).unsqueeze(1).expand(-1, 2)
    selected_trajectory = all_trajectory[batch_indices, sorted_indices[:, 0:2]]
    return {
        "goal_embed": goal_embed.detach().cpu().numpy(),
        "rgbd_embed_repeated": rgbd_embed_repeated.detach().cpu().numpy(),
        "goal_embed_repeated": goal_embed_repeated.detach().cpu().numpy(),
        "noise_pred_steps": np.stack(noise_pred_steps),
        "action_outputs": np.stack(action_outputs),
        "final_action": action.detach().cpu().numpy(),
        "critic_values": critic_values.detach().cpu().numpy(),
        "all_trajectory": all_trajectory.detach().cpu().numpy(),
        "selected_trajectory": selected_trajectory.detach().cpu().numpy(),
        "selected_indices": sorted_indices[:, 0:2].detach().cpu().numpy(),
    }


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

    with torch.no_grad():
        monolithic_rgbd = policy.rgbd_encoder(numpy_to_torch(images, args.device), numpy_to_torch(depths, args.device))
        cached_rgbd = cached_rgbd_embed(policy, images, depths, args.device)
        monolithic = run_with_rgbd_embed(
            policy,
            monolithic_rgbd,
            point_goal,
            initial_action,
            variance_noise,
            args.sample_num,
            args.device,
        )
        cached = run_with_rgbd_embed(
            policy,
            cached_rgbd,
            point_goal,
            initial_action,
            variance_noise,
            args.sample_num,
            args.device,
        )

    results = [
        compare(
            "rgbd_embed",
            monolithic_rgbd.detach().cpu().numpy(),
            cached_rgbd.detach().cpu().numpy(),
            args.atol,
            args.rtol,
        )
    ]
    for key in [
        "goal_embed",
        "rgbd_embed_repeated",
        "goal_embed_repeated",
        "noise_pred_steps",
        "action_outputs",
        "final_action",
        "critic_values",
        "all_trajectory",
        "selected_trajectory",
        "selected_indices",
    ]:
        results.append(compare(key, monolithic[key], cached[key], args.atol, args.rtol))

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
