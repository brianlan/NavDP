import argparse
import os
import sys

import onnx
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Export NavDP point-goal subgraphs to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to NavDP .ckpt file.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported ONNX files.")
    parser.add_argument("--batch-size", type=int, default=1, help="Runtime batch size.")
    parser.add_argument("--sample-num", type=int, default=16, help="Number of diffusion samples per batch item.")
    parser.add_argument("--device", default="cuda:0", help="Torch device used for export.")
    return parser.parse_args()


class RGBDEncoder(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.rgbd_encoder = policy.rgbd_encoder

    def forward(self, images, depths):
        return self.rgbd_encoder(images, depths)


class PointGoalDenoiser(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, last_actions, timestep, goal_embed, rgbd_embed):
        return self.policy.predict_noise(last_actions, timestep, goal_embed, rgbd_embed)


class Critic(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, trajectory, rgbd_embed):
        return self.policy.predict_critic(trajectory, rgbd_embed)


def export_and_check(module, args, output_path, input_names, output_names):
    torch.onnx.export(
        module,
        args,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
    )
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print(f"wrote {output_path} ({len(model.graph.node)} ONNX nodes)")


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested, but torch.cuda.is_available() is false.")

    # Needed for the RGBD encoder. The default PyTorch MHA fast path exports as
    # aten::_native_multi_head_attention, which is not supported by ONNX opset 17.
    torch.backends.mha.set_fastpath_enabled(False)

    navdp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "baselines", "navdp"))
    sys.path.insert(0, navdp_dir)
    from policy_network import NavDP_Policy

    os.makedirs(args.output_dir, exist_ok=True)
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

    batch = args.batch_size
    denoise_batch = args.batch_size * args.sample_num
    images = torch.rand(batch, 8, 224, 224, 3, device=args.device)
    depths = torch.rand(batch, 224, 224, 1, device=args.device)
    last_actions = torch.randn(denoise_batch, 24, 3, device=args.device)
    timestep = torch.tensor([9], dtype=torch.float32, device=args.device)
    goal_embed = torch.randn(denoise_batch, 1, 384, device=args.device)
    rgbd_embed = torch.randn(denoise_batch, 128, 384, device=args.device)

    with torch.no_grad():
        print("rgbd_encoder output:", RGBDEncoder(policy)(images, depths).shape)
        print("denoiser output:", PointGoalDenoiser(policy)(last_actions, timestep, goal_embed, rgbd_embed).shape)
        print("critic output:", Critic(policy)(last_actions, rgbd_embed).shape)

    suffix = f"b{batch}_s{args.sample_num}"
    export_and_check(
        RGBDEncoder(policy),
        (images, depths),
        os.path.join(args.output_dir, f"navdp_rgbd_encoder_{suffix}.onnx"),
        ["images", "depths"],
        ["rgbd_embed"],
    )
    export_and_check(
        PointGoalDenoiser(policy),
        (last_actions, timestep, goal_embed, rgbd_embed),
        os.path.join(args.output_dir, f"navdp_pointgoal_denoiser_{suffix}.onnx"),
        ["last_actions", "timestep", "goal_embed", "rgbd_embed"],
        ["noise_pred"],
    )
    export_and_check(
        Critic(policy),
        (last_actions, rgbd_embed),
        os.path.join(args.output_dir, f"navdp_critic_{suffix}.onnx"),
        ["trajectory", "rgbd_embed"],
        ["critic_values"],
    )


if __name__ == "__main__":
    main()
