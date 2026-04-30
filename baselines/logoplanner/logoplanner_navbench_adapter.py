"""LoGoPlanner adapter for NavArena-Bench PointNav evaluation.

This adapter bridges LoGoPlanner's HTTP REST server (logoplanner_server.py)
with NavArena-Bench's WebSocket NavigationModelServer interface, enabling
LoGoPlanner to be evaluated on the NavArena-Bench PointNav benchmark.

Architecture:
    NavArena-Bench runner ──WebSocket──► this adapter ──HTTP──► logoplanner_server.py
                                                                       │
                                                               LoGoPlanner_Agent
                                                             (GPU inference, CUDA)

Usage:
    # Step 1 – start the LoGoPlanner HTTP server (GPU machine):
    #   cd NavDP/baselines/logoplanner
    #   python logoplanner_server.py --port 43020 --checkpoint ./logoplanner_policy.ckpt
    #
    # Step 2 – run this adapter (same or different machine):
    #   python logoplanner_navbench_adapter.py \\
    #       --server-url http://127.0.0.1:43020 \\
    #       --port 8000

Coordinate conventions:
    - NavArena-Bench uses ENU world frame: yaw=0 → +X (East), CCW positive.
    - Robot body frame: +X = forward, +Y = left.
    - LoGoPlanner goal: (goal_x=forward, goal_y=lateral) in robot body frame.
    - World → robot-body rotation (computed once at episode start):
        fwd  =  cos(yaw0)*dx + sin(yaw0)*dy
        lat  = -sin(yaw0)*dx + cos(yaw0)*dy

LoGoPlanner is designed to do implicit localization from its visual-geometry
backbone + history queues, so the goal is expressed in the episode's *starting*
body frame and kept fixed thereafter — the model infers how far it has moved
on its own. We therefore compute the body-frame goal once on the first
prediction of each episode and reuse it, matching lekiwi_logoplanner_host.py.
"""

from __future__ import annotations

import argparse
import io
import json
import math

import numpy as np
import requests
from PIL import Image

from navarena_server.server import NavigationModelServer, serve


class LoGoPlannerNavBenchAdapter(NavigationModelServer):
    """NavArena-Bench PointNav agent backed by a running logoplanner_server.py."""

    def __init__(
        self,
        server_url: str,
        camera_intrinsic: list[list[float]],
        stop_threshold: float = 0.3,
        rgb_camera_key: str | None = None,
        depth_camera_key: str | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        """
        Args:
            server_url:        Base URL of logoplanner_server.py, e.g. "http://127.0.0.1:43020".
            camera_intrinsic:  3×3 camera intrinsic matrix as nested list [[fx,0,cx],[0,fy,cy],[0,0,1]].
                               Used by LoGoPlanner's trajectory visualisation; adjust to match your scenes.
            stop_threshold:    Value passed to /navigator_reset; LoGoPlanner issues a rotation-only
                               command when max trajectory value is below this threshold.
            rgb_camera_key:    Key into observation["rgb"] to pick the camera image (default: first key).
            depth_camera_key:  Key into observation["depth"] (default: first key).
            request_timeout:   HTTP request timeout in seconds.
        """
        self.server_url = server_url.rstrip("/")
        self.camera_intrinsic = camera_intrinsic
        self.stop_threshold = stop_threshold
        self.rgb_camera_key = rgb_camera_key
        self.depth_camera_key = depth_camera_key
        self.request_timeout = request_timeout
        self._goal_position: list[float] | None = None
        self._goal_robot_frame: tuple[float, float] | None = None

    # ------------------------------------------------------------------ #
    #  Lifecycle hooks                                                     #
    # ------------------------------------------------------------------ #

    async def on_episode_start(self, payload: dict, ctx) -> None:
        """Reset LoGoPlanner's memory queues and cache the goal for this episode."""
        task = payload.get("task", payload)
        goals = task.get("goals") or []
        if goals and isinstance(goals[0], dict) and goals[0].get("position") is not None:
            self._goal_position = list(goals[0]["position"])
        else:
            self._goal_position = None
        self._goal_robot_frame = None
        print(f"[adapter] episode_start: goal={self._goal_position}", flush=True)

        resp = requests.post(
            f"{self.server_url}/navigator_reset",
            json={
                "intrinsic": self.camera_intrinsic,
                "stop_threshold": self.stop_threshold,
                "batch_size": 1,
            },
            timeout=self.request_timeout,
        )
        resp.raise_for_status()

    async def on_episode_end(self, result: dict, ctx) -> None:
        """Finalize the per-episode mp4 so the last episode isn't left unclosed."""
        try:
            requests.post(
                f"{self.server_url}/finalize",
                timeout=self.request_timeout,
            )
        except Exception as exc:
            print(f"[adapter] finalize failed: {exc}", flush=True)

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #

    async def predict(self, observation: dict, ctx) -> dict:
        """Convert a NavBench observation to a LoGoPlanner HTTP call and return an action.

        Returns:
            dict with keys {"x", "y", "yaw"}:
                x   – forward displacement in robot frame (metres)
                y   – lateral displacement in robot frame (metres, positive = left)
                yaw – rotation increment (radians); always 0 for omnidirectional mode
        """
        rgb_buf, depth_buf = self._encode_images(observation)

        if self._goal_robot_frame is None:
            self._goal_robot_frame = self._compute_goal_robot_frame(observation)
            print(
                f"[adapter] initial body-frame goal "
                f"=({self._goal_robot_frame[0]:+.3f},{self._goal_robot_frame[1]:+.3f}) "
                f"(fixed for the rest of the episode)",
                flush=True,
            )
        goal_x, goal_y = self._goal_robot_frame

        traj = self._call_pointgoal_step(rgb_buf, depth_buf, goal_x, goal_y)

        # traj: list of shape (batch_size=1, predict_size, 3)
        traj_arr = np.asarray(traj)
        print(
            f"[adapter] goal_robot=({goal_x:+.3f},{goal_y:+.3f}) "
            f"traj.shape={traj_arr.shape} "
            f"wp[0]={traj_arr[0, 0].round(3).tolist()} "
            f"wp[4]={traj_arr[0, 4].round(3).tolist()} "
            f"wp[-1]={traj_arr[0, -1].round(3).tolist()} "
            f"fwd_range=[{traj_arr[0, :, 0].min():+.3f},{traj_arr[0, :, 0].max():+.3f}] "
            f"lat_range=[{traj_arr[0, :, 1].min():+.3f},{traj_arr[0, :, 1].max():+.3f}]",
            flush=True,
        )

        # Take the 5th waypoint of the first (only) environment.
        first_wp = traj[0][4]
        fwd = float(first_wp[0])
        lat = float(first_wp[1])
        yaw = float(first_wp[2])

        return {"waypoints": [{"x": fwd, "y": lat, "yaw": yaw}]}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _encode_images(self, observation: dict) -> tuple[io.BytesIO, io.BytesIO]:
        """Encode RGB and depth observations into PNG byte buffers for multipart upload."""
        # ── RGB: support both PIL.Image and numpy.ndarray ───────────────────
        rgb_dict: dict = observation.get("rgb", {})
        rgb_key = self.rgb_camera_key or next(iter(rgb_dict))
        rgb_raw = rgb_dict[rgb_key]
        if isinstance(rgb_raw, Image.Image):
            rgb_np = np.asarray(rgb_raw.convert("RGB"))
        else:
            rgb_np = np.asarray(rgb_raw)
            if rgb_np.ndim == 2:
                rgb_np = np.stack([rgb_np, rgb_np, rgb_np], axis=-1)
            if rgb_np.shape[-1] == 4:
                rgb_np = rgb_np[:, :, :3]
            if rgb_np.dtype != np.uint8:
                rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)

        # NavBench observations are typically RGB; LoGoPlanner expects BGR.
        rgb_bgr = rgb_np[:, :, ::-1].copy()   # H×W×3 uint8, BGR
        rgb_buf = io.BytesIO()
        Image.fromarray(rgb_bgr, mode="RGB").save(rgb_buf, format="PNG")
        rgb_buf.seek(0)

        # ── Depth: support both PIL.Image and numpy.ndarray ─────────────────
        depth_dict: dict = observation.get("depth", {})
        depth_key = self.depth_camera_key or next(iter(depth_dict))
        depth_raw = depth_dict[depth_key]
        depth_arr: np.ndarray = np.asarray(depth_raw, dtype=np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[:, :, 0]                   # (H, W)
        depth_int = (depth_arr * 10000).astype(np.int32)     # metres → 0.1 mm integer
        depth_buf = io.BytesIO()
        Image.fromarray(depth_int, mode="I").save(depth_buf, format="PNG")
        depth_buf.seek(0)

        return rgb_buf, depth_buf

    def _compute_goal_robot_frame(self, observation: dict) -> tuple[float, float]:
        """Transform the world-frame goal into the episode's *starting* body frame.

        Called exactly once per episode, on the first prediction. LoGoPlanner
        handles the shift from start-frame to current-frame internally via its
        implicit localization backbone.

        NavBench ENU convention (matches ROS standard):
            yaw = 0  →  robot faces +X (East)
            yaw increases counter-clockwise

        Robot body frame:
            +X = forward,  +Y = left

        Rotation (world → start-body):
            fwd =  cos(yaw0)*dx + sin(yaw0)*dy
            lat = -sin(yaw0)*dx + cos(yaw0)*dy
        """
        pose = observation.get("pose", {})
        robot_pos = pose.get("position", [0.0, 0.0, 0.0])
        robot_yaw: float = pose.get("yaw", 0.0)

        if self._goal_position is not None:
            goal_pos = self._goal_position
        else:
            goal = observation.get("goal", {})
            goal_pos = goal.get("position", [0.0, 0.0, 0.0])

        dx = goal_pos[0] - robot_pos[0]
        dy = goal_pos[1] - robot_pos[1]

        cos_y = math.cos(robot_yaw)
        sin_y = math.sin(robot_yaw)
        goal_fwd = cos_y * dx + sin_y * dy
        goal_lat = -sin_y * dx + cos_y * dy

        return goal_fwd, goal_lat

    def _call_pointgoal_step(
        self,
        rgb_buf: io.BytesIO,
        depth_buf: io.BytesIO,
        goal_x: float,
        goal_y: float,
    ) -> list:
        """POST to /pointgoal_step and return the parsed trajectory list."""
        files = {
            "image": ("image.png", rgb_buf, "image/png"),
            "depth": ("depth.png", depth_buf, "image/png"),
        }
        data = {
            "goal_data": json.dumps({
                "goal_x": [goal_x],   # list of length batch_size=1
                "goal_y": [goal_y],
            })
        }
        resp = requests.post(
            f"{self.server_url}/pointgoal_step",
            files=files,
            data=data,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        return resp.json()["trajectory"]


# ──────────────────────────────────────────────────────────────────────── #
#  CLI entry point                                                          #
# ──────────────────────────────────────────────────────────────────────── #

def _build_intrinsic(fx: float, fy: float, cx: float, cy: float) -> list[list[float]]:
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoGoPlanner → NavArena-Bench PointNav adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--server-url", default="http://127.0.0.1:43020",
        help="URL of the running logoplanner_server.py",
    )
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket bind host")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket bind port")
    parser.add_argument(
        "--stop-threshold", type=float, default=-10.0,
        help="LoGoPlanner stop threshold forwarded to /navigator_reset",
    )
    # Camera intrinsic (for LoGoPlanner trajectory visualisation).
    # Default matches the NavArena-Bench gs_env default camera (fx=fy=320, cx=320, cy=240).
    parser.add_argument("--fx", type=float, default=320.0, help="Camera intrinsic fx")
    parser.add_argument("--fy", type=float, default=320.0, help="Camera intrinsic fy")
    parser.add_argument("--cx", type=float, default=320.0, help="Camera intrinsic cx (principal point x)")
    parser.add_argument("--cy", type=float, default=240.0, help="Camera intrinsic cy (principal point y)")
    parser.add_argument(
        "--rgb-key", default=None,
        help="Camera key in observation['rgb'] (default: first available key)",
    )
    parser.add_argument(
        "--depth-key", default=None,
        help="Camera key in observation['depth'] (default: first available key)",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout (seconds)")
    args = parser.parse_args()

    intrinsic = _build_intrinsic(args.fx, args.fy, args.cx, args.cy)
    adapter = LoGoPlannerNavBenchAdapter(
        server_url=args.server_url,
        camera_intrinsic=intrinsic,
        stop_threshold=args.stop_threshold,
        rgb_camera_key=args.rgb_key,
        depth_camera_key=args.depth_key,
        request_timeout=args.timeout,
    )

    print(f"LoGoPlanner server : {args.server_url}")
    print(f"NavBench WebSocket : ws://{args.host}:{args.port}")
    serve(adapter, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
