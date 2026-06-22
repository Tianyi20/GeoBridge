#!/usr/bin/env python3
"""Replay saved WrenchEnv low-dimensional actions.

Expected episode layout:
    episode_xxxxxx/
        lowdim.npz          # contains action, robot0_eef_pos, robot0_eef_quat, ...
        episode_meta.json   # contains meta_seed, fps, success, num_steps, ...

The saved actions are parent/original-EE absolute pose actions:
    [x, y, z, qx, qy, qz, qw, gripper]

This matches WrenchEnv.step(), which calls:
    sim.solve_ik_and_apply(..., input_frame="parent_tcp")

python replay_wrench_actions.py /mnt/storage/DP_data/wrench_engagement/20260621_153408/episodes/episode_000001 --gui
"""



from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pybullet as p

try:
    import cv2
except ImportError:  # video recording is optional
    cv2 = None

# Make the script robust when launched from another working directory.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from WrenchEnv import WrenchEnv  # noqa: E402


def load_episode(episode_dir: Path) -> Tuple[np.lib.npyio.NpzFile, Dict]:
    episode_dir = Path(episode_dir)
    lowdim_path = episode_dir / "lowdim.npz"
    meta_path = episode_dir / "episode_meta.json"

    if not lowdim_path.exists():
        raise FileNotFoundError(f"Missing lowdim.npz: {lowdim_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing episode_meta.json: {meta_path}")

    lowdim = np.load(lowdim_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if "action" not in lowdim:
        raise KeyError(f"{lowdim_path} does not contain key 'action'. Keys={list(lowdim.keys())}")

    actions = np.asarray(lowdim["action"])
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(f"Expected action shape (T, 8), got {actions.shape} in {lowdim_path}")

    return lowdim, meta


def find_episode_dirs(path: Path) -> List[Path]:
    """Return one episode dir or all child episode dirs under a root."""
    path = Path(path)
    if (path / "lowdim.npz").exists() and (path / "episode_meta.json").exists():
        return [path]

    episode_dirs = sorted(
        pth for pth in path.glob("episode_*")
        if (pth / "lowdim.npz").exists() and (pth / "episode_meta.json").exists()
    )
    if not episode_dirs:
        raise FileNotFoundError(
            f"Could not find an episode under {path}. Expected either an episode dir "
            "or a root containing episode_*/lowdim.npz and episode_*/episode_meta.json."
        )
    return episode_dirs


def quat_angle_error(q1: np.ndarray, q2: np.ndarray, eps: float = 1e-8) -> float:
    """Smallest quaternion angular distance in radians for xyzw quaternions."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / max(np.linalg.norm(q1), eps)
    q2 = q2 / max(np.linalg.norm(q2), eps)
    dot = abs(float(np.dot(q1, q2)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def make_video_writer(video_path: Optional[Path], first_frame: np.ndarray, fps: int):
    if video_path is None:
        return None
    if cv2 is None:
        raise ImportError("cv2 is required for --save_video, but OpenCV is not installed.")

    video_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")
    return writer


def replay_one_episode(
    episode_dir: Path,
    *,
    connection_mode: int,
    sim_steps_per_action: Optional[int] = None,
    physics_hz: int = 120,
    stop_on_success: bool = False,
    max_actions: Optional[int] = None,
    save_video: bool = False,
    video_dir: Optional[Path] = None,
    if_FPSA: bool = False,
    randomize_image_noise: bool = True,
    randomize_objcolor: bool = True,
    randomize_lighting: bool = True,
    randomize_objpose: bool = True,
    randomize_distractors: bool = True,
    randomize_outlscene: bool = True,
    randomize_plane_height: bool = True,
    randomize_campose: bool = True,
) -> Dict:
    episode_dir = Path(episode_dir)
    lowdim, meta = load_episode(episode_dir)

    actions = np.asarray(lowdim["action"], dtype=np.float32)
    logged_pos = np.asarray(lowdim["robot0_eef_pos"], dtype=np.float32) if "robot0_eef_pos" in lowdim else None
    logged_quat = np.asarray(lowdim["robot0_eef_quat"], dtype=np.float32) if "robot0_eef_quat" in lowdim else None

    seed = int(meta.get("meta_seed", meta.get("seed", 42)))
    fps = int(meta.get("fps", 10))
    if sim_steps_per_action is None:
        sim_steps_per_action = max(1, int(round(float(physics_hz) / float(fps))))

    # Keep per-episode replay deterministic. The environment randomizers also receive this seed.
    np.random.seed(seed)
    random.seed(seed)

    env = WrenchEnv(
        sim_steps_per_action=sim_steps_per_action,
        connection_mode=connection_mode,
        seed=seed,
        if_FPSA=if_FPSA,
        randomize_image_noise=randomize_image_noise,
        randomize_objcolor=randomize_objcolor,
        randomize_lighting=randomize_lighting,
        randomize_objpose=randomize_objpose,
        randomize_distractors=randomize_distractors,
        randomize_outlscene=randomize_outlscene,
        randomize_plane_height=randomize_plane_height,
        randomize_campose=randomize_campose,
    )

    video_writer = None
    pos_errors: List[float] = []
    quat_errors: List[float] = []
    success_step: Optional[int] = None
    final_reward = False
    final_done = False

    try:
        obs = env.reset()
        if save_video:
            if video_dir is None:
                video_dir = episode_dir
            video_path = Path(video_dir) / f"{episode_dir.name}_action_replay.mp4"
            video_writer = make_video_writer(video_path, env.render(), fps=fps)
            video_writer.write(env.render()[..., ::-1])

        # Check initial reset state against recorded first lowdim state.
        if logged_pos is not None and len(logged_pos) > 0:
            init_pos_err = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - logged_pos[0]))
        else:
            init_pos_err = float("nan")

        n_actions = len(actions) if max_actions is None else min(len(actions), int(max_actions))
        for i in range(n_actions):
            obs, reward, done, info = env.step(actions[i])
            final_reward = bool(reward)
            final_done = bool(done)

            if video_writer is not None:
                video_writer.write(env.render()[..., ::-1])

            # The logged observation at index i+1 is the closest comparison target after replaying action i.
            # It will not be perfectly identical because collection generated 120Hz scripted targets but only
            # recorded 10Hz actions.
            j = i + 1
            if logged_pos is not None and j < len(logged_pos):
                pos_errors.append(float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - logged_pos[j])))
            if logged_quat is not None and j < len(logged_quat):
                quat_errors.append(quat_angle_error(np.asarray(obs["robot0_eef_quat"]), logged_quat[j]))

            if done and success_step is None:
                success_step = i
                if stop_on_success:
                    break

        summary = {
            "episode_dir": str(episode_dir),
            "seed": seed,
            "fps": fps,
            "sim_steps_per_action": int(sim_steps_per_action),
            "num_actions_loaded": int(len(actions)),
            "num_actions_replayed": int(i + 1 if len(actions) > 0 else 0),
            "recorded_success": bool(meta.get("success", False)),
            "replay_success": bool(final_done),
            "success_step": success_step,
            "initial_pos_error_to_log": init_pos_err,
            "mean_next_pos_error_to_log": float(np.mean(pos_errors)) if pos_errors else None,
            "max_next_pos_error_to_log": float(np.max(pos_errors)) if pos_errors else None,
            "mean_next_quat_error_rad_to_log": float(np.mean(quat_errors)) if quat_errors else None,
            "max_next_quat_error_rad_to_log": float(np.max(quat_errors)) if quat_errors else None,
        }
        return summary

    finally:
        if video_writer is not None:
            video_writer.release()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved parent-EE WrenchEnv actions from episode lowdim.npz.")
    parser.add_argument(
        "episode_path",
        type=Path,
        help="Path to one episode dir, or a root containing episode_*/lowdim.npz.",
    )
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI instead of DIRECT.")
    parser.add_argument(
        "--sim_steps_per_action",
        type=int,
        default=None,
        help="Physics steps per saved action. Default: round(120 / episode_meta['fps']).",
    )
    parser.add_argument("--physics_hz", type=int, default=120)
    parser.add_argument("--max_actions", type=int, default=None)
    parser.add_argument("--stop_on_success", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_dir", type=Path, default=None)
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=None,
        help="Optional JSON path for replay summaries. Default: <episode_path>/replay_summary.json for one episode, or <root>/replay_summary.json for many.",
    )

    # Debug switches. Defaults match collect_Wrenchdata.py / WrenchEnv.py.
    parser.add_argument("--if_FPSA", action="store_true", default=False)
    parser.add_argument("--no_image_noise", action="store_true")
    parser.add_argument("--no_objcolor", action="store_true")
    parser.add_argument("--no_lighting", action="store_true")
    parser.add_argument("--no_objpose", action="store_true")
    parser.add_argument("--no_distractors", action="store_true")
    parser.add_argument("--no_outlscene", action="store_true")
    parser.add_argument("--no_plane_height", action="store_true")
    parser.add_argument("--no_campose", action="store_true")

    args = parser.parse_args()

    episode_dirs = find_episode_dirs(args.episode_path)
    connection_mode = p.GUI if args.gui else p.DIRECT

    summaries = []
    for episode_dir in episode_dirs:
        print(f"\n=== Replaying {episode_dir} ===")
        summary = replay_one_episode(
            episode_dir,
            connection_mode=connection_mode,
            sim_steps_per_action=args.sim_steps_per_action,
            physics_hz=args.physics_hz,
            stop_on_success=args.stop_on_success,
            max_actions=args.max_actions,
            save_video=args.save_video,
            video_dir=args.video_dir,
            if_FPSA=args.if_FPSA,
            randomize_image_noise=not args.no_image_noise,
            randomize_objcolor=not args.no_objcolor,
            randomize_lighting=not args.no_lighting,
            randomize_objpose=not args.no_objpose,
            randomize_distractors=not args.no_distractors,
            randomize_outlscene=not args.no_outlscene,
            randomize_plane_height=not args.no_plane_height,
            randomize_campose=not args.no_campose,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    if args.summary_path is None:
        args.summary_path = Path(args.episode_path) / "replay_summary.json"
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved replay summary to: {args.summary_path}")


if __name__ == "__main__":
    main()
