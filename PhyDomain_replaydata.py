from pathlib import Path
import json
import numpy as np
import imageio.v2 as imageio
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from icecream import ic


def read_video_to_numpy(video_path):
    """
    读取 mp4，返回 shape = [T, H, W, C] 的 uint8 numpy array
    """
    reader = imageio.get_reader(str(video_path))
    frames = []
    for frame in reader:
        frames.append(frame)
    reader.close()
    return np.stack(frames, axis=0)

def collect_dataset_from_phydomain(data_root, out_path, only_success=True, use_eye_in_hand=False):
    data_root = Path(data_root)

    buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())

    episode_dirs = sorted(data_root.glob("episode_*"))

    num_total = 0
    num_added = 0
    num_skipped = 0

    for ep_idx, ep_dir in enumerate(episode_dirs):
        num_total += 1

        meta_path = ep_dir / "episode_meta.json"
        if not meta_path.exists():
            print(f"[skip] {ep_dir.name}: missing episode_meta.json")
            num_skipped += 1
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        success = bool(meta.get("success", False))
        if only_success and not success:
            print(f"[skip] {ep_dir.name}: success=False")
            num_skipped += 1
            continue

        agentview_path = ep_dir / "agentview.mp4"
        lowdim_path = ep_dir / "lowdim.npz"

        required_paths = [agentview_path, lowdim_path]

        if use_eye_in_hand:
            eye_in_hand_path = ep_dir / "eye_in_hand.mp4"
            required_paths.append(eye_in_hand_path)

        if not all(path.exists() for path in required_paths):
            print(f"[skip] {ep_dir.name}: missing data files")
            num_skipped += 1
            continue

        agentview_image = read_video_to_numpy(agentview_path)
        lowdim = np.load(lowdim_path)

        episode = {
            "agentview_image": agentview_image,
            "robot0_eef_pos": lowdim["robot0_eef_pos"],
            "robot0_eef_quat": lowdim["robot0_eef_quat"],
            "robot0_gripper_qpos": lowdim["robot0_gripper_qpos"],
            "action": lowdim["action"],
        }

        if use_eye_in_hand:
            episode["robot0_eye_in_hand_image"] = read_video_to_numpy(eye_in_hand_path)

        T = episode["action"].shape[0]

        keys_to_check = [
            "agentview_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]

        if use_eye_in_hand:
            keys_to_check.append("robot0_eye_in_hand_image")

        if not all(episode[k].shape[0] == T for k in keys_to_check):
            print(f"[skip] {ep_dir.name}: length mismatch")
            num_skipped += 1
            continue

        buffer.add_episode(episode)
        num_added += 1

        msg = (
            f"[add] {ep_dir.name}: "
            f"len={T}, "
            f"agentview={episode['agentview_image'].shape}, "
        )

        if use_eye_in_hand:
            msg += f"eye_in_hand={episode['robot0_eye_in_hand_image'].shape}, "

        msg += (
            f"action={lowdim['action'].shape}, "
            f"success={success}"
        )

        print(msg)

    print(f"\nDone. total={num_total}, added={num_added}, skipped={num_skipped}")

    buffer.save_to_path(out_path)
    return buffer


if __name__ == "__main__":
    data_root = "/mnt/storage/DP_data/pickup/episodes"
    out_path = "/mnt/storage/DP_data/pickup/pickup.zarr"

    use_eye_in_hand=False

    
    buffer = collect_dataset_from_phydomain(
        data_root,
        out_path,
        only_success=True,
        use_eye_in_hand=False,   # 没有 eye_in_hand.mp4 就设 False
    )
    print("done")

    buffer = ReplayBuffer.copy_from_path(out_path)
    ic(buffer.meta)
    ic(buffer.data.keys())
    action = buffer["action"]