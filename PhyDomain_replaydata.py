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


def collect_dataset_from_phydomain(data_root, out_path, only_success=True):
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

        # 1. 读视频
        agentview_path = ep_dir / "agentview.mp4"
        eye_in_hand_path = ep_dir / "eye_in_hand.mp4"
        lowdim_path = ep_dir / "lowdim.npz"

        if (not agentview_path.exists()) or (not eye_in_hand_path.exists()) or (not lowdim_path.exists()):
            print(f"[skip] {ep_dir.name}: missing data files")
            num_skipped += 1
            continue

        agentview_image = read_video_to_numpy(agentview_path)
        robot0_eye_in_hand_image = read_video_to_numpy(eye_in_hand_path)

        # 2. 读 lowdim
        lowdim = np.load(lowdim_path)

        episode = {
            "agentview_image": agentview_image,
            "robot0_eye_in_hand_image": robot0_eye_in_hand_image,
            "robot0_eef_pos": lowdim["robot0_eef_pos"],
            "robot0_eef_quat": lowdim["robot0_eef_quat"],
            "robot0_gripper_qpos": lowdim["robot0_gripper_qpos"],
            "action": lowdim["action"],
        }

        # 可选：做一下长度一致性检查
        T = episode["action"].shape[0]
        if not (
            episode["agentview_image"].shape[0] == T and
            episode["robot0_eye_in_hand_image"].shape[0] == T and
            episode["robot0_eef_pos"].shape[0] == T and
            episode["robot0_eef_quat"].shape[0] == T and
            episode["robot0_gripper_qpos"].shape[0] == T
        ):
            print(f"[skip] {ep_dir.name}: length mismatch")
            num_skipped += 1
            continue

        # 3. 加到 replay buffer
        buffer.add_episode(episode)
        num_added += 1

        print(
            f"[add] {ep_dir.name}: "
            f"len={episode['action'].shape[0]}, "
            f"agentview={episode['agentview_image'].shape}, "
            f"eye_in_hand={episode['robot0_eye_in_hand_image'].shape}, "
            f"action={lowdim['action'].shape}, "
            f"success={success}"
        )

    print(
        f"\nDone. total={num_total}, added={num_added}, skipped={num_skipped}"
    )

    buffer.save_to_path(out_path)
    return buffer


if __name__ == "__main__":
    data_root = "/home/iadc/PhyDomain/collected_episodes"
    out_path = "/home/iadc/PhyDomain/CupOnRack_fixed_angle.zarr"

    buffer = collect_dataset_from_phydomain(
        data_root,
        out_path,
        only_success=True
    )
    print("done")

    buffer = ReplayBuffer.copy_from_path(out_path)
    ic(buffer.meta)
    ic(buffer.data.keys())
    action = buffer["action"]