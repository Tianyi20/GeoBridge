from pathlib import Path
import json
import numpy as np
import imageio.v2 as imageio
import zarr
import concurrent.futures
import multiprocessing
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer


def read_video_to_numpy(video_path):
    reader = imageio.get_reader(str(video_path))
    frames = []
    for frame in reader:
        frames.append(frame)
    reader.close()
    return np.stack(frames, axis=0).astype(np.uint8)


def load_one_episode(args):
    ep_dir, only_success, use_eye_in_hand = args

    meta_path = ep_dir / "episode_meta.json"
    if not meta_path.exists():
        return None, f"[skip] {ep_dir.name}: missing episode_meta.json"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    success = bool(meta.get("success", False))
    if only_success and not success:
        return None, f"[skip] {ep_dir.name}: success=False"

    agentview_path = ep_dir / "agentview.mp4"
    lowdim_path = ep_dir / "lowdim.npz"

    required_paths = [agentview_path, lowdim_path]

    if use_eye_in_hand:
        eye_in_hand_path = ep_dir / "eye_in_hand.mp4"
        required_paths.append(eye_in_hand_path)

    if not all(path.exists() for path in required_paths):
        return None, f"[skip] {ep_dir.name}: missing data files"

    try:
        lowdim = np.load(lowdim_path)

        episode = {
            "agentview_image": read_video_to_numpy(agentview_path),
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
            return None, f"[skip] {ep_dir.name}: length mismatch"

        msg = (
            f"[add] {ep_dir.name}: "
            f"len={T}, "
            f"agentview={episode['agentview_image'].shape}, "
        )

        if use_eye_in_hand:
            msg += f"eye_in_hand={episode['robot0_eye_in_hand_image'].shape}, "

        msg += f"action={episode['action'].shape}, success={success}"

        return episode, msg

    except Exception as e:
        return None, f"[skip] {ep_dir.name}: error={repr(e)}"


def collect_dataset_from_phydomain_parallel(
    data_root,
    out_path,
    only_success=True,
    use_eye_in_hand=False,
    num_workers=None,
):
    data_root = Path(data_root)
    episode_dirs = sorted(data_root.glob("episode_*"))

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)

    buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())

    num_total = len(episode_dirs)
    num_added = 0
    num_skipped = 0

    tasks = [
        (ep_dir, only_success, use_eye_in_hand)
        for ep_dir in episode_dirs
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_one_episode, task) for task in tasks]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Loading episodes",
        ):
            episode, msg = future.result()
            # print(msg)

            if episode is None:
                num_skipped += 1
                continue

            buffer.add_episode(episode)
            num_added += 1

    print(f"\nDone. total={num_total}, added={num_added}, skipped={num_skipped}")

    buffer.save_to_path(out_path)
    return buffer


if __name__ == "__main__":
    data_root = "/mnt/storage/DP_data/pickup/20260613_221334/episodes"
    out_path = "./data/DP_data/bracket_v1/pickup.zarr"

    buffer = collect_dataset_from_phydomain_parallel(
        data_root,
        out_path,
        only_success=True,
        use_eye_in_hand=False,
        num_workers=8,
    )

    print("done")

    buffer = ReplayBuffer.copy_from_path(out_path)
    print(buffer.meta)
    print(buffer.data.keys())
    action = buffer["action"]