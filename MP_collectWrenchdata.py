"""
Multiprocessing data collector for WrenchSim.

Example:
  python MP_collectWrenchdata.py --num-episodes 3000 --num-processes 25 --restart-every 2

Design:
  - Parent process only schedules tasks and reports progress.
  - Each worker owns one independent PyBullet client.
  - Each episode starts from resetSimulation().
  - Episode ids and seeds are deterministic: seed = base_seed + episode_id + 1.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pybullet as pb
import pybullet_data as pd
from multiprocessing.connection import Connection
from pybullet_utils import bullet_client
from tqdm import tqdm

from WrenchSim import WrenchSim
from episode_writer import EpisodeWriter


_RESET = 1
_COLLECT = 2
_CLOSE = 3


@dataclass(frozen=True)
class CollectorConfig:
    num_episodes: int
    num_processes: int
    base_seed: int
    base_dir: Path
    time_step: float = 1.0 / 120.0
    fps: int = 10
    max_steps: int = 2000
    record_every_n_sim_steps: int = 12
    use_gui: bool = False
    overwrite: bool = False
    restart_every: int = 50


def default_base_dir() -> Path:
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("/mnt/storage/DP_data/wrench_engagement") / run_time / "episodes"


def build_scene_kwargs() -> Dict[str, Any]:
    return {
        "env_mesh_path": "./data/background/repaired_table/tabletop.obj",
        "manipulated_obj_path": "./data/objects/screw/screw.obj",
        "manipulated_obj_collision_path": "./data/objects/screw/screw_collision_asset.obj",
        "clipper_obj_path": "data/objects/clipper/clipper.obj",
        "initial_grasp_path": "data/objects/wrench/wrench_engage.yaml",
        "if_FPSA": False,
        "fpsa_aug_root": "./data/objects/bracket/fpsa_aug_outputs",
        "fpsa_include_base": True,
        "obj_pose_base": [0.7, -0.05, 0.25],
        "obj_euler_base": [0.0, 0.0, 0.0],
        "randomize_lighting": True,
        "randomize_outlscene": True,
        "outlscene_xyz_jit": 0.015,
        "outlscene_eul_jit": 0.001,
        "randomize_plane_height": True,
        "plane_height_jit": 0.002,
        "randomize_objpose": True,
        "obj_x_jit": 0.10,
        "obj_y_jit": 0.10,
        "obj_z_jit": 0.10,
        "obj_z_eul_jit": np.pi / 6,
        "randomize_campose": True,
        "cam_xyz_jit": 0.01,
        "cam_eul_jit": 0.005,
        "randomize_image_noise": True,
        "randomize_object_color": True,
        "object_color_mode": "bounded",
        "object_color_strength": 0.8,
        "randomize_distractors": True,
        "distractor_root": "/mnt/storage/GoogleScannedObjects",
        "distractor_num_range": (0, 5),
        "distractor_target_size_range": (0.06, 0.4),
        "distractor_workspace": ((-0.2, 0.8), (-0.72, 0.42)),
        "distractor_min_target_mask_pixels": 10,
    }


def setup_world(client: bullet_client.BulletClient, time_step: float) -> None:
    client.setAdditionalSearchPath(pd.getDataPath())
    client.setTimeStep(time_step)
    client.setGravity(0, 0, -9.8)
    client.setPhysicsEngineParameter(solverResidualThreshold=0)


def collect_one_episode(
    client: bullet_client.BulletClient,
    sim: WrenchSim,
    episode_dir: Path,
    meta_seed: int,
    fps: int,
    max_steps: int,
    record_every_n_sim_steps: int,
) -> bool:
    writer = EpisodeWriter(
        episode_dir,
        fps=fps,
        if_agent_view=True,
        if_eye_in_hand=False,
        extra_meta={"meta_seed": meta_seed},
    )

    sim.done = False
    sim_step = 0
    record_idx = 0

    try:
        while not sim.done and sim_step < max_steps:
            should_record = sim_step % record_every_n_sim_steps == 0
            if should_record:
                obs = sim.collect_observation(direct=False)

            sim.step()

            if should_record:
                action = sim.collect_action()
                writer.add_step(obs, action, record_idx / float(fps))
                record_idx += 1

            client.stepSimulation()
            sim_step += 1

        success = bool(sim.is_success())
        writer.close(success=success)
        return success
    except Exception:
        try:
            writer.close(success=False)
        except Exception:
            pass
        raise


def prepare_episode_dir(episode_dir: Path, overwrite: bool) -> Tuple[bool, str]:
    if episode_dir.exists():
        if not overwrite:
            return False, "exists"
        shutil.rmtree(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=False)
    return True, "new"


def collect_episode_in_worker(
    rank: int,
    client: bullet_client.BulletClient,
    cfg: CollectorConfig,
    episode_id: int,
) -> Dict[str, Any]:
    seed = cfg.base_seed + episode_id + 1
    episode_dir = cfg.base_dir / f"episode_{episode_id:06d}"

    should_collect, reason = prepare_episode_dir(episode_dir, cfg.overwrite)
    if not should_collect:
        return {
            "rank": rank,
            "episode_id": episode_id,
            "seed": seed,
            "status": "skipped",
            "reason": reason,
            "success": None,
        }

    client.resetSimulation()
    setup_world(client, cfg.time_step)

    try:
        client.configureDebugVisualizer(client.COV_ENABLE_RENDERING, 0)
    except Exception:
        pass

    np.random.seed(seed)
    sim = WrenchSim(client, offset=[0, 0, 0], control_dt=cfg.time_step, seed=seed)
    sim.make_scene(**build_scene_kwargs())
    sim.enable_high_quality_rendering()

    try:
        client.configureDebugVisualizer(client.COV_ENABLE_RENDERING, 1)
    except Exception:
        pass

    success = collect_one_episode(
        client=client,
        sim=sim,
        episode_dir=episode_dir,
        meta_seed=seed,
        fps=cfg.fps,
        max_steps=cfg.max_steps,
        record_every_n_sim_steps=cfg.record_every_n_sim_steps,
    )

    return {
        "rank": rank,
        "episode_id": episode_id,
        "seed": seed,
        "status": "ok",
        "success": success,
    }


def collector_worker(rank: int, child_pipe: Connection, cfg: CollectorConfig) -> None:
    log_dir = cfg.base_dir.parent / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_f = open(log_dir / f"worker_{rank}.log", "a", buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f

    print(f"\n===== worker {rank} started, pid={os.getpid()} =====", flush=True)

    client: Optional[bullet_client.BulletClient] = None
    episodes_since_restart = 0

    while True:
        try:
            message, payload = child_pipe.recv()
        except (EOFError, KeyboardInterrupt):
            break

        if message == _RESET:
            try:
                mode = pb.GUI if cfg.use_gui and rank == 0 else pb.DIRECT
                client = bullet_client.BulletClient(connection_mode=mode)
                setup_world(client, cfg.time_step)
                episodes_since_restart = 0
                child_pipe.send({"rank": rank, "status": "reset_ok"})
            except Exception:
                child_pipe.send({
                    "rank": rank,
                    "status": "reset_error",
                    "traceback": traceback.format_exc(),
                })
            continue

        if message == _COLLECT:
            assert client is not None, "Worker must receive _RESET before _COLLECT."
            episode_id = int(payload["episode_id"])

            try:
                result = collect_episode_in_worker(rank, client, cfg, episode_id)
            except Exception:
                try:
                    client.resetSimulation()
                    setup_world(client, cfg.time_step)
                except Exception:
                    pass
                result = {
                    "rank": rank,
                    "episode_id": episode_id,
                    "seed": cfg.base_seed + episode_id + 1,
                    "status": "error",
                    "success": False,
                    "traceback": traceback.format_exc(),
                }

            if result.get("status") != "skipped":
                episodes_since_restart += 1

            result["episodes_since_restart"] = episodes_since_restart
            result["restart_worker"] = cfg.restart_every > 0 and episodes_since_restart >= cfg.restart_every
            child_pipe.send(result)
            continue

        if message == _CLOSE:
            try:
                if client is not None:
                    client.disconnect()
            except Exception:
                pass
            child_pipe.send({"rank": rank, "status": "close_ok"})
            break

    child_pipe.close()


def start_one_worker(rank: int, cfg: CollectorConfig) -> Tuple[mp.Process, Connection]:
    parent_pipe, child_pipe = mp.Pipe()
    proc = mp.Process(target=collector_worker, args=(rank, child_pipe, cfg), daemon=False)
    proc.start()
    return proc, parent_pipe


def reset_one_worker(rank: int, pipe: Connection) -> None:
    pipe.send((_RESET, None))
    msg = pipe.recv()
    if msg.get("status") != "reset_ok":
        raise RuntimeError(f"Worker {rank} reset failed: {msg}")


def close_one_worker(proc: mp.Process, pipe: Connection) -> None:
    try:
        pipe.send((_CLOSE, None))
    except Exception:
        pass

    try:
        if pipe.poll(2.0):
            pipe.recv()
    except Exception:
        pass

    try:
        pipe.close()
    except Exception:
        pass

    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)


def restart_one_worker(
    rank: int,
    cfg: CollectorConfig,
    processes: List[mp.Process],
    parent_pipes: List[Connection],
) -> None:
    close_one_worker(processes[rank], parent_pipes[rank])
    processes[rank], parent_pipes[rank] = start_one_worker(rank, cfg)
    reset_one_worker(rank, parent_pipes[rank])


def start_workers(cfg: CollectorConfig) -> Tuple[List[mp.Process], List[Connection]]:
    processes: List[mp.Process] = []
    parent_pipes: List[Connection] = []

    for rank in range(cfg.num_processes):
        proc, pipe = start_one_worker(rank, cfg)
        processes.append(proc)
        parent_pipes.append(pipe)

    return processes, parent_pipes


def run_parent_scheduler(cfg: CollectorConfig) -> List[Dict[str, Any]]:
    cfg.base_dir.mkdir(parents=True, exist_ok=True)
    processes, parent_pipes = start_workers(cfg)
    results: List[Dict[str, Any]] = []

    try:
        for rank, pipe in enumerate(parent_pipes):
            reset_one_worker(rank, pipe)

        next_episode = 0
        active: Dict[int, int] = {}

        for rank, pipe in enumerate(parent_pipes):
            if next_episode >= cfg.num_episodes:
                break
            pipe.send((_COLLECT, {"episode_id": next_episode}))
            active[rank] = next_episode
            next_episode += 1

        with tqdm(total=cfg.num_episodes, desc="Collecting episodes") as pbar:
            while active:
                for rank, pipe in enumerate(parent_pipes):
                    if rank not in active or not pipe.poll(0.05):
                        continue

                    result = pipe.recv()
                    results.append(result)
                    active.pop(rank, None)
                    pbar.update(1)

                    if result.get("status") == "error":
                        pbar.write(
                            f"[worker {rank}] episode {result.get('episode_id')} failed; "
                            "see traceback in final summary."
                        )

                    if result.get("restart_worker") and next_episode < cfg.num_episodes:
                        pbar.write(
                            f"[worker {rank}] restarting after "
                            f"{result.get('episodes_since_restart')} collected episodes"
                        )
                        restart_one_worker(rank, cfg, processes, parent_pipes)
                        pipe = parent_pipes[rank]

                    if next_episode < cfg.num_episodes:
                        pipe.send((_COLLECT, {"episode_id": next_episode}))
                        active[rank] = next_episode
                        next_episode += 1

    finally:
        for proc, pipe in zip(processes, parent_pipes):
            close_one_worker(proc, pipe)

    return results


def parse_args() -> CollectorConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--num-processes", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--base-seed", type=int, default=43)
    parser.add_argument("--base-dir", type=Path, default=default_base_dir())
    parser.add_argument("--time-step", type=float, default=1.0 / 120.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--record-every-n-sim-steps", type=int, default=12)
    parser.add_argument("--use-gui", action="store_true", help="Only worker 0 uses GUI; others remain DIRECT.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing episode dirs before recollecting.")
    parser.add_argument("--restart-every", type=int, default=50, help="Restart each worker after this many collected episodes; <=0 disables restart.")
    args = parser.parse_args()

    if args.num_processes < 1:
        raise ValueError("--num-processes must be >= 1")
    if args.num_episodes < 1:
        raise ValueError("--num-episodes must be >= 1")
    if args.use_gui and args.num_processes > 1:
        print("[warning] --use-gui is best with --num-processes 1; only worker 0 will use GUI.")

    return CollectorConfig(
        num_episodes=args.num_episodes,
        num_processes=args.num_processes,
        base_seed=args.base_seed,
        base_dir=args.base_dir,
        time_step=args.time_step,
        fps=args.fps,
        max_steps=args.max_steps,
        record_every_n_sim_steps=args.record_every_n_sim_steps,
        use_gui=args.use_gui,
        overwrite=args.overwrite,
        restart_every=args.restart_every,
    )


def print_summary(results: List[Dict[str, Any]]) -> None:
    ok = [r for r in results if r.get("status") == "ok"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    errors = [r for r in results if r.get("status") == "error"]
    successes = [r for r in ok if r.get("success") is True]

    print("\n=== Multiprocessing collection summary ===")
    print(f"episodes returned : {len(results)}")
    print(f"collected ok      : {len(ok)}")
    print(f"successful grasps : {len(successes)} / {len(ok)}")
    print(f"skipped existing  : {len(skipped)}")
    print(f"errors            : {len(errors)}")

    if errors:
        print("\nFirst error traceback:")
        print(errors[0].get("traceback", "<no traceback>"))


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    config = parse_args()
    print(f"base_dir: {config.base_dir}")
    all_results = run_parent_scheduler(config)
    print_summary(all_results)
