"""
Multiprocessing data collector for PickUpSim.

Usage examples:
  python record_data_mp.py --num-episodes 100 --num-processes 8
  python record_data_mp.py --num-episodes 3000 --num-processes 20 --base-dir /mnt/storage/DP_data/pickup/episodes --base-seed 43

Design:
  - Parent process owns only task scheduling and progress reporting.
  - Each worker process owns exactly one independent PyBullet DIRECT client.
  - Each episode is collected in a fresh resetSimulation() world.
  - Episode ids and seeds are deterministic: seed = base_seed + episode_id + 1.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Keep numeric / BLAS libraries from oversubscribing every worker process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing as mp
from multiprocessing.connection import Connection

import numpy as np
import pybullet as pb
import pybullet_data as pd
from pybullet_utils import bullet_client
from tqdm import tqdm

from PickUpSim import PickUpSim
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
    fps: int = 20
    max_steps: int = 1000
    record_every_n_sim_steps: int = 6
    use_gui: bool = False
    overwrite: bool = False


def build_scene_kwargs() -> Dict[str, Any]:
    """Scene settings copied from the original single-process record_data.py."""
    return {
        "env_mesh_path": "./data/background/patched_table/tabletop.obj",
        "manipulated_obj_path": "./data/objects/banana/banana.obj",
        "initial_grasp_path": "./data/objects/banana/grasp.yaml",
        "obj_pose_base": [0.55, 0.0, 0.1],
        "obj_euler_base": [math.pi / 2, 0.0, math.pi / 2],
        "randomize_image_noise": True,
        "randomize_lighting": True,
        "randomize_objpose": True,
        "x_jitter_range": 0.15,
        "y_jitter_range": 0.2,
        "z_axis_rotation_range": np.pi,
        "randomize_distractors": True,
        "distractor_root": "/mnt/storage/GoogleScannedObjects",
        "distractor_num_range": (0, 4),
        "distractor_target_size_range": (0.06, 0.3),
        "distractor_workspace": ((0.05, 0.78), (-0.42, 0.42)),
        "distractor_min_target_mask_pixels": 10,
    }


def setup_world(client: bullet_client.BulletClient, time_step: float) -> None:
    """Apply per-world PyBullet settings after connect or resetSimulation."""
    client.setAdditionalSearchPath(pd.getDataPath())
    client.setTimeStep(time_step)
    client.setGravity(0, 0, -9.8)
    client.setPhysicsEngineParameter(solverResidualThreshold=0)


def collect_one_episode(
    sim: PickUpSim,
    episode_dir: Path,
    meta_seed: int,
    fps: int = 20,
    max_steps: int = 1000,
    record_every_n_sim_steps: int = 6,
) -> bool:
    """
    Collect one episode with the same observation/action order as record_data.py.

    Important multiprocessing change:
      use sim.bullet_client.stepSimulation(), not global pybullet.stepSimulation().
    """
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
        while (not sim.done) and sim_step < max_steps:
            if sim_step % record_every_n_sim_steps == 0:
                obs = sim.collect_observation(direct=False)

            # Compute and apply this control step.
            sim.step()

            if sim_step % record_every_n_sim_steps == 0:
                action = sim.collect_action()
                timestamp = record_idx / float(fps)
                writer.add_step(obs, action, timestamp)
                record_idx += 1

            sim.bullet_client.stepSimulation()
            sim_step += 1

        success = sim.is_success()
        writer.close(success=success)
        return bool(success)
    except Exception:
        # Try to close cleanly so partial metadata is not left open.
        try:
            writer.close(success=False)
        except Exception:
            pass
        raise


def prepare_episode_dir(episode_dir: Path, overwrite: bool) -> Tuple[bool, str]:
    """
    Returns (should_collect, reason).
    Existing directories are skipped unless --overwrite is set.
    """
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

    # Reset world per episode to avoid object/contact/state leakage.
    client.resetSimulation()
    setup_world(client, cfg.time_step)

    # Disable GUI rendering while loading, matching the batchsim pattern.
    try:
        client.configureDebugVisualizer(client.COV_ENABLE_RENDERING, 0)
    except Exception:
        pass

    np.random.seed(seed)

    sim = PickUpSim(client, offset=[0, 0, 0], control_dt=cfg.time_step, seed=seed)
    sim.make_scene(**build_scene_kwargs())
    sim.enable_high_quality_rendering()

    try:
        client.configureDebugVisualizer(client.COV_ENABLE_RENDERING, 1)
    except Exception:
        pass

    success = collect_one_episode(
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


def collector_worker(rank: int, num_processes: int, child_pipe: Connection, cfg: CollectorConfig) -> None:
    """Pipe-driven worker, adapted from PyBullet's batchsim3_grasp.py pattern."""
    client: Optional[bullet_client.BulletClient] = None

    while True:
        try:
            message, payload = child_pipe.recv()
        except (EOFError, KeyboardInterrupt):
            break

        if message == _RESET:
            try:
                connection_mode = pb.GUI if cfg.use_gui and rank == 0 else pb.DIRECT
                client = bullet_client.BulletClient(connection_mode=connection_mode)
                setup_world(client, cfg.time_step)
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
                child_pipe.send(result)
            except Exception:
                # Reset after failure so the next task starts from a clean physics world.
                try:
                    client.resetSimulation()
                    setup_world(client, cfg.time_step)
                except Exception:
                    pass
                child_pipe.send({
                    "rank": rank,
                    "episode_id": episode_id,
                    "seed": cfg.base_seed + episode_id + 1,
                    "status": "error",
                    "success": False,
                    "traceback": traceback.format_exc(),
                })
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


def start_workers(cfg: CollectorConfig) -> Tuple[List[mp.Process], List[Connection]]:
    processes: List[mp.Process] = []
    parent_pipes: List[Connection] = []

    for rank in range(cfg.num_processes):
        parent_pipe, child_pipe = mp.Pipe()
        proc = mp.Process(
            target=collector_worker,
            args=(rank, cfg.num_processes, child_pipe, cfg),
            daemon=False,
        )
        proc.start()
        parent_pipes.append(parent_pipe)
        processes.append(proc)

    return processes, parent_pipes


def run_parent_scheduler(cfg: CollectorConfig) -> List[Dict[str, Any]]:
    cfg.base_dir.mkdir(parents=True, exist_ok=True)
    processes, parent_pipes = start_workers(cfg)

    results: List[Dict[str, Any]] = []

    try:
        # RESET all workers first.
        for pipe in parent_pipes:
            pipe.send((_RESET, None))

        for pipe in parent_pipes:
            msg = pipe.recv()
            if msg.get("status") != "reset_ok":
                raise RuntimeError(f"Worker reset failed: {msg}")

        # Dynamic scheduling: give each free worker exactly one next episode.
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
                # Poll avoids blocking forever on a worker that is not the next one in rank order.
                for rank, pipe in enumerate(parent_pipes):
                    if rank not in active:
                        continue
                    if not pipe.poll(0.05):
                        continue

                    result = pipe.recv()
                    results.append(result)
                    active.pop(rank, None)
                    pbar.update(1)

                    if result.get("status") == "error":
                        pbar.write(
                            f"[worker {rank}] episode {result.get('episode_id')} failed; "
                            f"see traceback in final summary."
                        )

                    if next_episode < cfg.num_episodes:
                        pipe.send((_COLLECT, {"episode_id": next_episode}))
                        active[rank] = next_episode
                        next_episode += 1

    finally:
        # Ask every worker to close, even if collection failed.
        for pipe in parent_pipes:
            try:
                pipe.send((_CLOSE, None))
            except Exception:
                pass
        for pipe in parent_pipes:
            try:
                if pipe.poll(1.0):
                    pipe.recv()
            except Exception:
                pass
        for proc in processes:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)

    return results


def parse_args() -> CollectorConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--num-processes", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--base-seed", type=int, default=43)
    parser.add_argument("--base-dir", type=Path, default=Path("./DP_data/pickup/episodes"))
    parser.add_argument("--time-step", type=float, default=1.0 / 120.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--record-every-n-sim-steps", type=int, default=6)
    parser.add_argument("--use-gui", action="store_true", help="Only worker 0 uses GUI; others remain DIRECT.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing episode dirs before recollecting.")
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
    # spawn is safer than fork for PyBullet/OpenGL/multiprocessing.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    config = parse_args()
    all_results = run_parent_scheduler(config)
    print_summary(all_results)
