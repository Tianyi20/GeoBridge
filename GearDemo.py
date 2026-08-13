import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from GearSim import GearSim

import pkgutil


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# egl = pkgutil.get_loader('eglRenderer')
# if (egl):
#     pluginId = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
# else:
#     pluginId = p.loadPlugin("eglRendererPlugin")

time_step = 1.0 / 120.0
p.setTimeStep(time_step)
p.setGravity(0.0, 0.0, -9.8)

sim = GearSim(
    p,
    offset=[0.0, 0.0, 0.0],
    control_dt=time_step,
    seed=1222,
    randomize_initial_ee_pose=True,
)

sim.make_scene(
    env_mesh_path="./data/background/repaired_table/tabletop.obj",
    gear_path="./data/objects/gear_extraction/gear/gear.obj",
    supportor_path="./data/objects/gear_extraction/supportor/supportor.obj",
    initial_grasp_path="./data/objects/gear_extraction/gear/grasp.yaml",
    gear_collision_path=None,
    supportor_collision_path=None,

    if_FPSA_gear=True,
    fpsa_gear_aug_root="./data/objects/gear_extraction/gear/Gear_aug_outputs_uniform_scaling_baseline",
    fpsa_gear_include_base=True,

    # The gear origin is located at supportor-local (0, 0, 0.31).
    supportor_pose_base=[0.62, 0.10, 0.015],
    supportor_euler_base=[0.0, 0.0, 0.0],
    gear_to_support_pos=[0.0, 0.0, 0.32],
    gear_to_support_euler=[0.0, np.pi, np.pi],

    # Nearby low placement target, also expressed in the supportor frame.
    support_to_place_pos=[-0.15, 0.0, 0.1],
    support_to_place_euler=None,  # preserve the extracted gear orientation
    randomize_task_pose=True,
    task_x_jit=0.10,
    task_y_jit=0.06,
    task_z_jit=0.03,
    task_yaw_jit= 0.0,

    safe_approach=0.03,
    lift_height=0.05,
    place_approach_height=0.04,
    retreat_height=0.0,

    # Same visual-domain randomization configuration as AssemblyDemo.
    randomize_lighting=True,
    randomize_outlscene=True,
    outlscene_xyz_jit=0.015,
    outlscene_eul_jit=0.001,
    randomize_plane_height=True,
    plane_height_jit=0.003,
    randomize_campose=True,
    cam_xyz_jit=0.01,
    cam_eul_jit=0.005,
    randomize_fisheye_cam=True,
    fisheye_eyz_jit=0.005,
    fisheye_eul_jit=0.002,
    randomize_camera_intrinsic=True,
    randomize_image_noise=True,
    randomize_object_color=True,
    randomize_robot_texture=True,
    robot_texture_patterns=("checkers", "gradient", "noise", "plain"),
    object_color_mode="bounded",
    object_color_strength=0.5,
    randomize_distractors=True,
    distractor_root="/mnt/storage/GoogleScannedObjects",
    distractor_num_range=(0, 5),
    distractor_target_size_range=(0.1, 0.5),
    distractor_workspace=((-0.2, 1.3), (-0.72, 0.42)),
    distractor_clearance=0.06,
    distractor_path_clearance=0.06,
    distractor_min_target_mask_pixels=10,

    # Keep AssemblySim's post-grasp pose randomization, but use a conservative
    # range so the extracted gear does not get teleported into the support pin.
    fix_gear_to_gripper=True,
    randomize_object_in_hand_pose=True,
    object_in_hand_x_jit=0.002,
    object_in_hand_y_jit=0.002,
    object_in_hand_z_jit=0.002,
    object_in_hand_roll_jit= 0.002,
    object_in_hand_pitch_jit= 0.002,
    object_in_hand_yaw_jit= 0.002,
    object_in_hand_debug=False,
)

sim.enable_high_quality_rendering()

sim_step = 0
record_idx = 0
record_every_n_sim_steps = 12

try:
    while not sim.done:
        p.stepSimulation()
        sim.step()
        sim_step += 1
        time.sleep(0.01)

        if sim_step % record_every_n_sim_steps == 0:
            start = time.time()

            # eye_rgb = sim.get_eye_in_hand_image()
            # RGB_agent = sim.direct_get_agent_view()
            # cv2.imwrite("temp_fisheye.png",cv2.cvtColor(eye_rgb, cv2.COLOR_RGB2BGR))
            # cv2.imwrite("temp_agentview.png",cv2.cvtColor(RGB_agent, cv2.COLOR_RGBA2BGRA))

            stop = time.time()
            print("renderImage %f" % (stop - start))

            record_idx += 1
            print(record_idx, sim.state, sim.is_success(debug=False))

    print(sim.is_success(require_done=True, return_info=True, debug=True))

except KeyboardInterrupt:
    print("Stopped by user")
