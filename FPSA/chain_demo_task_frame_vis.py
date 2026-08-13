from icecream import ic

from FPSA_task_frame_vis import ShapeAugmentor
import numpy as np


if __name__ == "__main__":
    obj_path = "/home/iadc/GeoBridge/data/objects/wrench/high_quality/wrench.obj"
    #initial_grasp_path = "/home/iadc/GeoBridge/data/objects/wrench/wrench_v2/wrench_engage.yaml"

    augmentor = ShapeAugmentor(
        obj_path=obj_path,
        initial_grasp_path=None,
    )
    # Initial TCP
    T_init_tcp = np.eye(4)
    T_init_tcp[:3, 3] = np.array([0.06989, 0.0, 0.0], dtype=float)

    # Same constraint set as your original main.py.
    constraint_ids =  [33, 12, 944, 942, 148, 273, 149, 151, 316, 271, 150, 947, 270,
      951, 949, 32, 283, 13, 757, 152, 749, 153, 154, 524, 279, 771, 272, 946, 945,
      943, 21, 315, 11, 194, 195, 196, 22, 197, 848, 116, 114, 115, 8, 24, 296, 217,
      218, 219, 220, 221, 222, 223, 224, 950, 948, 185, 184, 6, 7, 109, 246, 245,
      244, 243, 242, 241, 240, 239, 238, 237, 236, 247, 290, 1247, 1238, 166, 162,
      1213, 284, 285, 286, 1131, 301, 1193, 1171, 1163, 1160, 855, 870, 866, 876,
      124, 1154, 304, 871, 129, 134, 268, 267, 266, 794, 177, 528, 167, 159, 537,
      311, 307, 10, 783, 604, 532, 657, 665, 180, 811, 820, 120, 607, 118, 539, 145,
      548, 520, 577, 131, 551, 538, 606, 183, 5, 4, 104, 28, 293, 17, 18, 294, 27,
      20, 295, 25, 521, 514, 509, 233, 234, 93, 61, 731, 726]

    T_origin = np.eye(4)
    T_origin[:3, 3] = [0.0, 0.0, 0.0]
    # ============================================================
    # Step 1: APAP refinement
    # ============================================================
    # This call starts from Step 1's result, not from the original mesh.
    # The displacements below are therefore incremental displacements on top of
    # the slippage output.

    jaw_move_ids = [5, 4, 104, 28, 293, 17, 18, 294, 27, 20, 295, 25, 521, 514, 509,
      233, 234, 93, 61, 731, 726]
    jaw_slippage_displacements = np.array([
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
        [0.0, 0.04, 0.04],
    ])

    V_after_slippage = augmentor.displacement_reshape(
        constraint_ids=constraint_ids,
        displace_idxs=jaw_move_ids,
        displacements=jaw_slippage_displacements,
        max_iters=200,
        reshape_method="APAP",
        input_name="step01_apap",
    )

    augmentor.write_augment_obj(
        output_path="step01_apap.obj",
        write_coacd=False,
    )

    augmentor.visualize_reshaped_mesh()

    print("V_after_slippage:", V_after_slippage.shape)

    # ============================================================
    # Step 2: slippage reshaping
    # ============================================================
    # This starts from the original mesh. After this call, augmentor.V_work is
    # automatically updated to the slippage result, so the next APAP call will
    # use the slippage-deformed mesh as input.
    constraint_ids = [33, 12, 944, 942, 148, 273, 149, 151, 316, 271, 150, 947, 270,
      951, 949, 32, 283, 13, 757, 152, 749, 153, 154, 524, 279, 771, 272, 946, 945,
      943, 21, 315, 11, 194, 195, 196, 22, 197, 848, 116, 114, 115, 8, 24, 296, 217,
      218, 219, 220, 221, 222, 223, 224, 950, 948, 185, 184, 6, 7, 109, 246, 245,
      244, 243, 242, 241, 240, 239, 238, 237, 236, 247, 290, 1247, 1238, 166, 162,
      1213, 284, 285, 286, 1131, 301, 1193, 1171, 1163, 1160, 855, 870, 866, 876,
      124, 1154, 304, 871, 129, 134, 268, 267, 266, 794, 177, 528, 167, 159, 537,
      311, 307, 10, 783, 604, 532, 657, 665, 180, 811, 820, 120, 607, 118, 539, 145,
      548, 520, 577, 131, 551, 538, 606, 183, 5, 4, 104, 28, 293, 17, 18, 294, 27,
      20, 295, 25, 521, 514, 509, 233, 234, 93, 61, 731, 726]
    apap_move_ids = [5, 4, 104, 28, 293, 17, 18, 294, 27, 20, 295, 25, 521, 514, 509,
      233, 234, 93, 61, 731, 726]
    apap_refine_displacements = np.array([
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
    ])

    V_final = augmentor.displacement_reshape(
        constraint_ids=constraint_ids,
        displace_idxs=apap_move_ids,
        displacements=apap_refine_displacements,
        max_iters=120,
        reshape_method="slippage",
        input_name="step02_slippage",
    )

    augmentor.write_augment_obj(
        output_path="step02_slippage.obj",
        write_coacd=True,
    )

    print("V_final:", V_final.shape)

    # ============================================================
    # Grasp transfer and visualization
    # ============================================================
    # self.V is still the original reference mesh. self.V_opt is the final mesh
    # after slippage -> APAP, so grasp transfer maps the original grasp to the
    # final chained deformation.


    T_new, anchor, debug = augmentor.transfer_grasp_SE3(
        T_grasp_old=T_origin,
        k_ring=3,
        use_distance_weights=True,
        quat_order="xyzw",
        patch_method="k_ring",
    )

    augmentor.visualize_deformed_grasp_pose(
        T_grasp_new=T_new,
        anchor=anchor,
        debug_info=debug,
        show_anchor=True,
        show_patch=True,
        show_old_grasp=True,
        T_grasp_old=T_origin,
    )

    # Compose the fixed TCP with the inverse transferred task frame using the
    # convention expected by the downstream wrench simulation.
    wrench_to_tcp_T = T_init_tcp @ np.linalg.inv(T_new)
    ic(wrench_to_tcp_T)

    # Publication-style Open3D figure.  This must run before changing V_opt
    # below, otherwise the saved patch correspondences would no longer share
    # the same frame as the shape-matching debug data.
    augmentor.visualize_task_frame_transfer(
        T_grasp_old=T_origin,
        T_new=T_new,
        T_tcp=T_init_tcp,
        wrench_to_tcp_T=wrench_to_tcp_T,
        anchor=anchor,
        debug_info=debug,
        show=True,
        export_path="shape_matching_task_frame_transfer.png",
        width=1800,
        height=1200,
    )

    # transform the origin of mesh by T_new
    augmentor.apply_transformation_to_mesh(np.linalg.inv(T_new))
    augmentor.visualize_reshaped_mesh()
    augmentor.write_augment_obj(
        output_path="transformed_mesh.obj",
        write_coacd=True,
    )

    # save wrench_to_tcp_T as a yaml file for later use in WrenchSim_equidistant_eye.py
    
