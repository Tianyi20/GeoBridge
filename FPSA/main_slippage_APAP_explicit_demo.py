from FPSA import ShapeAugmentor
import numpy as np


if __name__ == "__main__":
    obj_path = "/home/iadc/GeoBridge/data/objects/wrench/wrench_v2/wrench.obj"
    #initial_grasp_path = "/home/iadc/GeoBridge/data/objects/wrench/wrench_v2/wrench_engage.yaml"

    augmentor = ShapeAugmentor(
        obj_path=obj_path,
        initial_grasp_path=None,
    )

    # Same constraint set as your original main.py.
    constraint_ids = [59, 156, 58, 6, 0, 60, 61, 62, 63, 64, 57, 26, 21, 17, 13, 9,
      4, 5, 7, 11, 15, 23, 31, 40, 51, 157, 158, 162, 171, 195, 193, 189, 221, 196,
      205, 185, 207, 180, 182, 188, 187, 32, 22, 24, 55, 160, 159, 190, 191, 37, 35,
      29, 209, 212, 184, 202, 65, 14, 10, 52, 46, 44, 19, 18, 176, 172, 179, 168,
      167, 165, 89, 69, 67, 66, 68, 77, 76, 74, 75, 70, 71, 72, 73, 163, 88, 164,
      154]

    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 0.0]
    # ============================================================
    # Step 2: APAP refinement
    # ============================================================
    # This call starts from Step 1's result, not from the original mesh.
    # The displacements below are therefore incremental displacements on top of
    # the slippage output.

    jaw_move_ids = [77, 76, 74, 75, 70, 71, 72, 73, 163, 88, 164, 154]
    jaw_slippage_displacements = np.array([
        [0.0, -0.04, 0.0],
        [0.0, -0.04, 0.0],
        [0.0, -0.04, 0.0],
        [0.0, -0.04, 0.0],
        [0.0, -0.04, 0.0],
        [0.0, -0.04, 0.0],
        [0.0, -0.04, 0.0],
        [0.0, -0.04, 0.0],
        [0.0, -0.04, 0.0],
        [0.0, -0.04, 0.0],
        [0.0, -0.04, 0.0],
        [0.0, -0.04, 0.0],
    ])

    V_after_slippage = augmentor.displacement_reshape(
        constraint_ids=constraint_ids,
        displace_idxs=jaw_move_ids,
        displacements=jaw_slippage_displacements,
        max_iters=100,
        reshape_method="APAP",
        input_name="wrench_step02_APAP_refine",
    )

    augmentor.write_augment_obj(
        output_path="test_slippage_then_APAP.obj",
        write_coacd=False,
    )

    augmentor.visualize_reshaped_mesh()

    print("V_after_slippage:", V_after_slippage.shape)

    # ============================================================
    # Step 1: slippage reshaping
    # ============================================================
    # This starts from the original mesh. After this call, augmentor.V_work is
    # automatically updated to the slippage result, so the next APAP call will
    # use the slippage-deformed mesh as input.
    constraint_ids = [59, 156, 58, 6, 0, 60, 61, 62, 63, 64, 57, 26, 21, 17, 13, 9,
        4, 5, 7, 11, 15, 23, 31, 40, 51, 157, 158, 162, 171, 195, 193, 189, 221, 196,
        205, 185, 207, 180, 182, 188, 187, 32, 22, 24, 55, 160, 159, 190, 191, 37, 35,
        29, 209, 212, 184, 202, 65, 14, 10, 52, 46, 44, 19, 18, 176, 172, 179, 168,
        167, 165, 89, 69, 67, 66, 68, 73 ]

    apap_move_ids = [73]
    apap_refine_displacements = np.array([
        [-0.04, 0.0, 0.0],
        # [-0.04, 0.0, 0.0],
        # [-0.04, 0.0, 0.0],
        # [-0.04, 0.0, 0.0],
        # [-0.04, 0.0, 0.0],
        # [-0.04, 0.0, 0.0],
        # [-0.04, 0.0, 0.0],
        # [-0.04, 0.0, 0.0],
        # [-0.04, 0.0, 0.0],
        # [-0.04, 0.0, 0.0],
        # [-0.04, 0.0, 0.0],
        # [-0.04, 0.0, 0.0],

    ])

    V_final = augmentor.displacement_reshape(
        constraint_ids=constraint_ids,
        displace_idxs=apap_move_ids,
        displacements=apap_refine_displacements,
        max_iters=120,
        reshape_method="slippage",
        input_name="wrench_step01_slippage",
    )

    augmentor.write_augment_obj(
        output_path="test_step01_slippage.obj",
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
        T_grasp_old=T,
        k_ring=1,
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
        T_grasp_old=T,
    )

    print("new SE3 grasp pose:")
    print(T_new)
    print("shape matching fit error:", debug["fit_error_mean"], debug["fit_error_max"])
