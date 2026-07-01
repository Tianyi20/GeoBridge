from FPSA import ShapeAugmentor
import numpy as np
from icecream import ic

if __name__ == "__main__":
    obj_path = "/home/iadc/GeoBridge/data/objects/wrench/wrench_repaired.obj"
    initial_grasp_path = "/home/iadc/GeoBridge/data/objects/wrench/wrench_engage.yaml"

    augmentor = ShapeAugmentor(obj_path=obj_path, 
                               initial_grasp_path=initial_grasp_path)
    
    augmentor.displacement_reshape(constraint_ids = [
                                    0, 151, 153, 155, 164, 171, 178, 204,
                                    217, 229, 273, 281, 282, 320, 334, 340, 44, 510,
                                    512, 55, 56, 57, 583, 612, 621, 64, 65, 66, 68, 70,
                                    71, 72, 73, 85],
                                    displace_idxs = [
                                        155, 66, 68, 70, 71, 72, 73
                                    ],
                                    displacements = np.array([
                                        [0.0, -0.07, 0.0],
                                        [0.0, -0.07, 0.0],
                                        [0.0, -0.07, 0.0],
                                        [0.0, -0.07, 0.0],
                                        [0.0, -0.07, 0.0],
                                        [0.0, -0.07, 0.0],
                                        [0.0, -0.07, 0.0],
                                    ]),
                                    max_iters= 100,
                                    reshape_method= "APAP"
                                    )
    
    augmentor.write_augment_obj(output_path = "test.obj")

    T = np.eye(4)
    T[:3, 3] =  [0.16417, 0.0, 0.0]

    T_new, anchor, debug = augmentor.transfer_grasp_SE3(
        T_grasp_old= T,
        k_ring=5,
        use_distance_weights=True,
        quat_order="xyzw",
        patch_method= "xyz"
    )

    # T_new, anchor, debug = augmentor.transfer_initial_grasp_guess(
    #     k_ring=2,
    #     use_distance_weights=True,
    #     quat_order="xyzw",
    #     return_format="T",
    # )

    augmentor.visualize_deformed_grasp_pose(
        T_grasp_new=T_new,
        anchor=anchor,
        debug_info=debug,
        show_anchor=True,
        show_patch=True,
        show_old_grasp = True,
        T_grasp_old = T,
    )

    print("new SE3 grasp pose:")
    print(T_new)

    print("shape matching fit error:", debug["fit_error_mean"], debug["fit_error_max"])