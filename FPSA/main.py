from FPSA import ShapeAugmentor
import numpy as np
from icecream import ic

if __name__ == "__main__":
    obj_path = "/home/iadc/GeoBridge/data/objects/mug/original_mug/model.obj"
    initial_grasp_path = "/home/iadc/GeoBridge/data/objects/mug/original_mug/mug_grasp.yaml"

    augmentor = ShapeAugmentor(obj_path=obj_path, 
                               initial_grasp_path=initial_grasp_path)
    
    augmentor.displacement_reshape(constraint_ids= [1087, 1135, 1260, 1276, 3551],
                                   displace_idxs= [3551],
                                   displacements= np.array([0.0067756052628117945,
                                                -1.566278750761965e-08,
                                                -0.1021735721360982]),
                                    )
    augmentor.write_augment_obj(output_path = "test.obj")

    T_new, anchor, debug = augmentor.transfer_initial_grasp_guess(
        k_ring=2,
        use_distance_weights=True,
        quat_order="xyzw",
        return_format="T",
    )

    augmentor.visualize_deformed_grasp_pose(
        T_grasp_new=T_new,
        anchor=anchor,
        debug_info=debug,
        show_anchor=True,
        show_patch=True,
    )

    print("new SE3 grasp pose:")
    print(T_new)

    print("shape matching fit error:", debug["fit_error_mean"], debug["fit_error_max"])