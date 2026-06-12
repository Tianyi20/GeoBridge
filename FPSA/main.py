from FPSA import ShapeAugmentor
import numpy as np
from icecream import ic

if __name__ == "__main__":
    obj_path = "/home/iadc/GeoBridge/data/objects/bracket/bracket.obj"
    initial_grasp_path = "/home/iadc/GeoBridge/data/objects/bracket/bracket_grasp.yaml"

    augmentor = ShapeAugmentor(obj_path=obj_path, 
                               initial_grasp_path=initial_grasp_path)
    
    augmentor.displacement_reshape(constraint_ids= [339, 343, 345, 346, 846],
                                   displace_idxs= [846],
                                   displacements= np.array([0.21873721885442937,
                                                        -4.568995076824393e-09,
                                                        -8.609569257000194e-10]),
                                    max_iters= 100
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