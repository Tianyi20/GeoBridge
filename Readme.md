

我有 franka 的 URDF 文件，我想要根据我的 joints 的 position 和 gripper 的width 来导出他对应的 .ply mesh 文件。

franka urdf 在 franka_panda/panda.urdf . 已经包括了 gripper。

你需要帮我写这个 franka_export_mesh_by_joint.py。

我的目的是拿到 这个mesh 后和我的 real2sim 的 recon env 对齐，对齐你可以不用管，我手动对齐就行了。
下面是 joints angles 和 gripper width
ic| controller.get_joint_positions(): [np.float64(-0.07630907275429047),
                                       np.float64(0.16981874418467807),
                                       np.float64(-0.5780407272806453),
                                       np.float64(-1.427642524985215),
                                       np.float64(0.055147870278131755),
                                       np.float64(1.5863168275091382),
                                       np.float64(-1.0853228909191157)]
ic| controller.get_gripper_width(): 0.08021380007266998

```shell
python franka_export_mesh_by_joints.py   --urdf franka_panda/panda.urdf   --output franka_panda_pose.ply
```

 
```shell
python get_initial_grasp_guess.py data/objects/bracket/bracket.obj --out bracket_pose.yaml
```


```sh
blender -b --python usdz_to_obj.py -- \
  ./data/banana/\
  ./data/objects/banana \
  --apply-scale \
  --triangulate
```

```shell
blender --background --python bake_ply_vertex_color_blender.py
```

Note always `--restart-every` after 5 collections to prevent the process ran out of memory
```shell
python MP_collectdata.py --num-episodes 3000 --num-processes 24 --restart-every 2
```
