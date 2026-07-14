#!/usr/bin/env python3
"""
OBJ -> Blender Voxel Remesh -> OBJ

Usage:
blender --background --python remesh_obj_blender.py -- \
    --input /path/to/input.obj \
    --output /path/to/output_remesh.obj \
    --voxel-size 0.002 \
    --merge-distance 1e-7 \
    --triangulate
"""

import argparse
import os
import sys
import bpy


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--merge-distance", type=float, default=1e-7)
    parser.add_argument("--adaptivity", type=float, default=0.0)
    parser.add_argument("--triangulate", action="store_true")
    parser.add_argument("--smooth-shade", action="store_true")
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def select_active(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def import_obj(path):
    before = set(bpy.data.objects)

    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        bpy.ops.import_scene.obj(filepath=path)

    imported = [
        obj for obj in bpy.data.objects
        if obj not in before and obj.type == "MESH"
    ]

    if not imported:
        imported = [
            obj for obj in bpy.context.selected_objects
            if obj.type == "MESH"
        ]

    if not imported:
        raise RuntimeError(f"No mesh object imported from {path}")

    if len(imported) == 1:
        obj = imported[0]
        select_active(obj)
        return obj

    # 如果 OBJ 里有多个 object/group，合并成一个 mesh。
    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]
    bpy.ops.object.join()

    return bpy.context.object


def apply_scale(obj, scale):
    select_active(obj)
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def cleanup_mesh(obj, merge_distance):
    select_active(obj)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    try:
        bpy.ops.mesh.merge_by_distance(distance=merge_distance)
    except Exception:
        bpy.ops.mesh.remove_doubles(threshold=merge_distance)

    try:
        bpy.ops.mesh.delete_loose()
    except Exception:
        pass

    try:
        bpy.ops.mesh.dissolve_degenerate(threshold=merge_distance)
    except Exception:
        pass

    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def voxel_remesh(obj, voxel_size, adaptivity, smooth_shade):
    select_active(obj)

    obj.data.remesh_voxel_size = float(voxel_size)
    obj.data.remesh_voxel_adaptivity = float(adaptivity)

    try:
        bpy.ops.object.voxel_remesh()
    except Exception:
        mod = obj.modifiers.new(name="VoxelRemesh", type="REMESH")
        mod.mode = "VOXEL"
        mod.voxel_size = float(voxel_size)
        mod.adaptivity = float(adaptivity)
        bpy.ops.object.modifier_apply(modifier=mod.name)

    if smooth_shade:
        for poly in obj.data.polygons:
            poly.use_smooth = True


def triangulate_mesh(obj):
    select_active(obj)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(
        quad_method="BEAUTY",
        ngon_method="BEAUTY",
    )
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def export_obj(obj, path):
    select_active(obj)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=path,
            export_selected_objects=True,
            apply_modifiers=True,
            export_materials=False,
            export_uv=False,
            export_normals=True,
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=path,
            use_selection=True,
            use_mesh_modifiers=True,
            use_materials=False,
            use_uvs=False,
            use_normals=True,
        )


def print_stats(obj, name):
    mesh = obj.data
    print(
        f"[{name}] "
        f"vertices={len(mesh.vertices)}, "
        f"edges={len(mesh.edges)}, "
        f"faces={len(mesh.polygons)}"
    )


def main():
    args = parse_args()

    clear_scene()

    obj = import_obj(args.input)
    obj.name = "remeshed_obj"

    apply_scale(obj, args.scale)
    print_stats(obj, "imported")

    cleanup_mesh(obj, args.merge_distance)
    print_stats(obj, "cleaned")

    voxel_remesh(
        obj=obj,
        voxel_size=args.voxel_size,
        adaptivity=args.adaptivity,
        smooth_shade=args.smooth_shade,
    )
    print_stats(obj, "voxel_remeshed")

    cleanup_mesh(obj, args.merge_distance)
    print_stats(obj, "post_cleaned")

    if args.triangulate:
        triangulate_mesh(obj)
        print_stats(obj, "triangulated")

    export_obj(obj, args.output)
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()