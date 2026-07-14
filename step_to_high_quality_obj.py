#!/usr/bin/env python3
"""
STEP -> high-quality triangle OBJ using Gmsh OpenCASCADE.

Recommended for:
    SolidWorks STEP
        -> CAD-aware Gmsh surface triangulation
        -> clean triangle OBJ
        -> FPSA / APAP

Example:
    python step_to_high_quality_obj.py \
        --input wrench_attached.STEP \
        --output wrench_attached_gmsh.obj \
        --target-divisions 400 \
        --curvature 80 \
        --optimize-iters 5

If your STEP is in mm but your simulator/FPSA wants meters:
    python step_to_high_quality_obj.py \
        --input wrench_attached.STEP \
        --output wrench_attached_gmsh_m.obj \
        --target-divisions 20 \
        --curvature 8 \
        --scale-output 0.001

python step_to_high_quality_obj.py         
--input wrench_attached.STEP         
--output wrench_attached_gmsh_m.obj         
--target-divisions 20         
--curvature 8         
--scale-output 0.001

"""

import argparse
import math
import os
from pathlib import Path

import gmsh


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input STEP/STP file.")
    parser.add_argument("--output", required=True, help="Output OBJ file.")

    # Auto size control.
    parser.add_argument(
        "--target-divisions",
        type=float,
        default=400.0,
        help=(
            "Approximate bbox diagonal / target edge length. "
            "Larger = denser mesh. Try 250, 400, 600."
        ),
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=None,
        help="Optional absolute minimum edge size in STEP units.",
    )
    parser.add_argument(
        "--max-edge",
        type=float,
        default=None,
        help="Optional absolute maximum edge size in STEP units.",
    )
    parser.add_argument(
        "--min-edge-factor",
        type=float,
        default=0.30,
        help="Auto min edge = base_edge * this factor.",
    )
    parser.add_argument(
        "--max-edge-factor",
        type=float,
        default=1.60,
        help="Auto max edge = base_edge * this factor.",
    )

    # CAD curvature / feature preservation.
    parser.add_argument(
        "--curvature",
        type=float,
        default=80.0,
        help=(
            "Number of elements per 2*pi radians for curvature sizing. "
            "Larger preserves curved/fillet features better. Try 50, 80, 120."
        ),
    )
    parser.add_argument(
        "--min-circle-points",
        type=float,
        default=48.0,
        help="Minimum points used around circular curves if supported by this Gmsh build.",
    )
    parser.add_argument(
        "--min-curve-points",
        type=float,
        default=8.0,
        help="Minimum points per curve if supported by this Gmsh build.",
    )

    # Mesh algorithm.
    parser.add_argument(
        "--algorithm",
        default="frontal",
        choices=["frontal", "delaunay", "meshadapt", "auto"],
        help=(
            "2D surface meshing algorithm. "
            "frontal usually gives high quality; meshadapt is more robust for difficult CAD."
        ),
    )

    # Geometry cleanup. Default off because aggressive healing can modify tiny features.
    parser.add_argument(
        "--heal",
        action="store_true",
        help="Run OpenCASCADE shape healing. Use only if STEP has import/manifold issues.",
    )
    parser.add_argument(
        "--heal-tolerance",
        type=float,
        default=1e-8,
        help="OpenCASCADE healing tolerance in STEP units.",
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="Run OCC removeAllDuplicates. Useful for assemblies, but can alter topology.",
    )

    # Post mesh cleanup.
    parser.add_argument(
        "--optimize-iters",
        type=int,
        default=5,
        help="Relocate2D optimization iterations. Set 0 to disable.",
    )

    # Output scaling.
    parser.add_argument(
        "--scale-output",
        type=float,
        default=1.0,
        help=(
            "Scale only the exported OBJ vertex coordinates. "
            "Use 0.001 for mm STEP -> meter OBJ."
        ),
    )

    # Output controls.
    parser.add_argument(
        "--also-write-msh",
        action="store_true",
        help="Also write a .msh next to the OBJ for debugging in Gmsh.",
    )
    parser.add_argument(
        "--write-gmsh-obj",
        action="store_true",
        help=(
            "Also ask Gmsh to write OBJ directly. "
            "The script always writes a clean triangle-only OBJ manually."
        ),
    )

    return parser.parse_args()


def safe_set_number(name, value):
    try:
        gmsh.option.setNumber(name, float(value))
        return True
    except Exception as exc:
        print(f"[warn] could not set {name}={value}: {exc}")
        return False


def get_model_bbox():
    entities = gmsh.model.getEntities()
    if not entities:
        raise RuntimeError("No CAD entities found after STEP import.")

    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = -float("inf")

    for dim, tag in entities:
        try:
            bb = gmsh.model.getBoundingBox(dim, tag)
        except Exception:
            continue

        xmin = min(xmin, bb[0])
        ymin = min(ymin, bb[1])
        zmin = min(zmin, bb[2])
        xmax = max(xmax, bb[3])
        ymax = max(ymax, bb[4])
        zmax = max(zmax, bb[5])

    if not all(math.isfinite(v) for v in [xmin, ymin, zmin, xmax, ymax, zmax]):
        raise RuntimeError("Could not compute model bounding box.")

    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)

    return (xmin, ymin, zmin, xmax, ymax, zmax), diag


def algorithm_to_id(name):
    # Gmsh Mesh.Algorithm:
    # 1 MeshAdapt, 2 Automatic, 5 Delaunay, 6 Frontal-Delaunay
    if name == "meshadapt":
        return 1
    if name == "auto":
        return 2
    if name == "delaunay":
        return 5
    if name == "frontal":
        return 6
    raise ValueError(f"Unknown algorithm: {name}")


def configure_gmsh(args, diag):
    base_edge = diag / float(args.target_divisions)

    min_edge = args.min_edge
    max_edge = args.max_edge

    if min_edge is None:
        min_edge = base_edge * float(args.min_edge_factor)
    if max_edge is None:
        max_edge = base_edge * float(args.max_edge_factor)

    if min_edge <= 0 or max_edge <= 0:
        raise ValueError(f"Invalid mesh sizes: min_edge={min_edge}, max_edge={max_edge}")
    if min_edge > max_edge:
        raise ValueError(f"min_edge must be <= max_edge, got {min_edge} > {max_edge}")

    print(f"[mesh size] bbox_diag={diag:.10g}")
    print(f"[mesh size] base_edge={base_edge:.10g}")
    print(f"[mesh size] min_edge={min_edge:.10g}")
    print(f"[mesh size] max_edge={max_edge:.10g}")

    safe_set_number("General.Terminal", 1)

    # Keep first-order triangles for FPSA/Open3D/trimesh.
    safe_set_number("Mesh.ElementOrder", 1)
    safe_set_number("Mesh.SecondOrderLinear", 0)

    # Use triangular surface mesh.
    safe_set_number("Mesh.RecombineAll", 0)
    safe_set_number("Mesh.SubdivisionAlgorithm", 0)

    # Global size constraints.
    safe_set_number("Mesh.MeshSizeMin", min_edge)
    safe_set_number("Mesh.MeshSizeMax", max_edge)
    safe_set_number("Mesh.MeshSizeFactor", 1.0)

    # Preserve CAD points/curves and refine curved surfaces.
    safe_set_number("Mesh.MeshSizeFromPoints", 1)
    safe_set_number("Mesh.MeshSizeFromCurvature", args.curvature)
    safe_set_number("Mesh.MeshSizeExtendFromBoundary", 1)

    # These options exist in common Gmsh builds; safe_set avoids hard failure.
    safe_set_number("Mesh.MinimumCirclePoints", args.min_circle_points)
    safe_set_number("Mesh.MinimumCurvePoints", args.min_curve_points)

    # High-quality surface meshing.
    safe_set_number("Mesh.Algorithm", algorithm_to_id(args.algorithm))

    # Do not save only physical groups.
    safe_set_number("Mesh.SaveAll", 1)

    return min_edge, max_edge


def import_step(args):
    input_path = str(Path(args.input).expanduser().resolve())
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    print(f"[import] {input_path}")

    gmsh.model.add("step_to_obj")

    imported = gmsh.model.occ.importShapes(
        input_path,
        highestDimOnly=False,
        format="step",
    )

    print(f"[import] imported dimTags: {len(imported)}")

    gmsh.model.occ.synchronize()

    if args.heal:
        print(f"[heal] tolerance={args.heal_tolerance}")
        try:
            gmsh.model.occ.healShapes(
                [],
                args.heal_tolerance,
                True,   # fixDegenerated
                True,   # fixSmallEdges
                True,   # fixSmallFaces
                True,   # sewFaces
                True,   # makeSolids
            )
            gmsh.model.occ.synchronize()
        except Exception as exc:
            print(f"[warn] healShapes failed: {exc}")

    if args.remove_duplicates:
        print("[occ] removeAllDuplicates")
        try:
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
        except Exception as exc:
            print(f"[warn] removeAllDuplicates failed: {exc}")

    surfaces = gmsh.model.getEntities(2)
    volumes = gmsh.model.getEntities(3)

    print(f"[entities] surfaces={len(surfaces)}, volumes={len(volumes)}")

    if len(surfaces) == 0:
        raise RuntimeError("No surfaces found in STEP. Cannot generate surface mesh.")

    return imported


def optimize_surface_mesh(optimize_iters):
    if optimize_iters <= 0:
        return

    print(f"[optimize] Relocate2D iterations={optimize_iters}")

    try:
        gmsh.model.mesh.optimize("Relocate2D", False, int(optimize_iters))
    except TypeError:
        # Older Gmsh Python bindings may not expose all arguments.
        try:
            for _ in range(int(optimize_iters)):
                gmsh.model.mesh.optimize("Relocate2D")
        except Exception as exc:
            print(f"[warn] mesh optimize failed: {exc}")
    except Exception as exc:
        print(f"[warn] mesh optimize failed: {exc}")


def cleanup_mesh():
    for fn_name in ["removeDuplicateNodes", "removeDuplicateElements", "renumberNodes", "renumberElements"]:
        try:
            fn = getattr(gmsh.model.mesh, fn_name)
            fn()
            print(f"[mesh] {fn_name}")
        except Exception:
            pass


def get_element_num_nodes(element_type):
    props = gmsh.model.mesh.getElementProperties(element_type)
    # name, dim, order, numNodes, localNodeCoord, numPrimaryNodes
    return int(props[3])


def write_triangle_obj(output_path, scale_output=1.0):
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()

    tag_to_obj_index = {}
    vertices = []

    for i, tag in enumerate(node_tags):
        x = float(coords[3 * i + 0]) * scale_output
        y = float(coords[3 * i + 1]) * scale_output
        z = float(coords[3 * i + 2]) * scale_output
        vertices.append((x, y, z))
        tag_to_obj_index[int(tag)] = len(vertices)  # OBJ is 1-indexed.

    faces = []
    skipped_elements = {}

    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(2)

    for etype, elem_nodes in zip(element_types, element_node_tags):
        etype = int(etype)
        n = get_element_num_nodes(etype)
        elem_nodes = [int(v) for v in elem_nodes]

        if n == 3:
            for i in range(0, len(elem_nodes), 3):
                a = tag_to_obj_index[elem_nodes[i + 0]]
                b = tag_to_obj_index[elem_nodes[i + 1]]
                c = tag_to_obj_index[elem_nodes[i + 2]]
                if a != b and b != c and c != a:
                    faces.append((a, b, c))

        elif n == 4:
            # Should not happen with RecombineAll=0, but split quads if they appear.
            for i in range(0, len(elem_nodes), 4):
                a = tag_to_obj_index[elem_nodes[i + 0]]
                b = tag_to_obj_index[elem_nodes[i + 1]]
                c = tag_to_obj_index[elem_nodes[i + 2]]
                d = tag_to_obj_index[elem_nodes[i + 3]]
                if a != b and b != c and c != a:
                    faces.append((a, b, c))
                if a != c and c != d and d != a:
                    faces.append((a, c, d))
        else:
            skipped_elements[etype] = skipped_elements.get(etype, 0) + len(elem_nodes) // n

    if not vertices:
        raise RuntimeError("No vertices found in generated mesh.")
    if not faces:
        raise RuntimeError("No triangle faces found in generated mesh.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Generated by step_to_high_quality_obj.py using Gmsh\n")
        f.write("# Triangle-only OBJ for FPSA/APAP\n")
        f.write(f"# vertices {len(vertices)}\n")
        f.write(f"# faces {len(faces)}\n")

        for x, y, z in vertices:
            f.write(f"v {x:.12g} {y:.12g} {z:.12g}\n")

        for a, b, c in faces:
            f.write(f"f {a} {b} {c}\n")

    print(f"[obj] wrote {output_path}")
    print(f"[obj] vertices={len(vertices)}, triangle_faces={len(faces)}")

    if skipped_elements:
        print(f"[warn] skipped non-triangle element types: {skipped_elements}")

    return str(output_path)


def main():
    args = parse_args()

    gmsh.initialize()

    try:
        import_step(args)

        bbox, diag = get_model_bbox()
        print(
            "[bbox] "
            f"x=({bbox[0]:.10g}, {bbox[3]:.10g}) "
            f"y=({bbox[1]:.10g}, {bbox[4]:.10g}) "
            f"z=({bbox[2]:.10g}, {bbox[5]:.10g})"
        )

        configure_gmsh(args, diag)

        print("[mesh] generating 2D surface mesh")
        gmsh.model.mesh.generate(2)

        optimize_surface_mesh(args.optimize_iters)
        cleanup_mesh()

        output_path = write_triangle_obj(
            args.output,
            scale_output=float(args.scale_output),
        )

        if args.also_write_msh:
            msh_path = str(Path(output_path).with_suffix(".msh"))
            gmsh.write(msh_path)
            print(f"[msh] wrote {msh_path}")

        if args.write_gmsh_obj:
            gmsh_obj_path = str(Path(output_path).with_name(Path(output_path).stem + "_gmsh_writer.obj"))
            gmsh.write(gmsh_obj_path)
            print(f"[gmsh obj] wrote {gmsh_obj_path}")

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
