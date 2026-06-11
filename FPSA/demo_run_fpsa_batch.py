from FPSA_batch_randomizer import run_batch

if __name__ == "__main__":
    rows = run_batch(
        meta_path="fpsa_meta_bracket_demo.yaml",
        labels=["bracket_x_stretch", "bracket_x_shrink"],
        num_shapes=16,
        workers=8,
        seed=123,
        label_mode="balanced",
    )

    ok = [r for r in rows if r["ok"]]
    print(f"generated {len(ok)} shapes")
    if ok:
        print("first obj:", ok[0]["obj_path"])
        print("first grasp:", ok[0]["grasp_path"])
