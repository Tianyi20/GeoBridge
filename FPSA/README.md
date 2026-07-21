
Manually annotate mesh's deformation meta:

```shell
python FPSA_annotation_helper_gui_axis.py \
    --mesh /home/iadc/GeoBridge/data/objects/bracket/bracket.obj \
    --label bracket_y_stretch \
    --out bracket_y_stretch_annot.yaml
```

generate augmented shapes with transferred grasp pose according to fpsa_meta_bracket_demo.yaml

python FPSA_batch_randomizer.py   --meta fpsa_meta_bracket_demo.yaml

```shell
python FPSA_batch_randomizer.py   --meta fpsa_meta_bracket.yaml
```
