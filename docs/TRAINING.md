# Training AKROS on a single-class dataset

The training script expects ordinary YOLO labels, one line per object:

```text
0 center_x center_y width height
```

Coordinates are normalized to the original image size. Images without objects may have an empty or absent label file. A dataset may use a different source class ID; pass `--source-class` to map that class to AKROS class 0.

```text
my_dataset/
  data.yaml
  images/train/example.jpg
  images/val/example.jpg
  labels/train/example.txt
  labels/val/example.txt
```

```yaml
# my_dataset/data.yaml
path: .
train: images/train
val: images/val
names: [object]
```

The typical normalized object width and height should be estimated from **training labels only** and passed as `--prior-width` and `--prior-height`. The values below are an example for a plate dataset; change them for your dataset. Use `2x8` for horizontally elongated objects or `4x4` for more balanced object shapes. These are architecture choices, not a guarantee that one grid wins on every dataset.

Initial training:

```bash
python -m src.train_platedet \
  --data my_dataset/data.yaml --task plate \
  --architecture mergdet_m --model-size V5A --merg-block regular \
  --gradient-mode fixed --merg-fusion-mode multiscale \
  --height 192 --width 256 --output-stride 2x8 \
  --geometry-mode prior_log --prior-width 0.0836 --prior-height 0.0344 \
  --objectness-mode peak --objectness-target hard \
  --epochs 300 --batch-size 32 --learning-rate 0.002 \
  --augment --seed 0 --output runs/my_object/base.pt
```

Score-alignment continuation:

```bash
python -m src.train_platedet \
  --data my_dataset/data.yaml --task plate \
  --architecture mergdet_m --model-size V5A --merg-block regular \
  --gradient-mode fixed --merg-fusion-mode multiscale \
  --height 192 --width 256 --output-stride 2x8 \
  --geometry-mode prior_log --prior-width 0.0836 --prior-height 0.0344 \
  --objectness-mode peak --objectness-target iou --iou-weight 1 \
  --epochs 20 --batch-size 32 --learning-rate 0.0001 \
  --augment --seed 0 --init-checkpoint runs/my_object/base_last.pt \
  --output runs/my_object/final.pt
```

The raw training checkpoint contains optimizer and training state. `akros.AKROS` expects the compact public format found under `weights/`; package your own checkpoint with:

```bash
python scripts/pack_checkpoint.py --input runs/my_object/final_last.pt \
  --output weights/my_object.pt --label "my object" --dataset "my dataset"
```

For rigorous comparisons, fix the data split, training recipe, and seeds before evaluating results. Use validation images during development; preserve any reserved test split.
