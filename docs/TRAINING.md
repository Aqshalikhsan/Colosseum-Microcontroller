# Training Colosseum on a new dataset

Install with `python -m pip install git+https://github.com/Aqshalikhsan/Colosseum-Microcontroller.git`. The commands below also work from an editable source checkout (`python -m pip install -e .`).

Colosseum learns **one target class per checkpoint**. You can train it for plates, helmets, signs, or another object type using your own images and YOLO-format box annotations. The quick command below starts from random initialization, trains with a hard score target, continues with IoU-aligned scores, and writes a compact inference checkpoint.

The training script expects ordinary YOLO labels, one line per object:

```text
0 center_x center_y width height
```

Coordinates are normalized to the original image size, each in the range 0–1. Images without objects may have an empty or absent label file. A dataset may use a different source class ID; pass `--source-class` to select that class. Other classes are ignored.

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

Run from the repository root:

```bash
colosseum train --data my_dataset/data.yaml --name my_object \
  --label "my object" --grid 4x4
```

The script calculates median normalized width and height **from training boxes only**, after applying the same letterbox geometry used by the model. Choose `2x8` for horizontally elongated targets or `4x4` for more balanced shapes. The grid choice should be fixed before evaluating the result. Training uses CUDA when available and otherwise runs on CPU.

The default is one seed, 300 initial epochs and 20 score-alignment epochs. To run three seeds or change the budget:

```bash
colosseum train --data my_dataset/data.yaml --name my_object \
  --label "my object" --grid 4x4 --seeds 0 1 2 \
  --base-epochs 300 --score-epochs 20 --batch-size 32
```

For a dataset whose target is class 2, add `--source-class 2`. To supply your own priors, pass **both** `--prior-width` and `--prior-height`. For a smaller initial trial, lower `--base-epochs` and `--score-epochs`; that changes the training recipe.

Each seed produces `runs/custom/my_object/seedN/colosseum.pt`. For inference:

```bash
colosseum predict --weights runs/custom/my_object/seed0/colosseum.pt \
  --source path/to/image.jpg
```

## Manual training commands

The manual `src/` and `scripts/` commands below are for a source checkout. The packaged `colosseum train` command above performs both stages and packages the checkpoint automatically.

The commands below expose the underlying two-stage training script. The prior values are **an example for a plate dataset**; estimate them from your own training set or use the quick command above. Use `2x8` for this plate example and change `--task`, `--output-stride`, and the priors as needed.

Initial training:

```bash
python -m src.train_platedet \
  --data my_dataset/data.yaml --task custom \
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
  --data my_dataset/data.yaml --task custom \
  --architecture mergdet_m --model-size V5A --merg-block regular \
  --gradient-mode fixed --merg-fusion-mode multiscale \
  --height 192 --width 256 --output-stride 2x8 \
  --geometry-mode prior_log --prior-width 0.0836 --prior-height 0.0344 \
  --objectness-mode peak --objectness-target iou --iou-weight 1 \
  --epochs 20 --batch-size 32 --learning-rate 0.0001 \
  --augment --seed 0 --init-checkpoint runs/my_object/base_last.pt \
  --output runs/my_object/final.pt
```

The raw training checkpoint contains optimizer and training state. `colosseum.Colosseum` expects the compact public format found under `weights/`; package your own checkpoint with:

```bash
python scripts/pack_checkpoint.py --input runs/my_object/final_last.pt \
  --output weights/my_object.pt --label "my object" --dataset "my dataset"
```

For rigorous comparisons, fix the data split, training recipe, and seeds before evaluating results. Use validation images during development; preserve any reserved test split.
