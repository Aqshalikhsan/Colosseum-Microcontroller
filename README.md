# AKROS

**AKROS** is a compact, single-class object detector designed for microcontroller deployment. It uses fixed image-gradient features, residual convolution blocks, three-scale fusion, and separate confidence and box heads. The network has **81,403 parameters**. A checkpoint detects one target class; use a different checkpoint for a different class.

This repository provides **18 ready-to-use PyTorch checkpoints** (six datasets × three seeds), an image inference API and CLI, and the training code. Input is grayscale, letterboxed to **256 × 192** pixels. AKROS outputs bounding boxes, not text recognition.

## Quick start

Tested with Python 3.11.

```bash
git clone https://github.com/Aqshalikhsan/AKROS.git
cd AKROS
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python predict.py --weights weights/uc3m.pt --source path/to/image.jpg
```

Annotated images and machine-readable `predictions.json` are written to `runs/predict/`. Use `--source path/to/folder` for a folder of images, `--conf 0.25` to adjust the confidence cutoff, and `--device cuda:0` for a CUDA GPU.

Python API:

```python
from akros import AKROS

model = AKROS("weights/uc3m.pt")
for box in model.predict("path/to/image.jpg", conf=0.35):
    print(box.label, box.confidence, box.x1, box.y1, box.x2, box.y2)
```

The coordinates are pixels in the original image after EXIF orientation is applied. The prediction threshold only filters results; it does not change the model. Local confidence peaks are used to select candidate boxes.

## Model zoo

All included weights are **final checkpoints** from the reported three-seed training runs. The top-level weights are seed 0; seed 1 and 2 are in `weights/seeds/`. The complete training state was stripped; each file contains model tensors and the metadata required for inference. The `weights/manifest.json` file records SHA-256 hashes.

| Weights | Training dataset | Detects | Output stride |
|---|---|---|---|
| [`uc3m.pt`](weights/uc3m.pt) | UC3M-LP | license plate | 2 × 8 |
| [`gtsdb.pt`](weights/gtsdb.pt) | GTSDB | prohibitory traffic sign | 4 × 4 |
| [`shwd.pt`](weights/shwd.pt) | SHWD | safety helmet | 4 × 4 |
| [`ccpd.pt`](weights/ccpd.pt) | CCPD2019 | license plate | 2 × 8 |
| [`drone.pt`](weights/drone.pt) | Drone | drone | 4 × 4 |
| [`amlpr.pt`](weights/amlpr.pt) | AMLPR | license plate | 2 × 8 |

For example, use `weights/seeds/uc3m_seed1.pt` or `weights/seeds/uc3m_seed2.pt` to select another UC3M seed. The same naming convention applies to every dataset.

The dataset name identifies the training domain; three plate checkpoints are separate models trained on different images. Choose the checkpoint closest to the images you expect. These are floating-point PyTorch weights. They are not directly loadable by ESP32-S3 firmware; device deployment requires the separately quantized INT8 program and C runtime.

## Train on your data

AKROS uses YOLO-format single-class annotations. See [training instructions](docs/TRAINING.md) for the directory layout, `data.yaml`, initial training, and score-alignment continuation. Raw datasets and official test images are not included here.

## Reproducibility

- Input: grayscale, bilinear resize preserving aspect ratio, gray padding value 114.
- Architecture: fixed-gradient stem, regular bottleneck blocks, multiscale fusion, one confidence channel and four geometry channels.
- Geometry: normalized boxes with a prior-log width and height encoding; priors are saved in each checkpoint.
- Files under `src/` retain some internal legacy names because they are also used by the training and checkpoint export pipeline. The public model name is AKROS.

Please cite the AKROS paper when it becomes available. The datasets remain subject to their source terms; no raw dataset images are redistributed here.

## License

Code and the 18 AKROS weight files are released under the [MIT License](LICENSE). Third-party datasets are excluded.
