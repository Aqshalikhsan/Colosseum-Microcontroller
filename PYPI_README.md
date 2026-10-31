<p align="center">
  <img src="https://raw.githubusercontent.com/Aqshalikhsan/Colosseum-Microcontroller/main/assets/colosseum_banner.png" width="900" alt="Colosseum letter-tile logo">
</p>

Colosseum is a compact single-class object detection framework designed for microcontroller deployment. Install it to train a detector for your own target object. The package also includes 18 example checkpoints from six datasets and three seeds.

```bash
python -m pip install colosseum-microcontroller
```

```python
from colosseum import Colosseum

model = Colosseum()
checkpoints = model.train(
    data="my_dataset/data.yaml",
    name="my_object",
    label="my object",
    grid="4x4",
)
print(checkpoints[0])
print(model.metrics)  # AP50, AP75, AP50:95, precision, recall, F1
boxes = model.predict("path/to/image.jpg")
```

Training starts from scratch, saves a compact checkpoint and `metrics.json` under `runs/custom/my_object/seed0/`, and loads it for inference. Metrics are computed on the validation split. One checkpoint detects one target class. Your dataset uses images and normalized bounding-box text annotations; see the [training guide](https://github.com/Aqshalikhsan/Colosseum-Microcontroller/blob/main/docs/TRAINING.md) for the layout.

The command-line equivalent is:

```bash
colosseum train --data my_dataset/data.yaml --name my_object --label "my object" --grid 4x4
```

The bundled example names are `uc3m`, `gtsdb`, `shwd`, `ccpd`, `drone`, and `amlpr`, plus `_seed1` and `_seed2` variants. For inference you can pass either one of these names or the path to your own compact checkpoint. See the [repository](https://github.com/Aqshalikhsan/Colosseum-Microcontroller) for benchmarks. The weights are floating-point PyTorch checkpoints; ESP32-S3 deployment uses the separate INT8 runtime.

[Open the training notebook in Google Colab](https://colab.research.google.com/github/Aqshalikhsan/Colosseum-Microcontroller/blob/main/notebooks/Train_Colosseum.ipynb) to read a YOLO dataset ZIP from Google Drive, watch live epoch and batch progress, and save checkpoints back to Drive.
