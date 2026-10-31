<p align="center">
  <img src="https://raw.githubusercontent.com/Aqshalikhsan/Colosseum-Microcontroller/main/assets/colosseum_banner.png" width="900" alt="Colosseum letter-tile logo">
</p>

Colosseum is a compact single-class object detector designed for microcontroller deployment. The package provides PyTorch inference, a two-stage custom-training command, and 18 bundled checkpoints from six datasets and three seeds.

```bash
python -m pip install colosseum-microcontroller
colosseum predict --weights uc3m --source path/to/image.jpg
```

```python
from colosseum import Colosseum

model = Colosseum("uc3m")
boxes = model.predict("path/to/image.jpg")
```

For a new single-class YOLO dataset:

```bash
colosseum train --data my_dataset/data.yaml --name my_object --label "my object" --grid 4x4
```

The bundled names are `uc3m`, `gtsdb`, `shwd`, `ccpd`, `drone`, and `amlpr`, plus `_seed1` and `_seed2` variants. You can also pass a path to your own compact checkpoint. See the [repository](https://github.com/Aqshalikhsan/Colosseum-Microcontroller) for benchmarks, dataset layout, and the full training guide. The bundled weights are floating-point PyTorch checkpoints; ESP32-S3 deployment uses the separate INT8 runtime.
