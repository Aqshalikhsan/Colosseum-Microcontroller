"""Command-line Colosseum inference on an image or directory of images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from colosseum import Colosseum


SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="colosseum predict", description="Colosseum single-class image detection")
    parser.add_argument("--weights", default="uc3m", help="bundled model name or checkpoint path")
    parser.add_argument("--source", type=Path, required=True, help="image or directory")
    parser.add_argument("--output", type=Path, default=Path("runs/predict"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    args = parser.parse_args(argv)
    sources = (
        [args.source] if args.source.is_file()
        else sorted(p for p in args.source.iterdir() if p.suffix.lower() in SUFFIXES)
    )
    if not sources:
        parser.error("No supported images found in --source")
    args.output.mkdir(parents=True, exist_ok=True)
    model = Colosseum(args.weights, device=args.device)
    records = []
    for source in sources:
        detections = model.predict(source, conf=args.conf, max_det=args.max_det)
        with Image.open(source) as opened:
            annotated = ImageOps.exif_transpose(opened).convert("RGB")
        drawing = ImageDraw.Draw(annotated)
        for det in detections:
            drawing.rectangle((det.x1, det.y1, det.x2, det.y2), outline="red", width=3)
            drawing.text((det.x1, max(0, det.y1 - 12)), f"{det.label} {det.confidence:.2f}", fill="red")
        destination = args.output / f"{source.stem}.png"
        annotated.save(destination)
        records.append({"image": str(source), "output": str(destination),
                        "detections": [det.to_dict() for det in detections]})
        print(f"{source}: {len(detections)} detection(s) -> {destination}")
    (args.output / "predictions.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
