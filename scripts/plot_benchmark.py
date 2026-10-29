"""Recreate the benchmark PNGs from the tables in assets/*.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATASETS = ["UC3M", "GTSDB", "SHWD", "CCPD", "Drone", "AMLPR"]
MODELS = ["Colosseum", "ESPDet-Pico", "Tinyissimo v8", "YOLOX-Nano"]
COLORS = ["#1769AA", "#DB8A23", "#32885C", "#9954A8"]
METRICS = [("ap50", "AP50"), ("ap75", "AP75"), ("ap5095", "AP50:95")]


def read_rows(name: str) -> list[dict]:
    with (ASSETS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def accuracy_plot(rows: list[dict], precision: str) -> None:
    lookup = {(row["dataset"], row["model"]): row for row in rows if row["precision"] == precision}
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    x = np.arange(len(DATASETS))
    width = 0.19
    for axis, (field, label) in zip(axes, METRICS):
        for index, (model, color) in enumerate(zip(MODELS, COLORS)):
            selected = [lookup[(dataset, model)] for dataset in DATASETS]
            means = [float(row[field]) for row in selected]
            deviations = [float(row[field + "_sd"]) for row in selected]
            axis.bar(x + (index - 1.5) * width, means, width,
                     yerr=deviations, capsize=2, label=model, color=color,
                     error_kw={"elinewidth": 0.8, "capthick": 0.8})
        axis.set_ylabel(label, fontsize=12)
        axis.set_ylim(0, 1.08)
        axis.set_yticks(np.arange(0, 1.01, 0.2))
        axis.grid(axis="y", color="#DDE3E8", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(x, DATASETS, fontsize=12)
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.27), frameon=False)
    fig.suptitle(f"Six-dataset validation benchmark — {precision}", fontsize=19, weight="bold")
    fig.supxlabel("Three-seed mean ± sample SD · initial matched runs; SHWD and Drone reruns are discussed in the paper.", fontsize=10)
    destination = ASSETS / f"benchmark_{precision.lower()}.png"
    fig.savefig(destination, dpi=170, facecolor="white")
    plt.close(fig)
    print(destination)


def device_plot(rows: list[dict]) -> None:
    lookup = {row["model"]: row for row in rows}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    specs = [
        ("train_images_per_s", "Training throughput (images/s)"),
        ("pipeline_ms", "ESP32-S3 pipeline (ms)"),
        ("arena_kib", "Internal-SRAM arena (KiB)"),
        ("firmware_kib", "Firmware binary (KiB)"),
    ]
    for axis, (field, title) in zip(axes.flat, specs):
        values = [float(lookup[model][field]) for model in MODELS]
        bars = axis.barh(MODELS[::-1], values[::-1], color=COLORS[::-1], height=0.62)
        axis.set_title(title, fontsize=13, weight="bold", loc="left")
        axis.set_xlim(0, max(values) * 1.25)
        axis.grid(axis="x", color="#DDE3E8", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values[::-1]):
            axis.text(bar.get_width() + max(values) * 0.015,
                      bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=9)
    fig.suptitle("UC3M training pilot and ESP32-S3 deployment", fontsize=18, weight="bold")
    fig.supxlabel("Device: 240 MHz ESP32-S3, embedded frame, seed 0. Training throughput: separate RTX 5060 pilot.\nHigher throughput is better; lower device time and memory are better.", fontsize=9)
    destination = ASSETS / "benchmark_device.png"
    fig.savefig(destination, dpi=170, facecolor="white")
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    accuracy = read_rows("benchmark_results.csv")
    assert len(accuracy) == len(DATASETS) * len(MODELS) * 2
    accuracy_plot(accuracy, "Float")
    accuracy_plot(accuracy, "INT8")
    device_plot(read_rows("benchmark_device.csv"))
