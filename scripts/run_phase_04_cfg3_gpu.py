#!/usr/bin/env python3
"""
==============================================================================
BDD10K ACTIVE LEARNING LAB (PAPER V4.0) - CFG-3 ADAMW & COSINE LR (GPU MPS)
==============================================================================
Huấn luyện Cấu hình Cfg-3 trên Apple Silicon MPS GPU để tối đa hóa tốc độ:
- Cấu hình: AdamW Optimizer (lr0=0.001, weight_decay=0.01) + Cosine Annealing LR.
- Cơ chế Early Stopping (patience=8, max_epochs=35 trên fixed_val).
- Xuất kết quả vào paper_v4_10k/results/phase_04/runs/cfg_3_adamw_cosine/.
==============================================================================
"""

import os
import sys
import time
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "paper_v4_10k"))

import torch
from ultralytics import YOLO

from src.utils import set_seed
from src.train_engine import (
    clean_bdd10k_caches,
    get_splits_dir,
    get_results_dir,
)


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    set_seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 80)
    print("🔥 [GPU WORKER] BẮT ĐẦU HUẤN LUYỆN CFG-3: ADAMW & COSINE ANNEALING LR")
    print(f"💻 Thiết bị: {device.upper()} | Optimizer: AdamW | Cosine LR: True | Patience: 8")
    print("=" * 80)

    splits_dir = get_splits_dir()
    data_yaml = splits_dir / "bdd10k_splits.yaml"
    results_dir = get_results_dir() / "phase_04"
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cfg_name = "cfg_3_adamw_cosine"
    run_dir = runs_dir / cfg_name
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    model = YOLO("yolo11n.pt")

    t0 = time.time()
    try:
        train_res = model.train(
            data=str(data_yaml),
            epochs=35,
            patience=8,
            batch=16,
            imgsz=640,
            device=device,
            workers=0,
            optimizer="AdamW",
            lr0=0.001,
            cos_lr=True,
            project=str(runs_dir),
            name=cfg_name,
            exist_ok=True,
            seed=42,
            deterministic=True,
            verbose=False,
        )
    finally:
        clean_bdd10k_caches()

    duration = time.time() - t0
    best_weights = run_dir / "weights" / "best.pt"
    print(f"\n✓ Hoàn tất {cfg_name} trong {duration:.1f}s ({duration/60:.2f} phút)!")

    # Đánh giá trên fixed_val
    val_model = YOLO(str(best_weights if best_weights.exists() else "yolo11n.pt"))
    val_res = val_model.val(data=str(data_yaml), split="val", device=device, batch=16, workers=0, verbose=False)
    clean_bdd10k_caches()

    b = val_res.box
    metrics = {
        "precision": float(round(b.mp * 100.0, 2)),
        "recall": float(round(b.mr * 100.0, 2)),
        "map50": float(round(b.map50 * 100.0, 2)),
        "map50_95": float(round(b.map * 100.0, 2)),
    }
    print(f"📊 Kết quả Val {cfg_name}: mAP50 = {metrics['map50']:.2f}% | Recall = {metrics['recall']:.2f}% | Precision = {metrics['precision']:.2f}%")

    out_file = results_dir / "cfg_3_adamw_cosine_results.json"
    result_data = {
        "cfg_name": cfg_name,
        "device": device,
        "duration_sec": round(duration, 2),
        "metrics": metrics,
        "best_weights": str(best_weights.resolve()) if best_weights.exists() else "",
        "sha256": compute_sha256(best_weights) if best_weights.exists() else "",
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"📁 Đã lưu kết quả Cfg-3 tại: {out_file}")


if __name__ == "__main__":
    main()
