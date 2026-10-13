#!/usr/bin/env python3
"""
==============================================================================
BDD10K ACTIVE LEARNING LAB (PAPER V4.0) - CFG-4 FULL PROPOSED RECIPE (GPU MPS)
==============================================================================
Huấn luyện Cấu hình Đề xuất Toàn diện Cfg-4 trên Apple Silicon MPS GPU:
- Tích hợp cả 3 kỹ thuật tối ưu:
  1. Backbone Freezing: Khóa 10 layers đầu trong 5 epochs đầu, rã đông sau đó.
  2. AdamW Optimizer & Cosine Annealing Learning Rate (lr0=0.001, cos_lr=True).
  3. Class-Weighted Loss: Phạt nặng các lớp hiếm (train, motor, rider).
- Cơ chế Dynamic Early Stopping (patience=8, max_epochs=40 trên fixed_val).
- Xuất kết quả hoàn chỉnh vào paper_v4_10k/results/phase_04/runs/cfg_4_proposed/.
==============================================================================
"""

import os
import sys
import time
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

# Thêm root dự án vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "paper_v4_10k"))

import torch
from ultralytics import YOLO

from src.utils import set_seed, get_device
from src.train_engine import (
    compute_bdd10k_class_weights,
    clean_bdd10k_caches,
    unwrap_model_safe,
    get_splits_dir,
    get_results_dir,
    get_models_dir,
    BDD10K_CLASS_NAMES,
)


class FreezeWarmupCallback:
    """Callback khóa 10 layers đầu trong freeze_epochs đầu và mở khóa an toàn sau đó."""
    def __init__(self, freeze_layers: int = 10, freeze_epochs: int = 5):
        self.freeze_layers = freeze_layers
        self.freeze_epochs = freeze_epochs
        self.unfrozen = False

    def on_train_epoch_start(self, trainer):
        cur_epoch = getattr(trainer, "epoch", 0)
        if cur_epoch < self.freeze_epochs:
            for i, (name, param) in enumerate(trainer.model.named_parameters()):
                if any(f"model.{k}." in name for k in range(self.freeze_layers)):
                    param.requires_grad = False
        elif not self.unfrozen:
            print(f"\n[Cfg-4 Callback] Epoch {cur_epoch}: Mở khóa toàn bộ {self.freeze_layers} layers để tinh chỉnh sâu!")
            for name, param in trainer.model.named_parameters():
                if "dfl" not in name:
                    param.requires_grad = True
            self.unfrozen = True


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
    print("🔥 [GPU WORKER] BẮT ĐẦU HUẤN LUYỆN CFG-4: FULL PROPOSED RECIPE")
    print(f"💻 Thiết bị: {device.upper()} | AdamW + Cosine LR + Freeze 10L + Class Weights")
    print("=" * 80)

    splits_dir = get_splits_dir()
    data_yaml = splits_dir / "bdd10k_splits.yaml"
    results_dir = get_results_dir() / "phase_04"
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cw_info = compute_bdd10k_class_weights()
    class_weights = cw_info["weights_list"]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    cfg_name = "cfg_4_proposed"
    run_dir = runs_dir / cfg_name
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    model = YOLO("yolo11n.pt")

    # Đăng ký Freeze callback
    freeze_cb = FreezeWarmupCallback(freeze_layers=10, freeze_epochs=5)
    model.add_callback("on_train_epoch_start", freeze_cb.on_train_epoch_start)

    # Đăng ký Class weights callbacks
    def _inject_class_weights(trainer):
        target = getattr(trainer, "model", None)
        if target is not None:
            unwrapped = unwrap_model_safe(target)
            dev = next(unwrapped.parameters()).device
            unwrapped.class_weights = weights_tensor.to(dev)
            if hasattr(unwrapped, "criterion") and unwrapped.criterion is not None:
                unwrapped.criterion.class_weights = weights_tensor.to(dev).view(1, 1, -1)

    def _ensure_batch_class_weights(trainer):
        target = getattr(trainer, "model", None)
        if target is not None:
            unwrapped = unwrap_model_safe(target)
            if hasattr(unwrapped, "criterion") and unwrapped.criterion is not None:
                if getattr(unwrapped.criterion, "class_weights", None) is None:
                    dev = next(unwrapped.parameters()).device
                    unwrapped.criterion.class_weights = weights_tensor.to(dev).view(1, 1, -1)

    model.add_callback("on_train_start", _inject_class_weights)
    model.add_callback("on_train_batch_start", _ensure_batch_class_weights)

    t0 = time.time()
    try:
        train_res = model.train(
            data=str(data_yaml),
            epochs=40,
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

    out_file = results_dir / "cfg_4_proposed_results.json"
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
    print(f"📁 Đã lưu kết quả Cfg-4 tại: {out_file}")


if __name__ == "__main__":
    main()
