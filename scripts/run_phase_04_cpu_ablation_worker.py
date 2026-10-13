#!/usr/bin/env python3
"""
==============================================================================
BDD10K ACTIVE LEARNING LAB (PAPER V4.0) - CPU WORKER FOR ABLATION TOURNAMENT
==============================================================================
Worker CPU chạy song song độc lập trên 8 cores CPU Apple M1 Pro:
- Chạy Full Convergence cho các cấu hình đối chứng:
  * Cfg-0: Naive Baseline (SGD, lr0=0.01, linear lr, no freeze, no class-weights)
  * Cfg-2: Class-Weighted Loss (SGD, phạt nặng lớp hiếm: train, motor, rider)
  * Cfg-3: AdamW & Cosine LR (AdamW lr0=0.001, cos_lr=True, weight_decay=0.01)
- Tự động ngắt khi hội tụ qua Early Stopping (patience=8, max_epochs=35).
- Tách biệt 100% với GPU MPS để tránh nghẽn Metal buffer và đạt hiệu năng tối đa.
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
# Tận dụng 8 cores CPU cho tính toán song song
torch.set_num_threads(8)

from ultralytics import YOLO

from src.utils import set_seed
from src.train_engine import (
    compute_bdd10k_class_weights,
    clean_bdd10k_caches,
    unwrap_model_safe,
    get_splits_dir,
    get_results_dir,
    get_models_dir,
    BDD10K_CLASS_NAMES,
)


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def train_single_cpu_config(
    cfg_name: str,
    optimizer: str,
    lr0: float,
    cos_lr: bool,
    class_weights: Optional[List[float]],
    data_yaml: Path,
    runs_dir: Path,
    max_epochs: int = 35,
    patience: int = 8,
) -> Dict[str, Any]:
    """Huấn luyện 1 cấu hình cụ thể trên CPU với full convergence."""
    print("\n" + "=" * 70)
    print(f"⚙️ [CPU WORKER] BẮT ĐẦU HUẤN LUYỆN: {cfg_name.upper()}")
    print(f"   Optimizer: {optimizer} | lr0: {lr0} | cos_lr: {cos_lr} | Patience: {patience}")
    print("=" * 70)

    run_dir = runs_dir / cfg_name
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    model = YOLO("yolo11n.pt")

    # Đăng ký class weights nếu có
    if class_weights is not None:
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

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
            epochs=max_epochs,
            patience=patience,
            batch=16,
            imgsz=640,
            device="cpu",
            workers=4,
            optimizer=optimizer,
            lr0=lr0,
            cos_lr=cos_lr,
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
    val_res = val_model.val(data=str(data_yaml), split="val", device="cpu", batch=16, workers=2, verbose=False)
    clean_bdd10k_caches()

    b = val_res.box
    metrics = {
        "precision": float(round(b.mp * 100.0, 2)),
        "recall": float(round(b.mr * 100.0, 2)),
        "map50": float(round(b.map50 * 100.0, 2)),
        "map50_95": float(round(b.map * 100.0, 2)),
    }
    print(f"📊 Kết quả Val {cfg_name}: mAP50 = {metrics['map50']:.2f}% | Recall = {metrics['recall']:.2f}% | Precision = {metrics['precision']:.2f}%")

    return {
        "cfg_name": cfg_name,
        "duration_sec": round(duration, 2),
        "metrics": metrics,
        "best_weights": str(best_weights.resolve()) if best_weights.exists() else "",
    }


def main():
    set_seed(42)
    splits_dir = get_splits_dir()
    data_yaml = splits_dir / "bdd10k_splits.yaml"
    results_dir = get_results_dir() / "phase_04"
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cw_info = compute_bdd10k_class_weights()
    class_weights = cw_info["weights_list"]

    configs_to_run = [
        {
            "cfg_name": "cfg_0_naive",
            "optimizer": "SGD",
            "lr0": 0.01,
            "cos_lr": False,
            "class_weights": None,
        },
        {
            "cfg_name": "cfg_2_class_weights",
            "optimizer": "SGD",
            "lr0": 0.01,
            "cos_lr": False,
            "class_weights": class_weights,
        },
        {
            "cfg_name": "cfg_3_adamw_cosine",
            "optimizer": "AdamW",
            "lr0": 0.001,
            "cos_lr": True,
            "class_weights": None,
        },
    ]

    all_results = {}
    out_file = results_dir / "cpu_worker_ablation_results.json"

    for c in configs_to_run:
        res = train_single_cpu_config(
            cfg_name=c["cfg_name"],
            optimizer=c["optimizer"],
            lr0=c["lr0"],
            cos_lr=c["cos_lr"],
            class_weights=c["class_weights"],
            data_yaml=data_yaml,
            runs_dir=runs_dir,
            max_epochs=35,
            patience=8,
        )
        all_results[c["cfg_name"]] = res
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("🎉 CPU WORKER ĐÃ HOÀN TẤT XUẤT SẮC TOÀN BỘ CÁC CẤU HÌNH ĐỐI CHỨNG!")
    print(f"📁 Kết quả lưu tại: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
