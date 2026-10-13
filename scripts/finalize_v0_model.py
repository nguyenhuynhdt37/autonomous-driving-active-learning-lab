#!/usr/bin/env python3
"""
==============================================================================
BDD10K ACTIVE LEARNING LAB (PAPER V4.0) - FINALIZE V0 WINNER MODEL (CFG-3)
==============================================================================
Chính thức chốt Cfg-3 (AdamW & Cosine Annealing LR) làm Model V0 chủ lực:
1. Sao chép weights Cfg-3 vào paper_v4_10k/models/yolo11n_v0_best.pt
2. Chấm thi 4 lát cắt khó trên fixed_test (1.400 ảnh)
3. Chấm thi Public Benchmark trên data/bdd10k/images/val (1.000 ảnh)
4. Cập nhật v0_baseline_evaluation.json và models_lineage.json
==============================================================================
"""

import os
import sys
import time
import json
import shutil
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "paper_v4_10k"))

from ultralytics import YOLO
import torch

from src.utils import set_seed, get_device
from src.train_engine import (
    HardSliceEvaluator,
    clean_bdd10k_caches,
    get_splits_dir,
    get_results_dir,
    get_models_dir,
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
    print("🎯 CHÍNH THỨC CHỐT CFG-3 LÀM MODEL V0 CHỦ LỰC CHO PHASE 05")
    print("=" * 80)

    models_dir = get_models_dir()
    results_dir = get_results_dir() / "phase_04"
    splits_dir = get_splits_dir()

    cfg3_src = results_dir / "runs" / "cfg_3_adamw_cosine" / "weights" / "best.pt"
    yolo_dest = models_dir / "yolo11n_v0_best.pt"

    if not cfg3_src.exists():
        print(f"❌ Không tìm thấy weights Cfg-3 tại {cfg3_src}")
        return

    shutil.copy2(cfg3_src, yolo_dest)
    sha256_hash = compute_sha256(yolo_dest)
    print(f"✓ Đã cập nhật Checkpoint: {yolo_dest} ({round(yolo_dest.stat().st_size / (1024*1024), 2)} MB)")
    print(f"  SHA-256: {sha256_hash}")

    eval_model = YOLO(str(yolo_dest))

    # 1. Đo lại fixed_val
    print("\n-> Đang đo lại fixed_val (700 ảnh)...")
    data_yaml = splits_dir / "bdd10k_splits.yaml"
    val_res = eval_model.val(data=str(data_yaml), split="val", device=device, batch=16, workers=0, verbose=False)
    clean_bdd10k_caches()
    val_m50 = float(round(val_res.box.map50 * 100.0, 2))
    val_m50_95 = float(round(val_res.box.map * 100.0, 2))
    val_rec = float(round(val_res.box.mr * 100.0, 2))
    val_prec = float(round(val_res.box.mp * 100.0, 2))
    print(f"✓ Val Metrics: mAP50 = {val_m50:.2f}% | mAP50-95 = {val_m50_95:.2f}% | Recall = {val_rec:.2f}% | Precision = {val_prec:.2f}%")

    # 2. Đánh giá 4 lát cắt khó trên fixed_test
    print("\n-> Đang chấm thi 4 Lát Cắt Khó trên Fixed Test Set (1.400 ảnh)...")
    evaluator = HardSliceEvaluator()
    slice_res = evaluator.evaluate_model_on_slices(
        eval_model,
        test_images_dir=splits_dir / "fixed_test" / "images",
        device=device,
        batch_size=16,
    )
    clean_bdd10k_caches()

    print("✓ Kết quả 4 Lát Cắt Khó trên Fixed Test:")
    for s_name, sm in slice_res.get("slice_metrics", {}).items():
        print(f"   * Lát cắt {s_name:<10}: Recall = {sm.get('recall', 0):.2f}% | mAP50 = {sm.get('map50', 0):.2f}% (Count={sm.get('count')})")

    w_vec = slice_res.get("slice_weights_vector_w", {})
    print(f"\n🎯 Vector Trọng Số Định Hướng W_i cho Bước 5 (Uncertainty Mining):")
    for s_name, w_val in w_vec.items():
        print(f"   * W_{s_name:<10} = {w_val:.4f}")

    # 3. Chấm thi Public Benchmark bdd10k/val
    print("\n-> Đang chấm thi Public Benchmark (data/bdd10k/images/val 1.000 ảnh)...")
    pub_yaml_path = results_dir / "bdd10k_public_val.yaml"
    pub_val_eval = eval_model.val(data=str(pub_yaml_path), split="val", device=device, batch=16, workers=0, verbose=False)
    clean_bdd10k_caches()
    pub_res = {
        "num_benchmark_images": 1000,
        "map50": float(round(pub_val_eval.box.map50 * 100.0, 2)),
        "map50_95": float(round(pub_val_eval.box.map * 100.0, 2)),
        "recall": float(round(pub_val_eval.box.mr * 100.0, 2)),
        "precision": float(round(pub_val_eval.box.mp * 100.0, 2)),
        "status": "SUCCESS",
    }
    print(f"✓ Public Benchmark: mAP50 = {pub_res['map50']:.2f}% | Recall = {pub_res['recall']:.2f}% | Precision = {pub_res['precision']:.2f}%")

    # 4. Ghi nhận v0_baseline_evaluation.json
    final_report = {
        "phase": "04_seed_v0_training_full",
        "executed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "winner_model": {
            "name": "yolo11n_v0_best",
            "recipe": "cfg_3_adamw_cosine (Full Convergence - Official Winner)",
            "max_epochs": 35,
            "patience": 8,
            "val_metrics": {
                "map50": val_m50,
                "map50_95": val_m50_95,
                "recall": val_rec,
                "precision": val_prec,
            },
            "checkpoint_path": str(yolo_dest.resolve()),
            "sha256": sha256_hash,
        },
        "rtdetr_model": {
            "name": "rtdetr_v0_best",
            "checkpoint_path": str((models_dir / "rtdetr_v0_best.pt").resolve()),
            "sha256": compute_sha256(models_dir / "rtdetr_v0_best.pt") if (models_dir / "rtdetr_v0_best.pt").exists() else "",
        },
        "slice_evaluation_fixed_test": slice_res,
        "public_benchmark_bdd10k_val": pub_res,
    }

    report_file = results_dir / "v0_baseline_evaluation.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Đã cập nhật toàn bộ báo cáo: {report_file}")

    # Cập nhật models_lineage.json
    lineage_file = models_dir / "models_lineage.json"
    try:
        with open(lineage_file, "r", encoding="utf-8") as f:
            lineage_data = json.load(f)
    except Exception:
        lineage_data = {}

    lineage_data["yolo11n_v0_best"] = {
        "model_id": "yolo11n_v0_best",
        "filename": "yolo11n_v0_best.pt",
        "val_mAP50": val_m50,
        "val_recall": val_rec,
        "val_precision": val_prec,
        "fixed_test_mAP50": slice_res.get("slice_metrics", {}).get("day", {}).get("map50", val_m50),
        "sha256": sha256_hash,
        "recipe": "cfg_3_adamw_cosine_full_convergence_patience8",
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(lineage_file, "w", encoding="utf-8") as f:
        json.dump(lineage_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Đã cập nhật Model Lineage: {lineage_file}")


if __name__ == "__main__":
    main()
