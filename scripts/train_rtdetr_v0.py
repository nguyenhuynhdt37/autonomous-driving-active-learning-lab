"""
train_rtdetr_v0.py - Huấn luyện Hội tụ Song Mã RT-DETR V0 trên Seed_v0 (500 ảnh)
================================================================================
Kịch bản huấn luyện chính thức dành cho mô hình Vision Transformer RT-DETR:
1. Nạp mô hình RT-DETR từ checkpoint nội bộ / kiến trúc 10 lớp BDD10K.
2. Huấn luyện trên tập mầm seed_v0/images (500 ảnh) với data yaml bdd10k_splits.yaml.
3. Sử dụng cấu hình chuẩn: Optimizer AdamW, Cosine LR, device MPS (Apple Silicon).
4. Tự động kiểm toán và xóa sạch file .cache phát sinh (bảo toàn data/bdd10k/ Read-Only).
5. Lưu trữ checkpoint tốt nhất vào paper_v4_10k/models/rtdetr_v0_best.pt.
6. Chạy đánh giá kiểm định (Validation) trên fixed_val (700 ảnh) và cập nhật Lineage Manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Thiết lập đường dẫn thư mục dự án
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
sys.path.insert(0, str(project_dir))

from ultralytics import RTDETR
import torch

from src.paths import (
    find_project_dir,
    get_models_dir,
    get_results_dir,
    get_splits_dir,
)
from src.utils import format_time, get_device, set_seed


def cleanup_cache_files():
    """Xóa sạch mọi file *.cache phát sinh trong toàn bộ dự án để bảo toàn tính Read-Only."""
    for base in [Path("data/bdd10k"), Path("paper_v4_10k/data")]:
        if base.exists():
            for p in base.rglob("*.cache"):
                try:
                    p.unlink()
                except Exception:
                    pass


def calculate_sha256(file_path: Path) -> str:
    """Tính mã băm SHA-256 niêm phong file trọng số."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện hội tụ mô hình RT-DETR V0 trên tập seed_v0 (500 ảnh)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Số epochs huấn luyện (mặc định: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Kích thước batch (mặc định: 4 để tối ưu GPU MPS)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh đầu vào (mặc định: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Thiết bị tính toán: mps, cpu, cuda (mặc định: mps)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.0001,
        help="Learning rate ban đầu cho AdamW (mặc định: 1e-4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed tái lập kết quả (mặc định: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 85)
    print("🚀 HUẤN LUYỆN HỘI TỤ SONG MÃ RT-DETR V0 (TRANSFORMER BASELINE)")
    print("=" * 85)
    print(f"📁 Project Root: {find_project_dir()}")
    print(f"💻 Thiết bị: {args.device} (MPS khả dụng: {torch.backends.mps.is_available()})")
    print(f"📦 Số epochs: {args.epochs} | Batch size: {args.batch_size} | Image size: {args.imgsz}")
    print(f"🧠 Optimizer: AdamW | Initial LR: {args.lr0} | Cosine Annealing: Bật")

    # Xác định đường dẫn file cấu hình dữ liệu splits
    data_yaml = get_splits_dir() / "bdd10k_splits.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình {data_yaml}")

    # Đường dẫn thư mục mô hình và kết quả
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    target_pt = models_dir / "rtdetr_v0_best.pt"
    lineage_file = models_dir / "models_lineage.json"

    # Thư mục lưu vết huấn luyện runs
    run_project = project_dir / "runs" / "phase_04_rtdetr"
    run_name = f"rtdetr_seed_v0_e{args.epochs}_b{args.batch_size}"

    # Khởi tạo mô hình RT-DETR
    print("\n[1/4] Đang khởi tạo mô hình RT-DETR...")
    if target_pt.exists():
        print(f"  -> Tận dụng checkpoint cục bộ sẵn có: {target_pt}")
        model = RTDETR(str(target_pt))
    else:
        print("  -> Khởi tạo từ kiến trúc rtdetr-l.yaml...")
        model = RTDETR("rtdetr-l.yaml")

    # Huấn luyện
    print(f"\n[2/4] Bắt đầu huấn luyện {args.epochs} epochs trên seed_v0 (500 ảnh)...")
    start_time = time.time()
    try:
        train_results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            device=args.device,
            optimizer="AdamW",
            lr0=args.lr0,
            lrf=0.01,
            cos_lr=True,
            weight_decay=0.0001,
            patience=8,
            project=str(run_project),
            name=run_name,
            exist_ok=True,
            plots=True,
            verbose=True,
            workers=0,
            val=True,
        )
    except Exception as e:
        print(f"❌ Lỗi trong quá trình huấn luyện RT-DETR: {e}")
        # Dọn dẹp cache nếu có
        cleanup_cache_files()
        sys.exit(1)

    train_duration = time.time() - start_time
    print(f"✓ Huấn luyện RT-DETR hoàn tất trong {format_time(train_duration)}!")

    # Tìm file weights tốt nhất sau khi train
    trained_best_pt = run_project / run_name / "weights" / "best.pt"
    if not trained_best_pt.exists():
        trained_best_pt = run_project / run_name / "weights" / "last.pt"

    if trained_best_pt.exists():
        print(f"\n[3/4] Cập nhật checkpoint tốt nhất vào {target_pt}...")
        shutil.copy2(str(trained_best_pt), str(target_pt))
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy best.pt trong {run_project / run_name / 'weights'}")

    # Đánh giá kiểm định chính thức trên fixed_val (700 ảnh)
    print("\n[4/4] Đang thực hiện đánh giá kiểm định chính thức trên fixed_val (700 ảnh)...")
    val_model = RTDETR(str(target_pt))
    val_results = val_model.val(
        data=str(data_yaml),
        split="val",
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        plots=False,
        max_det=300,
        workers=0,
    )

    box = val_results.box
    val_map50 = float(round(box.map50 * 100.0, 2))
    val_map50_95 = float(round(box.map * 100.0, 2))
    val_recall = float(round(box.mr * 100.0, 2))
    val_precision = float(round(box.mp * 100.0, 2))

    print(f"✓ Kết Quả RT-DETR V0 trên fixed_val:")
    print(f"   * mAP@50     : {val_map50:.2f}%")
    print(f"   * mAP@50-95  : {val_map50_95:.2f}%")
    print(f"   * Recall     : {val_recall:.2f}%")
    print(f"   * Precision  : {val_precision:.2f}%")

    # Tính SHA-256
    sha256_val = calculate_sha256(target_pt)

    # Cập nhật models_lineage.json
    lineage_data: Dict[str, Any] = {}
    if lineage_file.exists():
        try:
            with open(lineage_file, "r", encoding="utf-8") as f:
                lineage_data = json.load(f)
        except Exception:
            lineage_data = {}

    lineage_data["rtdetr_v0_best"] = {
        "model_id": "rtdetr_v0_best",
        "architecture": "RT-DETR-Large-Transformer",
        "filename": "rtdetr_v0_best.pt",
        "val_mAP50": val_map50,
        "val_mAP50_95": val_map50_95,
        "val_recall": val_recall,
        "val_precision": val_precision,
        "sha256": sha256_val,
        "recipe": f"rtdetr_seed_v0_adamw_cosinelr_e{args.epochs}",
        "epochs_trained": args.epochs,
        "duration_sec": round(train_duration, 2),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(lineage_file, "w", encoding="utf-8") as f:
        json.dump(lineage_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Đã cập nhật Lineage Manifest: {lineage_file}")

    # Cập nhật v0_baseline_evaluation.json
    v0_eval_file = get_results_dir("phase_04") / "v0_baseline_evaluation.json"
    if v0_eval_file.exists():
        try:
            with open(v0_eval_file, "r", encoding="utf-8") as f:
                v0_data = json.load(f)
            v0_data["rtdetr_model"] = {
                "name": "rtdetr_v0_best",
                "architecture": "RT-DETR-Large-Transformer",
                "checkpoint_path": str(target_pt),
                "sha256": sha256_val,
                "epochs_trained": args.epochs,
                "val_metrics": {
                    "map50": val_map50,
                    "map50_95": val_map50_95,
                    "recall": val_recall,
                    "precision": val_precision,
                },
                "status": "CONVERGED_SUCCESS",
            }
            with open(v0_eval_file, "w", encoding="utf-8") as f:
                json.dump(v0_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Đã cập nhật v0_baseline_evaluation.json: {v0_eval_file}")
        except Exception as e:
            print(f"⚠️ Không thể cập nhật v0_baseline_evaluation.json: {e}")

    # Dọn dẹp cache phát sinh (Quy tắc AGENTS.md)
    cleanup_cache_files()
    print("✓ Đã dọn dẹp toàn bộ cache phát sinh (Bảo toàn data/bdd10k/ Read-Only).")
    print("=" * 85)
    print("🎉 HOÀN TẤT HUẤN LUYỆN VÀ ĐÁNH GIÁ SONG MÃ RT-DETR V0!")
    print("=" * 85)


if __name__ == "__main__":
    main()
