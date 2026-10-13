"""
run_phase_06_model_v1_training.py - Điều Phối Trọn Vẹn Quy Trình Phase 06 (Model V1 Evolution)
================================================================================================
Thực thi trọn vẹn Model-Centric Evolution cho Vòng Lặp 1:
1. Nạp Dataset V1 (750 ảnh) và Dataset Random V1 (750 ảnh).
2. Huấn luyện YOLO11n V1 Proposed với 4 lá chắn chống Overfit.
3. Huấn luyện YOLO11n V1 Random Baseline làm đối chứng khoa học.
4. Đánh giá độc lập trên Frozen Gold Test (1.400 ảnh).
5. Xuất bản Bảng phân rã chi tiết 10 lớp Per-class ra Markdown & JSON.
6. Vẽ và lưu 4 Biểu Đồ Khoa Học 300 DPI (AL Curve, Per-Class Gain, Loss Convergence, ROI).
7. Niêm phong Checkpoint yolo11n_v1_best.pt và cập nhật models_lineage.json.
8. Dọn sạch toàn bộ cache (*.cache), bảo toàn data/bdd10k/ Read-Only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "paper_v4_10k"))

from src.paths import get_results_dir, get_splits_dir, get_models_dir
from src.utils import get_device, set_seed
from src.train_engine import clean_bdd10k_caches
from src.train_v1_engine import ModelV1TrainingEngine, ModelEvaluationReport


def run_phase_06_pipeline(
    train_epochs: int = 8,
    batch_size: int = 16,
    fast_mode: bool = False,
    reuse_existing: bool = True,
) -> dict:
    t0_start = time.time()
    set_seed(42)
    dev = get_device()

    if fast_mode:
        train_epochs = 1

    print("=" * 80)
    print("🚀 BẮT ĐẦU PIPELINE PHASE 06 — MODEL-CENTRIC EVOLUTION V1 (GOLD TEST BENCHMARK)")
    print("=" * 80)
    print(f"  - Thiết bị tính toán: {dev}")
    print(f"  - Số Epochs huấn luyện: {train_epochs} (Fast Mode: {fast_mode})")

    engine = ModelV1TrainingEngine(device=dev)
    splits_dir = get_splits_dir()
    models_dir = get_models_dir()
    results_p6_dir = get_results_dir("phase_06")

    train_v1_yaml = splits_dir / "bdd10k_splits_v1.yaml"
    train_rnd_yaml = splits_dir / "bdd10k_splits_random_v1.yaml"

    # Kiểm tra checkpoint sẵn có hoặc huấn luyện mới
    v1_best_pt = models_dir / "yolo11n_v1_best.pt"
    active_run_weights = engine.runs_dir / "yolo11n_v1_active" / "weights" / "best.pt"
    p5_active_weights = get_results_dir("phase_05") / "runs" / "yolo11n_v1_active" / "weights" / "best.pt"

    # Di chuyển hoặc liên kết runs từ phase 05 sang phase 06 nếu chưa có
    if not (engine.runs_dir / "yolo11n_v1_active").exists() and (get_results_dir("phase_05") / "runs" / "yolo11n_v1_active").exists():
        import shutil
        shutil.copytree(get_results_dir("phase_05") / "runs" / "yolo11n_v1_active", engine.runs_dir / "yolo11n_v1_active", dirs_exist_ok=True)
    if not (engine.runs_dir / "yolo11n_v1_random").exists() and (get_results_dir("phase_05") / "runs" / "yolo11n_v1_random").exists():
        import shutil
        shutil.copytree(get_results_dir("phase_05") / "runs" / "yolo11n_v1_random", engine.runs_dir / "yolo11n_v1_random", dirs_exist_ok=True)

    # 1. Huấn luyện Model V1 Proposed (hoặc tái sử dụng nếu đã hội tụ)
    if reuse_existing and v1_best_pt.exists():
        print(f"\n[Bước 1/6] Tái sử dụng Checkpoint Model V1 đã hội tụ: {v1_best_pt}")
        v1_pt = v1_best_pt
    else:
        print(f"\n[Bước 1/6] Huấn luyện YOLO11n V1 Proposed trên 750 ảnh (train_v1)...")
        v1_pt = engine.train_single_model(
            data_yaml=train_v1_yaml,
            model_name="yolo11n_v1_active",
            epochs=train_epochs,
            batch_size=batch_size,
        )

    # 2. Huấn luyện Model V1 Random Baseline
    rnd_pt = engine.runs_dir / "yolo11n_v1_random" / "weights" / "best.pt"
    if not (reuse_existing and rnd_pt.exists()):
        print(f"\n[Bước 2/6] Huấn luyện Model V1 Random Baseline trên 750 ảnh (train_random_v1)...")
        rnd_pt = engine.train_single_model(
            data_yaml=train_rnd_yaml,
            model_name="yolo11n_v1_random",
            epochs=train_epochs,
            batch_size=batch_size,
        )

    # 3. Chấm thi độc lập trên Frozen Gold Test (1.400 ảnh)
    print("\n[Bước 3/6] Chấm thi độc lập 3 mô hình trên Frozen Gold Test (1.400 ảnh)...")
    v0_pt = models_dir / "yolo11n_v0_best.pt"
    v0_rep = engine.evaluate_model(v0_pt, model_name="YOLO11n V0 (500 Seed)", train_images_count=500, batch_size=batch_size)
    rnd_rep = engine.evaluate_model(rnd_pt, model_name="YOLO11n V1 Random (750 Random)", train_images_count=750, batch_size=batch_size)
    v1_rep = engine.evaluate_model(v1_pt, model_name="YOLO11n V1 Proposed (750 Active)", train_images_count=750, batch_size=batch_size)

    # 4. Xuất Bảng per-class 10 lớp ra Markdown & JSON
    print("\n[Bước 4/6] Xuất Bảng Phân Rã Chi Tiết 10 Lớp Per-Class...")
    table_md, table_json = engine.generate_comparison_tables(v0_rep, rnd_rep, v1_rep)

    # 5. Xuất 4 Biểu Đồ Khoa Học 300 DPI
    print("\n[Bước 5/6] Vẽ và Xuất Xưởng 4 Biểu Đồ Huấn Luyện 300 DPI Chuẩn Bài Báo V4.0...")
    figs = engine.export_figures(v0_rep, rnd_rep, v1_rep, active_runs_dir=engine.runs_dir / "yolo11n_v1_active")

    # 6. Niêm phong Checkpoint và cập nhật Lineage
    print("\n[Bước 6/6] Niêm phong Checkpoint yolo11n_v1_best.pt & Cập nhật Dòng Dõi...")
    lineage_path = engine.seal_lineage(v1_rep)

    # Dọn dẹp cache
    clean_bdd10k_caches()

    elapsed = time.time() - t0_start
    print("\n" + "=" * 80)
    print(f"🎉 HOÀN TẤT THÀNH CÔNG GIAI ĐOẠN PHASE 06 TRONG {elapsed:.2f}s!")
    print(f"  * mAP@50 Model V0 Anchor   : {v0_rep.macro_map50:.2f}%")
    print(f"  * mAP@50 Model Random      : {rnd_rep.macro_map50:.2f}%")
    print(f"  * mAP@50 Model V1 Proposed : {v1_rep.macro_map50:.2f}% (Tăng trưởng: +{v1_rep.macro_map50 - v0_rep.macro_map50:.2f}%)")
    print("=" * 80)

    summary = {
        "phase": "06_model_v1_evolution_and_benchmark",
        "status": "COMPLETED_SUCCESS",
        "elapsed_seconds": round(elapsed, 2),
        "models_evaluated": {
            "v0_anchor_map50": v0_rep.macro_map50,
            "random_v1_map50": rnd_rep.macro_map50,
            "proposed_v1_map50": v1_rep.macro_map50,
            "delta_map50_gain": round(v1_rep.macro_map50 - v0_rep.macro_map50, 2),
        },
        "figures": [str(p) for p in figs],
        "tables": {
            "markdown": str(table_md),
            "json": str(table_json),
        },
        "checkpoint": str((models_dir / "yolo11n_v1_best.pt").resolve()),
        "lineage": str(lineage_path),
    }

    with open(results_p6_dir / "phase_06_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Phase 06 Model V1 Training & Gold Benchmark")
    parser.add_argument("--epochs", type=int, default=8, help="Số epochs huấn luyện")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--fast", action="store_true", help="Chạy chế độ nhanh")
    parser.add_argument("--force-train", action="store_true", help="Bắt buộc train lại từ đầu")
    args = parser.parse_args()

    run_phase_06_pipeline(
        train_epochs=args.epochs,
        batch_size=args.batch,
        fast_mode=args.fast,
        reuse_existing=not args.force_train,
    )
