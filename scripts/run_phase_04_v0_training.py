"""
run_phase_04_v0_training.py - Script Thực Thi Toàn Bộ Đấu Trường Huấn Luyện Phase 04
====================================================================================
Quy trình thực thi hoàn chỉnh tự động hóa 100%:
1. Nạp tập mầm seed_v0 (500 ảnh) và bdd10k_metadata.json (Read-Only 100%).
2. Khảo sát phân bố 10 lớp giao thông và tính Class-Weighted Penalty Vector.
3. Chạy Đấu Trường Thực Nghiệm Cắt Bỏ (Ablation Tournament) 5 Cấu Hình Đối Đầu:
   - Cfg-0: Naive Baseline (SGD, No freeze, No class-weights, Linear LR)
   - Cfg-1: + Backbone Freezing (10 layers đầu trong 5 epochs)
   - Cfg-2: + Class-Weighted Loss (phạt nặng lớp hiếm: train, motor, rider)
   - Cfg-3: + AdamW & Cosine LR scheduler
   - Cfg-4: Full Proposed Recipe (Kết hợp cả 3 kỹ thuật + Early Stopping patience=10)
4. Tổng hợp Leaderboard định lượng & Chọn Quán Quân (Tournament Winner).
5. Huấn luyện Song Mã RT-DETR (Real-time Transformer) trên Proposed Recipe.
6. Đánh Giá Bài Thi Kép (Dual Ground-Truth Scoring Engine):
   - Đo trên Frozen Gold Test Set (fixed_test, 1.400 ảnh) -> Trích xuất ma trận Recall
     4 lát cắt khó (Night, Small, Occlusion, Day) để sinh Vector trọng số W_i.
   - Đối chuẩn phụ trên Public BDD10K Val (1.000 ảnh gốc, Read-Only 100%).
7. Xuất xưởng và Niêm phong Models vào paper_v4_10k/models/:
   - models/yolo11n_v0_best.pt
   - models/rtdetr_v0_best.pt
   - models/models_lineage.json (SHA-256 hash, metrics, lineage).
8. Vẽ biểu đồ 300 DPI và xuất báo cáo kết quả hoàn chỉnh sẵn sàng chèn vào Paper V4.0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Thêm đường dẫn project
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
sys.path.insert(0, str(project_dir))

from src.paths import (
    find_project_dir,
    get_bdd10k_metadata_file,
    get_figures_dir,
    get_models_dir,
    get_reports_dir,
    get_results_dir,
    get_splits_dir,
    get_tables_dir,
)
from src.train_engine import (
    AblationConfigFactory,
    AblationTournamentRunner,
    HardSliceEvaluator,
    ModelLineageManager,
    TrainEngine,
    TrainingConfig,
    compute_bdd10k_class_weights,
)
from src.utils import format_time, get_device, set_seed
from src.visualization import (
    plot_phase04_ablation_leaderboard,
    plot_phase04_class_weights_penalty,
    plot_phase04_slice_evaluation,
    save_figure,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 04: Huấn luyện Baseline Model V0 (Song mã YOLO11n + RT-DETR) & Đấu trường Cắt bỏ."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Số epochs tối đa cho mỗi cấu hình trong Đấu trường (mặc định: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Kích thước batch (mặc định: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Độ phân giải ảnh đầu vào (mặc định: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Thiết bị tính toán: auto, mps, cuda, hoặc cpu (mặc định: auto)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Chế độ kiểm thử nhanh (epochs=1 hoặc 2 cho CI/CD)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed tái lập 100% (mặc định: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 85)
    print("🚗 BDD10K AUTONOMOUS DRIVING ACTIVE LEARNING LAB - PAPER V4.0")
    print("📌 PHASE 04: HUẤN LUYỆN BASELINE MODEL V0 & ĐẤU TRƯỜNG CẮT BỎ CÔNG THỨC")
    print("=" * 85)

    proj_dir = find_project_dir()
    print(f"📁 Project Root: {proj_dir}")
    print(f"⚙️ Chế độ: {'FAST TEST (epochs=2)' if args.fast else f'FULL TOURNAMENT (epochs={args.epochs})'}")
    print(f"💻 Device: {args.device} (Phát hiện: {get_device()})")
    print(f"📦 Batch Size: {args.batch_size} | Image Size: {args.imgsz}")

    # Khởi tạo Runner
    runner = AblationTournamentRunner(
        tournament_epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        fast_mode=args.fast,
    )

    t0 = time.time()
    tournament_report = runner.run_tournament()
    total_duration = time.time() - t0

    # Vẽ và lưu biểu đồ khoa học 300 DPI
    print("\n[Biểu Đồ Khoa Học] Đang kết xuất các biểu đồ trực quan hóa 300 DPI...")
    fig_dir = get_figures_dir("phase_04")

    # 1. Biểu đồ Class Weights & Penalty
    cw_data = tournament_report["class_weights_analysis"]
    fig_cw = plot_phase04_class_weights_penalty(
        class_counts=cw_data["class_counts"],
        class_weights=cw_data["class_weights"],
    )
    save_figure(fig_cw, fig_dir / "01_class_distribution_and_penalty_weights.png")
    print("  ✓ Đã lưu: 01_class_distribution_and_penalty_weights.png")

    # 2. Biểu đồ Ablation Leaderboard
    leaderboard = tournament_report["leaderboard"]
    fig_lb = plot_phase04_ablation_leaderboard(leaderboard_data=leaderboard)
    save_figure(fig_lb, fig_dir / "02_ablation_tournament_recipes_comparison.png")
    print("  ✓ Đã lưu: 02_ablation_tournament_recipes_comparison.png")

    # 3. Biểu đồ Hard Slices & Vector W_i
    slice_data = tournament_report["slice_evaluation_fixed_test"]
    fig_slice = plot_phase04_slice_evaluation(
        slice_metrics=slice_data["slice_metrics"],
        slice_weights=slice_data["slice_weights_vector_w"],
    )
    save_figure(fig_slice, fig_dir / "03_hard_slices_recall_and_weights_vector.png")
    print("  ✓ Đã lưu: 03_hard_slices_recall_and_weights_vector.png")

    # Xuất bảng tổng hợp Markdown
    table_dir = get_tables_dir("phase_04")
    table_file = table_dir / "ablation_tournament_leaderboard.md"
    with open(table_file, "w", encoding="utf-8") as f:
        f.write("# Bảng Xếp Hạng Đấu Trường Cắt Bỏ Công Thức Huấn Luyện (Phase 04)\n\n")
        f.write("| Hạng | Tên Cấu Hình | Mô Tả Kỹ Thuật | mAP@50 (%) | mAP@50-95 (%) | Recall (%) | Precision (%) | Composite Score | Thời Gian |\n")
        f.write("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for rank, item in enumerate(leaderboard, start=1):
            is_win = " 🏆 (Winner)" if rank == 1 else ""
            f.write(
                f"| **#{rank}**{is_win} | `{item['name']}` | {item['description']} | "
                f"**{item['map50']:.2f}%** | {item['map50_95']:.2f}% | {item['recall']:.2f}% | "
                f"{item['precision']:.2f}% | **{item['composite_score']:.2f}** | {format_time(item['duration_sec'])} |\n"
            )
        f.write(f"\n*Tổng thời gian thực thi Phase 04:* **{format_time(total_duration)}**\n")
    print(f"  ✓ Đã lưu bảng kết quả: {table_file}")

    # Báo cáo tổng kết khoa học
    report_dir = get_reports_dir("phase_04")
    summary_report_file = report_dir / "phase_04_scientific_summary.md"
    winner = tournament_report["winner"]
    w_vec = slice_data["slice_weights_vector_w"]

    with open(summary_report_file, "w", encoding="utf-8") as f:
        f.write(f"""# Báo Cáo Khoa Học Phase 04: Huấn Luyện Baseline Model V0 & Đấu Trường Cắt Bỏ

## 1. Mục Tiêu Nghiên Cứu
Xây dựng mô hình mầm Model V0 đạt độ bền vững cao trên tập dữ liệu mầm hạt giống `seed_v0` (500 ảnh, 10% pool)
thông qua đấu trường cắt bỏ 5 công thức huấn luyện đối đầu, huấn luyện song mã (YOLO11n + RT-DETR),
và thực hiện bài thi kép trên Frozen Gold Test (1.400 ảnh) cùng Public BDD10K Val (1.000 ảnh).

## 2. Kết Quả Đấu Trường Cắt Bỏ
- **Quán quân:** `{winner['name']}` ({winner['description']})
- **Composite Benchmark Score:** {winner['composite_score']} điểm
- **Hiệu năng Validation:** mAP@50 = {winner['map50']:.2f}%, Recall = {winner['recall']:.2f}%

## 3. Vector Trọng Số Lát Cắt Khó Bàn Giao Phase 05
Dựa trên ma trận lỗi phát hiện trên 1.400 ảnh kiểm thử vàng độc lập (`fixed_test`),
vector trọng số lát cắt $W_i$ được chuẩn hóa như sau:
- **Night (Đêm):** $W_{{\\text{{night}}}} = {w_vec.get('night', 0.25):.4f}$ ({w_vec.get('night', 0.25)*100:.1f}%)
- **Small Object (Vật thể nhỏ):** $W_{{\\text{{small}}}} = {w_vec.get('small', 0.25):.4f}$ ({w_vec.get('small', 0.25)*100:.1f}%)
- **Occlusion (Che khuất):** $W_{{\\text{{occlusion}}}} = {w_vec.get('occlusion', 0.25):.4f}$ ({w_vec.get('occlusion', 0.25)*100:.1f}%)
- **Day (Ban ngày - Đối chứng):** $W_{{\\text{{day}}}} = {w_vec.get('day', 0.25):.4f}$ ({w_vec.get('day', 0.25)*100:.1f}%)

## 4. Kiểm Toán Lưu Trữ Xuất Xưởng (Model Lineage)
Các mô hình đã được niêm phong an toàn tại `paper_v4_10k/models/`:
- `yolo11n_v0_best.pt` (SHA-256: `{tournament_report['models_registered']['yolo11n']['sha256_hash']}`)
- `rtdetr_v0_best.pt` (SHA-256: `{tournament_report['models_registered']['rtdetr']['sha256_hash']}`)
- Manifest lineage: `paper_v4_10k/models/models_lineage.json`

Dữ liệu và mô hình V0 sẵn sàng 100% để chuyển giao sang **Phase 05: Active Learning Mining Iteration 1**!
""")
    print(f"  ✓ Đã lưu báo cáo khoa học: {summary_report_file}")

    print("\n" + "=" * 85)
    print(f"🎉 THỰC THI PHASE 04 THÀNH CÔNG TRONG {format_time(total_duration)}!")
    print("=" * 85)


if __name__ == "__main__":
    main()
