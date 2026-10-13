"""
run_phase_07_autonomous_loop.py - Script CLI Thực Thi Toàn Trình Phase 07 (End-to-End Autonomous Active Loop)
=============================================================================================================
Thực thi trọn gói quy trình tự động hóa:
1. Thiết lập Workspace độc lập tại paper_v4_10k/notebooks/autonomous_pipeline_loop/
2. Lọc trùng FastDedup trên 8.000 ảnh thô BDD10K (Read-Only).
3. Profiling 11 thuộc tính vật lý và 5 lát cắt khó.
4. Tách 4 tập với thuật toán Quán quân Hybrid Mix 80/20 & Group-Aware Video Isolation.
5. Huấn luyện và khóa mốc chuẩn Target_mAP (100% Random Toàn Phần).
6. Huấn luyện Model Mầm V0 (556 ảnh).
7. Giải đấu Song mã Thích ứng: Proposed Super-Fusion Active vs Uniform Random Baseline.
8. Kiểm soát bằng Bộ Rào Chắn Dừng 5 Tầng & Vòng Xác Nhận Chiến Thắng.
9. Xuất bản bảng Per-Class APA/IEEE (Markdown & JSON) và 3 Biểu đồ 300 DPI.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Thêm project root vào sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from src.autonomous_loop import (
    AutonomousLoopConfig,
    AutonomousLoopOrchestrator,
    clean_all_caches,
)
from src.paths import get_phase07_workspace_dir
from src.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 07: Pipeline Tự Động Hóa Toàn Trình Active Learning Tournament (BDD10K V4.0)"
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default=None,
        help="Thư mục workspace độc lập (mặc định: paper_v4_10k/notebooks/autonomous_pipeline_loop/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chạy ở chế độ mô phỏng / kiểm thử logic siêu tốc (Dry-Run / Verification Mode)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Số epoch tối đa cho mỗi lượt huấn luyện (mặc định: 8)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Patience cho Early Stopping trên fixed_val (mặc định: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (mặc định: 16)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=7,
        help="Số vòng lặp tối đa của giải đấu (mặc định: 7 vòng tương ứng 80%% pool)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Thiết bị tính toán: auto, cpu, mps, cuda (mặc định: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed ngẫu nhiên (mặc định: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ws_dir = Path(args.workspace_dir).resolve() if args.workspace_dir else get_phase07_workspace_dir()

    print("=" * 80)
    print("🚀 BẮT ĐẦU THỰC THI PHASE 07: END-TO-END AUTONOMOUS ACTIVE LOOP TOURNAMENT")
    print("=" * 80)
    print(f"📁 Workspace Độc Lập : {ws_dir}")
    print(f"💻 Thiết Bị Tính Toán: {args.device} (Phát hiện: {get_device().upper()})")
    print(f"⚡ Chế Độ Thực Thi   : {'MÔ PHỎNG / DRY-RUN (FAST VERIFICATION)' if args.dry_run else 'HUẤN LUYỆN TOÀN TRÌNH THỰC TẾ'}")
    print(f"🔄 Epochs / Patience : {args.epochs} / {args.patience}")
    print(f"🎯 Vòng Tối Đa       : {args.max_iterations} (Tỷ lệ động 10%%/vòng)")
    print("=" * 80)

    t_start = time.time()

    # Khởi tạo Cấu hình và Trọng tài điều phối
    config = AutonomousLoopConfig(
        workspace_dir=ws_dir,
        batch_ratio=0.10,
        max_iterations=args.max_iterations,
        epochs_per_iteration=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )

    orchestrator = AutonomousLoopOrchestrator(config)

    # Thực thi trọn gói pipeline
    verdict = orchestrator.run_full_pipeline(dry_run=args.dry_run)

    t_elapsed = round(time.time() - t_start, 2)
    clean_all_caches()

    print("\n" + "=" * 80)
    print("🏁 KẾT QUẢ CHUNG CUỘC GIẢI ĐẤU SONG MÃ PHASE 07")
    print("=" * 80)
    print(f"🏆 Trạng Thái Dừng Rào Chắn: {verdict.status.value}")
    print(f"📣 Lý Do Bản Án            : {verdict.verdict_message}")
    print(f"🛑 Vòng Dừng               : Vòng {verdict.stop_iteration} ({verdict.stop_pool_percentage}% Pool)")
    print(f"🎯 Trần 100% Target mAP    : {verdict.target_map50:.2f}% (5.565 ảnh)")
    print(f"🌱 Mốc Mầm V0 Baseline     : {verdict.v0_map50:.2f}% (556 ảnh)")
    print(f"⭐ Active Super-Fusion     : {verdict.active_final_map50:.2f}%")
    print(f"🎲 Random Baseline         : {verdict.random_final_map50:.2f}%")
    print(f"📈 Cách Biệt với Random    : +{verdict.map50_lead_over_random:.2f}% mAP50")
    print(f"⚡ Cách Biệt với Trần 100% : {'+' if verdict.map50_delta_to_target >= 0 else ''}{verdict.map50_delta_to_target:.2f}% mAP50")
    print(f"💰 ROI Tiết Kiệm Gán Nhãn  : Tiết kiệm {verdict.label_cost_reduction_percent:.1f}% ngân sách nhãn!")
    print(f"⏱️ Tổng Thời Gian Thực Thi : {t_elapsed} giây")
    print("=" * 80)


if __name__ == "__main__":
    main()
