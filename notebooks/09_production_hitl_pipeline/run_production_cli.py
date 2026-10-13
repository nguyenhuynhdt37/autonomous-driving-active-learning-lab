#!/usr/bin/env python3
"""
run_production_cli.py - Giao Diện CLI Vận Hành Production HITL Tự Động
=======================================================================
Hỗ trợ nạp trực tiếp toàn bộ tham số qua dòng lệnh mà KHÔNG CẦN sửa file YAML bằng tay:
  python run_production_cli.py \
      --model yolo11n.pt \
      --task detect \
      --raw-images workspace/data/raw_images \
      --raw-labels workspace/inbox/initial_labels.zip \
      --prelabel \
      --min-conf 0.70 \
      --watch-inbox
"""

import argparse
import sys
import time
from pathlib import Path

# Thêm đường dẫn project
base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

from src.pipeline import ProductionHITLPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Giao Diện Vận Hành Production HITL Pipeline Tự Động (Module 09)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Tên model hoặc checkpoint (VD: yolo11n.pt, yolo11n-seg.pt, yolo11n-pose.pt)")
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "segment", "pose"], help="Tác vụ học máy")
    parser.add_argument("--raw-images", type=str, default=None, help="Đường dẫn thư mục ảnh thô ban đầu")
    parser.add_argument("--raw-labels", type=str, default=None, help="Đường dẫn thư mục hoặc file .zip nhãn thô ban đầu (nếu có sẵn)")
    parser.add_argument("--seed-size", type=int, default=100, help="Số lượng ảnh tập mầm Seed V0 khởi đầu")
    parser.add_argument("--batch-size", type=int, default=100, help="Số lượng ảnh bốc mỗi vòng Active Mining")
    parser.add_argument("--prelabel", action="store_true", default=True, help="Bật gợi ý nhãn (Pre-annotations) khi xuất gói gán nhãn")
    parser.add_argument("--no-prelabel", dest="prelabel", action="store_false", help="Tắt pre-annotations, chỉ xuất ảnh sạch thuần túy")
    parser.add_argument("--min-conf", type=float, default=0.70, help="Ngưỡng độ tin cậy tối thiểu để giữ lại nhãn gợi ý (tránh làm rối mắt annotator)")
    parser.add_argument("--epochs", type=int, default=15, help="Số epochs fine-tune mỗi vòng lặp")
    parser.add_argument("--watch-inbox", action="store_true", help="Tự động lắng nghe thư mục inbox/ để nạp nhãn khi người dùng thả file zip vào")

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 KHỞI ĐỘNG PRODUCTION HITL PIPELINE (CLI MODE)")
    print(f"  - Model: {args.model} | Task: {args.task}")
    print(f"  - Raw Images: {args.raw_images or 'workspace/data/raw_images'}")
    print(f"  - Raw Labels: {args.raw_labels or 'Chưa nạp (Cold Start)'}")
    print(f"  - Pre-labeling: {'BẬT (Conf >= ' + str(args.min_conf) + ')' if args.prelabel else 'TẮT (Chỉ ảnh sạch)'}")
    print("=" * 80)

    # 1. Khởi tạo Pipeline
    pipeline = ProductionHITLPipeline(
        model_name=args.model,
        task=args.task,
        raw_images_dir=args.raw_images,
        raw_labels_path=args.raw_labels,
        prelabel_min_conf=args.min_conf,
    )

    # 2. Phân tách tập dữ liệu ban đầu
    splits_info = pipeline.setup_initial_splits(seed_v0_size=args.seed_size)

    # 3. Kiểm tra nhãn cho Seed V0
    current_it = 0
    if not pipeline.check_seed_labels_ready(split_name="seed_v0"):
        print("\n⚠️ THÔNG BÁO QUAN TRỌNG: Tập mầm seed_v0 chưa có nhãn!")
        # Xuất gói nhiệm vụ ra outbox
        zip_task = pipeline.export_labeling_task(
            iteration=0,
            include_prelabels=False,  # V0 chưa có model huấn luyện -> xuất ảnh sạch
        )
        print(f"👉 Hãy lấy file nhiệm vụ tại: {zip_task}")
        print(f"👉 Sau khi đội ngũ gán nhãn xong, hãy thả file .zip nhãn vào: {pipeline.inbox_dir}")

        if args.watch_inbox:
            print("\n⏳ Đang lắng nghe thư mục inbox/ (nhấn Ctrl+C để dừng)...")
            while not list(pipeline.inbox_dir.glob("*.zip")):
                time.sleep(3)
            # Tự động nạp
            pipeline.ingest_labels_from_inbox(iteration=0)
        else:
            print("\nChạy lại lệnh với cờ --watch-inbox hoặc nạp zip trực tiếp qua hàm ingest_labels_from_inbox().")
            sys.exit(0)
    else:
        print("\n✓ Đã phát hiện sẵn nhãn cho seed_v0 (từ raw_labels)! Tự động tiếp tục.")
        pipeline.consolidate_train_split(iteration=0)

    # 4. Huấn luyện Model V0
    v0_metrics = pipeline.train_model(iteration=0, epochs=args.epochs)
    prev_map = v0_metrics["map50"]

    # 5. Vòng lặp Active Mining
    for iteration in range(1, 6):
        print(f"\n{'='*80}\n🔄 BẮT ĐẦU VÒNG LẶP ACTIVE LEARNING V{iteration}\n{'='*80}")
        # Bốc batch mẫu khó
        mined = pipeline.mine_active_batch(current_iteration=iteration - 1, batch_size=args.batch_size)
        if not mined:
            print("Kho dữ liệu đã cạn kiệt, kết thúc vòng lặp.")
            break

        # Xuất gói kèm pre-labels (có rào chắn tin cậy)
        zip_task = pipeline.export_labeling_task(
            iteration=iteration,
            include_prelabels=args.prelabel,
            min_confidence=args.min_conf,
        )
        print(f"👉 Đã xuất xưởng gói nhiệm vụ vòng {iteration}: {zip_task}")
        print(f"👉 Hãy thả file zip nhãn hoàn thành vào: {pipeline.inbox_dir}")

        if args.watch_inbox:
            print(f"⏳ Đang chờ nhãn cho vòng {iteration}...")
            # Chờ có file zip mới
            while True:
                zips = list(pipeline.inbox_dir.glob("*.zip"))
                if zips:
                    break
                time.sleep(3)
            pipeline.ingest_labels_from_inbox(iteration=iteration)
        else:
            print(f"Tạm dừng vòng {iteration}. Sau khi có nhãn, nạp lại để tiếp tục.")
            break

        # Huấn luyện model V_k
        v_metrics = pipeline.train_model(iteration=iteration, epochs=args.epochs)
        curr_map = v_metrics["map50"]

        # Kiểm tra điều kiện dừng
        verdict = pipeline.stopping_checker.check(
            current_map50=curr_map,
            previous_map50=prev_map,
            current_budget_percent=round((args.seed_size + iteration * args.batch_size) / len(list(pipeline.raw_images_dir.glob('*.*'))) * 100, 1),
        )
        prev_map = curr_map

        print(f"\n[BẢN ÁN RÀO CHẮN DỪNG VÒNG {iteration}]: {verdict.status}")
        print(f"  {verdict.reason}")

        if verdict.should_stop:
            print(f"\n🏁 ĐÃ ĐẠT ĐIỀU KIỆN DỪNG HOÀN HẢO! Dừng vòng lặp.")
            break

    # 6. Xuất xưởng Production models
    print("\n📦 TIẾN HÀNH XUẤT XƯỞNG MODEL PRODUCTION (ONNX & OPENVIINO INT8)...")
    prod_outputs = pipeline.export_production_model()
    print("🎉 HOÀN TẤT VÒNG LẶP PRODUCTION TOÀN DIỆN!")
    print(f"  Outputs: {prod_outputs}")


if __name__ == "__main__":
    main()
