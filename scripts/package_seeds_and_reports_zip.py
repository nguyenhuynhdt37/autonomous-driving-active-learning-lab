"""
package_seeds_and_reports_zip.py - Đóng Gói Toàn Diện Seeds & Báo Cáo Chẩn Đoán Vào File ZIP
========================================================================================
Tập hợp toàn bộ:
1. reports/seeds/ : 7 file báo cáo Markdown chẩn đoán điểm yếu và tiến trình tiến hóa.
2. data/splits/seed_v0 -> seed_v5 :
   - images/ : Ảnh JPEG thực tế (dereference symlinks để mở được trên mọi máy tính).
   - labels/ : 100% Nhãn vàng Ground-Truth do Oracle cấp.
   - model_predictions_v{k-1}/ : Nhãn format YOLO do mô hình vòng trước detect sai/bỏ sót.
   - seed_manifest.json : Thống kê metadata, phân bố lớp và lát cắt khó.
3. README.md hướng dẫn cấu trúc dữ liệu cho đồng nghiệp làm báo cáo.
"""

import hashlib
import os
import sys
import time
import zipfile
from pathlib import Path

# Thêm project root vào sys.path
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
sys.path.insert(0, str(project_dir))


def compute_file_sha256(filepath: Path) -> str:
    """Tính mã băm SHA-256 của file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_package_readme() -> str:
    """Tạo nội dung file README.md hướng dẫn chi tiết cho đồng nghiệp làm báo cáo."""
    return """# 📦 BỘ DỮ LIỆU SEEDS & BÁO CÁO CHẨN ĐOÁN ACTIVE LEARNING (BDD10K LAB)
=============================================================================

Tài liệu hướng dẫn cấu trúc dữ liệu dành cho đồng nghiệp phục vụ viết báo cáo khoa học / đề tài.

## 1. 📂 CẤU TRÚC THƯ MỤC TRONG FILE NÉN

```text
bdd10k_seeds_and_reports/
├── README.md                              # Tài liệu hướng dẫn này
├── reports/                               # Trọn bộ 7 Báo cáo Chẩn đoán Điểm yếu (Markdown)
│   ├── seed_v0_diagnostic_report.md       # Chẩn đoán tập mầm ban đầu (Model V0)
│   ├── seed_v1_diagnostic_report.md       # Lỗi & Điểm yếu Model V0 detect trên Seed 1
│   ├── seed_v2_diagnostic_report.md       # Lỗi & Điểm yếu Model V1 detect trên Seed 2
│   ├── seed_v3_diagnostic_report.md       # Lỗi & Điểm yếu Model V2 detect trên Seed 3
│   ├── seed_v4_diagnostic_report.md       # Lỗi & Điểm yếu Model V3 detect trên Seed 4
│   ├── seed_v5_diagnostic_report.md       # Lỗi & Điểm yếu Model V4 detect trên Seed 5
│   └── seeds_summary_and_weakness_evolution.md # Tổng kết tiến trình tiến hóa & khắc phục điểm liệt
└── seeds/                                 # 6 Hạt giống dữ liệu Active Learning
    ├── seed_v0/                           # Tập mầm nền móng (500 ảnh)
    │   ├── images/                        # Toàn bộ ảnh vật lý (.jpg, 1280x720)
    │   ├── labels/                        # 100% Nhãn vàng Ground-Truth (format YOLO: class xc yc w h)
    │   └── seed_manifest.json             # Thống kê lớp và 6 lát cắt thử thách
    ├── seed_v1/                           # Hạt giống Vòng 1 (556 ảnh)
    │   ├── images/                        # Ảnh vật lý (.jpg)
    │   ├── labels/                        # 100% Nhãn vàng Ground-Truth (dùng để nạp huấn luyện)
    │   ├── model_predictions_v0/          # 🔥 BẰNG CHỨNG MODEL V0 DETECT TRÊN SEED 1
    │   │   ├── labels/                    # File .txt format YOLO kèm độ tự tin: class xc yc w h conf
    │   │   └── predictions_summary.json   # Thống kê tổng quan số box detect và conf TB
    │   └── seed_manifest.json
    ├── seed_v2/ (kèm model_predictions_v1/)
    ├── seed_v3/ (kèm model_predictions_v2/)
    ├── seed_v4/ (kèm model_predictions_v3/)
    └── seed_v5/ (kèm model_predictions_v4/)
```

---

## 2. 🔍 Ý NGHĨA KHOA HỌC CỦA CẤU TRÚC 2 TẦNG NHÃN

Trong mỗi seed (từ `seed_v1` đến `seed_v5`), bạn sẽ thấy 2 thư mục nhãn tách biệt rõ ràng:

1. **Thư mục `labels/` (Nhãn Vàng Ground-Truth do Oracle cấp)**:
   - Cột dữ liệu: `<class_id> <x_center> <y_center> <width> <height>` (chuẩn YOLO, tọa độ chuẩn hóa [0, 1]).
   - **Mục đích**: Đây là nhãn chuẩn xác 100% dùng để huấn luyện mô hình thế hệ tiếp theo ($V_k$). Tuyệt đối không chứa độ tự tin để chống ô nhiễm nhãn (Zero Label Noise).

2. **Thư mục `model_predictions_v{k-1}/labels/` (Nhãn do Mô Hình Tiền Nhiệm Tự Dự Đoán)**:
   - Cột dữ liệu: `<class_id> <x_center> <y_center> <width> <height> <confidence>`
   - **Mục đích**: Đây là bằng chứng thực nghiệm khách quan chứng minh mô hình vòng trước ($V_{k-1}$) còn rất non yếu trên chính 556 ảnh này:
     * **Bỏ sót (False Negatives - FN)**: Có trong `labels/` nhưng vắng mặt trong `model_predictions/`.
     * **Ảo giác (False Positives - FP)**: Không có trong `labels/` nhưng model lại tự vẽ box.
     * **Nhầm lẫn danh mục (Classification Confusion)**: Cùng vị trí (IoU >= 0.45) nhưng đoán nhầm lớp (ví dụ: nhãn thật là `rider` nhưng model đoán thành `person`, nhãn thật là `motor` nhưng đoán thành `bike`).
     * **Vùng bối rối (Uncertainty)**: Bounding box có độ tự tin thấp ($0.25 \\le \\text{conf} \\le 0.50$).

---

## 3. 🎯 10 LỚP ĐỐI TƯỢNG BDD10K VÀ ID TƯƠNG ỨNG

```text
0: person         1: rider          2: car            3: truck          4: bus
5: train          6: motor          7: bike           8: traffic light  9: traffic sign
```

---

## 4. 📊 CÁCH SỬ DỤNG LÀM BÁO CÁO

1. **Xem phân tích định lượng & định tính**: Đọc file `reports/seeds/seeds_summary_and_weakness_evolution.md` để lấy số liệu tổng hợp (mAP, F1-Score, tỷ lệ tiết kiệm 40% chi phí gán nhãn).
2. **Trích xuất ca lỗi điển hình**: Đọc từng file `seed_v{k}_diagnostic_report.md` để lấy các bảng so khớp IoU và top confusions.
3. **Minh họa trực quan hình ảnh**: Bạn có thể mở ảnh trong `seeds/seed_v{k}/images/` và vẽ đè 2 bounding boxes (màu xanh lá cho `labels/` Ground-Truth, màu đỏ cho `model_predictions/`) để đưa vào slide thuyết trình hoặc bài báo làm hình minh họa "Trước và Sau khi Active Learning cứu chữa".
"""


def package_seeds_and_reports(output_zip_path: Path) -> Path:
    """Thực hiện đóng gói toàn bộ ảnh, nhãn, predictions và reports vào file ZIP."""
    print("=" * 80)
    print("📦 BẮT ĐẦU ĐÓNG GÓI TOÀN DIỆN SEEDS & BÁO CÁO CHẨN ĐOÁN ACTIVE LEARNING")
    print("=" * 80)
    t0 = time.time()

    reports_dir = project_dir / "reports" / "seeds"
    splits_dir = project_dir / "data" / "splits"
    target_seeds = ["seed_v0", "seed_v1", "seed_v2", "seed_v3", "seed_v4", "seed_v5"]

    output_zip_path = Path(output_zip_path).resolve()
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Đếm sơ bộ số lượng file cần nén
    total_files_count = 0
    total_uncompressed_bytes = 0

    print("\n🔍 1. Quét kiểm tra dữ liệu đầu vào:")
    # Kiểm tra reports
    report_files = sorted(list(reports_dir.glob("*.md")))
    print(f"  • Thư mục reports/seeds : {len(report_files)} file Markdown")
    total_files_count += len(report_files)

    seed_dirs_map = {}
    for s_name in target_seeds:
        s_dir = splits_dir / s_name
        if s_dir.exists():
            imgs = list((s_dir / "images").glob("*.jpg"))
            lbls = list((s_dir / "labels").glob("*.txt"))
            pred_dirs = [d for d in s_dir.iterdir() if d.is_dir() and d.name.startswith("model_predictions")]
            pred_lbls = []
            pred_summaries = []
            for pd in pred_dirs:
                pred_lbls.extend(list((pd / "labels").glob("*.txt")))
                ps = pd / "predictions_summary.json"
                if ps.exists():
                    pred_summaries.append(ps)
            manifest = s_dir / "seed_manifest.json"

            seed_dirs_map[s_name] = {
                "images": imgs,
                "labels": lbls,
                "pred_labels": pred_lbls,
                "pred_summaries": pred_summaries,
                "manifest": manifest if manifest.exists() else None,
            }
            n_tot = len(imgs) + len(lbls) + len(pred_lbls) + len(pred_summaries) + (1 if manifest.exists() else 0)
            total_files_count += n_tot
            print(f"  • Thư mục {s_name:<9} : {len(imgs)} ảnh | {len(lbls)} nhãn GT | {len(pred_lbls)} nhãn detect | {n_tot} files")
        else:
            print(f"  ⚠️ Cảnh báo: Không tìm thấy {s_name} tại {s_dir}")

    print(f"\n📦 Tổng số file dự kiến đóng gói: {total_files_count:,} files (kèm README.md)")

    # Khởi tạo file ZIP
    print(f"\n🚀 2. Đang tiến hành nén vào file ZIP: {output_zip_path.name}")
    packed_count = 0

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Thêm README.md
        readme_content = create_package_readme()
        zf.writestr("bdd10k_seeds_and_reports/README.md", readme_content)
        packed_count += 1

        # 2. Thêm toàn bộ reports
        print("  • Đang nén các báo cáo Markdown...")
        for r_path in report_files:
            arcname = f"bdd10k_seeds_and_reports/reports/{r_path.name}"
            zf.write(r_path, arcname=arcname)
            packed_count += 1

        # 3. Thêm từng seed
        for s_name in target_seeds:
            if s_name not in seed_dirs_map:
                continue
            s_data = seed_dirs_map[s_name]
            print(f"  • Đang đóng gói {s_name} ({len(s_data['images'])} ảnh, {len(s_data['labels'])} nhãn GT, {len(s_data['pred_labels'])} nhãn detect)...")

            # Đóng gói ảnh (dereference symlinks)
            for img_p in s_data["images"]:
                real_img_path = img_p.resolve()
                if not real_img_path.exists():
                    continue
                arcname = f"bdd10k_seeds_and_reports/seeds/{s_name}/images/{img_p.name}"
                # Nén ảnh bằng ZIP_STORED để đạt tốc độ tối đa vì JPEG đã nén sẵn
                zf.write(real_img_path, arcname=arcname, compress_type=zipfile.ZIP_STORED)
                packed_count += 1

            # Đóng gói nhãn vàng Ground-Truth
            for lbl_p in s_data["labels"]:
                real_lbl_path = lbl_p.resolve()
                if not real_lbl_path.exists():
                    continue
                arcname = f"bdd10k_seeds_and_reports/seeds/{s_name}/labels/{lbl_p.name}"
                zf.write(real_lbl_path, arcname=arcname, compress_type=zipfile.ZIP_DEFLATED)
                packed_count += 1

            # Đóng gói nhãn model detect
            for pred_p in s_data["pred_labels"]:
                real_pred_path = pred_p.resolve()
                if not real_pred_path.exists():
                    continue
                parent_pred_name = pred_p.parent.parent.name  # model_predictions_v{k-1}
                arcname = f"bdd10k_seeds_and_reports/seeds/{s_name}/{parent_pred_name}/labels/{pred_p.name}"
                zf.write(real_pred_path, arcname=arcname, compress_type=zipfile.ZIP_DEFLATED)
                packed_count += 1

            # Đóng gói predictions_summary.json
            for ps_p in s_data["pred_summaries"]:
                parent_pred_name = ps_p.parent.name
                arcname = f"bdd10k_seeds_and_reports/seeds/{s_name}/{parent_pred_name}/predictions_summary.json"
                zf.write(ps_p, arcname=arcname, compress_type=zipfile.ZIP_DEFLATED)
                packed_count += 1

            # Đóng gói seed_manifest.json
            if s_data["manifest"]:
                arcname = f"bdd10k_seeds_and_reports/seeds/{s_name}/seed_manifest.json"
                zf.write(s_data["manifest"], arcname=arcname, compress_type=zipfile.ZIP_DEFLATED)
                packed_count += 1

    dur = time.time() - t0
    zip_size_bytes = output_zip_path.stat().st_size
    zip_size_mb = zip_size_bytes / (1024 * 1024)
    sha256_hash = compute_file_sha256(output_zip_path)

    print("\n" + "=" * 80)
    print("🎉 ĐÓNG GÓI HOÀN TẤT THÀNH CÔNG 100%!")
    print("=" * 80)
    print(f"📁 Tệp ZIP xuất xưởng  : {output_zip_path.as_posix()}")
    print(f"📊 Tổng số file đóng gói: {packed_count:,} files")
    print(f"💾 Kích thước file ZIP : {zip_size_mb:.2f} MB ({zip_size_bytes:,} bytes)")
    print(f"🔒 Mã băm SHA-256      : {sha256_hash}")
    print(f"⏱️ Thời gian thực thi  : {dur:.2f} giây")
    print("=" * 80)
    return output_zip_path


if __name__ == "__main__":
    out_zip = project_dir / "bdd10k_seeds_and_reports_pack.zip"
    package_seeds_and_reports(out_zip)
