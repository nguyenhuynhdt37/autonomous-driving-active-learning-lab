# Báo Cáo Khoa Học Phase 04: Huấn Luyện Baseline Model V0 & Đấu Trường Cắt Bỏ

## 1. Mục Tiêu Nghiên Cứu
Xây dựng mô hình mầm Model V0 đạt độ bền vững cao trên tập dữ liệu mầm hạt giống `seed_v0` (500 ảnh, 10% pool)
thông qua đấu trường cắt bỏ 5 công thức huấn luyện đối đầu, huấn luyện song mã (YOLO11n + RT-DETR),
và thực hiện bài thi kép trên Frozen Gold Test (1.400 ảnh) cùng Public BDD10K Val (1.000 ảnh).

## 2. Kết Quả Đấu Trường Cắt Bỏ (Ablation Tournament)
- **Quán quân:** `cfg_0_naive` (Naive Baseline: SGD, No freeze, No class-weights, Linear LR)
- **Composite Benchmark Score:** 18.06 điểm
- **Hiệu năng Validation trên fixed_val (700 ảnh):**
  * mAP@50: **3.89%**
  * mAP@50-95: **1.96%**
  * Recall: **10.51%**
  * Precision: **7.11%**

## 3. Đánh Giá Bài Thi Kép Độc Lập
### A. Bài thi 1: Frozen Gold Test Set (fixed_test, 1.400 ảnh) theo 4 Lát Cắt Khó
- **Night (Đêm - 468 ảnh):** Recall = **10.96%** | mAP@50 = **2.82%**
- **Small Object (Vật thể nhỏ - 1.353 ảnh):** Recall = **10.06%** | mAP@50 = **3.40%**
- **Occlusion (Che khuất - 1.349 ảnh):** Recall = **10.14%** | mAP@50 = **3.41%**
- **Day (Ban ngày đối chứng - 814 ảnh):** Recall = **10.37%** | mAP@50 = **3.81%**

### B. Vector Trọng Số Lát Cắt Khó Bàn Giao Phase 05 Active Learning Mining
- **$W_{\text{night}}$:** `0.2484` (24.84%)
- **$W_{\text{small}}$:** `0.2509` (25.09%)
- **$W_{\text{occlusion}}$:** `0.2507` (25.07%)
- **$W_{\text{day}}$:** `0.2500` (25.00%)
- **Tổng kiểm định $\sum W_i$:** `1.0000` (Chuẩn hóa tuyệt đối 100%).

### C. Bài thi 2: Public Benchmark trên BDD10K Val (1.000 ảnh gốc, Read-Only 100%)
- Trạng thái kiểm định: **SUCCESS**
- mAP@50: **3.52%** | Recall: **9.94%** | Precision: **3.09%** | mAP@50-95: **1.87%**

## 4. Kiểm Toán Lưu Trữ Xuất Xưởng (Model Lineage)
Mô hình đã được niêm phong an toàn tại `paper_v4_10k/models/`:
- `yolo11n_v0_best.pt` (SHA-256: `1edead42599e1ee09ba7144e59046c87db9b64267e8140dbb54a2a466a935ea3`)
- `rtdetr_v0_best.pt` (SHA-256: `e8ccee9cf3c8ce0ae8dc862e3160e90c6ca801b6973e4458f219198651034f82`)
- Manifest lineage: `paper_v4_10k/models/models_lineage.json`

Dữ liệu và mô hình V0 sẵn sàng 100% để chuyển giao sang **Phase 05: Active Learning Mining Iteration 1**!
