# 🏛️ QUY TẮC KIẾN TRÚC DỮ LIỆU & QUẢN LÝ DỰ ÁN (PROJECT DATA ARCHITECTURE RULES)

> [!IMPORTANT]
> Quy tắc kiến trúc dữ liệu và phương pháp luận nghiên cứu chuẩn mực nội bộ của dự án, được cố định qua chỉ thị `/learn`:
> Bắt buộc tuân thủ 100% trong toàn bộ các Phase, Module, Script và Notebook của dự án này.
> Không áp dụng global mà duy trì cục bộ trong phạm vi dự án.

---

## 1. Nguyên Tắc "Read-Only" Cho Dữ Liệu Gốc BDD10K
- Thư mục `data/bdd10k/` là **kho dữ liệu nguồn gốc bất biến (Immutable / Read-Only 100%)**.
- **Tuyệt đối KHÔNG**:
  - Không sửa đổi cấu trúc hoặc nội dung của `data/bdd10k/`.
  - Không ghi thêm bất kỳ file mới nào (kể cả metadata, nhãn, ảnh hay log) vào trong `data/bdd10k/`.
- Khi cần đối chuẩn (benchmark) hoặc giải quyết xung đột lọc trùng:
  - Tra cứu trực tiếp từ file metadata gốc: `data/bdd10k/bdd10k_metadata.json` ở chế độ **Read-Only**.

---

## 2. Kiến Trúc 3 Tầng Dữ Liệu Chuẩn Mực (`raw` — `processed` — `splits`)
Mọi thao tác quản lý dữ liệu trong dự án BẮT BUỘC tuân thủ phân tầng 3 thư mục:
- 📁 **`data/raw/`**: Vùng lưu trữ dữ liệu thô ban đầu (unlabeled raw pool, các batch thu thập mới).
- 📁 **`data/processed/`**: Vùng lưu trữ kết quả sau khi xử lý & lọc trùng (Phase 01 FastDedup):
  - `unique_pool/`: Kho ảnh duy nhất (đã lọc trùng sạch sẽ).
  - `dropped_duplicates/`: Các ảnh trùng lặp bị loại bỏ.
  - `reports/` (hoặc manifest): Lưu trữ báo cáo profiling, manifest JSON ghi vết dedup.
- 📁 **`data/splits/`**: Vùng phân chia tập dữ liệu cho Active Learning (Phase 02, 03, 04...):
  - `fixed_test/`: 1.400 ảnh (20%) Frozen Gold Test Set (cố định, cách ly tuyệt đối Zero Data Leakage).
  - `fixed_val/`: 700 ảnh (10%) Tập validation cố định dùng chung cho Early Stopping.
  - `seed_v0/`: Max 500 ảnh (tối thiểu 10% pool) Tập mầm ban đầu để huấn luyện Model V0.
  - `unlabeled_pool/`: 4.394 ảnh Kho ảnh chưa gán nhãn để Active Learning khai thác (Uncertainty & Diversity Mining).

---

## 3. Nghiêm Cấm Tạo Thư Mục `metadata/` và `ground_truth/` Riêng Lẻ
- **Tuyệt đối KHÔNG tạo thêm thư mục `metadata/` riêng biệt** (như `paper_v4_10k/metadata/`) làm phình dự án và phân mảnh cấu trúc.
- **Tuyệt đối KHÔNG tạo thêm thư mục `ground_truth/` riêng biệt** (như `data/ground_truth/` hoặc các file mock pairs).
- Metadata gốc: Đọc trực tiếp từ `data/bdd10k/bdd10k_metadata.json` (Read-Only).
- Artifacts / manifest kết quả sau xử lý: Lưu trữ đúng vào `data/processed/` hoặc `results/` tương ứng theo từng Phase.

---

## 4. Cơ Chế Symlink Materialization (Zero Storage Overhead)
- Mọi thao tác đổ dữ liệu vào `data/processed/` và `data/splits/` **BẮT BUỘC sử dụng Symbolic Link (Symlink)** trỏ về ảnh gốc tại `data/bdd10k/images/train/`.
- **Tuyệt đối KHÔNG sao chép (copy) ảnh vật lý**: Giữ mức tăng dung lượng đĩa = 0 KB và tốc độ khởi tạo tức thì.
- Symlinks hoàn toàn tương thích với các DataLoader chuẩn của PyTorch, torchvision, OpenCV và PIL.

---

## 5. Cơ Chế Ánh Xạ Metadata $O(1)$ Qua Tên Ảnh (`image_id`)
- Ở mọi Notebook và Script (từ Phase 01, 02, 03 trở đi), khi cần thông tin nhãn (bounding boxes, categories) hoặc thuộc tính môi trường (weather, scene, timeofday):
  - Đọc trực tiếp từ `data/bdd10k/bdd10k_metadata.json` ở chế độ **Read-Only**.
  - Xây dựng lookup map: `{item["name"]: item}` để truy xuất với độ phức tạp thời gian $O(1)$.
  - Ánh xạ trực tiếp từ tên file (`image_id`) của ảnh trong `data/processed/` hoặc `data/splits/`.

---

## 6. Nguyên Tắc "Hỏi Trước Khi Quyết Định" (No Unilateral Assumption — Always Consult First)
> [!IMPORTANT]
> Agent TUYỆT ĐỐI KHÔNG TỰ Ý CHỐT QUYẾT ĐỊNH THIẾT KẾ HOẶC ĐƯA RA GIẢ ĐỊNH ĐƠN PHƯƠNG.
- Mọi quyết định quan trọng về:
  + Tỷ lệ phân chia tập dữ liệu (`fixed_test`, `fixed_val`, `seed_v0`, `unlabeled_pool`), quy mô số lượng ảnh.
  + Lựa chọn thuật toán, tham số siêu hình (hyperparameters), hàm mục tiêu, ngưỡng lọc.
  + Thay đổi cấu trúc thư mục hoặc luồng pipeline.
- **Quy trình bắt buộc**:
  1. Agent phải phân tích kỹ lưỡng, đề xuất các phương án cụ thể (Options A, B, C...) kèm ưu/nhược điểm định lượng.
  2. **HỎI Ý KIẾN VÀ CHỜ NGƯỜI DÙNG DUYỆT (Review & Approve)** trước khi bắt tay vào triển khai code.
  3. Không tự động ngầm định hay âm thầm thay đổi mà không có sự đồng thuận của người dùng.

---

## 7. Định Hướng Nghiên Cứu Chuẩn Mực (Strict Research-Oriented Rigor)
> [!TIP]
> Luôn giải quyết bài toán theo tư duy nghiên cứu khoa học hàn lâm (Top-Tier Scientific Design):
1. **Đấu trường Thực nghiệm Song song (Parallel Strategy Tournament)**:
   - Không áp đặt chủ quan một phương pháp ngay từ đầu.
   - Luôn thiết lập giải đấu đối đầu giữa các ứng viên: Baseline truyền thống (Random) vs Đối chứng cắt bỏ (Direct Metadata Ablation) vs Các biến thể tối ưu ($k$-Center, Facility Location) vs Phương pháp đề xuất (Hybrid Mix).
2. **Hệ Thống Đo Lường Kép Độc Lập (Dual Ground-Truth Scoring Engine)**:
   - Trục 1: Chân lý khách quan bên ngoài từ `bdd10k_metadata.json` (Read-Only) để đo Quota Error và bảo toàn danh mục 10 lớp giao thông.
   - Trục 2: Không gian đặc trưng vật lý định lượng từ Phase 02 (`data_profiling_manifest.json`) để đo bán kính phủ cực đại ($R_{\text{cov}}$), điểm dàn đều tựa mô-đun ($F_{\text{score}}$), độ phân tán PCA.
   - Trục 3: Rào chắn an toàn (Gatekeeper): Zero Video Overlap và Quét chéo pHash Hamming Distance $< 5 = 0$.
3. **Minh Bạch Hóa Bản Chất "Dưới Nắp Capô"**:
   - Khi giải thích giải pháp, luôn bóc tách bản chất toán học, so sánh bằng hình ảnh đời thường trực quan và giải thích rõ tại sao phương pháp đó thắng cuộc.

---

## 8. Chuẩn Cơ Cấu Phân Chia 4 Tập Cho Active Learning
Mọi quy trình phân chia dữ liệu cho bài toán Active Learning trong dự án tuân thủ chuẩn 4 tập:
1. 📁 **`fixed_test/`**: **20% tổng ảnh (1.400 ảnh)** — Frozen Gold Test Set (đóng băng vĩnh viễn, bài thi cuối kỳ chung).
2. 📁 **`fixed_val/`**: **10% tổng ảnh (700 ảnh)** — Fixed Validation Set (đóng băng dùng chung cho Early Stopping).
3. 📁 **`seed_v0/`**: **Max 500 ảnh (tối thiểu 10% pool)** — Phân bổ cân bằng theo 4 lát cắt khó (35% Đêm, 20% Nhỏ, 20% Che, 25% Ngày) bằng thuật toán Hybrid Mix 80/20, làm tập mầm huấn luyện Model V0 khởi đầu.
4. 📁 **`unlabeled_pool/`**: **Phần còn lại (~4.394 ảnh)** — Kho mỏ dữ liệu chưa gán nhãn dành riêng cho Active Learning khai thác qua các vòng lặp (Iterations 1, 2, 3...).

---

## 9. Chuẩn Quản Lý Mô Hình Song Mã & Đối Chuẩn Baseline (Dual-Model Baseline Rigor)
> [!IMPORTANT]
> Dự án áp dụng cơ chế đánh giá song mã CNN vs Transformer xuyên suốt mọi Phase:
1. **Lưu trữ & Niêm phong**: Checkpoints chính thức lưu tại `paper_v4_10k/models/` (`yolo11n_v0_best.pt` và `rtdetr_v0_best.pt`), kèm manifest niêm phong `models/models_lineage.json` (ghi hash SHA-256, recipe, số epochs, metrics thực đo).
2. **Nghiêm cấm Placeholder / Mock Metrics**: Tuyệt đối không chấp nhận các giá trị rác như `mAP = 0.0%`. Mọi mô hình baseline bắt buộc phải được huấn luyện hội tụ thực sự (tối thiểu 10 epochs cho Transformer trên MPS) và kiểm định độc lập trên `fixed_val` (700 ảnh) để lấy benchmark chính thức.
3. **Phân định Vai trò Khoa học**:
   - **YOLO11n (Quán quân V0)**: Kiến trúc CNN nhẹ (~2.6M params), sở hữu inductive bias không gian mạnh mẽ, hội tụ xuất sắc trên tập mầm 500 ảnh (`mAP50 = 20.23%`, Precision 53.09%), đóng vai trò **Động cơ Khai thác (Mining Engine)** cho Active Learning.
   - **RT-DETR (Transformer Baseline)**: Kiến trúc Vision Transformer lớn (~32M params), phụ thuộc Global Attention và Hungarian Matching, cần nhiều dữ liệu/epochs hơn, đóng vai trò **Mô hình Đối trọng (Counterpart Benchmark)** kiểm tra độ bền vững qua các vòng lặp.

---

## 10. Bản Chất Phép Đo Macro-Average & Bắt Buộc Lưu Trữ Bảng Per-Class 10 Lớp
> [!TIP]
> Phân rã nguyên nhân sụt giảm điểm số vĩ mô để phục vụ chiến lược Active Learning:
1. **Bản chất Toán học**: Chỉ số `mAP@50`, `Recall`, `Precision` tổng (`all`) là **Macro-Average (trung bình cộng chia đều qua 10 lớp BDD10K)**.
2. **Hiệu ứng Class Imbalance Penalty**: Các lớp phổ biến (`car` đạt mAP50 > 60%, `person` ~ 37%) bị kéo tụt bởi các lớp hiếm ít mẫu (`train: 0%`, `rider: 2.2%`, `motor: 2.1%`).
3. **Bắt Buộc Lưu Trữ Bảng Per-Class 10 Lớp**: Trong mọi Phase và báo cáo, kết quả kiểm định PHẢI được lưu trữ đầy đủ bảng phân rã chi tiết 10 lớp vào cả 2 định dạng:
   - File Markdown: `results/phase_XX/tables/per_class_...md` (phục vụ copy trực tiếp vào bài báo).
   - File JSON: `results/phase_XX/tables/per_class_...json` và cập nhật vào file JSON tổng kết chính của Phase.

---

## 11. Quy Trình Kiểm Toán Chất Lượng Dữ Liệu & Nhãn Ngẫu Nhiên Định Kỳ (Random Data & BBox Alignment Audit)
> [!IMPORTANT]
> Khi có bất kỳ nghi vấn nào về chất lượng nhãn, sai lệch ánh xạ hoặc điểm số mô hình:
1. **Quy mô mẫu**: Bốc ngẫu nhiên độc lập tối thiểu 100 ảnh từ toàn bộ 4 tập trong `data/splits/`.
2. **5 Tiêu chí kiểm toán bắt buộc**:
   - Tỷ lệ Symlink hợp lệ (Zero Broken Links) = 100%.
   - Tỷ lệ ánh xạ ngược $O(1)$ vào `bdd10k_metadata.json` (Read-Only) = 100%.
   - Tính hợp lệ của Bounding Boxes (không tràn viền, không tọa độ âm, đúng 10 lớp) = 100%.
   - Sai số cực đại tọa độ giữa file `.txt` và JSON gốc $\Delta \text{pixel} = 0.000000$ px (khớp tuyệt đối).
   - Render hình ảnh trực quan đối chiếu cho các tập, lưu vào `results/audit/visual_samples/`.
3. **Lưu vết**: Xuất báo cáo Markdown và JSON tại `results/audit/`.

---

## 12. Kỷ Luật Dọn Dẹp Cache Tự Động (Zero Stale Cache Discipline)
- Sau bất kỳ thao tác huấn luyện hoặc đánh giá nào của Ultralytics/PyTorch, lập tức quét và xóa sạch mọi file `*.cache` trong toàn bộ repo (`data/bdd10k/`, `paper_v4_10k/data/`).
- Tuyệt đối không để rò rỉ bất kỳ file `.cache` nào vào thư mục gốc `data/bdd10k/` (bảo toàn nguyên tắc Read-Only 100%).
