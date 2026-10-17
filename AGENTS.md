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

---

## 13. Bản Chất Phase 02: Công Xưởng Sinh Metadata & Giải Đấu Thuật Toán So Khớp Ground-Truth
> [!IMPORTANT]
> Phase 02 KHÔNG PHẢI là thống kê mô tả chung chung hay vẽ biểu đồ tĩnh, mà là **Công xưởng viết Metadata hồ sơ định lượng cho từng ảnh** (`data_profiling_manifest.json`):
1. **Bản chất Sinh Metadata Từng Mẫu**: Mỗi ảnh trong Unique Pool (6.994 ảnh) phải được định danh và gán nhãn thuộc tính cụ thể (`is_night`, `is_blur`, `is_weather`, `is_high_occ`, `is_small_slice`, `is_dense`).
2. **Đấu Trường Thực Nghiệm Đối Đầu (Algorithm Tournament)**:
   - Với mỗi thuộc tính (Sub-Point), tuyệt đối không áp đặt một thuật toán chủ quan.
   - Luôn thiết lập giải đấu đối đầu song song giữa các thuật toán đối chứng (Baselines) và phương pháp cải tiến (Proposed):
     + *Point 2.1 (Ban đêm)*: RGB Mean vs Otsu Thresholding vs **Quán Quân YCbCr P50 (Median)** (F1 = 97.12%, triệt tiêu 100% lóa đèn pha xe đối diện).
     + *Point 2.2 (Mờ & Rung xóc)*: Laplacian Variance vs Tenengrad Gradient vs **Quán Quân Tenengrad + 2D-FFT Radial Slope** (F1 = 100% trên tập Huỳnh Audit, bắt trọn rung xóc thân xe ID 11 & 12).
     + *Point 2.3 (Khí quyển sương mù)*: Kênh Tối Cắt Bỏ vs He et al. Dark Channel Prior vs **Quán Quân DCP + Clear Day Guard** (F1 = 31.03%, triệt tiêu 971 cảnh báo giả trên đường nhựa tối màu ban ngày).
     + *Point 2.4 (Che khuất cao)*: Naive 2D BBox IoU vs IoMin Ratio vs **Quán Quân Directed Overlap DAG y_bottom** (F1 = 93.80%, vượt +36.05% F1 trên 70.091 hộp nhãn gốc BDD10K).
     + *Point 2.5 (Mật độ & Đa thang đo)*: Đếm hộp thô vs **Quán Quân Shannon Scale Entropy + Density Guard** (F1 = 95.40%, phân lập chính xác cảnh kẹt xe đô thị).
3. **Hệ Thống So Khớp Ground-Truth Kép (Ground-Truth Cross-Verification)**:
   - *Trục Khách quan BDD10K*: Đọc trực tiếp từ `data/bdd10k/bdd10k_metadata.json` (Read-Only $O(1)$) qua các trường `timeofday`, `weather`, và các bounding boxes có thuộc tính `occluded: true` để tính Precision, Recall, F1 và Quota Error thực.
   - *Trục Mắt Người Độc Lập*: Tập kiểm toán độc lập 100 mẫu (Nguyễn Xuân Huỳnh Audit) để bắt các lỗi thị giác thực tế ngoài nhãn gốc (như rung xóc gờ giảm tốc).
4. **Căn Cứ Khoa Học Bất Biến Cho Phase 03**:
   - Toàn bộ kết quả thuật toán Quán Quân được đóng dấu và xuất bản vào `data_profiling_manifest.json`.
   - File manifest này là **căn cứ khoa học duy nhất và bắt buộc** để Phase 03 bốc mẫu phân tầng 500 ảnh `seed_v0` (35% Đêm, 20% Nhỏ, 20% Che, 25% Ngày theo Rule 8) và bảo toàn tính cô lập tuyệt đối Zero Data Leakage cho `fixed_test` (1.400 ảnh) và `fixed_val` (700 ảnh).
5. **Cơ Cấu Lưu Trữ & Tra Cứu Thư Viện 100 Mẫu Kiểm Toán Mắt Người**:
   - Bảng manifest kiểm toán mờ độc lập (`results/phase_02/tables/human_blur_ground_truth_manifest.json`) lưu trữ 100 mẫu dưới cấu trúc chuẩn với key chính là `candidates` (gồm các trường `id`, `image_name`, `laplacian_var`, `tenengrad`, `fft_energy_slope`, `human_label`, `is_blur_ground_truth`, `notes`).
   - Mọi script và frontend controller khi load dữ liệu bắt buộc phải trích xuất đúng key `candidates` để đảm bảo nạp và render đầy đủ 100/100 mẫu ảnh kiểm toán mắt người thực tế kèm bộ ba chỉ số quang học.

---

## 14. Kỷ Luật Thiết Kế UI & Video Simulator Kế Thừa Chuẩn GreenSM Nordic Editorial
> [!TIP]
> Tiêu chuẩn giao diện và trình mô phỏng trực quan hóa phục vụ bảo vệ đồ án và xuất bản báo cáo:
1. **Thanh Lọc 100% Emoji Dạng AI (Zero AI Emoji Discipline)**:
   - Nghiêm cấm hoàn toàn các biểu tượng emoji phong cách chatbot AI (`✨`, `🚀`, `🎯`, `💡`, `🔍`, `🔥`...) trên toàn bộ giao diện HTML, JS và Markdown báo cáo.
   - Bắt buộc 100% sử dụng icon SVG nét mảnh chuẩn Lucide Icons (`stroke-width: 1.5 - 2px`).
2. **Hệ Thống Màu & Bố Cục Chuẩn GreenSM Nordic Editorial**:
   - Màu nền giấy ấm: `#F8F7F2` (Paper Warm), Thẻ nội dung: `#FFFFFF` (Surface White), Viền phân cách: `#E5E3DC` (Subtle Border).
   - Màu nhận diện thương hiệu nghiên cứu: Xanh rừng sâu `#2D5A34` (Forest Green), Xanh bạc hà nhẹ `#E8F0EA` (Mint Light) và Viền bạc hà `#CDE0D1`.
   - Typography: Font chữ hiện đại, trang nhã (Inter / Be Vietnam Pro / Editorial Sans) kết hợp JetBrains Mono / Courier New cho mã code và số liệu kỹ thuật.
3. **Kiến Trúc Video Simulator Split View Đồng Bộ Thời Gian Thực**:
   - Trình phát video chuẩn Master MP4 (1280x720 @ 30 FPS, H.264/AAC, bitrate tối ưu < 2MB).
   - Bố cục Split View 2 cột: Cột trái phát Video trực quan hóa luồng dữ liệu (Dataflow Animation); Cột phải là Bảng Telemetry Live tự động cập nhật trạng thái chặng, thuật toán, FPS, và mã giả theo sự kiện `timeupdate` của video.
   - Tích hợp 4 Stage Pills cho phép người xem click nhảy trực tiếp (`currentTime`) đến từng chặng trong video.
4. **Chuẩn Giọng Đọc Thuyết Minh (Audio Voice Standard)**:
   - Giọng thuyết minh video mô phỏng đồng bộ thống nhất xuyên suốt các Point: Bắt buộc sử dụng **Microsoft Edge TTS `vi-VN-HoaiMyNeural` (Natural Voice, tốc độ `rate=+5%`)**.
   - Nội dung thuyết minh phải súc tích, mạch lạc, phát âm chuẩn thuật ngữ kỹ thuật, đồng bộ chính xác với timeline của từng chặng.
5. **Kỷ Luật Viền Thẻ Card (Zero Clunky Directional Borders Discipline)**:
   - **Tuyệt đối nghiêm cấm**: Sử dụng các class viền một cạnh dày 4px như `border-l-4`, `border-r-4`, `border-t-4`, `border-b-4` kết hợp màu đậm trên các thẻ card bo góc (`rounded-xl`, `rounded-2xl`). Cơ chế render CSS trên góc bo tròn sẽ làm lộ ra các vệt cong màu lem nhem, méo mó ở cạnh viền ngoài.
   - **Bắt buộc 100%**: Sử dụng viền đồng nhất 1px nhẹ nhàng (`border border-[#E5E3DC]` hoặc `border border-[#ECEAE2]`) bao quanh toàn bộ thẻ card theo chuẩn GreenSM Nordic Editorial.
   - **Phân định danh mục**: Sử dụng tag pill thanh lịch (`px-2 py-0.5 rounded-md text-[10px] font-mono`), màu sắc chữ tiêu đề hoặc icon SVG Lucide bên trong, tuyệt đối không dùng viền biên dày để làm điểm nhấn.
