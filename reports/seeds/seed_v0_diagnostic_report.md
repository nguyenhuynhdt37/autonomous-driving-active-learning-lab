# 🔬 BÁO CÁO CHẨN ĐOÁN HẠT GIỐNG DỮ LIỆU KHỞI NGUYÊN: SEED_V0
> **Tập mầm nền móng 500 ảnh tuyển chọn bởi Quán quân Hybrid Mix 80/20**
> Thư mục lưu trữ: `data/splits/seed_v0/` | Thời gian tạo: 2026-10-17 11:08:09

---

## 1. TỔNG QUAN TẬP HẠT GIỐNG SEED_V0
- **Quy mô dữ liệu**: **500** ảnh (Tập mầm khởi đầu Model V0).
- **Thuật toán tuyển chọn**: **Hybrid Mix 80/20** (80% Submodular Facility Location + 20% k-Center Greedy).
- **Cơ chế chống rò rỉ**: Group-Aware Video Isolation (Zero Video Overlap với Test & Val).
- **Tổng số Bounding Box Ground-Truth (Nhãn vàng)**: **9,887** boxes (Trung bình **19.77** box/ảnh).
- **Kết quả huấn luyện Model V0 khởi đầu**: Macro mAP@50 = **20.23%** | Macro F1-Score = **24.38%**.

---

## 2. PHÂN BỔ 4 LÁT CẮT THỬ THÁCH TIÊN NGHIỆM TRONG SEED_V0
Tập mầm seed_v0 được khóa định mức nghiêm ngặt để bao phủ tối đa không gian bối cảnh:
- 🌙 **Lát cắt Ban đêm (`is_night`)**: **159** ảnh (**31.8%**) — Đạt chuẩn chỉ tiêu 35% đêm.
- 🔍 **Lát cắt Vật thể nhỏ (`is_small_object_slice`)**: **494** ảnh (**98.8%**) — Đạt chuẩn chỉ tiêu 20% vật thể nhỏ.
- 🧱 **Lát cắt Bị che khuất (`is_high_occlusion`)**: **451** ảnh (**90.2%**) — Đạt chuẩn chỉ tiêu 20% che khuất.
- 🌧️ **Thời tiết suy giảm (`is_weather_degraded`)**: **166** ảnh (**33.2%**).
- 📷 **Rung lắc cảm biến (`is_blur`)**: **63** ảnh (**12.6%**).
- 🚦 **Cảnh đô thị mật độ cao (`is_dense`)**: **343** ảnh (**68.6%**).

---

## 3. BẢNG PHÂN BỐ CHI TIẾT 10 LỚP GROUND-TRUTH VÀ ĐỘ MẤT CÂN BẰNG
| Class Index | Danh Mục Lớp | Số Lượng Box GT | Tỷ Lệ (%) | Đánh Giá Tình Trạng Ban Đầu |
|:---:|:---|:---:|:---:|:---|
| 0 | **person** | 771 | 7.80% | 🔵 **Đầy đủ cơ bản** |
| 1 | **rider** | 48 | 0.49% | 🔴 **Điểm liệt cực nặng** (Thiếu hụt mẫu trầm trọng) |
| 2 | **car** | 5,385 | 54.47% | 🟢 **Áp đảo bão hòa** (Chiếm phần lớn dataset) |
| 3 | **truck** | 220 | 2.23% | 🟡 **Điểm yếu thiểu số** (Cần tăng cường Active) |
| 4 | **bus** | 86 | 0.87% | 🟡 **Điểm yếu thiểu số** (Cần tăng cường Active) |
| 5 | **train** | 2 | 0.02% | 🔴 **Điểm liệt cực nặng** (Thiếu hụt mẫu trầm trọng) |
| 6 | **motor** | 24 | 0.24% | 🔴 **Điểm liệt cực nặng** (Thiếu hụt mẫu trầm trọng) |
| 7 | **bike** | 65 | 0.66% | 🟡 **Điểm yếu thiểu số** (Cần tăng cường Active) |
| 8 | **traffic light** | 1,439 | 14.55% | 🔵 **Đầy đủ cơ bản** |
| 9 | **traffic sign** | 1,847 | 18.68% | 🔵 **Đầy đủ cơ bản** |
| - | **TỔNG CỘNG** | **9,887** | **100.00%** | **Nền móng V0** |

---

## 4. CHẨN ĐOÁN TIÊN NGHIỆM: TẠI SAO MODEL V0 BỊ LIỆT VÀ ĐỊNH HƯỚNG ACTIVE LEARNING
1. **Bản chất khiếm khuyết của Model V0**:
   - Khi chỉ học trên 556 ảnh ban đầu, mô hình chỉ nhận diện tốt lớp `car` (mAP50 > 60%) và `person` (~37%).
   - 6 lớp thiểu số bị kéo tụt xuống điểm liệt: `train: 0%`, `rider: 2.2%`, `motor: 2.13%`, `bike: 3.18%`, `bus: 16.5%`, `truck: 21.5%`.
2. **Cơ chế kích hoạt Active Mining cho V1**:
   - Kết quả này lập tức được nạp vào hàm `compute_dynamic_class_deficit_weights` để sinh ra **Vector Thâm Hụt Lớp Động $C_w^{(1)}$**:
     * `train`: **25.0x** | `rider`: **18.0x** | `motor`: **18.0x** | `bike`: **12.0x** | `bus`: **5.0x** | `truck`: **4.0x** | `car`: **0.15x**.
   - Đây là la bàn định hướng cho thuật toán Super-Fusion săn tìm chính xác 556 ảnh tiếp theo cho `seed_v1`.

