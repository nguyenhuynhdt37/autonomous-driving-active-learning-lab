# BÁO CÁO KIỂM TOÁN LOGIC TOÀN TRÌNH (PIPELINE INTEGRITY AUDIT)
> **Chế độ**: Dry-Run Logic Audit | **Cam kết**: Zero Fake Metrics (Tuyệt đối không bịa đặt số liệu mAP)
> **Thời gian thực hiện**: 2026-10-17 12:04:37

---

## 1. BẢNG PHÂN BỔ NGÂN SÁCH DỮ LIỆU CHUẨN MỰC
| Hạng Mục Dữ Liệu | Số Lượng Ảnh | Tỷ Lệ (%) | Ghi Chú Kỹ Thuật |
|:---|:---:|:---:|:---|
| **Unique Pool Sạch (Phase 01)** | **6,994** | 100.0% | Kế thừa từ `data/processed/unique_pool` |
| **Frozen Gold Test** | **1,400** | 20.0% | Đóng băng vĩnh viễn (Zero Video Overlap) |
| **Fixed Validation** | **700** | 10.0% | Dùng chung cho Early Stopping |
| **Initial Unlabeled Bank** | **4,894** | 70.0% | Ngân sách còn lại sau Val/Test |
| **Seed V0 (20% Bank)** | **979** | 20.0% của Bank | Hybrid Mix 80/20 phân tầng 4 lát cắt khó |
| **Unlabeled Mining Pool** | **3,915** | 80.0% của Bank | Kho ảnh chưa gán nhãn cho Active Learning |
| **Ngân sách chọn mẫu $K$** | **250** ảnh/vòng | - | Tích lũy V0=979, V1=1229, V2=1479... |

---

## 2. KẾT QUẢ KIỂM TOÁN 5 TIÊU CHÍ AN TOÀN
- 🛡️ **Zero Video Overlap Isolation**: **PASSED (ĐẠT)** (Không có video sequence nào bị cắt chéo giữa Test, Val và Train Bank).
- 🔗 **Symlink Materialization (0 KB)**: **PASSED (ĐẠT)** (100% symlinks hợp lệ, không broken link, unlabeled pool không rò rỉ nhãn vàng).
- 📖 **Quality-Gated Oracle GT O(1)**: **100.0%** khớp nhãn vàng gốc từ `bdd10k_metadata.json`.
- 🧠 **Động cơ tuyển chọn C4 Super-Fusion**: Bốc thành công **250** ảnh khó, Top-K Margin Uncertainty trung bình = **0.9800**.
- 🚦 **Bộ rào chắn 5 tầng Stopping Guard**: Hoạt động bình thường, sẵn sàng kiểm soát huấn luyện.

---

## 3. PHÂN BỐ LÁT CẮT KHÓ TRONG TOP-250 ẢNH ĐƯỢC TUYỂN CHỌN
- 🌙 Ảnh ban đêm (`is_night`): **205** ảnh
- 📷 Ảnh mờ rung cảm biến (`is_blur`): **85** ảnh
- 🧱 Ảnh che khuất cao (`is_high_occlusion`): **250** ảnh
- 🔍 Ảnh vật thể nhỏ (`is_small_object_slice`): **250** ảnh
- 🚦 Cảnh giao thông dày đặc (`is_dense`): **209** ảnh

*Báo cáo được niêm phong tự động bởi AutonomousLoopOrchestrator trong chế độ Dry-Run Logic Audit.*
