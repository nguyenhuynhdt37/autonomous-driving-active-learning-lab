# Bảng Xếp Hạng Đấu Trường Cắt Bỏ Công Thức Huấn Luyện (Phase 04)

| Hạng | Tên Cấu Hình | Mô Tả Kỹ Thuật | mAP@50 (%) | mAP@50-95 (%) | Recall (%) | Precision (%) | Composite Score | Thời Gian |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** 🏆 (Winner) | `cfg_1_freeze` | + Backbone Freezing: Đóng băng 10 layers đầu trong 5 epochs | **3.64%** | 1.92% | 10.48% | 3.24% | **5.35** | 1m 9.3s |
| **#2** | `cfg_4_proposed` | Full Proposed Recipe: Kết hợp cả 3 kỹ thuật + Early Stopping patience=10 | **3.06%** | 2.00% | 8.07% | 11.79% | **4.35** | 1m 8.5s |
| **#3** | `cfg_0_naive` | Naive Baseline: SGD, No freeze, No class-weights, Linear LR | **2.92%** | 1.51% | 7.95% | 3.10% | **4.15** | 1m 27.1s |
| **#4** | `cfg_3_adamw_cosine` | + AdamW & Cosine LR: Tối ưu hội tụ đa thang độ mượt mà | **2.78%** | 1.55% | 3.79% | 13.14% | **2.84** | 1m 19.6s |
| **#5** | `cfg_2_class_weights` | + Class-Weighted Loss: Phạt nặng lớp hiếm (train, motor, rider) | **1.12%** | 0.82% | 4.49% | 9.11% | **2.07** | 1m 20.8s |

*Tổng thời gian thực thi Phase 04:* **16m 32.7s**
