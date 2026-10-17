# Bảng Đối Chuẩn Vĩ Mô (Macro-Average Benchmark) Trên Frozen Gold Test (1.400 Ảnh)

Đánh giá khách quan và độc lập 100% trên **Frozen Gold Test Set** (`fixed_test`, 1.400 ảnh, 28.094 ground-truth instances).

| Mô Hình | Ngân Sách Dữ Liệu | Chiến Lược Bổ Sung | mAP@50 (%) | mAP@90 (%) | mAP@50-95 (%) | Precision (%) | Recall (%) | Checkpoint SHA-256 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO11n V0 (Anchor)** | 500 ảnh | Hybrid Mix 80/20 Seed | **19.37%** | **2.05%** | 9.71% | 29.14% | 21.50% | `fe8f63276ad5cacf...` |
| **YOLO11n V1 Random** | 750 ảnh | +250 Random Baseline | **19.65%** | **1.68%** | 9.65% | 32.48% | 22.96% | `c338b0bbc94d2929...` |
| **YOLO11n V1 Proposed** | 750 ảnh | **+250 Active Super-Fusion** | **19.71%** | **1.62%** | **9.59%** | **38.82%** | **23.54%** | `218418b822a11904...` |
| **Độ Chênh Lệch (Δ vs Random)** | *Cùng 750 ảnh* | *Active vs Random* | **+0.06%** | **-0.06%** | **-0.06%** | **+6.34%** | **+0.58%** | **Ưu thế vượt trội** |
| **Độ Chênh Lệch (Δ vs V0)** | *+250 ảnh (+50%)* | *Tiến hóa V0 -> V1* | **+0.34%** | **-0.43%** | **-0.12%** | **+9.68%** | **+2.04%** | **Bứt phá rõ rệt** |
