# BDD10K Dataset (Berkeley DeepDrive Detection Subset)

Bộ dữ liệu chuẩn mực **BDD10K** được giải nén từ file lưu trữ chính thức `archive (1).zip`.

## 1. Cấu Trúc Dữ Liệu
- `images/`:
  - `train/`: **7.000 ảnh**
  - `val/`: **1.000 ảnh**
  - `test/`: **2.000 ảnh**
  - **Tổng cộng**: **10.000 bức ảnh độ phân giải cao**.
- `labels/`:
  - `train/`: **7.000 file nhãn YOLO txt**
  - `val/`: **1.000 file nhãn YOLO txt**
- `bdd10k_labels/`:
  - `bdd10k_labels_train.json` (JSON gốc UC Berkeley chứa weather, scene, timeofday, bbox)
  - `bdd10k_labels_val.json`
- `bdd10k_metadata.json`:
  - File tổng hợp metadata cho **8.000 ảnh** có nhãn, sẵn sàng làm Ground-Truth để đối chuẩn thuật toán phân tích lát cắt (Hard-Slice Benchmark).
- `bdd10k.yaml`: File cấu hình dataset YOLO gốc 10 lớp:
  `["person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign"]`.
