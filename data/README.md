# 🗄️ Cấu Trúc Dữ Liệu Nghiên Cứu — BDD10K Active Learning Lab (V4.0)

> **Nguyên tắc cốt lõi**: **Toàn bộ tập Test và Val để sẵn trong `bdd10k`, tuyệt đối không tự chia (No Custom Splits).**  
> Dữ liệu được tổ chức tinh gọn, trực diện, không tạo các thư mục trung gian vòng vèo (`splits/`, `raw/`), bảo đảm 100% tính minh bạch và chuẩn mực học thuật quốc tế.

---

## 1. Bản Đồ Dữ Liệu Tinh Gọn (Clean Data Architecture)

```text
paper_v4_10k/data/
├── bdd10k/                           # NGUỒN DỮ LIỆU CHÍNH THỨC CỦA PAPER
│   ├── images/
│   │   ├── train/                    # 7.000 ảnh gốc BDD10K (Nguồn ảnh thô cho FastDedup & Active Learning)
│   │   ├── val/                      # 1.000 ảnh gốc BDD10K (Tập kiểm định chuẩn quốc tế - CÓ SẴN TRONG BDD10K)
│   │   └── test/                     # 2.000 ảnh gốc BDD10K (Tập kiểm thử chuẩn quốc tế - CÓ SẴN TRONG BDD10K)
│   │
│   ├── labels/                       # Nhãn tọa độ Bounding Box YOLO (.txt)
│   │   ├── train/                    # 7.000 nhãn train (Làm Automated Oracle tra cứu khi Active Mining chọn ảnh)
│   │   └── val/                      # 1.000 nhãn val (Tính mAP@50, mAP@50-95 và đánh giá các lát cắt)
│   │
│   ├── bdd10k_metadata.json          # 8.000 metadata ảnh có ground-truth thời tiết, ánh sáng, bối cảnh
│   └── bdd10k.yaml                   # Cấu hình dataset Ultralytics YOLO chuẩn
│
├── processed/                        # KẾT QUẢ TIỀN XỬ LÝ (FASTDEDUP OUTPUTS)
│   ├── unique_pool/                  # Tập ảnh duy nhất sau khi lọc trùng qua pHash (HD < 5)
│   └── dropped_duplicates/           # Tập ảnh trùng lặp bị loại bỏ
│
└── ground_truth/                     # CSDL ĐỐI CHUẨN NHÃN CHÂN LÝ MICRO-BENCHMARK
    └── ground_truth_dedup_pairs.json # 6 cặp ảnh trùng lặp thực tế đã xác thực phục vụ benchmark Bước 1
```

---

## 2. Vì Sao Cấu Trúc Này Là Tối Ưu Nhất?

1. **Không Rò Rỉ Dữ Liệu (Zero Data Leakage)**:
   - Tập `val` (1.000 ảnh) và `test` (2.000 ảnh) nằm tách bạch hoàn toàn trong `bdd10k/images/val` và `bdd10k/images/test`.
   - Vòng lặp lọc trùng (FastDedup) và Học chủ động (Active Learning) chỉ khai thác trên 7.000 ảnh của `bdd10k/images/train`. Tập kiểm định và kiểm thử không bao giờ bị đụng chạm hay học lén.

2. **Không Tự Chia Tách (No Custom Splits)**:
   - Dùng trực tiếp Official Validation Split của Berkeley DeepDrive giúp kết quả của bài báo có thể so sánh trực tiếp với mọi nghiên cứu khác trên thế giới mà không sợ bị nghi ngờ chọn mẫu thiên lệch (*selection bias*).

3. **Gọn Gàng & Dễ Hiểu**:
   - Mọi script và notebook đều đọc trực tiếp từ `data/bdd10k/`, loại bỏ hoàn toàn các symlink ảo vòng vèo gây khó hiểu cho người đọc.
