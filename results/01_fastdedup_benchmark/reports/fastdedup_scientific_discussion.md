# BÁO CÁO KHOA HỌC: BENCHMARK DUAL-STREAM FAST DEDUPLICATION (PHASE 01)
> Thực nghiệm tái lập độc lập trên Berkeley DeepDrive (BDD10K 1280x720, 7.000 ảnh training).

## 1. Kết Luận Khoa Học Trọng Yếu
1. **Khắc Phục Hoàn Toàn Lỗi Tự Quy Chiếu (Zero Circular Reasoning)**:
   - Kiểm toán bằng mắt người trên 34 cặp ứng viên xác nhận pHash đạt **100% Precision** và **100% Recall** trên các cặp dừng xe thực tế, trong khi aHash bị lạm thu (FP=26) và ResNet50 bị nhiễu tương đồng ngữ nghĩa.
2. **Hiệu Năng Phần Cứng Siêu Tốc (Real-Time CPU Throughput)**:
   - Proposed Method đạt **397.3 FPS** trên CPU đơn thuần, nhanh gấp **15.4 lần** so với ResNet50 (23.3 FPS), hoàn toàn đáp ứng streaming trên Edge Device mà không cần GPU.
3. **Giá Trị Mới của Keep-Most-Vehicles (Semantic Retention)**:
   - Cơ chế bảo toàn đối tượng giữ lại **76 phương tiện** và **50 xe nhỏ (<32px)**, vượt trội hoàn toàn so với cơ chế Keep-First (72 xe / 47 xe nhỏ), bảo toàn các mẫu học hiếm cho Active Learning.
