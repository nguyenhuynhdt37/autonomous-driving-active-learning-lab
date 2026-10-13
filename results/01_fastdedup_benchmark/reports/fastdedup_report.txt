# BÁO CÁO KHOA HỌC: BENCHMARK DUAL-STREAM FAST DEDUPLICATION (PHASE 01)

## 1. Tóm Tắt Định Lượng (Executive Summary)
- Tập dữ liệu đánh giá: Berkeley DeepDrive (BDD10K) 1280x720, 7.000 ảnh training qua 598 video sequences.
- Phương pháp đề xuất: Dual-Stream pHash kết hợp cơ chế giải quyết xung đột Keep-Most-Vehicles (HD < 5).

## 2. Kết Quả Đối Chuẩn Chính
1. **Thông Lượng & Tốc Độ Tính Toán**:
   - aHash và dHash đạt tốc độ cao nhất (~440-450 FPS), tuy nhiên nhạy cảm với thay đổi độ sáng bối cảnh.
   - ResNet50 Deep Feature Embedding chỉ đạt **23.3 FPS** trên CPU (chậm hơn 13.4 lần so với pHash), đòi hỏi tài nguyên GPU và weights 97.7 MB, không khả thi cho Edge Device streaming.
   - Proposed Method đạt **360.3 FPS** trên CPU, hoàn toàn đáp ứng xử lý thời gian thực (Zero GPU).

2. **Độ Chuẩn Xác Lọc Trùng (Quality & Pareto Threshold)**:
   - Ngưỡng Hamming Distance HD = 5 chứng minh là điểm cân bằng Pareto tối ưu giữa Precision và Recall.

3. **Bảo Toàn Đối Tượng (Keep-Most-Vehicles Novelty)**:
   - Cơ chế Keep-Most-Vehicles chứng minh bảo toàn vượt trội số lượng BBox phương tiện và Small Objects so với Keep-First và Random-Drop.

## 3. Kết Luận
Giải pháp Dual-Stream pHash + Keep-Most-Vehicles là lựa chọn khoa học và thực tiễn tối ưu cho Bước 1 của Pipeline.
