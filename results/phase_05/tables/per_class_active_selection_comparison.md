# Bảng Đối Chuẩn Chi Tiết Săn Lùng 6 Lớp Điểm Liệt BDD10K Giữa Các Chiến Lược

Đánh giá trực tiếp trên nhãn gốc `bdd10k_metadata.json` (Read-Only) cho từng batch K = 250 ảnh.

| Chiến Lược Tuyển Chọn | `truck` | `bus` | `motor` | `bike` | `rider` | `train` | **Tổng Vật Thể Hiếm** | CPS (Hiếm/Ảnh) | Bán Kính Phủ $R_{\text{cov}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `C1_RandomSampling` | 91 | 39 | 9 | 30 | 21 | 0 | 190 | 0.76 | 5.86 |
| `C2_TopK_PureUncertainty` | 89 | 45 | 8 | 51 | 25 | 2 | 220 | 0.88 | 7.48 |
| `C3_PureCoreSet_KCenter` | 113 | 35 | 24 | 70 | 47 | 3 | 292 | 1.17 | 3.16 |
| `C4_Super_Fusion_Architecture` | 122 | 88 | 89 | 192 | 155 | 9 | **655** | **2.62** | 6.45 |