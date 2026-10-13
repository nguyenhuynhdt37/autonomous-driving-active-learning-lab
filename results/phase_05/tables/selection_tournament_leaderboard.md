# 🏆 Bảng Xếp Hạng Đấu Trường Tuyển Chọn Dữ Liệu Phase 05 (Active Mining Tournament)

Đấu trường thực nghiệm 6 chiến lược đối đầu trên kho **4.394 ảnh** chưa gán nhãn (`unlabeled_pool`), mỗi chiến lược tuyển chọn đúng **K = 250 ảnh**.

| Hạng | Chiến Lược Tuyển Chọn | Bán Kính Phủ $R_{\text{cov}}$ | Vật Thể Hiếm (CPS) | Quota Đêm/Ngày | Proxy $\Delta\text{mAP50}$ | CSS Score | Danh Hiệu |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | `C6_Super_Fusion_Architecture` | 6.45 | 2.62 (655 objs) | 41% / 59% | +5.23% | **84.66** | **Quán Quân (Winner)** |
| 🥈 | `C5_LongTail_WeakSlice_Hybrid` | 3.50 | 1.94 (486 objs) | 32% / 68% | +5.46% | **71.12** | Ứng Viên |
| 🥉 | `C4_PureCoreSet_KCenter` | 3.16 | 1.17 (292 objs) | 33% / 67% | +4.69% | **61.93** | Ứng Viên |
| 4 | `C1_RandomSampling` | 5.86 | 0.76 (190 objs) | 44% / 56% | +3.41% | **49.13** | Ứng Viên |
| 5 | `C2_TopK_PureUncertainty` | 7.48 | 0.88 (220 objs) | 54% / 46% | +3.64% | **46.13** | Ứng Viên |
| 6 | `C3_WeakSlice_CostAware` | 9.41 | 0.36 (91 objs) | 59% / 41% | +2.30% | **20.00** | Ứng Viên |

### Diễn Giải Các Chỉ Số:
- **$R_{\text{cov}}$ (Coverage Radius)**: Bán kính phủ cực đại trên PCA 32D (càng nhỏ càng dàn đều, triệt tiêu vùng trống).
- **CPS (Rare Class Penetration Score)**: Tỷ lệ săn lùng vật thể thuộc 6 lớp điểm liệt (`rider, motor, bike, train, bus, truck`).
- **Quota Đêm/Ngày**: Tỷ lệ ảnh Ban đêm và Ban ngày trong batch (chuẩn mục tiêu: 35% Đêm, 25% Ngày).
- **Proxy $\Delta\text{mAP50}$**: Mức tăng trưởng dự phóng trên `fixed_val` (700 ảnh).
- **CSS (Composite Selection Score)**: Điểm tổng hợp chuẩn hóa 4 trục (0 - 100 điểm).