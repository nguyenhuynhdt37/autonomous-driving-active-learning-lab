# Bảng Xếp Hạng Đấu Trường Tuyển Chọn Dữ Liệu Phase 05 (Active Mining Tournament)

Đấu trường thực nghiệm 6 chiến lược đối đầu trên kho **4.394 ảnh** chưa gán nhãn (`unlabeled_pool`), mỗi chiến lược tuyển chọn đúng **K = 250 ảnh**.
Tất cả chỉ số đánh giá được đối chuẩn trực tiếp 100% trên nhãn gốc BDD10K (Read-Only) và không gian đặc trưng PCA 32D.

| Hạng | Chiến Lược Tuyển Chọn | Bán Kính Phủ $R_{\text{cov}}$ | Vật Thể Hiếm BDD10K (CPS) | Quota Đêm/Ngày | Tín Hiệu Bất Định ($U + D$) | CSS Score | Danh Hiệu |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `C6_Super_Fusion_Architecture` | 6.45 | 2.62 (655 objs) | 41% / 59% | 0.7114 | **84.21** | **Quán Quân (Winner)** |
| 2 | `C5_LongTail_WeakSlice_Hybrid` | 3.50 | 1.94 (486 objs) | 32% / 68% | 0.6241 | **52.33** | Ứng Viên |
| 3 | `C4_PureCoreSet_KCenter` | 3.16 | 1.17 (292 objs) | 33% / 67% | 0.6088 | **46.02** | Ứng Viên |
| 4 | `C1_RandomSampling` | 5.86 | 0.76 (190 objs) | 44% / 56% | 0.6172 | **43.95** | Ứng Viên |
| 5 | `C2_TopK_PureUncertainty` | 7.48 | 0.88 (220 objs) | 54% / 46% | 0.6387 | **41.92** | Ứng Viên |
| 6 | `C3_WeakSlice_CostAware` | 9.41 | 0.36 (91 objs) | 59% / 41% | 0.6747 | **32.85** | Ứng Viên |

### Diễn Giải Các Tiêu Chí Đo Lường Dữ Liệu Thật:
- **$R_{\text{cov}}$ (Coverage Radius)**: Bán kính phủ cực đại trên không gian PCA 32D (càng nhỏ càng dàn đều, triệt tiêu vùng trống dữ liệu).
- **CPS (Rare Class Penetration Score)**: Tỷ lệ săn lùng vật thể thuộc 6 lớp điểm liệt BDD10K gốc (`rider, motor, bike, train, bus, truck`) qua tra cứu $O(1)$ vào `bdd10k_metadata.json`.
- **Quota Đêm/Ngày**: Tỷ lệ ảnh Ban đêm và Ban ngày trong batch 250 ảnh (chuẩn mục tiêu: 35% Đêm, 25% Ngày theo Rule 8 AGENTS.md).
- **Tín Hiệu Bất Định ($U + D$)**: Trung bình cộng giữa Margin Uncertainty $U_i$ (YOLO11n V0) và QBC Disagreement $D_i^{\text{QBC}}$ (giữa YOLO11n và RT-DETR) của 250 ảnh được tuyển chọn.
- **CSS (Composite Selection Score)**: Điểm tổng hợp chuẩn hóa 4 trục dữ liệu thật (30% Rare CPS + 30% Coverage + 20% Quota + 20% Uncertainty).