"""
standardize_phase05_notebook.py
Cập nhật khắc họa rõ nét 4 nhóm trụ cột thử thách vật lý trong Notebook 05.
"""

from pathlib import Path
import nbformat

def update_notebook_05():
    base_dir = Path(__file__).resolve().parent.parent
    nb_path = base_dir / "notebooks" / "05_active_mining_tournament_and_hitl.ipynb"
    
    print(f"📖 Đang đọc notebook 05 tại: {nb_path}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    cell_0_text = (
        "# 🎯 BDD10K Active Learning Lab — Phase 05: Active Selection Tournament & Quality-Gated Data Curation\n"
        "> **Dự án**: End-to-End Human-in-the-Loop Active Learning Lab — BDD10K Autonomous Driving Benchmark (Paper V4.0)  \n"
        "> **Tác giả**: Nguyễn Xuân Huỳnh  \n"
        "> **Repository**: [BuildPhase/tang/paper_v4_10k](https://github.com/nguyenhuynhdt37/plant-disease-detection-ai)  \n"
        "> **Quy chuẩn kiến trúc**: Tuân thủ nghiêm ngặt `AGENTS.md` (Read-Only 100% `data/bdd10k/`, Symlink Materialization 0 KB đĩa, Tra cứu $O(1)$ qua `image_id`, Cấp 100% Nhãn Vàng BDD10K bảo đảm Zero Label Noise).\n\n"
        "---\n\n"
        "## 1. Giới Thiệu & Mục Tiêu Nghiên Cứu Data-Centric Của Phase 05\n"
        "Sau khi hoàn tất **Phase 04**, mô hình Quán quân khởi đầu **YOLO11n V0** đạt `mAP@50 = 20.23%` nhưng bộc lộ điểm yếu cốt tử: **6 lớp giao thông bị rơi vào vùng điểm liệt** (`train: 0.0%`, `rider: 2.2%`, `motor: 2.13%`, `bike: 3.18%`, `bus: 16.5%`, `truck: 21.5%`) do mất cân bằng dữ liệu cực đoan 2.600:1 trong tập mầm `seed_v0` (500 ảnh).\n\n"
        "### 🏛️ 4 Nhóm Trụ Cột Thử Thách Vật Lý Đa Yếu Tố (Multi-Factor Physical Challenges):\n"
        "Để bứt phá khỏi bẫy hiệu năng của Random Sampling, thuật toán tuyển chọn mẫu thông minh của Phase 05 được thiết kế nhằm đồng thời giải quyết **4 Trụ Cột Thử Thách Cốt Lõi**:\n"
        "1. ⚖️ **Trụ Cột 1 - Mất Cân Bằng Nhãn Lớp Cực Đoan (Long-Tail Class Imbalance 2.600:1)**: Các lớp hiếm và yếu thế (Vulnerable Road Users - VRU) như `train, rider, motor, bike, bus, truck` bị lớp áp đảo `car` đè bẹp; cần cơ chế phạt thâm hụt lớp động $C_w^{(k)}(c)$ dựa trên F1 và Recall để kéo các lớp liệt lên.\n"
        "2. 🌧️ **Trụ Cột 2 - Môi Trường Ánh Sáng & Thời Tiết Khắc Nghiệt ($E(x_i)$)**: Môi trường ban đêm thiếu sáng (`night`, tăng x1.35), mưa gió sương mù trơn trượt (`rainy/weather degraded`, tăng x1.25), và các nút giao lộ tắc nghẽn dày đặc (`dense traffic`, tăng x1.20).\n"
        "3. 📷 **Trụ Cột 3 - Rung Lắc Cảm Biến & Motion Blur ($B(x_i)$)**: Rung giật camera xe trên mặt đường gồ ghề làm nhòe ảnh (`laplacian_variance < 120`, tăng x1.35) cùng hiện tượng chói lóa đèn xe ban đêm; thử thách độ bền suy luận của mô hình trước nhiễu vật lý.\n"
        "4. 🔍 **Trụ Cột 4 - Hình Học Bounding Box Nhỏ & Che Khuất ($G_w(o)$)**: Các đối tượng tầm xa có kích thước siêu nhỏ ($< 32 \\times 32$ pixels, trọng số 1.5x), đối tượng bị che khuất một phần (`occluded`, trọng số 1.4x), và đối tượng bị tràn viền cắt góc (`truncated`, trọng số 1.2x).\n\n"
        "### 🎯 Các Mục Tiêu Cốt Lõi Của Phase 05:\n"
        "1. **Khai Phá Kho 4.394 Ảnh Chưa Gán Nhãn (`unlabeled_pool`)**: Quét inference toàn phần bằng song mã CNN (YOLO11n V0) và Transformer (RT-DETR V0), trích xuất Margin Uncertainty $U_i$, Bất đồng kiến trúc $D_i^{\\text{QBC}}$ và không gian đặc trưng PCA 32D.\n"
        "2. **Đấu Trường Tuyển Chọn 6 Chiến Lược Đối Đầu (K = 250 ảnh)**:\n"
        "   - **C1 (`RandomSampling`)**: Mốc sàn ngẫu nhiên (Floor Baseline).\n"
        "   - **C2 (`TopK_PureUncertainty`)**: Top-K bối rối (bộc lộ bẫy tụ cụm outlier).\n"
        "   - **C3 (`WeakSlice_CostAware`)**: Bốc theo ma trận lát cắt khó $W_i$ từ Phase 04.\n"
        "   - **C4 (`PureCoreSet_KCenter`)**: $k$-Center Greedy thuần túy cực tiểu hóa bán kính phủ $R_{\\text{cov}}$.\n"
        "   - **C5 (`LongTail_WeakSlice_Hybrid`)**: Đặc trị 6 lớp điểm liệt ($C_w$) kết hợp $k$-Center Greedy.\n"
        "   - **C6 (`Super_Fusion_Architecture`)**: 🏆 **Quán quân đề xuất**: Tổ hợp đa mục tiêu $[0.6 \\cdot U_i + 0.4 \\cdot D_{\\text{QBC}}] \\times C_w \\times W_i / (1 + 0.05 \\cdot N_{\\text{boxes}})$ kết hợp thuật toán **Submodular Facility Location (Lazy Greedy)** trên 4 Trụ Cột Thử Thách.\n"
        "3. **Chính Sách Cổng Kiểm Soát Chất Lượng (Quality-Gated Pre-Labeling Policy)**:\n"
        "   - Thẩm định độ chín của mô hình theo từng lớp: Tắt hoàn toàn pre-label ở các lớp yếu để chống box rác; chỉ mở pre-label ở lớp đạt ngưỡng ($Precision \\ge 75\\%$).\n"
        "   - Đo lường chỉ số **Annotation Effort Reduction (AER %)** tiết kiệm công sức thực tế.\n"
        "4. **Vật Liệu Hóa Dataset V1 (750 ảnh) & Cấp 100% Nhãn Vàng BDD10K**:\n"
        "   - Cấp 100% nhãn vàng chuẩn BDD10K để triệt tiêu hoàn toàn rủi ro Confirmation Bias / Error Drift.\n"
        "   - Rút 250 ảnh khỏi `unlabeled_pool/` (còn lại đúng 4.144 ảnh).\n"
    )

    nb.cells[0].source = cell_0_text

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print("✅ Đã cập nhật Notebook 05 thành công với 4 Trụ Cột Thử Thách Vật Lý!")

if __name__ == "__main__":
    update_notebook_05()
