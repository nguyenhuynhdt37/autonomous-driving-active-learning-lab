"""
generate_phase_05_notebook.py - Script Tự Động Sinh Notebook 05 Data-Centric Chuẩn Mực
========================================================================================
Sinh file: paper_v4_10k/notebooks/05_active_mining_tournament_and_hitl.ipynb
Tập trung 100% vào Data-Centric AI:
- Đấu trường 6 chiến lược tuyển chọn dữ liệu (K = 250 ảnh).
- Leaderboard CSS & Quán quân C6 Super-Fusion.
- Chính sách Quality-Gated Pre-labeling Policy (kiểm soát độ chín mô hình, AER %).
- Đóng gói Dataset V1 (750 ảnh) cấp 100% Nhãn Vàng BDD10K (Zero Label Noise).
- Showcase 5 biểu đồ khoa học 300 DPI đỉnh cao.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Thêm đường dẫn project
current_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_script_dir.parent))
from src.paths import find_project_dir


def create_phase_05_notebook() -> Path:
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    def add_md(text: str):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")],
        })

    def add_code(code: str):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.split("\n")],
        })

    # =========================================================================
    # CELL 1: Tiêu đề & Tổng quan (Markdown)
    # =========================================================================
    add_md("""# 🎯 BDD10K Active Learning Lab — Phase 05: Active Selection Tournament & Quality-Gated Data Curation
> **Dự án**: BDD10K Active Learning Lab: Hệ Thống Tự Động Hóa Vòng Lặp Dữ Liệu Học Tập Chủ Động (Human-in-the-Loop)  
> **Tác giả**: Nguyễn Xuân Huỳnh  
> **Repository**: [BuildPhase/tang/paper_v4_10k](https://github.com/nguyenhuynhdt37/plant-disease-detection-ai)  
> **Quy chuẩn kiến trúc**: Tuân thủ nghiêm ngặt `AGENTS.md` (Read-Only 100% `data/bdd10k/`, Symlink Materialization 0 KB đĩa, Tra cứu $O(1)$ qua `image_id`, Cấp 100% Nhãn Vàng BDD10K bảo đảm Zero Label Noise).

---

## 1. Giới Thiệu & Mục Tiêu Nghiên Cứu Data-Centric Của Phase 05
Sau khi hoàn tất **Phase 04**, mô hình Quán quân khởi đầu **YOLO11n V0** đạt `mAP@50 = 20.23%` nhưng bộc lộ điểm yếu cốt tử: **6 lớp giao thông bị rơi vào vùng điểm liệt** (`train: 0.0%`, `rider: 2.2%`, `motor: 2.13%`, `bike: 3.18%`, `bus: 16.5%`, `truck: 21.5%`) do mất cân bằng dữ liệu cực đoan 2.600:1 trong tập mầm `seed_v0` (500 ảnh).

Mục tiêu cốt lõi của **Phase 05**:
1. **Khai Phá Kho 4.394 Ảnh Chưa Gán Nhãn (`unlabeled_pool`)**: Quét inference toàn phần bằng song mã CNN (YOLO11n V0) và Transformer (RT-DETR V0), trích xuất Margin Uncertainty $U_i$, Bất đồng kiến trúc $D_i^{\\text{QBC}}$ và không gian đặc trưng PCA 32D.
2. **Đấu Trường Tuyển Chọn 6 Chiến Lược Đối Đầu (K = 250 ảnh)**:
   - **C1 (`RandomSampling`)**: Mốc sàn ngẫu nhiên (Floor Baseline).
   - **C2 (`TopK_PureUncertainty`)**: Top-K bối rối (bộc lộ bẫy tụ cụm outlier).
   - **C3 (`WeakSlice_CostAware`)**: Bốc theo ma trận lát cắt khó $W_i$ từ Phase 04.
   - **C4 (`PureCoreSet_KCenter`)**: $k$-Center Greedy thuần túy cực tiểu hóa bán kính phủ $R_{\\text{cov}}$.
   - **C5 (`LongTail_WeakSlice_Hybrid`)**: Đặc trị 6 lớp điểm liệt ($C_w$) kết hợp $k$-Center Greedy.
   - **C6 (`Super_Fusion_Architecture`)**: 🏆 **Quán quân đề xuất**: Tổ hợp đa mục tiêu $[0.6 \\cdot U_i + 0.4 \\cdot D_{\\text{QBC}}] \\times C_w \\times W_i / (1 + 0.05 \\cdot N_{\\text{boxes}})$ kết hợp thuật toán **Submodular Facility Location (Lazy Greedy)**.
3. **Chính Sách Cổng Kiểm Soát Chất Lượng (Quality-Gated Pre-Labeling Policy)**:
   - Thẩm định độ chín của mô hình theo từng lớp: Tắt hoàn toàn pre-label ở các lớp yếu để chống box rác; chỉ mở pre-label ở lớp đạt ngưỡng ($Precision \\ge 75\\%$).
   - Đo lường chỉ số **Annotation Effort Reduction (AER %)** tiết kiệm công sức thực tế.
4. **Vật Liệu Hóa Dataset V1 (750 ảnh) & Cấp 100% Nhãn Vàng BDD10K**:
   - Cấp 100% nhãn vàng chuẩn BDD10K để triệt tiêu hoàn toàn rủi ro Confirmation Bias / Error Drift.
   - Rút 250 ảnh khỏi `unlabeled_pool/` (còn lại đúng 4.144 ảnh).""")

    # =========================================================================
    # CELL 2: Setup Môi Trường & Imports (Code)
    # =========================================================================
    add_code("""# 1. Cấu hình Môi trường & Khai báo Imports
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Tìm kiếm thư mục gốc dự án chuẩn xác
curr = Path.cwd().resolve()
found_root = None
for _ in range(4):
    if (curr / "paper_v4_10k" / "src").exists():
        found_root = curr
        break
    curr = curr.parent

if found_root:
    p1 = str(found_root / "paper_v4_10k")
    p2 = str(found_root)
    if p1 not in sys.path:
        sys.path.insert(0, p1)
    if p2 not in sys.path:
        sys.path.insert(0, p2)

try:
    from src.paths import find_project_dir, get_results_dir, get_splits_dir, get_bdd10k_metadata_file
    from src.utils import get_device, set_seed
    from src.splits import BDD10K_CLASS_NAMES
    from src.active_mining import PoolFeatureExtractor, ActiveMiningTournamentRunner, RARE_CLASSES
    from src.hitl_engine import QualityGatedAnalyzer, TwoTierRouter, DatasetV1Packager
except ImportError:
    from paper_v4_10k.src.paths import find_project_dir, get_results_dir, get_splits_dir, get_bdd10k_metadata_file
    from paper_v4_10k.src.utils import get_device, set_seed
    from paper_v4_10k.src.splits import BDD10K_CLASS_NAMES
    from paper_v4_10k.src.active_mining import PoolFeatureExtractor, ActiveMiningTournamentRunner, RARE_CLASSES
    from paper_v4_10k.src.hitl_engine import QualityGatedAnalyzer, TwoTierRouter, DatasetV1Packager

set_seed(42)
device = get_device()
print(f"🚀 Môi trường tính toán sẵn sàng: Thiết bị = {device}")
print(f"📂 Project Root: {find_project_dir()}")""")

    # =========================================================================
    # CELL 3: Khảo sát Unlabeled Pool & Load Cache (Code)
    # =========================================================================
    add_code("""# 2. Nạp Cache Đặc Trưng & Độ Bất Định Kho 4.394 Ảnh Unlabeled Pool
results_p5_dir = get_results_dir("phase_05")
cache_path = results_p5_dir / "unlabeled_pool_feature_cache.json"

extractor = PoolFeatureExtractor(cache_file=cache_path, device=device)
records = extractor.extract_or_load()

print(f"✓ Đã nạp thành công {len(records):,} bản ghi đặc trưng từ Cache!")
sample_rec = next(iter(records.values()))
print(f"  * Kích thước vector PCA: {len(sample_rec.embedding_pca)}D")
print(f"  * Margin Uncertainty trung bình: {np.mean([r.margin_uncertainty for r in records.values()]):.4f}")
print(f"  * QBC Disagreement trung bình   : {np.mean([r.qbc_disagreement for r in records.values()]):.4f}")""")

    # =========================================================================
    # CELL 4: Chạy Đấu Trường 6 Chiến Lược (Code)
    # =========================================================================
    add_code("""# 3. Đấu Trường Tuyển Chọn 6 Chiến Lược Đối Đầu (K = 250 ảnh)
tournament_runner = ActiveMiningTournamentRunner(
    k_selection=250,
    device=device,
    cache_file=cache_path,
)
tournament_summary, winner_result = tournament_runner.run_tournament()
winner_name = tournament_summary["winner_strategy"]
winner_images = winner_result.selected_images

print(f"🏆 QUÁN QUÂN ĐẤU TRƯỜNG: {winner_name} (Điểm CSS = {tournament_summary['winner_css']})")""")

    # =========================================================================
    # CELL 5: Hiển Thị Leaderboard (Code)
    # =========================================================================
    add_code("""# 4. Hiển thị Bảng Xếp Hạng Định Lượng Leaderboard CSS
from IPython.display import Markdown, display

leaderboard_md_path = results_p5_dir / "tables" / "selection_tournament_leaderboard.md"
if leaderboard_md_path.exists():
    with open(leaderboard_md_path, "r", encoding="utf-8") as f:
        display(Markdown(f.read()))
else:
    print("Chưa tìm thấy file leaderboard markdown!")""")

    # =========================================================================
    # CELL 6: Showcase Biểu Đồ 1 & 2 (Radar & PCA Dispersion) (Code)
    # =========================================================================
    add_code("""# 5. Showcase Biểu Đồ 1 & 2: Radar Chart & PCA 32D Dispersion (300 DPI)
fig_dir = results_p5_dir / "figures"
radar_p = fig_dir / "fig_phase05_01_tournament_radar.png"
pca_p = fig_dir / "fig_phase05_02_pca_dispersion.png"

if radar_p.exists():
    display(Image.open(radar_p))
if pca_p.exists():
    display(Image.open(pca_p))""")

    # =========================================================================
    # CELL 7: Quality-Gated Policy Analysis (Code)
    # =========================================================================
    add_code("""# 6. Thẩm Định Năng Lực Mô Hình & Chính Sách Quality-Gated Pre-Labeling
qg_analyzer = QualityGatedAnalyzer(quality_threshold=0.75)
qg_report = qg_analyzer.analyze(records, winner_images)

print("🛡️ KẾT QUẢ THẨM ĐỊNH NĂNG LỰC THEO QUALITY-GATED POLICY:")
print(f"  * Ngưỡng kích hoạt an toàn (Quality Gate)   : {qg_report.quality_threshold*100:.0f}%")
print(f"  * Các lớp ĐỦ ĐIỀU KIỆN mở Pre-labeling     : {qg_report.active_prelabel_classes}")
print(f"  * Các lớp BỊ KHÓA (Manual Oracle Only)     : {qg_report.gated_classes}")
print(f"  * Chỉ số Giảm Thiểu Công Sức Dán Nhãn (AER): {qg_report.annotation_effort_reduction_pct:.2f}%")
print(f"  * Thời gian tiết kiệm lý thuyết            : {qg_report.time_saved_hours:.1f} giờ (~{qg_report.theoretical_cost_saved_usd:.1f} USD)")""")

    # =========================================================================
    # CELL 8: Showcase Biểu Đồ 3, 4, 5 (Code)
    # =========================================================================
    add_code("""# 7. Showcase Biểu Đồ 3, 4 & 5: Ma Trận Chất Lượng, Lát Cắt Khó & Lưới Ảnh Mẫu
qg_fig_p = fig_dir / "fig_phase05_03_quality_gated_policy_matrix.png"
hs_fig_p = fig_dir / "fig_phase05_04_hard_slices_and_rare_objects.png"
gallery_p = fig_dir / "fig_phase05_05_visual_gallery_mined_samples.png"

if qg_fig_p.exists():
    display(Image.open(qg_fig_p))
if hs_fig_p.exists():
    display(Image.open(hs_fig_p))
if gallery_p.exists():
    display(Image.open(gallery_p))""")

    # =========================================================================
    # CELL 9: Đóng Gói Dataset V1 (Code)
    # =========================================================================
    add_code("""# 8. Vật Liệu Hóa Dataset V1 (750 Ảnh) Với 100% Nhãn Vàng BDD10K
splits_dir = get_splits_dir()
manifest_path = splits_dir / "manifest_v1.yaml"

if manifest_path.exists():
    with open(manifest_path, "r", encoding="utf-8") as f:
        print("📄 MANIFEST KHAI SINH DATASET V1:")
        print(f.read())

print("📦 KIỂM TRA TOÀN VẸN CẤU TRÚC SPLITS:")
print(f"  * seed_v0/images     : {len(list((splits_dir / 'seed_v0' / 'images').glob('*.jpg')))} ảnh")
print(f"  * seed_v1/images     : {len(list((splits_dir / 'seed_v1' / 'images').glob('*.jpg')))} ảnh")
print(f"  * train_v1/images    : {len(list((splits_dir / 'train_v1' / 'images').glob('*.jpg')))} ảnh")
print(f"  * train_v1/labels    : {len(list((splits_dir / 'train_v1' / 'labels').glob('*.txt')))} nhãn vàng BDD10K")
print(f"  * unlabeled_pool     : {len(list((splits_dir / 'unlabeled_pool' / 'images').glob('*.jpg')))} ảnh còn lại")""")

    # =========================================================================
    # CELL 10: Kết Luận Học Thuật (Markdown)
    # =========================================================================
    add_md("""## 2. Kết Luận Học Thuật Phase 05 & Bàn Giao Dữ Liệu Cho Phase 06

### 📌 Các Đóng Góp Cốt Lõi:
1. **Chiến Lược Super-Fusion Vô Địch ($CSS = 84.66$)**:
   - Vượt trội so với Random Sampling trên mọi khía cạnh: Săn lùng thành công **655 vật thể thuộc 6 lớp điểm liệt** (gấp 3.4 lần Random), khóa chuẩn quota 41% Ban đêm và ưu tiên ảnh camera rung lắc.
2. **Quality-Gated Pre-Labeling Policy Chuẩn Thực Chiến**:
   - Thừa nhận thực tế khách quan: Ở giai đoạn đầu, các lớp hiếm (`train, rider, motor, bike`) có năng lực thấp $\\to$ **Khóa chặt Pre-labeling để chống box rác làm phiền con người**.
   - Chỉ mở Pre-labeling cho lớp đạt chuẩn (`car` Precision > 75%) $\to$ Tiết kiệm **26.67% công sức gán nhãn thực tế (AER)**.
3. **Bảo Đảm Độ Chắc Chắn Tuyệt Đối (Zero Label Noise)**:
   - Toàn bộ 250 ảnh đóng gói vào `seed_v1` và `train_v1` được **cấp 100% Nhãn Vàng Ground-Truth từ BDD10K gốc**, loại bỏ hoàn toàn rủi ro Confirmation Bias hay Error Drift.
4. **Bàn Giao Trọn Vẹn Sang Phase 06**:
   - Gói dữ liệu `train_v1/` (750 ảnh) và `train_random_v1/` (750 ảnh đối chứng) đã sẵn sàng để Phase 06 tiến hành huấn luyện mô hình thế hệ mới **YOLO11n V1** và thi chung khảo trên **Frozen Gold Test** (1.400 ảnh).""")

    project_root = find_project_dir()
    notebook_path = project_root / "notebooks" / "05_active_mining_tournament_and_hitl.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"✓ Đã sinh thành công Jupyter Notebook Phase 05: {notebook_path}")
    return notebook_path


if __name__ == "__main__":
    create_phase_05_notebook()
