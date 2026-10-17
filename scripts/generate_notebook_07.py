"""
generate_notebook_07.py - Script tái cấu trúc và chuẩn hóa hoàn toàn Notebook 07
================================================================================
Thiết lập chính xác 13 cells khoa học mạch lạc, tuân thủ 100% các nguyên tắc:
1. Zero Fake Metrics (không số liệu giả mạo)
2. Ngân sách chuẩn: 6.994 Unique Pool, 1.400 Test, 700 Val, 4.894 Bank, 979 Seed V0 (20%), 3.915 Unlabeled Pool, K=250.
3. Kế thừa trọn vẹn Phase 01 -> 06.
4. Trọn bộ 6 metrics: mAP50, mAP95, Precision, Recall, F1, CF-Score.
5. Đường dẫn splits: data/splits/autonomous_loop/ (symlink 0 KB đĩa).
"""

import json
from pathlib import Path


def build_notebook_07() -> dict:
    cells = []

    # --------------------------------------------------------------------------
    # CELL 0: MARKDOWN - TUYÊN NGÔN NGHIÊN CỨU & KIẾN TRÚC TOÀN TRÌNH
    # --------------------------------------------------------------------------
    c0 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🏛️ BDD10K V4.0 — PHASE 07: PIPELINE TỰ ĐỘNG HÓA TOÀN TRÌNH\n",
            "## (END-TO-END AUTONOMOUS ACTIVE LOOP TOURNAMENT WITH CONTINUOUS FINE-TUNING & 5-TIER STOPPING GUARD)\n",
            "\n",
            "> **Dự án**: BDD10K Active Learning Lab: Hệ Thống Tự Động Hóa Vòng Lặp Dữ Liệu Học Tập Chủ Động (Human-in-the-Loop)  \n",
            "> **Tác giả**: Nhóm Nghiên Cứu BDD10K Autonomous Driving Active Learning Lab (Paper V4.0)  \n",
            "> **Workspace Độc Lập**: `notebooks/08_autonomous_pipeline_loop/`  \n",
            "> **Thư Mục Splits Chuẩn Hóa**: `data/splits/autonomous_loop/` (Symlink Materialization 0 KB đĩa)  \n",
            "> **Bảo Tồn Dữ Liệu**: `data/bdd10k/` Read-Only 100% bất biến, kế thừa sạch từ `data/processed/unique_pool` (6.994 ảnh).  \n",
            "> **Kỷ Luật Khoa Học Tối Cao**: **ZERO FAKE METRICS** — Tuyệt đối không tự bịa đặt benchmark hoặc sinh dummy checkpoints.\n",
            "\n",
            "---\n",
            "\n",
            "### 📋 Tuyên Ngôn Nghiên Cứu & Câu Hỏi Khoa Học Cốt Lõi:\n",
            "Làm thế nào để tích hợp toàn bộ các công nghệ Quán quân xuất sắc nhất từ **Phase 01 ➔ 06** thành một **Hệ Thống Tự Động Hóa Toàn Trình (Autonomous Active Loop Orchestrator)** độc lập, có khả năng tự vận hành vòng lặp tuần hoàn, tối ưu ngân sách gán nhãn mà vẫn bảo đảm độ chuẩn xác nhận diện vượt trội?\n",
            "\n",
            "1. **FastDedup** (Phase 01): Kế thừa trực tiếp Unique Pool **6.994 ảnh sạch** (loại bỏ video sequence trùng lặp, pHash Hamming Distance < 3).\n",
            "2. **Multimodal Profiler** (Phase 02): Kế thừa trực tiếp 5 trục quang học định lượng (`data_profiling_manifest.json`) cho từng ảnh.\n",
            "3. **Phân chia 4 tập Độc Lập** (Phase 03): Group-Aware Video Isolation (Zero Video Overlap) kết hợp Hybrid Mix 80/20:\n",
            "   - `fixed_test`: **1.400 ảnh (20.0%)** Frozen Gold Test đóng băng vĩnh viễn (Zero Data Leakage).\n",
            "   - `fixed_val`: **700 ảnh (10.0%)** Fixed Validation đóng băng dùng chung cho Early Stopping.\n",
            "   - `initial_unlabeled_bank`: **4.894 ảnh (70.0%)** ($6.994 - 1.400 - 700$).\n",
            "   - `seed_v0`: **979 ảnh (chính xác 20.0% của Unlabeled Bank)** làm tập mầm huấn luyện Model V0.\n",
            "   - `unlabeled_pool`: **3.915 ảnh** dành riêng cho Active Mining qua các vòng lặp.\n",
            "   - Ngân sách tuyển chọn mỗi vòng: **$K = 250$ ảnh** (Tập train tích lũy: V0=979, V1=1.229, V2=1.479, V3=1.729, V4=1.979, V5=2.229 ảnh).\n",
            "4. **Mô Hình Khai Thác Cfg-3 Quán Quân** (Phase 04): YOLO11n (AdamW + Cosine LR + Backbone Freezing cho V0).\n",
            "5. **Động Cơ Tuyển Chọn Quán Quân C4 Super-Fusion** (Phase 05):\n",
            "   - Margin Uncertainty $U(x)$ từ Model inference vòng trước.\n",
            "   - Vector điều phối phạt thích ứng Phase 04 Cfg-3: $W_i = \\max(1.0, 0.40 / \\max(\\text{mAP}_{50, i}, 0.02))$.\n",
            "   - Submodular Facility Location (Lazy Greedy) trên không gian đặc trưng **PCA 32D** bảo đảm độ dàn đều không gian.\n",
            "   - Lát cắt thử thách môi trường $G(x)$ (Đêm, Mờ rung, Che khuất, Sương mù, Dày đặc).\n",
            "6. **Quality-Gated Oracle Ground-Truth** (Phase 05 & 06): Tra cứu $O(1)$ nhãn vàng sạch từ `bdd10k_metadata.json`, triệt tiêu 100% Label Noise, khóa Pre-labeling nếu $F_1$ lớp hiếm < 75%.\n",
            "7. **Trọn Bộ 6 Metrics Đánh Giá Toàn Diện**:\n",
            "   - $\\text{mAP@50}$ (Macro & Per-Class 10 lớp)\n",
            "   - $\\text{mAP@50:95}$ (map95 COCO)\n",
            "   - $\\text{Precision}$ (P)\n",
            "   - $\\text{Recall}$ (R)\n",
            "   - $\\text{F1-Score}$: $F_1 = \\frac{2 \\cdot P \\cdot R}{P + R + 1e-7}$\n",
            "   - $\\text{CF-Score}$ (Composite Fitness Score): $\\text{CF-Score} = 0.40 \\cdot \\text{mAP}_{50} + 0.30 \\cdot \\text{mAP}_{50:95} + 0.30 \\cdot \\text{Recall}$\n",
            "8. **Bộ Rào Chắn Dừng 5 Tầng & Vòng Xác Nhận** (Confirmation Round $k+1$).\n"
        ]
    }
    cells.append(c0)

    # --------------------------------------------------------------------------
    # CELL 1: CODE - KHỞI TẠO MÔI TRƯỜNG & THIẾT LẬP WORKSPACE
    # --------------------------------------------------------------------------
    c1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Khởi tạo môi trường, auto-detect workspace và cấu hình MPS device\n",
            "import os\n",
            "import sys\n",
            "import json\n",
            "import time\n",
            "from pathlib import Path\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "import torch\n",
            "\n",
            "# Tự động định vị thư mục gốc dự án linh hoạt\n",
            "curr = Path.cwd().resolve()\n",
            "candidates = [curr, curr.parent, curr.parent.parent, curr.parent.parent.parent]\n",
            "project_root = None\n",
            "for c in candidates:\n",
            "    if (c / \"src\").exists() and (c / \"data\").exists():\n",
            "        project_root = c.resolve()\n",
            "        break\n",
            "\n",
            "if project_root is None:\n",
            "    raise RuntimeError(\"Không tìm thấy thư mục gốc dự án chứa src/ và data/!\")\n",
            "\n",
            "if str(project_root) not in sys.path:\n",
            "    sys.path.insert(0, str(project_root))\n",
            "\n",
            "from src.paths import (\n",
            "    find_project_dir,\n",
            "    get_phase07_workspace_dir,\n",
            "    get_autonomous_splits_dir,\n",
            "    get_bdd10k_dir,\n",
            "    get_data_dir,\n",
            ")\n",
            "from src.autonomous_loop import (\n",
            "    AutonomousLoopConfig,\n",
            "    AutonomousLoopOrchestrator,\n",
            "    StoppingGuard,\n",
            "    TournamentStoppingStatus,\n",
            "    IterationMetrics,\n",
            "    clean_all_caches,\n",
            "    clean_bdd10k_caches,\n",
            "    compute_phase04_class_deficit_weights,\n",
            ")\n",
            "from src.utils import get_device, set_seed\n",
            "\n",
            "set_seed(42)\n",
            "device = get_device()\n",
            "workspace_dir = get_phase07_workspace_dir()\n",
            "splits_dir = get_autonomous_splits_dir()\n",
            "\n",
            "print(f\"📁 Project Root      : {project_root}\")\n",
            "print(f\"📁 Workspace Phase 07: {workspace_dir}\")\n",
            "print(f\"📁 Splits Directory  : {splits_dir}\")\n",
            "print(f\"💻 Computing Device  : {device.upper()}\")\n",
            "if torch.backends.mps.is_available():\n",
            "    print(\"🚀 Apple Silicon MPS Hardware Acceleration: ENABLED\")\n"
        ]
    }
    cells.append(c1)

    # --------------------------------------------------------------------------
    # CELL 2: MARKDOWN - SƠ ĐỒ LUỒNG DỮ LIỆU & QUẢN LÝ NGÂN SÁCH
    # --------------------------------------------------------------------------
    c2 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. Sơ Đồ Luồng Dữ Liệu 12 Bước Toàn Trình & Phân Bổ Ngân Sách Chuẩn Mực\n",
            "\n",
            "Hệ thống tuân thủ cấu trúc phân bổ dữ liệu chuẩn mực cho Active Learning:\n",
            "- **Tổng Unique Pool**: **6.994 ảnh (100.0%)** sạch từ Phase 01 FastDedup.\n",
            "- **Frozen Gold Test**: **1.400 ảnh (20.0%)** đóng băng vĩnh viễn (Group-Aware Video Isolation Zero Leakage).\n",
            "- **Fixed Validation**: **700 ảnh (10.0%)** đóng băng dùng chung cho Early Stopping.\n",
            "- **Initial Unlabeled Bank**: **4.894 ảnh (70.0%)** ngân sách còn lại sau Val/Test ($6.994 - 1.400 - 700$).\n",
            "- **Seed V0**: **979 ảnh (chính xác 20.0% của Unlabeled Bank)** bốc bằng Hybrid Mix 80/20 phân tầng 4 lát cắt khó.\n",
            "- **Unlabeled Mining Pool còn lại**: **3.915 ảnh** dành cho Active Mining qua các vòng lặp.\n",
            "- **Ngân sách tuyển chọn mỗi vòng**: **$K = 250$ ảnh** ($V_0 = 979 \\to V_1 = 1.229 \\to V_2 = 1.479 \\to V_3 = 1.729 \\to V_4 = 1.979 \\to V_5 = 2.229$ ảnh).\n"
        ]
    }
    cells.append(c2)

    # --------------------------------------------------------------------------
    # CELL 3: CODE - NẠP UNIQUE POOL 6.994 ẢNH & PROFILING MANIFEST
    # --------------------------------------------------------------------------
    c3 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Kế thừa trực tiếp Unique Pool 6.994 ảnh và Profiling Manifest Phase 02 (không chạy lại)\n",
            "up_dir = get_data_dir() / \"processed\" / \"unique_pool\"\n",
            "clean_images = sorted([p.name for p in up_dir.glob(\"*.jpg\")])\n",
            "print(f\"📦 Unique Pool Sạch Kế Thừa: {len(clean_images):,} ảnh (.jpg symlinks 0 KB)\")\n",
            "\n",
            "prof_manifest_path = get_data_dir() / \"processed\" / \"reports\" / \"data_profiling_manifest.json\"\n",
            "with open(prof_manifest_path, \"r\", encoding=\"utf-8\") as f:\n",
            "    profiling_manifest = json.load(f)\n",
            "print(f\"📖 Data Profiling Manifest: {len(profiling_manifest):,} bản ghi 5 trục quang học định lượng\")\n",
            "\n",
            "# Khảo sát phân bổ thuộc tính trong 6.994 ảnh sạch\n",
            "n_night = sum(1 for v in profiling_manifest.values() if v.get(\"is_night\"))\n",
            "n_small = sum(1 for v in profiling_manifest.values() if v.get(\"is_small_object_slice\"))\n",
            "n_occ = sum(1 for v in profiling_manifest.values() if v.get(\"is_high_occlusion\"))\n",
            "n_blur = sum(1 for v in profiling_manifest.values() if v.get(\"is_blur\"))\n",
            "print(f\"   • Ảnh Ban Đêm (is_night)      : {n_night:,} ảnh ({n_night/len(clean_images)*100:.1f}%)\")\n",
            "print(f\"   • Ảnh Box Nhỏ (is_small_slice): {n_small:,} ảnh ({n_small/len(clean_images)*100:.1f}%)\")\n",
            "print(f\"   • Ảnh Che Khuất (is_occluded) : {n_occ:,} ảnh ({n_occ/len(clean_images)*100:.1f}%)\")\n",
            "print(f\"   • Ảnh Mờ Rung (is_blur)       : {n_blur:,} ảnh ({n_blur/len(clean_images)*100:.1f}%)\")\n"
        ]
    }
    cells.append(c3)

    # --------------------------------------------------------------------------
    # CELL 4: CODE - PHÂN TÁCH 4 TẬP ĐỘC LẬP BẰNG SYMLINKS 0 KB
    # --------------------------------------------------------------------------
    c4 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Phân tách 4 tập độc lập với Group-Aware Video Isolation và Symlinks 0 KB đĩa\n",
            "config = AutonomousLoopConfig(\n",
            "    workspace_dir=workspace_dir,\n",
            "    splits_dir=splits_dir,\n",
            "    initial_unlabeled_ratio=0.20,\n",
            "    batch_size_k=250,\n",
            "    max_iterations=5,\n",
            "    device=device,\n",
            "    seed=42,\n",
            ")\n",
            "orchestrator = AutonomousLoopOrchestrator(config)\n",
            "orchestrator.prepare_data_and_profiling()\n",
            "orchestrator.partition_and_materialize_splits()\n",
            "\n",
            "# Kiểm toán Zero Video Overlap (cách ly rò rỉ video sequence)\n",
            "test_vids = {name.split(\"-\")[0] for name in orchestrator.fixed_test_images}\n",
            "val_vids = {name.split(\"-\")[0] for name in orchestrator.fixed_val_images}\n",
            "bank_vids = {name.split(\"-\")[0] for name in orchestrator.initial_unlabeled_bank_images}\n",
            "\n",
            "assert len(test_vids & val_vids) == 0, \"Lỗi rò rỉ video giữa test và val!\"\n",
            "assert len(test_vids & bank_vids) == 0, \"Lỗi rò rỉ video giữa test và bank!\"\n",
            "assert len(val_vids & bank_vids) == 0, \"Lỗi rò rỉ video giữa val và bank!\"\n",
            "print(\"\\n🛡️ KIỂM TOÁN AN TOÀN ZERO VIDEO OVERLAP: PASSED (100% CÁCH LY TUYỆT ĐỐI)\")\n",
            "print(f\"  • Video IDs Fixed Test : {len(test_vids):,} video chuỗi\")\n",
            "print(f\"  • Video IDs Fixed Val  : {len(val_vids):,} video chuỗi\")\n",
            "print(f\"  • Video IDs Train Bank : {len(bank_vids):,} video chuỗi\")\n"
        ]
    }
    cells.append(c4)

    # --------------------------------------------------------------------------
    # CELL 5: MARKDOWN - THUẬT TOÁN QUÁN QUÂN C4 SUPER-FUSION & BỘ 6 METRICS
    # --------------------------------------------------------------------------
    c5 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. Động Cơ Tuyển Chọn Quán Quân C4 Super-Fusion & Bộ 6 Metrics Chuẩn APA/IEEE\n",
            "\n",
            "#### A. Công Thức Tuyển Chọn Đa Yếu Tố C4 Super-Fusion:\n",
            "Thuật toán kết hợp 4 thành phần định lượng giúp mô hình giải quyết triệt để các ca khó và các lớp hiếm:\n",
            "$$S(x_i) = U(x_i) \\cdot \\left(\\sum_{o \\in \\text{preds}(x_i)} W_i(\\text{class}_o) \\cdot G_w(o)\\right) \\cdot E(x_i) \\cdot B(x_i)$$\n",
            "Trong đó:\n",
            "1. **Margin Uncertainty $U(x_i)$**: Đo độ bối rối của mô hình trên vật thể bất định nhất.\n",
            "2. **Vector Phạt Thích Ứng $W_i$ (Phase 04 Cfg-3)**: Phạt nặng các lớp có mAP thấp:\n",
            "   $$W_i = \\max\\left(1.0, \\, \\frac{0.40}{\\max(\\text{mAP}_{50, i}, \\, 0.02)}\\right)$$\n",
            "3. **Hình học $G_w(o)$**: Trọng số ưu tiên vật thể nhỏ ($< 32\\times 32$px), che khuất và tràn viền.\n",
            "4. **Môi trường & Rung lắc $E(x_i), B(x_i)$**: Boost mạnh cho ảnh đêm, sương mù, motion blur.\n",
            "5. **Submodular Facility Location trên PCA 32D**: Dàn đều bán kính phủ trong không gian đặc trưng, tránh co cụm vào một khung cảnh lặp lại.\n",
            "\n",
            "#### B. Quality-Gated Oracle Ground-Truth (Read-Only $O(1)$):\n",
            "- Tra cứu nhãn vàng sạch từ `data/bdd10k/bdd10k_metadata.json` ở chế độ Read-Only $O(1)$.\n",
            "- **Khóa Pre-labeling tự động nếu $F_1$ các lớp hiếm < 75%** $\\implies$ Triệt tiêu 100% rủi ro Label Noise và Error Drift.\n",
            "\n",
            "#### C. Bộ 6 Metrics Đánh Giá Toàn Diện:\n",
            "1. **`mAP@50`**: Độ chuẩn xác nhận diện vật thể tại ngưỡng IoU 0.50.\n",
            "2. **`mAP@50:95` (map95)**: Độ chuẩn xác vị trí nghiêm ngặt theo chuẩn COCO (IoU 0.50 ➔ 0.95).\n",
            "3. **`Precision`**: Tỷ lệ phát hiện đúng trên tổng số hộp dự đoán.\n",
            "4. **`Recall`**: Khả năng bao phủ, bắt trọn vật thể (đặc biệt các lớp hiếm như train, rider, motor).\n",
            "5. **`F1-Score`**: Trung bình điều hòa cân bằng: $F_1 = \\frac{2 \\cdot P \\cdot R}{P + R + 1e-7}$.\n",
            "6. **`CF-Score` (Composite Fitness Score)**: Chỉ số sức mạnh tổng hợp:\n",
            "   $$\\text{CF-Score} = 0.40 \\cdot \\text{mAP}_{50} + 0.30 \\cdot \\text{mAP}_{50:95} + 0.30 \\cdot \\text{Recall}$$\n"
        ]
    }
    cells.append(c5)

    # --------------------------------------------------------------------------
    # CELL 6: CODE - CẤU HÌNH VÀ KHỞI TẠO ORCHESTRATOR
    # --------------------------------------------------------------------------
    c6 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Khởi tạo AutonomousLoopOrchestrator với cấu hình tham số minh bạch\n",
            "print(\"⚙️ Khởi tạo AutonomousLoopOrchestrator...\")\n",
            "print(f\"   • Ngân sách Seed V0       : {len(orchestrator.seed_v0_images):,} ảnh (20.0% Bank 4.894 ảnh)\")\n",
            "print(f\"   • Unlabeled Pool khả dụng : {len(orchestrator.unlabeled_pool_images):,} ảnh\")\n",
            "print(f\"   • Batch Size mỗi vòng K   : {config.batch_size_k} ảnh\")\n",
            "print(f\"   • Số vòng lặp tối đa      : {config.max_iterations} vòng (V1 -> V5)\")\n",
            "print(f\"   • Thiết bị tính toán      : {config.device.upper()}\")\n",
            "\n",
            "# Khởi tạo vector trọng số phạt thích ứng W_i ban đầu cho V0\n",
            "initial_w = compute_phase04_class_deficit_weights(None)\n",
            "print(\"\\n⚖️ Vector Phạt Thích Ứng Ban Đầu W_i (Phase 04 Cfg-3 Quán Quân):\")\n",
            "for cname in [\"train\", \"rider\", \"motor\", \"bike\", \"bus\", \"truck\", \"car\"]:\n",
            "    print(f\"   • {cname:13s}: {initial_w.get(cname, 1.0):.2f}x\")\n"
        ]
    }
    cells.append(c6)

    # --------------------------------------------------------------------------
    # CELL 7: CODE - THỰC THI TOÀN TRÌNH (DRY-RUN LOGIC AUDIT VS PRODUCTION TRAINING)
    # --------------------------------------------------------------------------
    c7 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Thực thi Vòng Lặp Tự Động Hóa (Hỗ trợ cờ DRY_RUN với Cam Kết ZERO FAKE METRICS)\n",
            "# - DRY_RUN = True : Chạy kiểm toán logic toàn trình (Symlinks, Video Isolation, Oracle GT, Bộ chọn mẫu, Rào chắn dừng).\n",
            "#                    Tuyệt đối không sinh điểm số mAP bịa đặt hay file checkpoint rác.\n",
            "# - DRY_RUN = False: Kích hoạt PyTorch Ultralytics huấn luyện thật trên Apple Silicon MPS, đo trực tiếp trên 1.400 ảnh fixed_test.\n",
            "DRY_RUN = True\n",
            "\n",
            "print(f\"🚀 Khởi chạy Autonomous Active Loop (DRY_RUN = {DRY_RUN})...\")\n",
            "verdict = orchestrator.run_full_pipeline(dry_run=DRY_RUN)\n",
            "\n",
            "print(\"\\n🏁 KẾT QUẢ THỰC THI HOÀN TẤT:\")\n",
            "print(f\"   • Trạng thái : [{verdict.status.value}]\")\n",
            "print(f\"   • Thông điệp : {verdict.verdict_message}\")\n"
        ]
    }
    cells.append(c7)

    # --------------------------------------------------------------------------
    # CELL 8: MARKDOWN - BÁO CÁO ĐÁNH GIÁ & BẢNG SỐ LIỆU KHOA HỌC
    # --------------------------------------------------------------------------
    c8 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4. Báo Cáo Kiểm Toán Kỹ Thuật & Bảng Đối Chuẩn Per-Class 6 Metrics\n",
            "\n",
            "Bảng phân tích toàn diện 10 lớp giao thông BDD10K với đầy đủ bộ 6 metrics:\n",
            "- `mAP@50`, `mAP@50:95`, `Precision`, `Recall`, `F1-Score`, `CF-Score`\n",
            "- Vector điều phối phạt thích ứng $W_i$.\n"
        ]
    }
    cells.append(c8)

    # --------------------------------------------------------------------------
    # CELL 9: CODE - RENDER BẢNG PHÂN BỔ & PER-CLASS 6 METRICS
    # --------------------------------------------------------------------------
    c9 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Render Bảng phân bổ ngân sách dữ liệu & Bảng kiểm toán kỹ thuật\n",
            "from IPython.display import Markdown, display\n",
            "\n",
            "audit_report_path = workspace_dir / \"results\" / \"tables\" / \"pipeline_logic_audit_report.md\"\n",
            "if audit_report_path.exists():\n",
            "    with open(audit_report_path, \"r\", encoding=\"utf-8\") as f:\n",
            "        content = f.read()\n",
            "    display(Markdown(content))\n",
            "else:\n",
            "    print(\"Chưa có file báo cáo kiểm toán kỹ thuật.\")\n",
            "\n",
            "# Hiển thị Bảng Per-Class 10 Lớp Trọn Bộ 6 Metrics Chuẩn APA/IEEE\n",
            "v0_table_file = workspace_dir / \"results\" / \"tables\" / \"per_class_v0_baseline.json\"\n",
            "if v0_table_file.exists():\n",
            "    with open(v0_table_file, \"r\", encoding=\"utf-8\") as f:\n",
            "        v0_data = json.load(f)\n",
            "    tbl = v0_data.get(\"per_class_table\", [])\n",
            "    macro_map50 = v0_data.get(\"macro_map50\", 0.0)\n",
            "    macro_map50_95 = v0_data.get(\"macro_map50_95\", 0.0)\n",
            "    macro_prec = v0_data.get(\"macro_precision\", 0.0)\n",
            "    macro_rec = v0_data.get(\"macro_recall\", 0.0)\n",
            "    macro_f1 = v0_data.get(\"macro_f1\", 0.0)\n",
            "    macro_cfscore = v0_data.get(\"macro_cfscore\", 0.0)\n",
            "    print(\"\\n📋 Bảng Per-Class 10 Lớp (Model Mầm V0) Với Trọn Bộ 6 Metrics Chuẩn APA/IEEE:\")\n",
            "    header = f\"| {'Index':^5} | {'Class Name':^13} | {'Support':^8} | {'mAP@50':^8} | {'mAP@50:95':^10} | {'Prec':^7} | {'Recall':^7} | {'F1':^7} | {'CF-Score':^9} | {'Weight':^7} |\"\n",
            "    print(header)\n",
            "    print(\"|\" + \"-\" * (len(header) - 2) + \"|\")\n",
            "    for r in tbl:\n",
            "        print(f\"| {r['class_index']:^5} | {r['class_name']:<13} | {r['support']:>8} | {r['map50']:>7.2f}% | {r['map50_95']:>9.2f}% | {r['precision']:>6.1f}% | {r['recall']:>6.1f}% | {r['f1_score']:>6.1f}% | {r['cfscore']:>8.2f} | {r['dynamic_weight']:>6.2f}x |\")\n",
            "    print(f\"\\n🏆 MACRO AVERAGE: mAP@50={macro_map50:.2f}%, map95={macro_map50_95:.2f}%, Prec={macro_prec:.2f}%, Recall={macro_rec:.2f}%, F1={macro_f1:.2f}%, CF-Score={macro_cfscore:.2f}\")\n",
            "else:\n",
            "    print(\"⚠️ Chưa tìm thấy file bảng per_class_v0_baseline.json.\")\n"
        ]
    }
    cells.append(c9)

    # --------------------------------------------------------------------------
    # CELL 10: CODE - TRỰC QUAN HÓA THÀNH PHẦN CHỌN MẪU THỰC TẾ
    # --------------------------------------------------------------------------
    c10 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Trực quan hóa thực nghiệm: Phân tích các thành phần chọn mẫu thực tế của Top-250 ảnh tuyển chọn\n",
            "with open(workspace_dir / \"results\" / \"tables\" / \"data_allocation_audit.json\", \"r\", encoding=\"utf-8\") as f:\n",
            "    audit_data = json.load(f)\n",
            "\n",
            "hard_slices = audit_data.get(\"hard_slices_in_selection\", {})\n",
            "if hard_slices:\n",
            "    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)\n",
            "    labels = [\"Night (Đêm)\", \"Blur (Mờ rung)\", \"Occlusion (Che)\", \"Small (Nhỏ)\", \"Dense (Kẹt xe)\"]\n",
            "    counts = [\n",
            "        hard_slices.get(\"is_night\", 0),\n",
            "        hard_slices.get(\"is_blur\", 0),\n",
            "        hard_slices.get(\"is_high_occlusion\", 0),\n",
            "        hard_slices.get(\"is_small_object_slice\", 0),\n",
            "        hard_slices.get(\"is_dense\", 0),\n",
            "    ]\n",
            "    colors = [\"#1A237E\", \"#E65100\", \"#37474F\", \"#00695C\", \"#880E4F\"]\n",
            "    bars = ax.bar(labels, counts, color=colors, edgecolor=\"black\", linewidth=1.0, width=0.55)\n",
            "    for bar, c in zip(bars, counts):\n",
            "        yval = bar.get_height()\n",
            "        pct = (c / audit_data.get(\"selected_batch_k\", 250)) * 100.0\n",
            "        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 3, f\"{c} ảnh\\n({pct:.1f}%)\", ha=\"center\", va=\"bottom\", fontsize=9.5, fontweight=\"bold\")\n",
            "\n",
            "    ax.set_title(f\"Phân Bố Các Lát Cắt Khó Trong Top-{audit_data.get('selected_batch_k', 250)} Ảnh Được Tuyển Chọn Bởi C4 Super-Fusion\", fontsize=11, fontweight=\"bold\", pad=12)\n",
            "    ax.set_ylabel(\"Số lượng ảnh tuyển chọn\", fontsize=10)\n",
            "    ax.set_ylim(0, max(counts) * 1.35)\n",
            "    ax.grid(axis=\"y\", linestyle=\"--\", alpha=0.5)\n",
            "    plt.tight_layout()\n",
            "    plot_save_path = workspace_dir / \"results\" / \"figures\" / \"selection_hard_slices_breakdown.png\"\n",
            "    plot_save_path.parent.mkdir(parents=True, exist_ok=True)\n",
            "    plt.savefig(plot_save_path, dpi=300)\n",
            "    plt.show()\n",
            "    print(f\"✅ Đã xuất bản biểu đồ phân tích tuyển chọn tại: {plot_save_path}\")\n"
        ]
    }
    cells.append(c10)

    # --------------------------------------------------------------------------
    # CELL 11: CODE - KIỂM TOÁN ZERO CACHE BDD10K & NIÊM PHONG MANIFEST
    # --------------------------------------------------------------------------
    c11 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 8. Kiểm toán Kỷ luật Zero Stale Cache BDD10K & Niêm phong Lineage\n",
            "cleaned_caches = clean_all_caches()\n",
            "bdd_cache_files = list(get_bdd10k_dir().rglob(\"*.cache\"))\n",
            "assert len(bdd_cache_files) == 0, f\"Phát hiện {len(bdd_cache_files)} cache rò rỉ vào data/bdd10k/!\"\n",
            "\n",
            "print(\"🔒 KIỂM TOÁN ZERO STALE CACHE BDD10K: PASSED (100% READ-ONLY BẤT BIẾN)\")\n",
            "print(f\"   • Số file cache đã dọn sạch tự động: {cleaned_caches}\")\n",
            "print(f\"   • File cache rò rỉ trong data/bdd10k/: {len(bdd_cache_files)} (Khớp tuyệt đối 0 file)\")\n",
            "\n",
            "# Niêm phong Manifest toàn trình\n",
            "manifest_file = workspace_dir / \"results\" / \"pipeline_execution_manifest.json\"\n",
            "manifest_data = {\n",
            "    \"pipeline_name\": \"Phase 07 - End-to-End Autonomous Active Loop\",\n",
            "    \"status\": verdict.status.value,\n",
            "    \"dry_run\": DRY_RUN,\n",
            "    \"unique_pool_size\": len(orchestrator.unique_images),\n",
            "    \"fixed_test_size\": len(orchestrator.fixed_test_images),\n",
            "    \"fixed_val_size\": len(orchestrator.fixed_val_images),\n",
            "    \"seed_v0_size\": len(orchestrator.seed_v0_images),\n",
            "    \"unlabeled_pool_size\": len(orchestrator.unlabeled_pool_images),\n",
            "    \"batch_k_size\": config.batch_size_k,\n",
            "    \"timestamp\": time.strftime(\"%Y-%m-%dT%H:%M:%SZ\", time.gmtime()),\n",
            "}\n",
            "with open(manifest_file, \"w\", encoding=\"utf-8\") as f:\n",
            "    json.dump(manifest_data, f, indent=2)\n",
            "print(f\"📄 Đã niêm phong execution manifest tại: {manifest_file}\")\n"
        ]
    }
    cells.append(c11)

    # --------------------------------------------------------------------------
    # CELL 12: MARKDOWN - KẾT LUẬN KHOA HỌC & BÀN LUẬN HITL
    # --------------------------------------------------------------------------
    c12 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5. Kết Luận Khoa Học & Bàn Luận Human-in-the-Loop\n",
            "\n",
            "1. **Tính Toàn Vẹn & Kỷ Luật Nghiên Cứu (Zero Fake Metrics)**:\n",
            "   - Hệ thống kiên quyết loại bỏ mọi số liệu bịa đặt và dummy checkpoints. Mọi đánh giá logic đều dựa trên dữ liệu thực tế đã được materialization qua Symlinks 0 KB.\n",
            "2. **Cơ Chế Phân Bổ Ngân Sách Chuẩn Mực**:\n",
            "   - Việc phân chia Seed V0 lấy đúng **20.0% của Unlabeled Bank ($979$ ảnh)** sau khi đã cách ly 1.400 Test và 700 Val bằng Group-Aware Video Isolation bảo đảm tính đại diện thống kê cao và triệt tiêu hoàn toàn Data Leakage.\n",
            "3. **Hiệu Lực Của Vector Phạt Thích Ứng $W_i$ (Phase 04 Cfg-3 Quán Quân)**:\n",
            "   - Công thức $W_i = \\max(1.0, 0.40 / \\max(\\text{mAP50}_i, 0.02))$ hướng động cơ Super-Fusion tập trung giải cứu các lớp điểm liệt (`train`, `rider`, `motor`), hạn chế bốc trùng lặp thừa thãi các lớp đã bão hòa (`car`).\n",
            "4. **Bộ 6 Metrics Toàn Diện**:\n",
            "   - Việc tích hợp đồng thời `mAP@50`, `mAP@50:95`, `Precision`, `Recall`, `F1-Score` và `CF-Score` đem lại góc nhìn đa chiều, phản ánh chuẩn xác chất lượng phát hiện vật thể giao thông trong thế giới thực.\n"
        ]
    }
    cells.append(c12)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5,
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb


if __name__ == "__main__":
    nb = build_notebook_07()
    out_path = Path("notebooks/08_autonomous_pipeline_loop/07_end_to_end_autonomous_active_loop.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✅ Đã tạo thành công Notebook 07 với {len(nb['cells'])} cells tại: {out_path}")
