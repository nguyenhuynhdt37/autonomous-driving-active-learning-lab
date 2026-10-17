"""
paths.py - Quản lý và định vị đường dẫn thư mục tự động (Path Discovery Helpers)
==================================================================================
Thiết kế theo chuẩn định vị thông minh của repo plant-disease-detection-ai:
Tự động tìm kiếm thư mục gốc `paper_v4_10k` từ bất kỳ vị trí thực thi nào
(Root directory, thư mục con notebooks/, terminal scripts, hay Google Colab).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def find_project_dir(start: Optional[str | Path] = None) -> Path:
    """
    Tự động tìm thư mục gốc `paper_v4_10k` chứa `src/` và `notebooks/`.
    
    Tìm kiếm ngược lên cây thư mục từ vị trí gọi hàm hoặc `start`.
    """
    search_roots = []
    if start is not None:
        start_path = Path(start).resolve()
        search_roots.extend([start_path, *start_path.parents])

    cwd = Path.cwd().resolve()
    search_roots.extend([cwd, *cwd.parents])

    seen = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)

        # Trường hợp 1: root chính là paper_v4_10k và có thư mục src/
        if root.name == "paper_v4_10k" and (root / "src").exists():
            return root

        # Trường hợp 2: root chứa thư mục con paper_v4_10k
        candidate = root / "paper_v4_10k"
        if candidate.exists() and (candidate / "src").exists():
            return candidate

        # Trường hợp 3: đang ở trong thư mục dự án chứa src và notebooks
        if (root / "src").exists() and (root / "notebooks").exists():
            return root

    # Fallback an toàn nếu không tìm thấy
    fallback = Path(__file__).resolve().parent.parent
    if (fallback / "src").exists():
        return fallback

    raise FileNotFoundError(
        "Không tìm thấy thư mục gốc `paper_v4_10k` chứa src/. "
        "Hãy chạy script hoặc notebook từ repo `paper_v4_10k`."
    )


def get_data_dir() -> Path:
    """Trả về đường dẫn tới thư mục data/."""
    d = find_project_dir() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_metadata_dir() -> Path:
    """
    Trả về đường dẫn chứa metadata chuẩn BDD10K (Read-Only).
    Quy định: Metadata nằm trực tiếp trong data/bdd10k/, không tạo thư mục metadata/ riêng lẻ.
    """
    return get_bdd10k_dir()


def get_bdd10k_metadata_file() -> Path:
    """Trả về đường dẫn file metadata gốc bdd10k_metadata.json trong data/bdd10k/."""
    return get_bdd10k_dir() / "bdd10k_metadata.json"


def get_results_dir(phase_name: Optional[str] = None) -> Path:
    """Trả về đường dẫn tới thư mục results/ hoặc thư mục phase cụ thể."""
    base = find_project_dir() / "results"
    if phase_name:
        p = base / phase_name
        p.mkdir(parents=True, exist_ok=True)
        return p
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_figures_dir(phase_name: str) -> Path:
    """Trả về đường dẫn thư mục lưu ảnh biểu đồ results/{phase_name}/figures/."""
    p = get_results_dir(phase_name) / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_tables_dir(phase_name: str) -> Path:
    """Trả về đường dẫn thư mục lưu bảng số liệu results/{phase_name}/tables/."""
    p = get_results_dir(phase_name) / "tables"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_reports_dir(phase_name: str) -> Path:
    """Trả về đường dẫn thư mục lưu báo cáo results/{phase_name}/reports/."""
    p = get_results_dir(phase_name) / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_bdd10k_dir() -> Path:
    """Trả về đường dẫn tới thư mục data/bdd10k/."""
    return get_data_dir() / "bdd10k"


def get_bdd10k_train_images_dir() -> Path:
    """Trả về đường dẫn tới tập ảnh huấn luyện chính thức BDD10K (7.000 ảnh)."""
    return get_bdd10k_dir() / "images" / "train"


def get_bdd10k_train_labels_dir() -> Path:
    """Trả về đường dẫn tới tập nhãn huấn luyện chính thức BDD10K (7.000 nhãn YOLO)."""
    return get_bdd10k_dir() / "labels" / "train"


def get_bdd10k_val_images_dir() -> Path:
    """Trả về đường dẫn tới tập ảnh kiểm định chính thức BDD10K (1.000 ảnh)."""
    return get_bdd10k_dir() / "images" / "val"


def get_bdd10k_val_labels_dir() -> Path:
    """Trả về đường dẫn tới tập nhãn kiểm định chính thức BDD10K (1.000 nhãn YOLO)."""
    return get_bdd10k_dir() / "labels" / "val"


def get_bdd10k_test_images_dir() -> Path:
    """Trả về đường dẫn tới tập ảnh kiểm thử chính thức BDD10K (2.000 ảnh)."""
    return get_bdd10k_dir() / "images" / "test"


def get_processed_data_dir() -> Path:
    """Trả về đường dẫn tới thư mục data/processed/."""
    d = get_data_dir() / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_raw_data_dir() -> Path:
    """Trả về đường dẫn tới thư mục data/raw/."""
    d = get_data_dir() / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_splits_dir() -> Path:
    """Trả về đường dẫn tới thư mục data/splits/."""
    d = get_data_dir() / "splits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_models_dir() -> Path:
    """Trả về đường dẫn tới thư mục lưu trữ models/ (paper_v4_10k/models/)."""
    d = find_project_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_phase07_workspace_dir() -> Path:
    """Trả về đường dẫn tới workspace tự quản độc lập Phase 07 (notebooks/autonomous_pipeline_loop/)."""
    project_dir = find_project_dir()
    d = project_dir / "notebooks" / "autonomous_pipeline_loop"
    if d.exists():
        return d
    cand = project_dir / "notebooks" / "08_autonomous_pipeline_loop"
    if cand.exists():
        return cand
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_autonomous_splits_dir() -> Path:
    """Trả về đường dẫn thư mục splits chuẩn hóa cho Autonomous Active Loop (data/splits/autonomous_loop/)."""
    d = get_splits_dir() / "autonomous_loop"
    d.mkdir(parents=True, exist_ok=True)
    return d


