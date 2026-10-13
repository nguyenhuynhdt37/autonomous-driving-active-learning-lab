"""
precompute_phase_05_cache.py - Sinh trước Feature Cache cho Phase 05
====================================================================
Quét toàn bộ 4.394 ảnh unlabeled pool bằng YOLO11n V0 và RT-DETR V0,
tính PCA 32D và lưu vào results/phase_05/unlabeled_pool_feature_cache.json.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "paper_v4_10k"))

from src.active_mining import PoolFeatureExtractor
from src.utils import set_seed, get_device
from src.train_engine import clean_bdd10k_caches


def main():
    set_seed(42)
    device = get_device()
    print(f"Bắt đầu trích xuất cache trên thiết bị: {device}")
    
    extractor = PoolFeatureExtractor(
        device=device,
        batch_size=16,
    )
    t0 = time.time()
    records = extractor.compute_and_cache_features()
    clean_bdd10k_caches()
    
    print(f"✓ Đã hoàn tất {len(records)} bản ghi trong {time.time() - t0:.2f}s!")


if __name__ == "__main__":
    main()
