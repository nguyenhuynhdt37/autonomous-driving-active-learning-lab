"""
utils.py - Các tiện ích hỗ trợ thực nghiệm tái lập và cấu hình hệ thống
========================================================================
Thiết kế theo phong cách repo plant-disease-detection-ai:
Quản lý seed ngẫu nhiên, tự động nhận diện phần cứng (MPS, CUDA, CPU).
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int = 42) -> None:
    """Cố định Random Seed để đảm bảo tính tái lập 100% của mọi thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Tự động phát hiện và trả về thiết bị tính toán tối ưu trên máy chủ."""
    if HAS_TORCH:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def format_time(seconds: float) -> str:
    """Định dạng thời gian chạy thành chuỗi dễ đọc (hh:mm:ss.ms)."""
    if seconds < 1.0:
        return f"{seconds * 1000.0:.1f}ms"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem_sec = seconds % 60
    return f"{minutes}m {rem_sec:.1f}s"
