"""
metrics.py - Hệ Thống Chỉ Số Đo Đạc Khoa Học (V4.0 Paper-Ready)
===============================================================
Cung cấp các phép đo định lượng chuẩn hóa:
  1. compute_dedup_precision_recall_f1: Đánh giá độ chuẩn xác phát hiện cặp trùng lặp.
  2. compute_vehicle_retention: Đánh giá khả năng bảo toàn đối tượng phương tiện & Small Object.
  3. compute_speed_profile: Đo thông lượng FPS, Latency và RAM qua nhiều lượt chạy lặp lại.
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

import numpy as np
import pandas as pd


def normalize_pair(img1: str, img2: str) -> Tuple[str, str]:
    """Chuẩn hóa cặp tên ảnh theo thứ tự bảng chữ cái để đảm bảo tính bất biến giao hoán."""
    name1 = Path(img1).name
    name2 = Path(img2).name
    return (name1, name2) if name1 < name2 else (name2, name1)


def compute_dedup_precision_recall_f1(
    predicted_pairs: Set[Tuple[str, str]],
    ground_truth_pairs: Set[Tuple[str, str]]
) -> Dict[str, float]:
    """
    Tính Precision, Recall và F1-Score cho bài toán phát hiện cặp ảnh trùng lặp.
    
    Đầu ra dict: precision, recall, f1, tp, fp, fn, total_predicted, total_gt.
    """
    norm_pred = {normalize_pair(p[0], p[1]) for p in predicted_pairs if p[0] != p[1]}
    norm_gt = {normalize_pair(p[0], p[1]) for p in ground_truth_pairs if p[0] != p[1]}

    tp = len(norm_pred.intersection(norm_gt))
    fp = len(norm_pred.difference(norm_gt))
    fn = len(norm_gt.difference(norm_pred))

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision * 100.0, 2),
        "recall": round(recall * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_predicted": len(norm_pred),
        "total_gt": len(norm_gt)
    }


def compute_vehicle_retention(
    kept_image_paths: List[str],
    metadata_db: Dict[str, Any],
    vehicle_classes: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Tính số lượng bounding boxes phương tiện và vật thể nhỏ được bảo toàn.
    """
    if vehicle_classes is None:
        vehicle_classes = {"car", "truck", "bus", "motor", "bike"}

    total_vehicles = 0
    small_vehicles = 0
    occluded_vehicles = 0
    by_class_counts: Dict[str, int] = {cls: 0 for cls in vehicle_classes}

    img_w, img_h = 1280.0, 720.0
    small_thresh_area = 0.01 * (img_w * img_h)

    for img_path in kept_image_paths:
        stem = Path(img_path).stem
        name = Path(img_path).name

        entry = metadata_db.get(stem) or metadata_db.get(name) or metadata_db.get(img_path)
        if not entry or "objects" not in entry:
            continue

        for obj in entry["objects"]:
            cat = obj.get("category", "").lower()
            if cat in vehicle_classes:
                total_vehicles += 1
                by_class_counts[cat] = by_class_counts.get(cat, 0) + 1

                box = obj.get("box2d", {})
                w = abs(box.get("x2", 0) - box.get("x1", 0))
                h = abs(box.get("y2", 0) - box.get("y1", 0))
                area = w * h

                if area < small_thresh_area or (w < 32 and h < 32):
                    small_vehicles += 1

                if obj.get("occluded", False):
                    occluded_vehicles += 1

    return {
        "total_vehicles": total_vehicles,
        "small_vehicles": small_vehicles,
        "occluded_vehicles": occluded_vehicles,
        "by_class": by_class_counts,
        "num_kept_images": len(kept_image_paths)
    }


def compute_speed_profile(
    method,
    image_paths: List[str],
    n_runs: int = 3
) -> Dict[str, float]:
    """
    Đo đạc chính xác tốc độ thông lượng (FPS), độ trễ (Latency ms) và RAM tiêu thụ.
    """
    if not image_paths:
        return {"fps_median": 0.0, "fps_std": 0.0, "latency_ms": 0.0, "peak_ram_mb": 0.0}

    num_images = len(image_paths)
    fps_records: List[float] = []

    # Warm-up
    warmup_samples = image_paths[:min(5, num_images)]
    for p in warmup_samples:
        _ = method.compute_signature(p)

    tracemalloc.start()
    for _ in range(n_runs):
        start_time = time.perf_counter()
        for p in image_paths:
            _ = method.compute_signature(p)
        elapsed = time.perf_counter() - start_time

        if elapsed > 0:
            fps_records.append(num_images / elapsed)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak / (1024 * 1024)

    fps_median = float(np.median(fps_records))
    fps_std = float(np.std(fps_records))
    latency_ms = (1000.0 / fps_median) if fps_median > 0 else 0.0

    return {
        "fps_median": round(fps_median, 1),
        "fps_std": round(fps_std, 2),
        "latency_ms": round(latency_ms, 2),
        "peak_ram_mb": round(peak_ram_mb, 2),
        "sample_count": num_images
    }
