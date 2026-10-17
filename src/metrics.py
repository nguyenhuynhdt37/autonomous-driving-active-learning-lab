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


def evaluate_candidate_pairs_against_human_ground_truth(
    predicted_pairs: Set[Tuple[str, str]],
    human_manifest: Dict[str, Any] | str | Path,
    positive_labels: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Đánh giá độ chuẩn xác phát hiện cặp trùng lặp đối chiếu trực tiếp với Human-Verified Ground Truth.
    
    Args:
        predicted_pairs: Tập các cặp ảnh dự đoán là trùng lặp.
        human_manifest: Dict hoặc đường dẫn tới file JSON manifest đã qua con người kiểm toán.
        positive_labels: Các nhãn được coi là Positive (Mặc định: {'Duplicate', 'Near-duplicate'}).
        
    Returns:
        Dict chứa precision, recall, f1, tp, fp, fn, breakdown per class.
    """
    import json
    if positive_labels is None:
        positive_labels = {"Duplicate", "Near-duplicate"}

    if isinstance(human_manifest, (str, Path)):
        with open(human_manifest, "r", encoding="utf-8") as f:
            human_manifest = json.load(f)

    candidates = human_manifest.get("candidates", [])
    candidate_lookup = {}
    gt_positives = set()
    gt_negatives = set()

    for c in candidates:
        pair_key = normalize_pair(c["img1"], c["img2"])
        candidate_lookup[pair_key] = c
        label = c.get("human_label", "Uncertain")
        if label in positive_labels:
            gt_positives.add(pair_key)
        elif label == "Not duplicate":
            gt_negatives.add(pair_key)

    norm_pred = {normalize_pair(p[0], p[1]) for p in predicted_pairs if p[0] != p[1]}
    pred_in_candidates = norm_pred.intersection(set(candidate_lookup.keys()))

    tp = len(pred_in_candidates.intersection(gt_positives))
    fp = len(pred_in_candidates.intersection(gt_negatives))
    fn = len(gt_positives.difference(pred_in_candidates))

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
        "total_predicted_in_candidates": len(pred_in_candidates),
        "total_gt_positives": len(gt_positives),
        "total_gt_negatives": len(gt_negatives)
    }


def generate_autonomous_driving_robustness_benchmark(
    sample_images: List[str | Path],
    output_dir: str | Path,
    n_positive_per_type: int = 10,
    n_negatives: int = 50,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Sinh tập kiểm định độc lập mô phỏng các hiện tượng quang học thực tế của camera xe tự hành:
      - Motion blur (xe rung lắc)
      - Exposure shift (đèn pha / phơi sáng chênh lệch)
      - JPEG compression (nén video transmission)
      - Sensor noise (nhiễu cảm biến ban đêm)
      - Mild crop & zoom (thay đổi vị trí tương đối)
    Kèm các cặp negative từ các khung hình khác nhau.
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    import io

    rng = np.random.default_rng(seed)
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    pert_dir = out_path / "perturbed_samples"
    pert_dir.mkdir(parents=True, exist_ok=True)

    available_images = [Path(p) for p in sample_images if Path(p).exists()]
    if len(available_images) < 20:
        raise ValueError(f"Cần tối thiểu 20 ảnh đầu vào hợp lệ, hiện có: {len(available_images)}")

    perturbations = ["motion_blur", "exposure_shift", "jpeg_compression", "sensor_noise", "mild_crop"]
    benchmark_pairs = []

    # 1. Sinh các cặp Positive
    chosen_pos = rng.choice(available_images, size=min(len(available_images), n_positive_per_type * len(perturbations)), replace=False)
    idx = 0
    for p_type in perturbations:
        for _ in range(n_positive_per_type):
            src_p = chosen_pos[idx]
            idx += 1
            src_name = src_p.stem

            img_cv = cv2.imread(str(src_p))
            if img_cv is None:
                continue

            h, w = img_cv.shape[:2]
            perturbed_cv = img_cv.copy()

            if p_type == "motion_blur":
                kernel_size = 9
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
                kernel = kernel / kernel_size
                perturbed_cv = cv2.filter2D(img_cv, -1, kernel)
            elif p_type == "exposure_shift":
                alpha = 1.25  # Tăng tương phản
                beta = 30     # Tăng độ sáng
                perturbed_cv = np.clip(alpha * img_cv.astype(np.float32) + beta, 0, 255).astype(np.uint8)
            elif p_type == "jpeg_compression":
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 35]
                _, encimg = cv2.imencode(".jpg", img_cv, encode_param)
                perturbed_cv = cv2.imdecode(encimg, 1)
            elif p_type == "sensor_noise":
                noise = rng.normal(0, 18, img_cv.shape).astype(np.float32)
                perturbed_cv = np.clip(img_cv.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            elif p_type == "mild_crop":
                crop_ratio = 0.90
                ch, cw = int(h * crop_ratio), int(w * crop_ratio)
                sy = (h - ch) // 2
                sx = (w - cw) // 2
                cropped = img_cv[sy:sy + ch, sx:sx + cw]
                perturbed_cv = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

            out_pert_name = f"{src_name}_pert_{p_type}.jpg"
            out_pert_path = pert_dir / out_pert_name
            cv2.imwrite(str(out_pert_path), perturbed_cv)

            benchmark_pairs.append({
                "pair_id": f"pos_{len(benchmark_pairs) + 1}",
                "img1_name": src_p.name,
                "img1_path": str(src_p),
                "img2_name": out_pert_name,
                "img2_path": str(out_pert_path),
                "label": 1,
                "pair_type": "positive",
                "perturbation": p_type
            })

    # 2. Sinh các cặp Negative
    chosen_neg_idx1 = rng.choice(len(available_images), size=n_negatives, replace=True)
    chosen_neg_idx2 = rng.choice(len(available_images), size=n_negatives, replace=True)
    for i in range(n_negatives):
        if chosen_neg_idx1[i] == chosen_neg_idx2[i]:
            chosen_neg_idx2[i] = (chosen_neg_idx2[i] + 1) % len(available_images)
        p1 = available_images[chosen_neg_idx1[i]]
        p2 = available_images[chosen_neg_idx2[i]]

        benchmark_pairs.append({
            "pair_id": f"neg_{i + 1}",
            "img1_name": p1.name,
            "img1_path": str(p1),
            "img2_name": p2.name,
            "img2_path": str(p2),
            "label": 0,
            "pair_type": "negative",
            "perturbation": "different_scene"
        })

    benchmark_manifest = {
        "benchmark_name": "Autonomous Driving Optical Robustness Benchmark",
        "total_pairs": len(benchmark_pairs),
        "total_positives": sum(1 for p in benchmark_pairs if p["label"] == 1),
        "total_negatives": sum(1 for p in benchmark_pairs if p["label"] == 0),
        "pairs": benchmark_pairs
    }

    manifest_file = out_path / "robustness_benchmark_manifest.json"
    import json
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_manifest, f, indent=2, ensure_ascii=False)

    return benchmark_manifest


def evaluate_robustness_benchmark(
    methods: Dict[str, Any],
    benchmark_manifest: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Đánh giá độc lập các phương pháp trên Autonomous Driving Robustness Benchmark.
    Tính toán ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1 và độ bền vững theo từng loại biến đổi.
    """
    pairs = benchmark_manifest.get("pairs", [])
    y_true = np.array([p["label"] for p in pairs])
    p_types = [p.get("perturbation", "") for p in pairs]

    eval_results = {}

    for m_name, m_obj in methods.items():
        # Dự đoán nhãn và trích xuất khoảng cách
        dists = []
        preds = []
        for p in pairs:
            s1 = m_obj.compute_signature(p["img1_path"])
            s2 = m_obj.compute_signature(p["img2_path"])
            d = float(m_obj.distance(s1, s2))
            dists.append(d)
            preds.append(1 if m_obj.is_duplicate(s1, s2) else 0)

        dists = np.array(dists)
        preds = np.array(preds)

        # Tính TP, FP, TN, FN
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        acc = ((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0

        # Phân rã độ bền vững theo từng biến đổi (Recall trên từng perturbation)
        breakdown = {}
        for pt in set(p_types):
            if pt == "different_scene":
                continue
            mask = np.array([pt == curr_pt and y == 1 for curr_pt, y in zip(p_types, y_true)])
            if np.sum(mask) > 0:
                retained_ratio = float(np.sum(preds[mask] == 1) / np.sum(mask))
                breakdown[pt] = round(retained_ratio * 100.0, 1)

        # Chuẩn hóa điểm similarity score = 1 - normalized_dist để tính PR/ROC curves
        max_d = np.max(dists) if np.max(dists) > 0 else 1.0
        sim_scores = 1.0 - (dists / max_d)

        eval_results[m_name] = {
            "accuracy": round(acc * 100.0, 2),
            "precision": round(precision * 100.0, 2),
            "recall": round(recall * 100.0, 2),
            "f1": round(f1 * 100.0, 2),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "breakdown": breakdown,
            "distances": dists.tolist(),
            "sim_scores": sim_scores.tolist(),
            "preds": preds.tolist()
        }

    return {"results": eval_results, "y_true": y_true.tolist(), "perturbations": p_types}


def bootstrap_metric_ci(
    y_true: np.ndarray | List[int],
    y_pred: np.ndarray | List[int],
    metric_name: str = "f1",
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Tính khoảng tin cậy 95% Bootstrap Confidence Interval cho phép đo Precision, Recall, F1 hoặc Accuracy.
    Trả về (median, ci_lower, ci_upper).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    boot_scores = []

    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n, size=n)
        yt_b = y_true[boot_idx]
        yp_b = y_pred[boot_idx]

        tp = np.sum((yp_b == 1) & (yt_b == 1))
        fp = np.sum((yp_b == 1) & (yt_b == 0))
        tn = np.sum((yp_b == 0) & (yt_b == 0))
        fn = np.sum((yp_b == 0) & (yt_b == 1))

        if metric_name == "precision":
            sc = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        elif metric_name == "recall":
            sc = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        elif metric_name == "accuracy":
            sc = ((tp + tn) / n) if n > 0 else 0.0
        else:  # f1
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            sc = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        boot_scores.append(sc * 100.0)

    median = float(np.median(boot_scores))
    ci_lower = float(np.percentile(boot_scores, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))

    return round(median, 2), round(ci_lower, 2), round(ci_upper, 2)
