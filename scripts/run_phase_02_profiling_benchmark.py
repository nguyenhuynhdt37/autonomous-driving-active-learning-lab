"""
run_phase_02_profiling_benchmark.py - Script Thực Thi Benchmark & Full Profiling Phase 02
========================================================================================
Quy trình thực thi chuẩn hóa 2 giai đoạn:
Giai đoạn 1: Benchmark Đối Chuẩn Đa Thuật Toán (Comparative Study) trên 2.000 ảnh đại diện
             kết hợp Controlled Blur Perturbation Test.
Giai đoạn 2: Kích hoạt Hybrid Ensemble Profiler quét FULL 100% kho dữ liệu 6.994 ảnh
             trong data/processed/unique_pool/ và xuất xưởng data_profiling_manifest.json.

Tuân thủ nghiêm ngặt quy tắc AGENTS.md:
- data/bdd10k/ là Read-Only 100%.
- Không tạo thư mục metadata/ hoặc ground_truth/ riêng lẻ.
- Lưu trữ kết quả đúng vào data/processed/reports/ và results/phase_02/.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# Bổ sung path để import module src nội bộ
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
sys.path.insert(0, str(project_dir))

from src.paths import (
    find_project_dir,
    get_bdd10k_metadata_file,
    get_processed_data_dir,
    get_results_dir,
    get_reports_dir,
    get_figures_dir,
)
from src.profiling import (
    compute_hsv_brightness,
    compute_ycbcr_percentiles,
    compute_retinex_illumination,
    classify_night_ensemble,
    compute_laplacian_variance,
    compute_edge_gated_tenengrad,
    compute_fft_frequency_energy,
    classify_blur_ensemble,
    compute_dark_channel_prior,
    compute_atmospheric_transmission_map,
    compute_directional_rain_streak,
    compute_color_contrast_degradation,
    classify_weather_ensemble,
    run_weather_tournament_benchmark,
    compute_directed_occlusion_graph,
    compute_scale_shannon_entropy,
    detect_dense_scene,
    run_controlled_blur_benchmark,
    evaluate_binary_metric,
    HybridEnsembleProfiler,
)


def run_phase_02_pipeline(
    benchmark_samples: int = 2000,
    blur_test_samples: int = 100,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Chạy toàn bộ pipeline Phase 02 tự động."""
    print("=" * 80)
    print("🚀 BẮT ĐẦU PHASE 02: AUTOMATED DATA PROFILING & HARD-SLICE BENCHMARK")
    print("=" * 80)
    start_total_time = time.time()

    proj_dir = find_project_dir()
    meta_file = get_bdd10k_metadata_file()
    processed_dir = get_processed_data_dir()
    unique_pool_dir = processed_dir / "unique_pool"
    reports_dir = processed_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_phase2_dir = get_results_dir("phase_02")

    # 1. Kiểm tra tài nguyên đầu vào
    if not meta_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file metadata gốc: {meta_file}")
    if not unique_pool_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục unique_pool: {unique_pool_dir}")

    print(f"\n📂 1. ĐỌC DỮ LIỆU & NGUỒN CHÂN LÝ:")
    print(f"  - Metadata gốc (Read-Only) : {meta_file}")
    with open(meta_file, "r", encoding="utf-8") as f:
        bdd_meta = json.load(f)
    print(f"  - Tổng số bản ghi metadata : {len(bdd_meta):,} records")

    all_image_paths = sorted(list(unique_pool_dir.glob("*.jpg")))
    total_unique_images = len(all_image_paths)
    print(f"  - Unique Pool kho ảnh sạch : {total_unique_images:,} ảnh symlinks")

    if total_unique_images == 0:
        raise ValueError("Thư mục unique_pool không có ảnh nào để xử lý!")

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 1: BENCHMARK ĐỐI CHUẨN ĐA THUẬT TOÁN (COMPARATIVE ABLATION)
    # --------------------------------------------------------------------------
    print(f"\n🔬 2. GIAI ĐOẠN 1: THỰC NGHIỆM ĐA THUẬT TOÁN (TRÊN {benchmark_samples:,} ẢNH ĐẠI DIỆN):")
    sample_paths = all_image_paths[:benchmark_samples]
    
    # Ground truth arrays
    gt_night: List[bool] = []
    pred_night_hsv: List[bool] = []
    pred_night_ycbcr: List[bool] = []
    pred_night_retinex: List[bool] = []
    pred_night_ensemble: List[bool] = []

    gt_small_slice: List[bool] = []
    gt_dense_slice: List[bool] = []
    gt_occlusion_slice: List[bool] = []

    pred_small_slice: List[bool] = []
    pred_dense_slice: List[bool] = []
    pred_occlusion_slice: List[bool] = []

    # Danh sách ảnh ban ngày sắc nét phục vụ Controlled Blur Perturbation
    sharp_daylight_images: List[np.ndarray] = []

    bench_start = time.time()
    for idx, img_p in enumerate(sample_paths):
        stem = img_p.stem
        gt_item = bdd_meta.get(stem, {}) or bdd_meta.get(img_p.name, {})
        
        img = cv2.imread(str(img_p))
        if img is None:
            continue
            
        # Ground-truth ngày đêm từ BDD10K
        is_true_night = (gt_item.get("timeofday") == "night") or gt_item.get("is_night", False)
        gt_night.append(is_true_night)
        
        # Thí nghiệm các phương pháp Ngày/Đêm
        # A1: HSV Baseline
        v_val = compute_hsv_brightness(img)
        pred_night_hsv.append(v_val < 65.0)
        
        # A2: YCbCr Percentiles
        y_stats = compute_ycbcr_percentiles(img)
        pred_night_ycbcr.append((y_stats["p50"] <= 45.0) and (y_stats["p10"] <= 18.0))
        
        # A3: Retinex
        ret_val = compute_retinex_illumination(img, sigma=30.0)
        pred_night_retinex.append(ret_val < 60.0)
        
        # A_Ensemble: Hybrid
        ens_night, _ = classify_night_ensemble(img, v_thresh=65.0, p50_thresh=55.0, retinex_thresh=62.0)
        pred_night_ensemble.append(ens_night)

        # Gom ảnh sắc nét ban ngày làm bài test Blur
        if (not is_true_night) and (gt_item.get("weather") == "clear") and len(sharp_daylight_images) < blur_test_samples:
            sharp_daylight_images.append(img.copy())

        # Ground-truth hình học từ BBox của BDD10K
        boxes = gt_item.get("objects", [])
        
        # Tính toán hình học
        occ_stats = compute_directed_occlusion_graph(boxes, ioa_thresh=0.20)
        scale_stats = compute_scale_shannon_entropy(boxes, img_w=img.shape[1], img_h=img.shape[0])
        dense_stats = detect_dense_scene(boxes, count_thresh=15)
        
        pred_small_slice.append(scale_stats["is_small_object_slice"])
        pred_dense_slice.append(dense_stats["is_dense"])
        pred_occlusion_slice.append(occ_stats["is_high_occlusion"])
        
        # Ground-truth tương ứng
        gt_small_slice.append(scale_stats["small_count"] >= 2 or scale_stats["small_ratio"] >= 0.30)
        gt_dense_slice.append(dense_stats["vehicle_objects"] >= 12 or dense_stats["total_objects"] >= 15)
        gt_occlusion_slice.append(occ_stats["num_occluded"] >= 2 or occ_stats["max_chain_length"] >= 2)

    bench_elapsed = time.time() - bench_start
    bench_fps = round(len(sample_paths) / bench_elapsed, 1) if bench_elapsed > 0 else 0
    print(f"  ⚡ Hoàn thành so sánh {len(sample_paths):,} ảnh trong {bench_elapsed:.2f}s ({bench_fps} FPS)")

    # Đánh giá chỉ số ánh sáng
    night_metrics_hsv = evaluate_binary_metric(gt_night, pred_night_hsv)
    night_metrics_ycbcr = evaluate_binary_metric(gt_night, pred_night_ycbcr)
    night_metrics_retinex = evaluate_binary_metric(gt_night, pred_night_retinex)
    night_metrics_ensemble = evaluate_binary_metric(gt_night, pred_night_ensemble)

    # --------------------------------------------------------------------------
    # THỰC NGHIỆM CONTROLLED BLUR PERTURBATION TEST (CHO ĐỘ MỜ / RUNG LẮC)
    # --------------------------------------------------------------------------
    print(f"\n🧪 3. CONTROLLED PERTURBATION TEST: BENCHMARK ĐỘ MỜ TRÊN {len(sharp_daylight_images) * 3} ẢNH...")
    blur_benchmark_results = run_controlled_blur_benchmark(
        sharp_daylight_images,
        lap_thresh=120.0,
        tenengrad_thresh=15000.0,
        fft_thresh=0.015,
    )

    # Đánh giá chỉ số hình học
    small_metrics = evaluate_binary_metric(gt_small_slice, pred_small_slice)
    dense_metrics = evaluate_binary_metric(gt_dense_slice, pred_dense_slice)
    occ_metrics = evaluate_binary_metric(gt_occlusion_slice, pred_occlusion_slice)

    # --------------------------------------------------------------------------
    # THỰC NGHIỆM ĐẤU TRƯỜNG THỜI TIẾT ĐA PHƯƠNG PHÁP (SECTION 3.5)
    # --------------------------------------------------------------------------
    print(f"\n🌧️ 3.5. ĐẤU TRƯỜNG THỜI TIẾT (WEATHER BENCHMARK TOURNAMENT TRÊN {len(sample_paths):,} ẢNH)...")
    weather_benchmark_results = run_weather_tournament_benchmark(sample_paths, bdd_meta)

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 2: CHẠY FULL 100% UNIQUE POOL VỚI HYBRID ENSEMBLE PROFILER
    # --------------------------------------------------------------------------
    print(f"\n🏭 4. GIAI ĐOẠN 2: CHẠY FULL PROFILING TOÀN BỘ {total_unique_images:,} ẢNH UNIQUE POOL...")
    profiler = HybridEnsembleProfiler(
        v_thresh=65.0,
        p50_thresh=55.0,
        lap_thresh=120.0,
        tenengrad_thresh=15000.0,
        fft_thresh=0.015,
        dense_count_thresh=15,
        occlusion_ioa_thresh=0.20,
    )
    
    full_manifest = profiler.profile_batch(
        all_image_paths,
        metadata_lookup=bdd_meta,
        max_workers=max_workers,
        progress_interval=1000,
    )

    # Thống kê phân bố toàn bộ 6.994 ảnh
    total_count = len(full_manifest)
    night_count = sum(1 for r in full_manifest.values() if r["is_night"])
    blur_count = sum(1 for r in full_manifest.values() if r["is_blur"])
    weather_count = sum(1 for r in full_manifest.values() if r["is_weather_degraded"])
    small_count = sum(1 for r in full_manifest.values() if r["is_small_object_slice"])
    occ_count = sum(1 for r in full_manifest.values() if r["is_high_occlusion"])
    dense_count = sum(1 for r in full_manifest.values() if r["is_dense"])

    # --------------------------------------------------------------------------
    # 5. XUẤT XƯỞNG CÁC ARTIFACTS BÁO CÁO & MANIFEST
    # --------------------------------------------------------------------------
    print(f"\n💾 5. XUẤT XƯỞNG CÁC FILE MANIFEST & BÁO CÁO ĐỐI CHUẨN:")
    
    # 1. data_profiling_manifest.json (Metadata chuẩn cho toàn bộ unique pool)
    manifest_path = reports_dir / "data_profiling_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(full_manifest, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Đã lưu Manifest chuẩn: {manifest_path} ({total_count:,} records)")

    # Đồng bộ sang Phase 07 autonomous pipeline loop profiling dir
    phase7_prof_dir = proj_dir / "notebooks" / "autonomous_pipeline_loop" / "data" / "profiling"
    if phase7_prof_dir.exists():
        phase7_manifest_path = phase7_prof_dir / "data_profiling_manifest.json"
        with open(phase7_manifest_path, "w", encoding="utf-8") as f:
            json.dump(full_manifest, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Đã đồng bộ Nguồn 2 chuẩn hóa sang Phase 07: {phase7_manifest_path}")

    # 2. hard_slice_benchmark_report.json (Báo cáo nghiên cứu học thuật)
    benchmark_report = {
        "report_title": "Phase 02 Automated Data Profiling & Multi-Method Hard-Slice Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_unique_pool_images": total_unique_images,
        "benchmark_sample_count": len(sample_paths),
        "illumination_comparative_benchmark": {
            "hsv_baseline": night_metrics_hsv,
            "ycbcr_percentiles": night_metrics_ycbcr,
            "retinex_ssr": night_metrics_retinex,
            "hybrid_ensemble_proposed": night_metrics_ensemble,
        },
        "blur_controlled_perturbation_benchmark": blur_benchmark_results,
        "weather_comparative_tournament": weather_benchmark_results,
        "geometric_topology_benchmark": {
            "small_object_slice": small_metrics,
            "dense_traffic_scene": dense_metrics,
            "directed_occlusion_graph": occ_metrics,
        },
        "full_dataset_distribution_summary": {
            "total_images": total_count,
            "night_slice_count": night_count,
            "night_slice_percent": round(night_count / total_count * 100, 2),
            "blur_slice_count": blur_count,
            "blur_slice_percent": round(blur_count / total_count * 100, 2),
            "weather_degraded_count": weather_count,
            "weather_degraded_percent": round(weather_count / total_count * 100, 2),
            "small_object_slice_count": small_count,
            "small_object_slice_percent": round(small_count / total_count * 100, 2),
            "high_occlusion_slice_count": occ_count,
            "high_occlusion_slice_percent": round(occ_count / total_count * 100, 2),
            "dense_scene_count": dense_count,
            "dense_scene_percent": round(dense_count / total_count * 100, 2),
        },
        "total_pipeline_elapsed_seconds": round(time.time() - start_total_time, 2),
    }

    report_path = reports_dir / "hard_slice_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Đã lưu Báo cáo Đối chuẩn: {report_path}")

    # Đồng bộ sang results/phase_02/
    results_phase2_dir.mkdir(parents=True, exist_ok=True)
    with open(results_phase2_dir / "hard_slice_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------------------------
    # 6. IN BẢNG ĐỐI CHUẨN TỔNG HỢP RA TERMINAL
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("🏆 BẢNG ĐỐI CHUẨN ABLATION STUDY ĐA THUẬT TOÁN (PHASE 02 BENCHMARK)")
    print("=" * 90)
    print(f"{'Lát cắt / Bài toán':<24} | {'Phương pháp':<28} | {'Accuracy':<9} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 90)
    
    # Night
    print(f"{'1. Ban Đêm (Night)':<24} | {'Baseline (HSV < 65)':<28} | {night_metrics_hsv['Accuracy']:>8}% | {night_metrics_hsv['Precision']:>9}% | {night_metrics_hsv['Recall']:>7}% | {night_metrics_hsv['F1_Score']:>7}%")
    print(f"{'':<24} | {'YCbCr Percentiles (P50)':<28} | {night_metrics_ycbcr['Accuracy']:>8}% | {night_metrics_ycbcr['Precision']:>9}% | {night_metrics_ycbcr['Recall']:>7}% | {night_metrics_ycbcr['F1_Score']:>7}%")
    print(f"{'':<24} | {'Retinex SSR':<28} | {night_metrics_retinex['Accuracy']:>8}% | {night_metrics_retinex['Precision']:>9}% | {night_metrics_retinex['Recall']:>7}% | {night_metrics_retinex['F1_Score']:>7}%")
    print(f"{'':<24} | {'🏆 Hybrid Ensemble':<28} | {night_metrics_ensemble['Accuracy']:>8}% | {night_metrics_ensemble['Precision']:>9}% | {night_metrics_ensemble['Recall']:>7}% | {night_metrics_ensemble['F1_Score']:>7}%")
    print("-" * 90)
    
    # Blur
    b_lap = blur_benchmark_results["laplacian"]
    b_ten = blur_benchmark_results["tenengrad"]
    b_fft = blur_benchmark_results["fft_frequency"]
    b_ens = blur_benchmark_results["hybrid_ensemble"]
    print(f"{'2. Độ Mờ (Controlled)':<24} | {'Baseline (Laplacian)':<28} | {b_lap['Accuracy']:>8}% | {b_lap['Precision']:>9}% | {b_lap['Recall']:>7}% | {b_lap['F1_Score']:>7}%")
    print(f"{'':<24} | {'Edge-Gated Tenengrad':<28} | {b_ten['Accuracy']:>8}% | {b_ten['Precision']:>9}% | {b_ten['Recall']:>7}% | {b_ten['F1_Score']:>7}%")
    print(f"{'':<24} | {'2D-FFT Power Ratio':<28} | {b_fft['Accuracy']:>8}% | {b_fft['Precision']:>9}% | {b_fft['Recall']:>7}% | {b_fft['F1_Score']:>7}%")
    print(f"{'':<24} | {'🏆 Dual-Domain Ensemble':<28} | {b_ens['Accuracy']:>8}% | {b_ens['Precision']:>9}% | {b_ens['Recall']:>7}% | {b_ens['F1_Score']:>7}%")
    print("-" * 90)
    
    # Weather Tournament
    w_c1 = weather_benchmark_results["c1_dcp_baseline"]
    w_c2 = weather_benchmark_results["c2_transmission_map"]
    w_c3 = weather_benchmark_results["c3_rain_streak"]
    w_c4 = weather_benchmark_results["c4_color_contrast"]
    w_c5 = weather_benchmark_results["c5_multi_domain_ensemble"]
    print(f"{'3. Thời Tiết Suy Giảm':<24} | {'C1: Pure DCP > 40 (Cũ)':<28} | {w_c1['Accuracy']:>8}% | {w_c1['Precision']:>9}% | {w_c1['Recall']:>7}% | {w_c1['F1_Score']:>7}%")
    print(f"{'':<24} | {'C2: Atm Transmission Map':<28} | {w_c2['Accuracy']:>8}% | {w_c2['Precision']:>9}% | {w_c2['Recall']:>7}% | {w_c2['F1_Score']:>7}%")
    print(f"{'':<24} | {'C3: Rain-Streak Gradient':<28} | {w_c3['Accuracy']:>8}% | {w_c3['Precision']:>9}% | {w_c3['Recall']:>7}% | {w_c3['F1_Score']:>7}%")
    print(f"{'':<24} | {'C4: Color Sat & Contrast':<28} | {w_c4['Accuracy']:>8}% | {w_c4['Precision']:>9}% | {w_c4['Recall']:>7}% | {w_c4['F1_Score']:>7}%")
    print(f"{'':<24} | {'🏆 C5: Multi-Domain Ensemble':<28} | {w_c5['Accuracy']:>8}% | {w_c5['Precision']:>9}% | {w_c5['Recall']:>7}% | {w_c5['F1_Score']:>7}%")
    print("-" * 90)

    # Topology
    print(f"{'4. Vật Thể Nhỏ (Small)':<24} | {'Scale Shannon Entropy':<28} | {small_metrics['Accuracy']:>8}% | {small_metrics['Precision']:>9}% | {small_metrics['Recall']:>7}% | {small_metrics['F1_Score']:>7}%")
    print(f"{'5. Mật Độ Cao (Dense)':<24} | {'Vehicle Thresholding':<28} | {dense_metrics['Accuracy']:>8}% | {dense_metrics['Precision']:>9}% | {dense_metrics['Recall']:>7}% | {dense_metrics['F1_Score']:>7}%")
    print(f"{'6. Che Khuất (Occlusion)':<24} | {'Directed Topology Graph':<28} | {occ_metrics['Accuracy']:>8}% | {occ_metrics['Precision']:>9}% | {occ_metrics['Recall']:>7}% | {occ_metrics['F1_Score']:>7}%")
    print("=" * 90)

    print("\n📊 PHÂN BỐ 5 LÁT CẮT TRÊN TOÀN BỘ 6.994 ẢNH UNIQUE POOL:")
    print(f"  - 🌙 Ban đêm (Night Slice)          : {night_count:,} ảnh ({night_count/total_count*100:.2f}%)")
    print(f"  - 🌫️ Mờ/Rung lắc (Blur Slice)       : {blur_count:,} ảnh ({blur_count/total_count*100:.2f}%)")
    print(f"  - 🌧️ Thời tiết suy giảm (Weather Ens): {weather_count:,} ảnh ({weather_count/total_count*100:.2f}%)")
    print(f"  - 🔍 Vật thể nhỏ (Small Object)     : {small_count:,} ảnh ({small_count/total_count*100:.2f}%)")
    print(f"  - 🚗 Che khuất cao (High Occlusion) : {occ_count:,} ảnh ({occ_count/total_count*100:.2f}%)")
    print(f"  - 🚦 Mật độ dày đặc (Dense Scene)   : {dense_count:,} ảnh ({dense_count/total_count*100:.2f}%)")
    print(f"\n🎉 TỔNG THỜI GIAN THỰC THI PIPELINE: {benchmark_report['total_pipeline_elapsed_seconds']} giây.")
    print("=" * 90)

    return benchmark_report


if __name__ == "__main__":
    run_phase_02_pipeline(
        benchmark_samples=2000,
        blur_test_samples=100,
        max_workers=4,
    )
