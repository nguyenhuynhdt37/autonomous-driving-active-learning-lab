"""
run_phase_03_frozen_splits.py - Script Thực Thi Toàn Bộ Đấu Trường Phase 03
=============================================================================
Quy trình thực thi hoàn chỉnh tự động hóa 100%:
1. Nạp 6.994 ảnh từ unique_pool và bdd10k_metadata.json (Read-Only 100%).
2. Khởi tạo GroupAwareVideoSplitter gom 4.277 chuỗi video sequences.
3. Chạy Đấu Trường Thực Nghiệm Song Song (Parallel Tournament) 6 Chiến Lược:
   - C1: Random Video Split (Baseline ngẫu nhiên).
   - C2: Direct Metadata Phase 02 Stratified (Lấy thẳng theo tags Phase 02).
   - C3: Pure k-Center Greedy (Bám biên Minimax).
   - C4: Pure Facility Location (Dàn đều mật độ tự nhiên).
   - C5: 🏆 Hybrid Mix 80/20 (80% Facility + 20% k-Center).
   - C6: 🥈 Hybrid Mix 70/30 (70% Facility + 30% k-Center).
4. Chấm điểm qua Dual Ground-Truth Scoring Engine & Xếp hạng Leaderboard.
5. Tự động chọn ra Winner đạt Composite Benchmark Score (CBS) cao nhất.
6. Kích hoạt AntiLeakageAuditor: Kiểm toán rò rỉ video & Quét chéo pHash HD < 5 = 0.
7. Kích hoạt SymlinkSplitMaterializer: Tạo cấu trúc thư mục và symlinks 0 KB đĩa.
8. Sinh cấu hình YOLO bdd10k_splits.yaml và Niêm phong SHA-256 Checksum Seal.
9. Xuất toàn bộ báo cáo JSON và vẽ biểu đồ 300 DPI sẵn sàng chèn vào Paper V4.0.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Thêm đường dẫn project
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent
sys.path.insert(0, str(project_dir))

from src.paths import (
    find_project_dir,
    get_bdd10k_metadata_file,
    get_bdd10k_train_images_dir,
    get_bdd10k_train_labels_dir,
    get_processed_data_dir,
    get_splits_dir,
    get_results_dir,
)
from src.splits import (
    BDD10K_CLASS_NAMES,
    GroupAwareVideoSplitter,
    RandomGroupSplitter,
    DirectMetadataStratifiedSplitter,
    StratifiedKCenterSplitter,
    SubmodularFacilityLocationSplitter,
    HybridMixedSplitter,
    StrategyTournamentEvaluator,
    AntiLeakageAuditor,
    SymlinkSplitMaterializer,
    SHA256ChecksumSealer,
    generate_yolo_splits_config,
)
from src.visualization import (
    plot_strategy_tournament_comparison,
    plot_pca_diversity_dispersion,
    plot_class_distribution_coverage,
    save_figure,
)


def run_phase_03_pipeline() -> Dict[str, Any]:
    """Chạy toàn bộ pipeline Phase 03."""
    print("=" * 85)
    print("🚀 BẮT ĐẦU PHASE 03: SHARED PRE-BRANCHING FROZEN TEST & VAL SPLITS (TOURNAMENT)")
    print("=" * 85)
    start_total_time = time.time()

    proj_dir = find_project_dir()
    processed_dir = get_processed_data_dir()
    manifest_path = processed_dir / "reports" / "data_profiling_manifest.json"
    metadata_path = get_bdd10k_metadata_file()
    unique_pool_dir = processed_dir / "unique_pool"
    splits_dir = get_splits_dir()

    # Thư mục lưu kết quả & báo cáo
    splits_reports_dir = splits_dir / "reports"
    splits_figures_dir = splits_reports_dir / "figures"
    splits_reports_dir.mkdir(parents=True, exist_ok=True)
    splits_figures_dir.mkdir(parents=True, exist_ok=True)

    results_p3_dir = get_results_dir("phase_03")
    results_p3_reports = results_p3_dir / "reports"
    results_p3_figures = results_p3_dir / "figures"
    results_p3_reports.mkdir(parents=True, exist_ok=True)
    results_p3_figures.mkdir(parents=True, exist_ok=True)

    # 1. Kiểm tra tài nguyên đầu vào
    if not manifest_path.exists():
        raise FileNotFoundError(f"Không tìm thấy manifest Phase 02: {manifest_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Không tìm thấy metadata gốc BDD10K: {metadata_path}")
    if not unique_pool_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục unique_pool: {unique_pool_dir}")

    # 2. Khởi tạo GroupAwareVideoSplitter
    print("\n📂 1. TỔNG HỢP VIDEO SEQUENCES & CHUẨN HÓA KHÔNG GIAN ĐẶC TRƯNG 11 CHIỀU:")
    splitter = GroupAwareVideoSplitter(
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        random_seed=42
    )
    print(f"  - Tổng số ảnh sạch trong unique_pool : {splitter.total_images_count:,} ảnh")
    print(f"  - Tổng số chuỗi video sequences     : {splitter.total_videos_count:,} sequences")
    print(f"  - Kích thước không gian đặc trưng    : {splitter.norm_image_features.shape} (11 continuous features)")
    print(f"  - RBF Kernel Sigma Heuristic         : {splitter.rbf_sigma:.4f}")

    # 3. Khởi chạy Tournament 6 Chiến Lược
    evaluator = StrategyTournamentEvaluator(splitter)
    eval_records, results_map, winner_name = evaluator.run_tournament()

    # In Bảng Leaderboard
    print("\n" + "=" * 90)
    print("🏆 BẢNG XẾP HẠNG LEADERBOARD ĐẤU TRƯỜNG PHÂN CHIA DỮ LIỆU (TOURNAMENT BENCHMARK)")
    print("=" * 90)
    print(f"{'Hạng':<5} | {'Chiến Lược Tuyển Chọn':<38} | {'QAE %':<7} | {'R_cov':<6} | {'F_score':<7} | {'Classes':<7} | {'Leak':<6} | {'CBS':<6} | {'Quyết Định'}")
    print("-" * 90)
    for ev in eval_records:
        rank_str = str(ev["rank"])
        name_str = ev["strategy_name"][:38]
        qae_str = f"{ev['quota_absolute_error']*100:.1f}%"
        rcov_str = f"{ev['coverage_radius']:.2f}"
        fac_str = f"{ev['facility_location_score']:.3f}"
        cls_str = f"{ev['classes_present_count']}/10"
        leak_str = "PASS" if ev["gatekeeper_pass"] else "FAIL"
        cbs_str = f"{ev['composite_benchmark_score']:.1f}"
        dec_str = ev["decision"]
        print(f"{rank_str:<5} | {name_str:<38} | {qae_str:<7} | {rcov_str:<6} | {fac_str:<7} | {cls_str:<7} | {leak_str:<6} | {cbs_str:<6} | {dec_str}")
    print("=" * 90)

    winner_result = results_map[winner_name]
    winner_eval = eval_records[0]

    print(f"\n🥇 CHIẾN LƯỢC CHIẾN THẮNG ĐƯỢC CHỌN:")
    print(f"  - Tên chiến lược  : {winner_name}")
    print(f"  - Điểm CBS        : {winner_eval['composite_benchmark_score']:.1f} / 100")
    print(f"  - Số ảnh Test     : {winner_result.num_test:,} ảnh (20.0%)")
    print(f"  - Số ảnh Val      : {winner_result.num_val:,} ảnh (10.0%)")
    print(f"  - Số ảnh Seed v0  : {winner_result.num_seed:,} ảnh (~10.2% pool / 7.1% tổng)")
    print(f"  - Số ảnh Pool     : {winner_result.num_pool:,} ảnh (62.8%)")

    # 4. Kiểm toán rò rỉ kép 4 chiều (AntiLeakageAuditor)
    auditor = AntiLeakageAuditor(pool_images_dir=unique_pool_dir)
    audit_report = auditor.audit_splits(winner_result, max_workers=8)

    if audit_report["overall_status"] != "PASS":
        raise RuntimeError("CẢNH BÁO: Phát hiện rò rỉ dữ liệu hoặc trùng lặp pHash!")

    # 5. Symlink Materialization
    raw_imgs_dir = get_bdd10k_train_images_dir()
    raw_lbls_dir = get_bdd10k_train_labels_dir()

    materializer = SymlinkSplitMaterializer(
        splits_dir=splits_dir,
        raw_images_dir=raw_imgs_dir,
        raw_labels_dir=raw_lbls_dir
    )
    mat_summary = materializer.materialize(winner_result)

    # 6. Sinh cấu hình YOLO và SHA-256 Checksum Seal
    yaml_path = generate_yolo_splits_config(splits_dir)

    # Tổng hợp manifest chi tiết của split chiến thắng
    splits_manifest = {
        "dataset_name": "BDD10K Autonomous Driving Lab (Paper V4.0)",
        "phase": "03_shared_frozen_test_val_splits",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "winning_strategy": winner_name,
        "composite_benchmark_score": winner_eval["composite_benchmark_score"],
        "counts": {
            "total_images": winner_result.total_images,
            "fixed_test": winner_result.num_test,
            "fixed_val": winner_result.num_val,
            "seed_v0": winner_result.num_seed,
            "unlabeled_pool": winner_result.num_pool,
            "fixed_test_percent": round(winner_result.num_test / winner_result.total_images * 100, 2),
            "fixed_val_percent": round(winner_result.num_val / winner_result.total_images * 100, 2),
            "seed_v0_percent": round(winner_result.num_seed / winner_result.total_images * 100, 2),
            "unlabeled_pool_percent": round(winner_result.num_pool / winner_result.total_images * 100, 2),
        },
        "audit_verification": audit_report,
        "slices_quota_metrics": winner_eval["quota_ratios"],
        "physical_space_metrics": {
            "coverage_radius": winner_eval["coverage_radius"],
            "mean_nn_distance": winner_eval["mean_nn_distance"],
            "facility_location_score": winner_eval["facility_location_score"],
            "pca_spread_ratio": winner_eval["pca_spread_ratio"],
        },
        "test_images": winner_result.test_images,
        "val_images": winner_result.val_images,
        "seed_images": winner_result.seed_images,
        "pool_images": winner_result.pool_images,
    }

    # Niêm phong SHA-256 Seal
    sealer = SHA256ChecksumSealer(splits_dir=splits_dir)
    checksum_path = sealer.seal_dataset(splits_manifest)

    # 7. Lưu toàn bộ báo cáo JSON
    manifest_out = splits_reports_dir / "splits_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(splits_manifest, f, indent=2, ensure_ascii=False)

    tournament_out = splits_reports_dir / "strategy_tournament_benchmark.json"
    with open(tournament_out, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=2, ensure_ascii=False)

    audit_out = splits_reports_dir / "frozen_splits_audit_report.json"
    with open(audit_out, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    # Sao lưu sang results/phase_03/reports/
    for fn, obj in [
        ("splits_manifest.json", splits_manifest),
        ("strategy_tournament_benchmark.json", eval_records),
        ("frozen_splits_audit_report.json", audit_report)
    ]:
        with open(results_p3_reports / fn, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    print(f"\n📊 7. XUẤT CÁC BÁO CÁO JSON THÀNH CÔNG:")
    print(f"  - Splits Manifest     : {manifest_out}")
    print(f"  - Tournament Benchmark: {tournament_out}")
    print(f"  - Audit Report        : {audit_out}")
    print(f"  - YOLO Config         : {yaml_path}")
    print(f"  - SHA-256 Seal        : {checksum_path}")

    # 8. Vẽ và lưu các biểu đồ 300 DPI
    print(f"\n🎨 8. XUẤT CÁC BIỂU ĐỒ TRỰC QUAN HÓA KHOA HỌC (300 DPI):")

    # Biểu đồ 1: Tournament Comparison
    fig1 = plot_strategy_tournament_comparison(eval_records)
    save_figure(fig1, "strategy_tournament_benchmark.png", output_dir=splits_figures_dir)
    save_figure(fig1, "strategy_tournament_benchmark.png", output_dir=results_p3_figures)

    # Biểu đồ 2: PCA Dispersion
    test_indices = [splitter.all_images.index(img) for img in winner_result.test_images]
    val_indices = [splitter.all_images.index(img) for img in winner_result.val_images]
    seed_indices = [splitter.all_images.index(img) for img in winner_result.seed_images]
    pool_indices = [splitter.all_images.index(img) for img in winner_result.pool_images]

    fig2 = plot_pca_diversity_dispersion(
        norm_features=splitter.norm_image_features,
        test_indices=test_indices,
        val_indices=val_indices,
        pool_indices=pool_indices,
        seed_indices=seed_indices
    )
    save_figure(fig2, "pca_test_diversity_dispersion.png", output_dir=splits_figures_dir)
    save_figure(fig2, "pca_test_diversity_dispersion.png", output_dir=results_p3_figures)

    # Biểu đồ 3: Class Distribution Coverage
    class_dist_dict = {
        "Full Pool": evaluator.full_class_counts,
        "Fixed Test": winner_eval["test_class_counts"],
        "Fixed Val": {c: 0 for c in BDD10K_CLASS_NAMES},
        "Seed V0": {c: 0 for c in BDD10K_CLASS_NAMES},
        "Unlabeled Pool": {c: 0 for c in BDD10K_CLASS_NAMES}
    }
    for vid in winner_result.val_videos:
        for c, cnt in splitter.video_records[vid].class_counts.items():
            class_dist_dict["Fixed Val"][c] += cnt
    for vid in winner_result.seed_videos:
        for c, cnt in splitter.video_records[vid].class_counts.items():
            class_dist_dict["Seed V0"][c] += cnt
    for vid in winner_result.pool_videos:
        for c, cnt in splitter.video_records[vid].class_counts.items():
            class_dist_dict["Unlabeled Pool"][c] += cnt

    fig3 = plot_class_distribution_coverage(class_dist_dict)
    save_figure(fig3, "class_distribution_coverage.png", output_dir=splits_figures_dir)
    save_figure(fig3, "class_distribution_coverage.png", output_dir=results_p3_figures)

    total_elapsed = time.time() - start_total_time
    print(f"\n🎉 HOÀN THÀNH TOÀN BỘ PHASE 03 TRONG {total_elapsed:.2f} GIÂY!")
    print("=" * 85)

    return {
        "winner_strategy": winner_name,
        "cbs_score": winner_eval["composite_benchmark_score"],
        "total_elapsed_seconds": round(total_elapsed, 2),
        "audit_pass": audit_report["overall_status"] == "PASS"
    }


if __name__ == "__main__":
    run_phase_03_pipeline()
