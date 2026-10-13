"""
run_phase_05_active_tournament.py - Điều Phối Trọn Vẹn Quy Trình Phase 05 (Data-Centric Active Mining)
=====================================================================================================
Chuyên biệt hóa 100% cho Data-Centric AI (Tuyển chọn, Kiểm soát chất lượng & Đóng gói dữ liệu):
1. Nạp Feature & Uncertainty Cache của 4.394 ảnh Unlabeled Pool.
2. Đấu trường 6 chiến lược tuyển chọn (K = 250 ảnh).
3. Đánh giá Leaderboard CSS và chọn Quán quân Super-Fusion (Winner C6).
4. Phân tích Quality-Gated Policy: Khóa chặt pre-label ở các lớp yếu, đo lường chỉ số tiết kiệm công sức (AER %).
5. Đóng gói xuất xưởng Dataset V1 (750 ảnh) bằng Symlink Materialization (0 KB đĩa, 100% Nhãn Vàng BDD10K).
6. Rút 250 ảnh khỏi unlabeled_pool/ (còn 4.144 ảnh), niêm phong manifest_v1.yaml và checksum SHA-256.
7. Xuất bản trọn bộ 5 Biểu Đồ Khoa Học 300 DPI phục vụ bài báo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "paper_v4_10k"))

from src.paths import (
    find_project_dir,
    get_bdd10k_metadata_file,
    get_bdd10k_train_images_dir,
    get_models_dir,
    get_results_dir,
    get_splits_dir,
)
from src.utils import get_device, set_seed
from src.splits import BDD10K_CLASS_NAMES
from src.train_engine import clean_bdd10k_caches
from src.active_mining import (
    CLASS_DEFICIT_WEIGHTS,
    RARE_CLASSES,
    PoolItemRecord,
    PoolFeatureExtractor,
    SelectionResult,
    ActiveMiningTournamentRunner,
    RandomSamplingStrategy,
)
from src.hitl_engine import (
    ConsensusChecker,
    TwoTierRouter,
    DatasetV1Packager,
    QualityGatedAnalyzer,
    QualityGatedReport,
)
from src.visualization import (
    plot_selection_tournament_radar,
    plot_pca_dispersion_distribution,
    plot_quality_gated_policy_matrix,
    plot_hard_slices_and_rare_objects,
    plot_visual_gallery_mined_samples,
    save_figure,
)


def run_phase_05_pipeline(
    k_selection: int = 250,
    batch_size: int = 16,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Thực thi trọn vẹn toàn bộ quy trình Data-Centric của Phase 05.
    """
    t_pipeline_start = time.time()
    set_seed(42)
    dev = get_device() if device == "auto" else device

    print("=" * 80)
    print("🚀 BẮT ĐẦU PIPELINE PHASE 05 — ACTIVE MINING TOURNAMENT & DATA CURATION (DATA-CENTRIC)")
    print("=" * 80)
    print(f"  - Thiết bị tính toán: {dev}")
    print(f"  - Quy mô tuyển chọn K: {k_selection} ảnh")

    results_p5_dir = get_results_dir("phase_05")
    tables_dir = results_p5_dir / "tables"
    figures_dir = results_p5_dir / "figures"
    splits_dir = get_splits_dir()

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG HOẶC NẠP CACHE UNLABELED POOL
    # --------------------------------------------------------------------------
    print("\n[Bước 1/6] Nạp Feature & Uncertainty Cache của 4.394 ảnh Unlabeled Pool...")
    cache_path = results_p5_dir / "unlabeled_pool_feature_cache.json"
    extractor = PoolFeatureExtractor(cache_file=cache_path, device=dev, batch_size=batch_size)
    records = extractor.extract_or_load()
    print(f"  ✓ Sẵn sàng {len(records):,} bản ghi đặc trưng (PCA 32D, Margin Uncertainty, QBC Disagreement).")

    # --------------------------------------------------------------------------
    # BƯỚC 2 & 3: ĐẤU TRƯỜNG 6 CHIẾN LƯỢC TUYỂN CHỌN & CHỌN QUÁN QUÂN
    # --------------------------------------------------------------------------
    print(f"\n[Bước 2/6 & 3/6] Khởi động Đấu Trường Tuyển Chọn 6 Chiến Lược (K = {k_selection} ảnh)...")
    tournament_runner = ActiveMiningTournamentRunner(
        k_selection=k_selection,
        device=dev,
        cache_file=cache_path,
    )
    tournament_summary, winner_result = tournament_runner.run_tournament()
    winner_name = tournament_summary["winner_strategy"]
    winner_images = winner_result.selected_images
    print(f"  ✓ Quán quân được chọn: {winner_name} với điểm CSS = {tournament_summary['winner_css']}")

    # Đồng thời bốc 250 ảnh của C1 Random để phục vụ đối chứng khoa học
    random_strat = RandomSamplingStrategy(seed=42)
    random_result = random_strat.select(records, k=k_selection)
    random_images = random_result.selected_images

    # Lưu Leaderboard ra Markdown và JSON
    leaderboard_md_path = tables_dir / "selection_tournament_leaderboard.md"
    leaderboard_json_path = tables_dir / "selection_tournament_leaderboard.json"

    with open(leaderboard_json_path, "w", encoding="utf-8") as f:
        json.dump(tournament_summary, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# 🏆 Bảng Xếp Hạng Đấu Trường Tuyển Chọn Dữ Liệu Phase 05 (Active Mining Tournament)",
        "",
        f"Đấu trường thực nghiệm 6 chiến lược đối đầu trên kho **{len(records):,} ảnh** chưa gán nhãn (`unlabeled_pool`), mỗi chiến lược tuyển chọn đúng **K = {k_selection} ảnh**.",
        "",
        "| Hạng | Chiến Lược Tuyển Chọn | Bán Kính Phủ $R_{\\text{cov}}$ | Vật Thể Hiếm (CPS) | Quota Đêm/Ngày | Proxy $\\Delta\\text{mAP50}$ | CSS Score | Danh Hiệu |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    rank_icons = ["🥇", "🥈", "🥉", "4", "5", "6"]
    for idx, item in enumerate(tournament_summary["leaderboard"]):
        icon = rank_icons[idx] if idx < len(rank_icons) else str(idx + 1)
        strat_name = item.get("strategy_name", "")
        title = "**Quán Quân (Winner)**" if strat_name == winner_name else "Ứng Viên"
        r_cov = item.get("r_cov_pca32d", 0.0)
        cps = item.get("rare_class_cps", 0.0)
        rare_objs = item.get("total_rare_objects", 0)
        night_r = item.get("night_ratio", 0.0)
        day_r = item.get("day_ratio", 0.0)
        p_delta = item.get("proxy_delta_map50", 0.0)
        css = item.get("composite_selection_score_css", 0.0)
        md_lines.append(
            f"| {icon} | `{strat_name}` | {r_cov:.2f} | "
            f"{cps:.2f} ({rare_objs} objs) | "
            f"{night_r:.0f}% / {day_r:.0f}% | "
            f"+{p_delta:.2f}% | **{css:.2f}** | {title} |"
        )
    md_lines.extend([
        "",
        "### Diễn Giải Các Chỉ Số:",
        "- **$R_{\\text{cov}}$ (Coverage Radius)**: Bán kính phủ cực đại trên PCA 32D (càng nhỏ càng dàn đều, triệt tiêu vùng trống).",
        "- **CPS (Rare Class Penetration Score)**: Tỷ lệ săn lùng vật thể thuộc 6 lớp điểm liệt (`rider, motor, bike, train, bus, truck`).",
        "- **Quota Đêm/Ngày**: Tỷ lệ ảnh Ban đêm và Ban ngày trong batch (chuẩn mục tiêu: 35% Đêm, 25% Ngày).",
        "- **Proxy $\\Delta\\text{mAP50}$**: Mức tăng trưởng dự phóng trên `fixed_val` (700 ảnh).",
        "- **CSS (Composite Selection Score)**: Điểm tổng hợp chuẩn hóa 4 trục (0 - 100 điểm).",
    ])
    with open(leaderboard_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  ✓ Đã lưu Leaderboard vào {leaderboard_md_path}")

    # --------------------------------------------------------------------------
    # BƯỚC 4: CHÍNH SÁCH QUALITY-GATED PRE-LABELING & PHÂN TÍCH AER
    # --------------------------------------------------------------------------
    print("\n[Bước 4/6] Đánh giá Chính Sách Quality-Gated Pre-Labeling & Tiết Kiệm Công Sức...")
    qg_analyzer = QualityGatedAnalyzer(quality_threshold=0.75)
    qg_report = qg_analyzer.analyze(records, winner_images)

    router = TwoTierRouter(spot_check_ratio=0.10, rollback_error_threshold=0.05)
    hitl_results = router.route_and_annotate(records, winner_images, seed=42)

    print(f"  ✓ Kết quả thẩm định năng lực mô hình theo từng lớp:")
    print(f"    * Lớp mở Pre-labeling (Precision >= 75%) : {qg_report.active_prelabel_classes}")
    print(f"    * Lớp Gated (Manual Oracle Only, Precision < 75%): {qg_report.gated_classes}")
    print(f"    * Chỉ số AER (Annotation Effort Reduction)   : {qg_report.annotation_effort_reduction_pct}%")
    print(f"    * Thời gian tiết kiệm lý thuyết               : {qg_report.time_saved_hours:.1f} giờ (~{qg_report.theoretical_cost_saved_usd:.1f} USD)")

    hitl_report_path = results_p5_dir / "hitl_routing_report.json"
    hitl_summary_to_save = {
        "quality_gated_policy": asdict(qg_report),
        "hitl_routing_stats": {
            "tier_1_auto_count": hitl_results["tier_1_auto_count"],
            "tier_2_oracle_count": hitl_results["tier_2_oracle_count"],
            "spot_check_count": hitl_results["spot_check_count"],
            "spot_check_errors": hitl_results["spot_check_errors"],
            "spot_check_error_rate": hitl_results["spot_check_error_rate"],
            "rollback_triggered": hitl_results["rollback_triggered"],
        },
        "sample_annotations": [asdict(a) for a in hitl_results["annotations"][:10]],
    }
    with open(hitl_report_path, "w", encoding="utf-8") as f:
        json.dump(hitl_summary_to_save, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Đã lưu báo cáo định tuyến & chính sách chất lượng vào {hitl_report_path}")

    # --------------------------------------------------------------------------
    # BƯỚC 5: ĐÓNG GÓI XUẤT XƯỞNG DATASET V1 (100% NHÃN VÀNG BDD10K)
    # --------------------------------------------------------------------------
    print("\n[Bước 5/6] Vật liệu hóa Dataset V1 (750 ảnh) bằng Symlink Materialization...")
    packager = DatasetV1Packager()
    packager.restore_unlabeled_pool(list(records.keys()))
    pkg_results = packager.package_dataset_v1(hitl_results)
    print(f"  ✓ Xuất xưởng thành công {pkg_results['total_train_images']} ảnh huấn luyện vào data/splits/train_v1/!")
    print(f"  ✓ 100% Nhãn Huấn Luyện Cấp Từ Ground-Truth BDD10K Gốc (Zero Label Noise).")
    print(f"  ✓ Kho unlabeled_pool còn lại: {pkg_results['unlabeled_pool_remaining']:,} ảnh (đã rút {pkg_results['removed_from_pool']} ảnh).")
    print(f"  ✓ Tăng dung lượng đĩa thực tế: 0 KB (100% Symbolic Links).")

    # Đóng gói tập đối chứng train_random_v1 (500 Seed v0 + 250 Random)
    train_random_dir = splits_dir / "train_random_v1"
    (train_random_dir / "images").mkdir(parents=True, exist_ok=True)
    (train_random_dir / "labels").mkdir(parents=True, exist_ok=True)

    for d_rnd in [(train_random_dir / "images"), (train_random_dir / "labels")]:
        for item in d_rnd.iterdir():
            if item.is_symlink() or item.is_file():
                item.unlink()

    # Symlink 500 Seed v0
    for p in (splits_dir / "seed_v0" / "images").glob("*.jpg"):
        dst_img = train_random_dir / "images" / p.name
        dst_img.symlink_to(p.resolve())
        lbl_src = splits_dir / "seed_v0" / "labels" / f"{p.stem}.txt"
        if lbl_src.exists():
            dst_lbl = train_random_dir / "labels" / f"{p.stem}.txt"
            dst_lbl.symlink_to(lbl_src.resolve())

    # Symlink 250 Random
    bdd_train_img_d = find_project_dir() / "data" / "bdd10k" / "images" / "train"
    bdd_train_lbl_d = find_project_dir() / "data" / "bdd10k" / "labels" / "train"
    for img_name in random_images:
        stem = Path(img_name).stem
        src_img = bdd_train_img_d / img_name
        dst_img = train_random_dir / "images" / img_name
        if not dst_img.exists() and src_img.exists():
            dst_img.symlink_to(src_img)
        src_lbl = bdd_train_lbl_d / f"{stem}.txt"
        dst_lbl = train_random_dir / "labels" / f"{stem}.txt"
        if not dst_lbl.exists() and src_lbl.exists():
            dst_lbl.symlink_to(src_lbl)

    yaml_random_path = splits_dir / "bdd10k_splits_random_v1.yaml"
    with open(yaml_random_path, "w", encoding="utf-8") as f:
        f.write(f"""# BDD10K Random Baseline Dataset Config
path: {splits_dir.resolve()}
train: train_random_v1/images
val: fixed_val/images
test: fixed_test/images
names:
  0: person
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motor
  7: bike
  8: traffic light
  9: traffic sign
""")

    # --------------------------------------------------------------------------
    # BƯỚC 6: XUẤT BẢN TRỌN BỘ 5 BIỂU ĐỒ KHOA HỌC 300 DPI CỦA PHASE 05
    # --------------------------------------------------------------------------
    print("\n[Bước 6/6] Vẽ và Xuất Xưởng 5 Biểu Đồ Khoa Học 300 DPI Chuẩn Bài Báo V4.0...")

    # 1. Radar Chart 6 chiến lược
    fig_radar = plot_selection_tournament_radar(
        tournament_summary["leaderboard"],
        title="Đấu Trường Tuyển Chọn 6 Chiến Lược Active Mining (Composite Benchmark Radar)"
    )
    p_radar = save_figure(fig_radar, "fig_phase05_01_tournament_radar.png", figures_dir, dpi=300)

    # 2. PCA 32D Dispersion plot
    all_embeddings_pca = np.array([records[img].embedding_pca for img in sorted(records.keys())], dtype=np.float32)
    selected_indices = [sorted(records.keys()).index(img) for img in winner_images]
    fig_pca = plot_pca_dispersion_distribution(
        all_embeddings_pca,
        selected_indices,
        winner_name=winner_name,
        title="Phân Bố Rải Thảm PCA 32D: 250 Ảnh Quán Quân Trên Nền 4.394 Ảnh Unlabeled Pool"
    )
    p_pca = save_figure(fig_pca, "fig_phase05_02_pca_dispersion.png", figures_dir, dpi=300)

    # 3. Quality-Gated Policy Matrix
    fig_qg = plot_quality_gated_policy_matrix(
        qg_report.per_class_precision,
        quality_threshold=qg_report.quality_threshold,
        aer_pct=qg_report.annotation_effort_reduction_pct,
        title="Chính Sách Cổng Kiểm Soát Chất Lượng (Quality-Gated Pre-Labeling Policy)"
    )
    p_qg = save_figure(fig_qg, "fig_phase05_03_quality_gated_policy_matrix.png", figures_dir, dpi=300)

    # 4. Hard Slices & Rare Objects
    prof_path = find_project_dir() / "paper_v4_10k" / "data" / "processed" / "reports" / "data_profiling_manifest.json"
    prof_data = {}
    if prof_path.exists():
        with open(prof_path, "r", encoding="utf-8") as f:
            prof_data = json.load(f)

    hsv_vals = [prof_data.get(img, {}).get("scores", {}).get("brightness_v_mean", 85.0) for img in winner_images]
    lap_vals = [records[img].laplacian_variance for img in winner_images]
    rare_counts: Dict[str, int] = {c: 0 for c in RARE_CLASSES}
    gt_meta_file = get_bdd10k_metadata_file()
    with open(gt_meta_file, "r", encoding="utf-8") as f:
        gt_meta = json.load(f)

    for img in winner_images:
        stem = Path(img).stem
        for obj in gt_meta.get(stem, {}).get("objects", []):
            cat = obj.get("category", "")
            if cat in rare_counts:
                rare_counts[cat] += 1

    fig_hs = plot_hard_slices_and_rare_objects(
        hsv_vals,
        lap_vals,
        rare_counts,
        title="Đa Chiều Lát Cắt Khó & Phân Bố 6 Lớp Điểm Liệt Được Khai Thác"
    )
    p_hs = save_figure(fig_hs, "fig_phase05_04_hard_slices_and_rare_objects.png", figures_dir, dpi=300)

    # 5. Visual Gallery 4 mẫu điển hình
    sample_images: List[Dict[str, Any]] = []
    # Night sample
    night_cands = [img for img in winner_images if records[img].is_night]
    night_img = night_cands[0] if night_cands else winner_images[0]
    sample_images.append({
        "path": str((bdd_train_img_d / night_img).resolve()),
        "tag": "Thiếu Sáng / Ban Đêm",
        "num_boxes": records[night_img].num_boxes,
        "rare_classes": "Night Slice"
    })

    # Blur sample
    blur_cands = [img for img in winner_images if records[img].is_blur]
    blur_img = blur_cands[0] if blur_cands else winner_images[1]
    sample_images.append({
        "path": str((bdd_train_img_d / blur_img).resolve()),
        "tag": "Rung Lắc / Mờ Ống Kính",
        "num_boxes": records[blur_img].num_boxes,
        "rare_classes": "Blur Slice"
    })

    # Dense small object sample
    small_cands = [img for img in winner_images if records[img].is_small_object_slice]
    small_img = small_cands[0] if small_cands else winner_images[2]
    sample_images.append({
        "path": str((bdd_train_img_d / small_img).resolve()),
        "tag": "Mật Độ Dày / Vật Thể Nhỏ",
        "num_boxes": records[small_img].num_boxes,
        "rare_classes": "Small VRU"
    })

    # Occlusion sample
    occ_cands = [img for img in winner_images if records[img].is_high_occlusion]
    occ_img = occ_cands[0] if occ_cands else winner_images[3]
    sample_images.append({
        "path": str((bdd_train_img_d / occ_img).resolve()),
        "tag": "Che Khuất Phức Tạp",
        "num_boxes": records[occ_img].num_boxes,
        "rare_classes": "Occluded"
    })

    fig_gallery = plot_visual_gallery_mined_samples(
        sample_images,
        title="Lưới Trực Quan 4 Mẫu Điển Hình Được Khai Thác (Night, Blur, Small, Occlusion)"
    )
    p_gallery = save_figure(fig_gallery, "fig_phase05_05_visual_gallery_mined_samples.png", figures_dir, dpi=300)

    # Dọn dẹp cache
    removed_caches = clean_bdd10k_caches()

    total_pipeline_time = time.time() - t_pipeline_start
    print("\n" + "=" * 80)
    print(f"🎉 HOÀN TẤT THÀNH CÔNG PHASE 05 DATA CURATION TRONG {total_pipeline_time:.2f}s!")
    print("=" * 80)

    summary_final = {
        "phase": "05_active_mining_tournament_and_hitl",
        "status": "COMPLETED_SUCCESS",
        "total_runtime_seconds": round(total_pipeline_time, 2),
        "winner_strategy": winner_name,
        "winner_css": tournament_summary["winner_css"],
        "dataset_v1": pkg_results,
        "quality_gated_report": asdict(qg_report),
        "figures": [str(p_radar), str(p_pca), str(p_qg), str(p_hs), str(p_gallery)],
    }

    summary_file = results_p5_dir / "phase_05_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_final, f, indent=2, ensure_ascii=False)

    return summary_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Phase 05 Data-Centric Active Mining")
    parser.add_argument("--k", type=int, default=250, help="Số lượng ảnh tuyển chọn")
    args = parser.parse_args()

    run_phase_05_pipeline(k_selection=args.k)
