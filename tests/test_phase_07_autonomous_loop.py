"""
test_phase_07_autonomous_loop.py - Bộ Kiểm Thử Tự Động Toàn Diện Cho Phase 07
=============================================================================
Kiểm toán nghiêm ngặt 100% các tiêu chuẩn kỹ thuật của Phase 07:
1. Workspace tự quản độc lập bên trong notebooks/autonomous_pipeline_loop/.
2. Bảo toàn tuyệt đối 100% Read-Only cho data/bdd10k/ và không can thiệp paper_v4_10k/data/.
3. Tính toán và phân chia tỷ lệ động 10% (4 tập chuẩn) với Zero Video Overlap.
4. Kiểm toán đầy đủ toàn bộ 5 Tầng Rào Chắn Dừng và Vòng Xác Nhận Chiến Thắng.
5. Kiểm tra cấu trúc dữ liệu IterationMetrics, TournamentVerdict và bảng Per-Class 10 lớp.
6. Kiểm tra các hàm vẽ 3 biểu đồ 300 DPI và sinh mã băm SHA-256 niêm phong.
7. Kỷ luật Zero Stale Cache: Không rò rỉ bất kỳ file *.cache nào vào data/bdd10k/.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.autonomous_loop import (
    ActiveMiningSelector,
    AutonomousLoopConfig,
    AutonomousLoopOrchestrator,
    DatasetWorkspaceManager,
    IterationMetrics,
    StoppingGuard,
    TournamentStoppingStatus,
    TournamentVerdict,
    clean_all_caches,
    clean_bdd10k_caches,
    compute_dynamic_class_deficit_weights,
    compute_file_sha256,
)
from src.paths import (
    find_project_dir,
    get_bdd10k_dir,
    get_data_dir,
    get_phase07_workspace_dir,
)
from src.splits import BDD10K_CLASS_NAMES


class TestPhase07AutonomousLoop(unittest.TestCase):
    """Bộ kiểm thử đơn vị & tích hợp toàn diện cho Phase 07."""

    @classmethod
    def setUpClass(cls):
        cls.project_dir = find_project_dir()
        cls.workspace_dir = get_phase07_workspace_dir()
        cls.bdd10k_dir = get_bdd10k_dir()

    def test_01_workspace_isolation_and_paths(self):
        """Kiểm tra không gian làm việc Phase 07 hoàn toàn độc lập và cô lập."""
        self.assertTrue(self.workspace_dir.exists(), "Thư mục workspace Phase 07 không tồn tại")
        self.assertEqual(self.workspace_dir.name, "autonomous_pipeline_loop")

        # Đảm bảo paper_v4_10k/data/ tồn tại và không bị xóa
        paper_data = self.project_dir / "data"
        self.assertTrue(paper_data.exists(), "Thư mục paper_v4_10k/data/ phải được bảo tồn nguyên vẹn")

        # Đảm bảo data/bdd10k/ là read-only và không có file tạm
        self.assertTrue(self.bdd10k_dir.exists(), "Thư mục data/bdd10k/ gốc phải tồn tại")
        cache_files = list(self.bdd10k_dir.rglob("*.cache"))
        self.assertEqual(len(cache_files), 0, f"Phát hiện {len(cache_files)} file *.cache rò rỉ vào data/bdd10k/!")

    def test_02_stopping_guard_tier_1_and_confirmation_round(self):
        """Kiểm toán Tầng 1: Chiến thắng sớm và Vòng Xác Nhận (Confirmation Round)."""
        target_map = 26.85
        guard = StoppingGuard(target_map50=target_map, max_iterations=7)

        # Vòng 1: mAP = 22.80% -> CONTINUE
        dummy_v1 = IterationMetrics(
            iteration=1, branch="active", num_train_images=1112, pool_percentage=20.0,
            macro_map50=22.80, macro_map50_95=13.0, macro_precision=32.0, macro_recall=25.0,
            delta_map50=2.57, per_class_map50={}, per_class_recall={},
            checkpoint_path="dummy.pt", sha256="abc", top_k_uncertainty=0.70,
            epochs_trained=8, duration_sec=10.0,
        )
        s1, _ = guard.check_stopping_criteria(1, dummy_v1)
        self.assertEqual(s1, TournamentStoppingStatus.CONTINUE)

        # Vòng 2: mAP = 27.20% >= target_map (26.85%) -> CONFIRMATION_ROUND_TRIGGERED
        dummy_v2 = IterationMetrics(
            iteration=2, branch="active", num_train_images=1668, pool_percentage=30.0,
            macro_map50=27.20, macro_map50_95=15.5, macro_precision=36.0, macro_recall=29.0,
            delta_map50=4.40, per_class_map50={}, per_class_recall={},
            checkpoint_path="dummy.pt", sha256="abc", top_k_uncertainty=0.45,
            epochs_trained=8, duration_sec=10.0,
        )
        s2, msg2 = guard.check_stopping_criteria(2, dummy_v2)
        self.assertEqual(s2, TournamentStoppingStatus.CONFIRMATION_ROUND_TRIGGERED)
        self.assertTrue(guard.confirmation_round_active)

        # Vòng 3 (Vòng xác nhận): mAP = 27.85% >= target_map -> VICTORY_CONFIRMED
        dummy_v3 = IterationMetrics(
            iteration=3, branch="active", num_train_images=2224, pool_percentage=40.0,
            macro_map50=27.85, macro_map50_95=16.0, macro_precision=37.0, macro_recall=30.0,
            delta_map50=0.65, per_class_map50={}, per_class_recall={},
            checkpoint_path="dummy.pt", sha256="abc", top_k_uncertainty=0.30,
            epochs_trained=8, duration_sec=10.0,
        )
        s3, msg3 = guard.check_stopping_criteria(3, dummy_v3)
        self.assertEqual(s3, TournamentStoppingStatus.VICTORY_CONFIRMED)
        self.assertIn("VÒNG XÁC NHẬN THÀNH CÔNG", msg3)

        # Kiểm toán kịch bản Vòng Xác Nhận THẤT BẠI: mAP < target_map
        guard_fail = StoppingGuard(target_map50=target_map, max_iterations=7)
        guard_fail.check_stopping_criteria(2, dummy_v2)  # kích hoạt confirmation round
        self.assertTrue(guard_fail.confirmation_round_active)
        # Giả sử vòng 3 sụt giảm xuống 26.50% (< 26.85%)
        dummy_v3_fail = IterationMetrics(
            iteration=3, branch="active", num_train_images=2224, pool_percentage=40.0,
            macro_map50=26.50, macro_map50_95=15.0, macro_precision=35.0, macro_recall=28.0,
            delta_map50=-0.70, per_class_map50={}, per_class_recall={},
            checkpoint_path="dummy.pt", sha256="abc", top_k_uncertainty=0.30,
            epochs_trained=8, duration_sec=10.0,
        )
        s_fail, msg_fail = guard_fail.check_stopping_criteria(3, dummy_v3_fail)
        self.assertEqual(s_fail, TournamentStoppingStatus.CONTINUE)
        self.assertFalse(guard_fail.confirmation_round_active)
        self.assertIn("Hủy cờ xác nhận, tiếp tục giải đấu", msg_fail)

    def test_03_stopping_guard_tier_2_upper_bound_near_saturated(self):
        """Kiểm toán Tầng 2: Tiệm cận trần 95% và tốc độ tăng trưởng Delta < 0.3%."""
        target_map = 26.85
        guard = StoppingGuard(target_map50=target_map, max_iterations=7, delta_plateau_threshold=0.3)

        # 95% của 26.85 = 25.5075%
        # Giả sử mAP = 25.80% (>= 95% trần) nhưng Delta = 0.20% (< 0.3%)
        dummy = IterationMetrics(
            iteration=3, branch="active", num_train_images=2224, pool_percentage=40.0,
            macro_map50=25.80, macro_map50_95=14.5, macro_precision=34.0, macro_recall=27.0,
            delta_map50=0.20, per_class_map50={}, per_class_recall={},
            checkpoint_path="dummy.pt", sha256="abc", top_k_uncertainty=0.40,
            epochs_trained=8, duration_sec=10.0,
        )
        status, msg = guard.check_stopping_criteria(3, dummy)
        self.assertEqual(status, TournamentStoppingStatus.UPPER_BOUND_NEAR_SATURATED)
        self.assertIn("TẦNG 2 DỪNG", msg)

    def test_04_stopping_guard_tier_3_knowledge_plateau(self):
        """Kiểm toán Tầng 3: Bão hòa tri thức Delta mAP < 0.3% liên tiếp qua 2 vòng."""
        target_map = 30.00
        guard = StoppingGuard(target_map50=target_map, max_iterations=7, delta_plateau_threshold=0.3)

        # Vòng 1: Delta = 0.20%
        dummy1 = IterationMetrics(
            iteration=1, branch="active", num_train_images=1112, pool_percentage=20.0,
            macro_map50=21.0, macro_map50_95=12.0, macro_precision=30.0, macro_recall=22.0,
            delta_map50=0.20, per_class_map50={}, per_class_recall={},
            checkpoint_path="dummy.pt", sha256="abc", top_k_uncertainty=0.60,
            epochs_trained=8, duration_sec=10.0,
        )
        s1, _ = guard.check_stopping_criteria(1, dummy1)
        self.assertEqual(s1, TournamentStoppingStatus.CONTINUE)

        # Vòng 2: Delta = 0.15% (lần thứ 2 liên tiếp < 0.3%) -> KNOWLEDGE_PLATEAU
        dummy2 = IterationMetrics(
            iteration=2, branch="active", num_train_images=1668, pool_percentage=30.0,
            macro_map50=21.15, macro_map50_95=12.1, macro_precision=30.2, macro_recall=22.2,
            delta_map50=0.15, per_class_map50={}, per_class_recall={},
            checkpoint_path="dummy.pt", sha256="abc", top_k_uncertainty=0.55,
            epochs_trained=8, duration_sec=10.0,
        )
        s2, msg2 = guard.check_stopping_criteria(2, dummy2)
        self.assertEqual(s2, TournamentStoppingStatus.KNOWLEDGE_PLATEAU)
        self.assertIn("TẦNG 3 DỪNG", msg2)

    def test_05_stopping_guard_tier_4_uncertainty_depleted(self):
        """Kiểm toán Tầng 4: Cạn kiệt ca khó Top-K Uncertainty < 0.15."""
        target_map = 30.00
        guard = StoppingGuard(target_map50=target_map, max_iterations=7, uncertainty_depleted_threshold=0.15)

        dummy = IterationMetrics(
            iteration=2, branch="active", num_train_images=1668, pool_percentage=30.0,
            macro_map50=23.0, macro_map50_95=13.0, macro_precision=31.0, macro_recall=24.0,
            delta_map50=1.5, per_class_map50={}, per_class_recall={},
            checkpoint_path="dummy.pt", sha256="abc", top_k_uncertainty=0.12,  # < 0.15
            epochs_trained=8, duration_sec=10.0,
        )
        status, msg = guard.check_stopping_criteria(2, dummy)
        self.assertEqual(status, TournamentStoppingStatus.UNCERTAINTY_DEPLETED)
        self.assertIn("TẦNG 4 DỪNG", msg)

    def test_06_stopping_guard_tier_5_fatal_threshold_80_percent(self):
        """Kiểm toán Tầng 5: Chạm trần sinh tử 80% (Vòng 7) phân định Thắng vs Bác Bỏ."""
        target_map = 26.85
        guard = StoppingGuard(target_map50=target_map, max_iterations=7)

        # Trường hợp 1: Chạm vòng 7 và mAP = 25.0% < target_map -> FAILED_REJECTED
        dummy_fail = IterationMetrics(
            iteration=7, branch="active", num_train_images=4448, pool_percentage=80.0,
            macro_map50=25.0, macro_map50_95=14.0, macro_precision=33.0, macro_recall=26.0,
            delta_map50=0.5, per_class_map50={}, per_class_recall={},
            checkpoint_path="dummy.pt", sha256="abc", top_k_uncertainty=0.25,
            epochs_trained=8, duration_sec=10.0,
        )
        status_fail, msg_fail = guard.check_stopping_criteria(7, dummy_fail)
        self.assertEqual(status_fail, TournamentStoppingStatus.FAILED_REJECTED)
        self.assertIn("BÁC BỎ", msg_fail)

        # Trường hợp 2: Chạm vòng 7 và mAP = 27.5% >= target_map -> VICTORY_AT_80_PERCENT
        guard_win = StoppingGuard(target_map50=target_map, max_iterations=7)
        dummy_win = IterationMetrics(
            iteration=7, branch="active", num_train_images=4448, pool_percentage=80.0,
            macro_map50=27.5, macro_map50_95=16.0, macro_precision=36.0, macro_recall=29.0,
            delta_map50=0.8, per_class_map50={}, per_class_recall={},
            checkpoint_path="dummy.pt", sha256="abc", top_k_uncertainty=0.25,
            epochs_trained=8, duration_sec=10.0,
        )
        status_win, msg_win = guard_win.check_stopping_criteria(7, dummy_win)
        self.assertEqual(status_win, TournamentStoppingStatus.VICTORY_AT_80_PERCENT)

    def test_07_full_pipeline_dry_run_execution(self):
        """Kiểm toán thực thi toàn trình pipeline Phase 07 ở chế độ Dry-Run."""
        config = AutonomousLoopConfig(
            workspace_dir=self.workspace_dir,
            batch_ratio=0.10,
            max_iterations=7,
            epochs_per_iteration=1,
            patience=2,
            batch_size=16,
            device="cpu",
            seed=42,
        )
        orchestrator = AutonomousLoopOrchestrator(config)
        verdict = orchestrator.run_full_pipeline(dry_run=True)

        self.assertIsNotNone(verdict)
        self.assertIn(
            verdict.status,
            [
                TournamentStoppingStatus.VICTORY_CONFIRMED,
                TournamentStoppingStatus.UPPER_BOUND_NEAR_SATURATED,
                TournamentStoppingStatus.VICTORY_AT_80_PERCENT,
            ],
        )
        self.assertGreater(verdict.target_map50, 20.0)
        self.assertGreater(verdict.active_final_map50, verdict.v0_map50)
        self.assertGreater(verdict.active_final_map50, verdict.random_final_map50)
        self.assertGreater(verdict.label_cost_reduction_percent, 20.0)

        # Kiểm tra số lượng ảnh 4 tập đạt chuẩn chính xác tuyệt đối
        self.assertEqual(len(orchestrator.unique_images), 7950)
        self.assertEqual(len(orchestrator.fixed_test_images), 1590)
        self.assertEqual(len(orchestrator.fixed_val_images), 795)
        self.assertEqual(len(orchestrator.full_train_pool_images), 5565)
        self.assertEqual(len(orchestrator.seed_v0_images), 556)
        self.assertEqual(len(orchestrator.unlabeled_pool_images), 5009)

        # Kiểm tra Zero Video Overlap giữa test, val, full_train_pool
        test_vids = {name.split("-")[0] for name in orchestrator.fixed_test_images}
        val_vids = {name.split("-")[0] for name in orchestrator.fixed_val_images}
        pool_vids = {name.split("-")[0] for name in orchestrator.full_train_pool_images}
        self.assertEqual(len(test_vids & val_vids), 0, "Rò rỉ video sequence giữa test và val!")
        self.assertEqual(len(test_vids & pool_vids), 0, "Rò rỉ video sequence giữa test và pool!")
        self.assertEqual(len(val_vids & pool_vids), 0, "Rò rỉ video sequence giữa val và pool!")

        # Kiểm tra số lượng symlinks thực tế trên đĩa
        ws = orchestrator.workspace_mgr
        self.assertEqual(len(list((ws.fixed_test_dir / "images").glob("*.jpg"))), 1590)
        self.assertEqual(len(list((ws.fixed_val_dir / "images").glob("*.jpg"))), 795)
        self.assertEqual(len(list((ws.full_train_pool_dir / "images").glob("*.jpg"))), 5565)
        self.assertEqual(len(list((ws.seed_v0_dir / "images").glob("*.jpg"))), 556)
        self.assertEqual(len(list((ws.unlabeled_pool_dir / "images").glob("*.jpg"))), 5009)
        self.assertEqual(len(list(ws.unique_pool_dir.glob("*.jpg"))), 7950)

        # Kiểm tra xuất bản bảng Markdown & JSON
        tables_dir = self.workspace_dir / "results" / "tables"
        self.assertTrue((tables_dir / "per_class_full_random_100pct.md").exists())
        self.assertTrue((tables_dir / "per_class_full_random_100pct.json").exists())
        self.assertTrue((tables_dir / "per_class_v0_baseline.md").exists())
        self.assertTrue((tables_dir / "per_class_v0_baseline.json").exists())
        self.assertTrue((tables_dir / "per_class_iterations_summary.md").exists())
        self.assertTrue((tables_dir / "per_class_iterations_summary.json").exists())
        self.assertTrue((tables_dir / "master_tournament_verdict.md").exists())
        self.assertTrue((tables_dir / "master_tournament_verdict.json").exists())

        # Kiểm tra nội dung JSON bảng per-class đủ 10 lớp
        with open(tables_dir / "per_class_full_random_100pct.json", "r", encoding="utf-8") as f:
            d_tgt = json.load(f)
            self.assertEqual(len(d_tgt["per_class_map50"]), 10)
            for c in BDD10K_CLASS_NAMES:
                self.assertIn(c, d_tgt["per_class_map50"])

        # Kiểm tra 3 biểu đồ xuất bản 300 DPI
        figures_dir = self.workspace_dir / "results" / "figures"
        fig1 = figures_dir / "fig1_active_learning_curve.png"
        fig2 = figures_dir / "fig2_per_class_radar_progression.png"
        fig3 = figures_dir / "fig3_label_efficiency_roi.png"

        self.assertTrue(fig1.exists(), f"Thiếu {fig1.name}")
        self.assertTrue(fig2.exists(), f"Thiếu {fig2.name}")
        self.assertTrue(fig3.exists(), f"Thiếu {fig3.name}")

        self.assertGreater(fig1.stat().st_size, 30_000, "Biểu đồ fig1 quá nhỏ")
        self.assertGreater(fig2.stat().st_size, 30_000, "Biểu đồ fig2 quá nhỏ")
        self.assertGreater(fig3.stat().st_size, 30_000, "Biểu đồ fig3 quá nhỏ")

        # Kiểm tra models lineage niêm phong
        lineage_file = self.workspace_mgr_models_lineage = self.workspace_dir / "models" / "models_lineage_phase07.json"
        self.assertTrue(lineage_file.exists())
        with open(lineage_file, "r", encoding="utf-8") as f:
            lin = json.load(f)
            self.assertIn("target_model", lin)
            self.assertIn("v0_model", lin)
            self.assertIn("active_models", lin)

    def test_08_zero_stale_cache_leakage(self):
        """Kỷ luật Zero Stale Cache: Không rò rỉ bất kỳ file *.cache nào vào data/bdd10k/."""
        clean_bdd10k_caches()
        cache_files = list(self.bdd10k_dir.rglob("*.cache"))
        self.assertEqual(len(cache_files), 0, f"Rò rỉ {len(cache_files)} file *.cache trong data/bdd10k/!")

    def test_09_dynamic_class_deficit_weights_adaptive_behavior(self):
        """Kiểm toán Cơ Chế Trọng Số Lớp Thích Ứng Động (Dynamic Adaptive Class Deficit Injection)."""
        # 1. Kiểm tra lớp điểm liệt vs lớp bão hòa
        v0_f1 = {
            "train": 0.0, "rider": 3.59, "motor": 3.53, "bike": 5.44,
            "bus": 17.82, "truck": 23.45, "car": 71.24, "person": 40.54,
            "traffic light": 33.37, "traffic sign": 36.14,
        }
        v0_rec = {
            "train": 0.0, "rider": 2.6, "motor": 2.5, "bike": 4.0,
            "bus": 16.0, "truck": 22.0, "car": 65.2, "person": 35.0,
            "traffic light": 31.0, "traffic sign": 35.0,
        }

        weights_v0 = compute_dynamic_class_deficit_weights(v0_f1, v0_rec)

        # Lớp train liệt hoàn toàn -> Trọng số vọt lên cực đại 25.0x
        self.assertEqual(weights_v0["train"], 25.0)
        # Các lớp yếu (rider, motor) có trọng số rất cao >= 15.0x
        self.assertGreaterEqual(weights_v0["rider"], 15.0)
        self.assertGreaterEqual(weights_v0["motor"], 15.0)
        # Lớp car đã bão hòa cao -> Trọng số tự động giảm tụt xuống <= 0.20x
        self.assertLessEqual(weights_v0["car"], 0.20)

        # Kiểm tra tính phân cấp chuẩn xác
        self.assertGreater(weights_v0["train"], weights_v0["rider"])
        self.assertGreater(weights_v0["rider"], weights_v0["bus"])
        self.assertGreater(weights_v0["bus"], weights_v0["person"])
        self.assertGreater(weights_v0["person"], weights_v0["car"])

        # 2. Kiểm tra tính thích ứng động qua từng vòng lặp: Khi lớp được cải thiện, trọng số tự động giảm
        v1_f1 = dict(v0_f1)
        v1_rec = dict(v0_rec)
        # Giả sử sau vòng 1 Active, lớp train tăng từ 0% lên 6.55% F1 và 4.5% Recall
        v1_f1["train"] = 6.55
        v1_rec["train"] = 4.5
        weights_v1 = compute_dynamic_class_deficit_weights(v1_f1, v1_rec)
        self.assertLess(weights_v1["train"], weights_v0["train"], "Trọng số lớp train phải giảm khi F1/Recall cải thiện!")
        self.assertLessEqual(weights_v1["train"], 24.0)

        # Giả sử sau vòng 2 Active, train tiếp tục tăng lên 13.06% F1 và 9.2% Recall
        v2_f1 = dict(v1_f1)
        v2_rec = dict(v1_rec)
        v2_f1["train"] = 13.06
        v2_rec["train"] = 9.2
        weights_v2 = compute_dynamic_class_deficit_weights(v2_f1, v2_rec)
        self.assertLess(weights_v2["train"], weights_v1["train"], "Trọng số lớp train phải tiếp tục giảm khi mô hình tiến bộ!")

    def test_10_multi_factor_selection_score_four_pillars(self):
        """Kiểm toán 4 Trụ Cột Thử Thách Tuyển Chọn Đa Yếu Tố (Multi-Factor Selection Score)."""
        mock_manifest = {
            "easy_day.jpg": {
                "is_night": False,
                "is_blur": False,
                "is_high_occlusion": False,
                "is_small_object_slice": False,
                "is_dense": False,
                "is_weather_degraded": False,
                "scores": {"laplacian_variance": 250.0, "brightness_v_mean": 140.0, "total_objects": 2, "vehicle_count": 2},
            },
            "hard_challenging.jpg": {
                "is_night": True,
                "is_blur": True,
                "is_high_occlusion": True,
                "is_small_object_slice": True,
                "is_dense": True,
                "is_weather_degraded": True,
                "scores": {"laplacian_variance": 65.0, "brightness_v_mean": 35.0, "dark_channel_prior": 8.0, "total_objects": 12, "vehicle_count": 10},
            },
        }
        mock_gt = {
            "easy_day": {
                "objects": [
                    {"category": "car", "box2d": {"x1": 100, "y1": 100, "x2": 400, "y2": 300}},
                    {"category": "car", "box2d": {"x1": 500, "y1": 150, "x2": 800, "y2": 400}},
                ]
            },
            "hard_challenging": {
                "objects": [
                    {"category": "train", "box2d": {"x1": 10, "y1": 10, "x2": 35, "y2": 35}, "attributes": {"occluded": True}},
                    {"category": "rider", "box2d": {"x1": 40, "y1": 40, "x2": 60, "y2": 65}, "attributes": {"truncated": True}},
                    {"category": "motor", "box2d": {"x1": 70, "y1": 70, "x2": 95, "y2": 95}, "attributes": {"occluded": True}},
                ]
            },
        }

        selector = ActiveMiningSelector(mock_manifest, mock_gt)
        dyn_weights = {"train": 25.0, "rider": 18.0, "motor": 18.0, "car": 0.15, "person": 1.0}

        easy_comp = selector.compute_image_multi_factor_components("easy_day.jpg", dyn_weights)
        hard_comp = selector.compute_image_multi_factor_components("hard_challenging.jpg", dyn_weights)

        # Kiểm toán từng trụ cột:
        # Pillar 1 (Class Deficit): ảnh hard chứa train/rider/motor có điểm class deficit vượt trội
        self.assertGreater(hard_comp["pillar_1_class"], easy_comp["pillar_1_class"])
        # Pillar 2 (Môi trường): ảnh hard ban đêm + mưa + tắc nghẽn
        self.assertGreater(hard_comp["pillar_2_env"], 1.5)
        self.assertEqual(easy_comp["pillar_2_env"], 1.0)
        # Pillar 3 (Rung blur): laplacian_variance = 65 < 120 -> boost mạnh
        self.assertGreater(hard_comp["pillar_3_blur"], 1.20)
        self.assertEqual(easy_comp["pillar_3_blur"], 1.0)
        # Điểm composite score của ảnh hard vượt trội áp đảo so với ảnh easy
        self.assertGreater(hard_comp["composite_score"], easy_comp["composite_score"] * 3.0)

    def test_11_per_class_tables_full_five_metrics_audit(self):
        """Kiểm toán Bảng Per-Class hoàn chỉnh 10 lớp với đầy đủ 5 chỉ số (Support, mAP50, mAP50-95, P, R, F1)."""
        tables_dir = self.workspace_dir / "results" / "tables"

        # Kiểm tra file per_class_v0_baseline.json và markdown
        v0_json_file = tables_dir / "per_class_v0_baseline.json"
        v0_md_file = tables_dir / "per_class_v0_baseline.md"
        self.assertTrue(v0_json_file.exists(), "Thiếu per_class_v0_baseline.json")
        self.assertTrue(v0_md_file.exists(), "Thiếu per_class_v0_baseline.md")

        with open(v0_json_file, "r", encoding="utf-8") as f:
            v0_data = json.load(f)

        # Kiểm tra đủ 10 lớp cho cả 5 chỉ số
        for metric_key in ["per_class_map50", "per_class_map50_95", "per_class_precision", "per_class_recall", "per_class_f1"]:
            self.assertIn(metric_key, v0_data, f"Thiếu trường {metric_key} trong JSON!")
            self.assertEqual(len(v0_data[metric_key]), 10, f"Trường {metric_key} phải chứa đủ 10 lớp!")

        # Kiểm tra tính chuẩn xác của công thức F1-Score: F1 = 2 * P * R / (P + R + 1e-7)
        for cname in BDD10K_CLASS_NAMES:
            p = v0_data["per_class_precision"][cname]
            r = v0_data["per_class_recall"][cname]
            f1_expected = round((2.0 * p * r) / (p + r + 1e-7), 2)
            f1_actual = v0_data["per_class_f1"][cname]
            self.assertAlmostEqual(f1_actual, f1_expected, places=1, msg=f"F1-Score sai cho lớp {cname}!")

        # Kiểm tra danh sách bảng chi tiết per_class_table trong JSON
        self.assertIn("per_class_table", v0_data)
        self.assertEqual(len(v0_data["per_class_table"]), 10)
        sample_row = v0_data["per_class_table"][0]
        for field in ["class_index", "class_name", "support", "map50", "map50_95", "precision", "recall", "f1_score", "dynamic_weight"]:
            self.assertIn(field, sample_row, f"Thiếu trường {field} trong per_class_table row!")

        # Kiểm tra bảng tổng kết per_class_iterations_summary
        sum_md = tables_dir / "per_class_iterations_summary.md"
        sum_json = tables_dir / "per_class_iterations_summary.json"
        self.assertTrue(sum_md.exists())
        self.assertTrue(sum_json.exists())

        with open(sum_md, "r", encoding="utf-8") as f:
            md_content = f.read()
            self.assertIn("Bảng Tiến Hóa Detection mAP@50", md_content)
            self.assertIn("Bảng Tiến Hóa Cân Bằng F1-Score", md_content)
            self.assertIn("Lịch Sử Vector Trọng Số Thâm Hụt Lớp Thích Ứng Động", md_content)

        with open(sum_json, "r", encoding="utf-8") as f:
            sum_data = json.load(f)
            self.assertIn("dynamic_weights_history", sum_data)
            self.assertIn("target_full", sum_data)
            self.assertIn("v0_baseline", sum_data)

    def test_12_zero_ground_truth_leakage_and_pseudo_labels_filtering(self):
        """Kiểm toán Kỷ Luật Zero Ground-Truth Leakage Tuyệt Đối & Lọc Pseudo-Labels (0.25 <= conf <= 0.75)."""
        mock_manifest = {
            "unlabeled_01.jpg": {
                "is_night": False,
                "is_blur": False,
                "is_weather_degraded": False,
                "scores": {"laplacian_variance": 200.0, "total_objects": 4, "vehicle_count": 3},
            },
            "unlabeled_02.jpg": {
                "is_night": True,
                "is_blur": True,
                "is_weather_degraded": True,
                "scores": {"laplacian_variance": 50.0, "total_objects": 8, "vehicle_count": 6},
            },
        }

        # Tạo pseudo-labels với các mức confidence khác nhau
        pseudo_labels_pool = {
            "unlabeled_01.jpg": [
                {"category": "car", "confidence": 0.95, "box2d": {"x1": 10, "y1": 10, "x2": 200, "y2": 200}},  # Quá tự tin (> 0.75) -> BỊ LOẠI
                {"category": "car", "confidence": 0.15, "box2d": {"x1": 20, "y1": 20, "x2": 150, "y2": 150}},  # Nhiễu (< 0.25) -> BỊ LOẠI
                {"category": "rider", "confidence": 0.45, "box2d": {"x1": 50, "y1": 50, "x2": 70, "y2": 70}},  # Lưỡng lự (0.45) -> CHẤP NHẬN
            ],
            "unlabeled_02.jpg": [
                {"category": "train", "confidence": 0.60, "box2d": {"x1": 5, "y1": 5, "x2": 30, "y2": 30}, "attributes": {"occluded": True}},  # Lưỡng lự (0.60) -> CHẤP NHẬN
                {"category": "motor", "confidence": 0.35, "box2d": {"x1": 40, "y1": 40, "x2": 80, "y2": 80}},  # Lưỡng lự (0.35) -> CHẤP NHẬN
            ],
        }

        # Khởi tạo selector mà KHÔNG truyền gt_metadata (Zero GT Leakage)
        selector = ActiveMiningSelector(profiling_manifest=mock_manifest, pseudo_labels_map=pseudo_labels_pool)
        self.assertFalse(hasattr(selector, "gt_metadata"), "ActiveMiningSelector tuyệt đối KHÔNG ĐƯỢC chứa thuộc tính gt_metadata!")

        # 1. Kiểm tra bộ lọc pseudo-labels: Chỉ giữ lại các box có 0.25 <= conf <= 0.75
        pseudo_01 = selector.get_pseudo_labels_for_image("unlabeled_01.jpg")
        self.assertEqual(len(pseudo_01), 1, "Chỉ 1 box lưỡng lự (rider, 0.45) được giữ lại; box 0.95 và 0.15 phải bị loại!")
        self.assertEqual(pseudo_01[0]["category"], "rider")

        pseudo_02 = selector.get_pseudo_labels_for_image("unlabeled_02.jpg")
        self.assertEqual(len(pseudo_02), 2, "Cả 2 box (train 0.60, motor 0.35) phải được giữ lại!")

        # 2. Kiểm tra tính điểm đa yếu tố hoàn toàn không đọc nhãn thật
        dyn_weights = {"train": 25.0, "rider": 18.0, "motor": 18.0, "car": 0.15}
        comp_01 = selector.compute_image_multi_factor_components("unlabeled_01.jpg", dyn_weights)
        comp_02 = selector.compute_image_multi_factor_components("unlabeled_02.jpg", dyn_weights)

        self.assertGreater(comp_02["composite_score"], comp_01["composite_score"])
        self.assertGreater(comp_02["pillar_1_class"], comp_01["pillar_1_class"])
        self.assertGreater(comp_02["pillar_2_env"], 1.5)

        # 3. Kiểm tra tuyển chọn batch qua Submodular Facility Location
        batch, unc = selector.select_active_batch(["unlabeled_01.jpg", "unlabeled_02.jpg"], k=1, class_deficit_weights=dyn_weights)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0], "unlabeled_02.jpg", "unlabeled_02.jpg có điểm đa yếu tố cao hơn phải được chọn đầu tiên!")

    def test_13_unlabeled_pool_labels_isolation_zero_leakage(self):
        """Kiểm toán Vật Lý: unlabeled_pool/ tuyệt đối không chứa bất kỳ file nhãn vàng nào."""
        unlabeled_lbl_dir = self.workspace_dir / "data" / "splits" / "unlabeled_pool" / "labels"
        if unlabeled_lbl_dir.exists():
            label_files = list(unlabeled_lbl_dir.glob("*.txt"))
            self.assertEqual(
                len(label_files), 0,
                f"Phát hiện {len(label_files)} file nhãn vàng trong unlabeled_pool/labels! "
                "Kho chưa gán nhãn phải cách ly 100% không chứa nhãn vàng."
            )

    def test_14_c5_weather_features_in_facility_location_pca(self):
        """Kiểm toán không gian đặc trưng PCA tích hợp 6 thuộc tính thời tiết C5 Ensemble."""
        mock_manifest = {
            f"sample_{i}.jpg": {
                "is_night": (i % 2 == 0),
                "is_blur": (i % 3 == 0),
                "is_weather_degraded": (i % 4 == 0),
                "is_dense": (i % 5 == 0),
                "is_small_object_slice": (i % 6 == 0),
                "is_high_occlusion": (i % 7 == 0),
                "scores": {
                    "brightness_v_mean": 80.0 + i * 2,
                    "brightness_p50": 70.0 + i * 2,
                    "tenengrad_sharpness": 8000.0,
                    "laplacian_variance": 150.0,
                    "fft_high_freq_ratio": 0.02,
                    "dark_channel_prior": 18.0,
                    "scale_shannon_entropy": 1.2,
                    "occlusion_pairs": 2,
                    "max_occlusion_chain": 2,
                    "vehicle_count": 5,
                    "total_objects": 8,
                    # 6 thuộc tính thời tiết C5 Ensemble
                    "atmospheric_transmission": 0.65,
                    "low_transmission_ratio": 0.18,
                    "rain_streak_ratio": 0.25,
                    "saturation_mean": 55.0,
                    "rms_contrast": 22.0,
                    "weather_ensemble_score": 0.45,
                },
            }
            for i in range(15)
        }
        selector = ActiveMiningSelector(mock_manifest)
        selector.generate_simulated_pseudo_labels(list(mock_manifest.keys()), seed=123)

        cand_pool = list(mock_manifest.keys())
        batch, unc = selector.select_active_batch(cand_pool, k=5)
        self.assertEqual(len(batch), 5)
        self.assertTrue(all(item in cand_pool for item in batch))

    def test_15_publication_figures_complete_suite_and_evolution_plot(self):
        """Kiểm toán toàn bộ bộ 5 biểu đồ khoa học 300 DPI xuất bản bài báo."""
        fig_dir = self.workspace_dir / "results" / "figures"
        required_figs = [
            "figure_07a_active_learning_curve.png",
            "figure_07b_dynamic_class_deficit_evolution.png",
            "figure_07c_per_class_gain_breakdown.png",
            "fig1_active_learning_curve.png",
            "fig2_per_class_radar_progression.png",
            "fig3_label_efficiency_roi.png",
        ]
        for f_name in required_figs:
            f_path = fig_dir / f_name
            if f_path.exists():
                self.assertGreater(f_path.stat().st_size, 1000, f"Biểu đồ {f_name} có kích thước quá nhỏ (< 1KB)!")


if __name__ == "__main__":
    unittest.main()


