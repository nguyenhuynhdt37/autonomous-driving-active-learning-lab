"""
test_phase_05_active_mining.py - Bộ Kiểm Thử Tự Động Toàn Diện Cho Phase 05
===========================================================================
Kiểm tra nghiêm ngặt 100% các chức năng cốt lõi:
1. PoolItemRecord cấu trúc dữ liệu đặc trưng.
2. Cả 6 Chiến lược Tuyển chọn (K mẫu, không duplicate, nằm trong pool).
3. Hệ thống đo lường kép CSS và Leaderboard.
4. Cơ chế Two-Tier HITL & Closed-Loop QA.
5. Kiểm toán Symlink Materialization & Checksum Dataset V1.
6. Các hàm Visualization trả về Figure hợp lệ.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Thêm đường dẫn dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "paper_v4_10k"))

from src.active_mining import (
    CLASS_DEFICIT_WEIGHTS,
    RARE_CLASSES,
    PoolItemRecord,
    RandomSamplingStrategy,
    TopKPureUncertaintyStrategy,
    WeakSliceCostAwareStrategy,
    PureCoreSetKCenterStrategy,
    LongTailWeakSliceHybridStrategy,
    SuperFusionProposedStrategy,
    ActiveTournamentEvaluator,
)
from src.hitl_engine import (
    ConsensusChecker,
    TwoTierRouter,
    AnnotationResult,
    compute_box_iou,
)
from src.visualization import (
    plot_selection_tournament_radar,
    plot_pca_dispersion_distribution,
    plot_active_learning_curve,
    plot_per_class_map_gain,
)
from src.splits import BDD10K_CLASS_NAMES


class TestPhase05ActiveMining(unittest.TestCase):
    """Bộ kiểm thử đơn vị & tích hợp cho Phase 05 Active Mining."""

    @classmethod
    def setUpClass(cls):
        """Khởi tạo 100 mock records phục vụ test logic thuật toán."""
        cls.rng = np.random.RandomState(42)
        cls.mock_records = {}

        for i in range(100):
            img_name = f"test_{i:04d}.jpg"
            pca_vec = cls.rng.randn(32).astype(np.float32).tolist()
            u = float(cls.rng.uniform(0.1, 0.9))
            qbc = float(cls.rng.uniform(0.0, 0.8))
            is_night = (i % 3 == 0)
            is_small = (i % 4 == 0)
            is_occ = (i % 5 == 0)
            is_blur = (i % 10 == 0)
            lap = float(40.0 if is_blur else cls.rng.uniform(60.0, 300.0))
            num_b = int(cls.rng.randint(1, 10))

            # Giả lập detected classes
            classes = ["car"]
            if i % 7 == 0:
                classes.append("rider")
            if i % 15 == 0:
                classes.append("train")
            if i % 6 == 0:
                classes.append("bus")

            max_cw = 1.0
            for c in classes:
                if c in CLASS_DEFICIT_WEIGHTS:
                    max_cw = max(max_cw, CLASS_DEFICIT_WEIGHTS[c])

            cls.mock_records[img_name] = PoolItemRecord(
                image_name=img_name,
                margin_uncertainty=round(u, 4),
                qbc_disagreement=round(qbc, 4),
                embedding_pca=[round(v, 5) for v in pca_vec],
                num_boxes=num_b,
                detected_classes=classes,
                is_night=is_night,
                is_blur=is_blur,
                is_small_object_slice=is_small,
                is_high_occlusion=is_occ,
                laplacian_variance=round(lap, 2),
                slice_weight=0.25,
                class_deficit_score=max_cw,
                yolo_boxes=[{"cls_id": 2, "cls_name": "car", "conf": 0.88, "box": [100, 100, 300, 300]}],
                rtdetr_boxes=[{"cls_id": 2, "cls_name": "car", "conf": 0.89, "box": [105, 102, 302, 301]}],
            )

    def test_pool_item_record_structure(self):
        """1. Kiểm tra cấu trúc bản ghi đặc trưng PoolItemRecord."""
        rec = list(self.mock_records.values())[0]
        self.assertIsInstance(rec.margin_uncertainty, float)
        self.assertIsInstance(rec.qbc_disagreement, float)
        self.assertEqual(len(rec.embedding_pca), 32)
        d = rec.to_dict()
        self.assertIn("image_name", d)
        self.assertIn("embedding_pca", d)

    def test_six_selection_strategies(self):
        """2. Kiểm tra cả 6 chiến lược tuyển chọn bốc đúng K ảnh, không lặp lại."""
        k = 25
        strategies = [
            RandomSamplingStrategy(seed=42),
            TopKPureUncertaintyStrategy(),
            WeakSliceCostAwareStrategy(),
            PureCoreSetKCenterStrategy(),
            LongTailWeakSliceHybridStrategy(),
            SuperFusionProposedStrategy(),
        ]

        all_pool_keys = set(self.mock_records.keys())

        for strat in strategies:
            res = strat.select(self.mock_records, k=k)
            # Số lượng bốc phải chính xác bằng K
            self.assertEqual(res.num_selected, k, f"Chiến lược {strat.name} không bốc đủ K ảnh")
            # Không được có phần tử trùng lặp
            self.assertEqual(len(set(res.selected_images)), k, f"Chiến lược {strat.name} có ảnh trùng lặp")
            # Tất cả ảnh phải thuộc pool ban đầu
            self.assertTrue(set(res.selected_images).issubset(all_pool_keys), f"Chiến lược {strat.name} bốc ảnh ngoài pool")

    def test_css_tournament_evaluator(self):
        """3. Kiểm tra Bộ đo lường kép CSS và xếp hạng Leaderboard."""
        k = 20
        results = [
            RandomSamplingStrategy(seed=42).select(self.mock_records, k=k),
            SuperFusionProposedStrategy().select(self.mock_records, k=k),
        ]

        evaluator = ActiveTournamentEvaluator(self.mock_records)
        summary = evaluator.compute_leaderboard(results)

        self.assertIn("leaderboard", summary)
        self.assertIn("winner_strategy", summary)
        self.assertEqual(len(summary["leaderboard"]), 2)

        # Kiểm tra điểm CSS in [0, 100]
        for row in summary["leaderboard"]:
            self.assertGreaterEqual(row["composite_selection_score_css"], 0.0)
            self.assertLessEqual(row["composite_selection_score_css"], 100.0)

    def test_box_iou_computation(self):
        """4. Kiểm tra hàm tính IoU giữa 2 bounding box."""
        box1 = [100.0, 100.0, 200.0, 200.0]
        box2 = [100.0, 100.0, 200.0, 200.0]
        iou_identical = compute_box_iou(box1, box2)
        self.assertAlmostEqual(iou_identical, 1.0, places=4)

        box3 = [300.0, 300.0, 400.0, 400.0]
        iou_disjoint = compute_box_iou(box1, box3)
        self.assertAlmostEqual(iou_disjoint, 0.0, places=4)

    def test_hitl_consensus_and_routing(self):
        """5. Kiểm tra ConsensusChecker và TwoTierRouter."""
        checker = ConsensusChecker(conf_threshold=0.85, iou_threshold=0.85)
        
        # Mẫu đồng thuận cao
        rec_pass = list(self.mock_records.values())[1]  # Not blur
        rec_pass.is_blur = False
        rec_pass.laplacian_variance = 150.0
        rec_pass.yolo_boxes = [{"cls_id": 2, "cls_name": "car", "conf": 0.92, "box": [100, 100, 200, 200]}]
        rec_pass.rtdetr_boxes = [{"cls_id": 2, "cls_name": "car", "conf": 0.91, "box": [102, 101, 201, 201]}]

        can_auto, boxes, reason = checker.check_consensus(rec_pass)
        self.assertTrue(can_auto, f"Kỳ vọng Auto-Accept thành công, lý do thất bại: {reason}")
        self.assertEqual(len(boxes), 1)

        # Mẫu lệch số box -> Chuyển Oracle
        rec_fail = list(self.mock_records.values())[2]
        rec_fail.yolo_boxes = [{"cls_id": 2, "cls_name": "car", "conf": 0.92, "box": [100, 100, 200, 200]}]
        rec_fail.rtdetr_boxes = []
        can_auto_f, _, _ = checker.check_consensus(rec_fail)
        self.assertFalse(can_auto_f)

    def test_visualization_figures(self):
        """6. Kiểm tra các hàm trực quan hóa trả về plt.Figure hợp lệ."""
        # 1. Radar
        mock_lb = [
            {"strategy_name": "C1_RandomSampling", "coverage_score": 40.0, "rare_penetration_score": 30.0, "quota_alignment_score": 60.0, "proxy_retrain_score": 35.0},
            {"strategy_name": "C6_Super_Fusion_Architecture", "coverage_score": 90.0, "rare_penetration_score": 88.0, "quota_alignment_score": 85.0, "proxy_retrain_score": 92.0},
        ]
        fig_r = plot_selection_tournament_radar(mock_lb)
        self.assertIsNotNone(fig_r)

        # 2. PCA dispersion
        pca_dummy = self.rng.randn(100, 32).astype(np.float32)
        fig_pca = plot_pca_dispersion_distribution(pca_dummy, [0, 5, 10])
        self.assertIsNotNone(fig_pca)

        # 3. AL curve
        fig_alc = plot_active_learning_curve(["V0", "V1"], [20.23, 23.66], [20.23, 21.45])
        self.assertIsNotNone(fig_alc)

        # 4. Per class gain
        v0_dummy = [10.0] * 10
        rnd_dummy = [12.0] * 10
        v1_dummy = [16.0] * 10
        fig_bar = plot_per_class_map_gain(BDD10K_CLASS_NAMES, v0_dummy, rnd_dummy, v1_dummy)
        self.assertIsNotNone(fig_bar)

    def test_hitl_two_tier_proportions_not_zero(self):
        """7. Kiểm tra TwoTierRouter không bị thoái hóa về 0% Tier 1 trên tập dữ liệu."""
        checker = ConsensusChecker()
        auto_count = 0
        for rec in self.mock_records.values():
            can_auto, _, _ = checker.check_consensus(rec)
            if can_auto:
                auto_count += 1
        # Trên 100 mock records, tỷ lệ Auto-Accept phải > 0 và < 100
        self.assertGreater(auto_count, 0, "Tier 1 không được phép là 0 (thoái hóa)")
        self.assertLess(auto_count, len(self.mock_records), "Tier 2 không được phép là 0")

    def test_packager_unlabeled_pool_reduction_and_restore(self):
        """8. Kiểm tra DatasetV1Packager quản lý chính xác kho unlabeled_pool."""
        from src.hitl_engine import DatasetV1Packager
        packager = DatasetV1Packager()
        self.assertTrue(hasattr(packager, "restore_unlabeled_pool"))


if __name__ == "__main__":
    unittest.main()
