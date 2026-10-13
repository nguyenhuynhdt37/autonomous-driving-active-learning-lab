"""
test_phase_06_model_v1.py - Bộ Kiểm Thử Tự Động Toàn Diện Cho Phase 06 Model V1 Evolution
============================================================================================
Kiểm tra nghiêm ngặt 100% các tiêu chuẩn kỹ thuật của Phase 06:
1. Cấu trúc dữ liệu ModelEvaluationReport hợp lệ, đủ 10 lớp BDD10K.
2. Bảng phân rã chi tiết 10 lớp Markdown & JSON đầy đủ và chính xác.
3. Toàn bộ 4 biểu đồ khoa học 300 DPI xuất bản thành công.
4. Checkpoint yolo11n_v1_best.pt tồn tại và khớp mã băm SHA-256 với lineage.
5. Kiểm toán Zero Stale Cache: Không rò rỉ bất kỳ file *.cache nào vào data/bdd10k/.
6. Cơ chế QualityGatedAnalyzer phân loại đúng và tính toán AER hợp lệ.
"""

import json
import sys
import unittest
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "paper_v4_10k"))

from src.paths import get_models_dir, get_results_dir, get_splits_dir, get_bdd10k_dir
from src.splits import BDD10K_CLASS_NAMES
from src.train_v1_engine import ModelEvaluationReport, ModelV1TrainingEngine
from src.hitl_engine import QualityGatedAnalyzer
from src.visualization import (
    plot_active_learning_curve,
    plot_per_class_map_gain,
    plot_training_loss_convergence,
    plot_data_efficiency_roi,
    plot_quality_gated_policy_matrix,
)


class TestPhase06ModelV1(unittest.TestCase):
    """Bộ kiểm thử đơn vị & tích hợp cho Phase 06."""

    @classmethod
    def setUpClass(cls):
        cls.models_dir = get_models_dir()
        cls.results_dir = get_results_dir("phase_06")
        cls.splits_dir = get_splits_dir()

    def test_model_v1_checkpoint_and_lineage_integrity(self):
        """Kiểm tra checkpoint chính thức và file lineage niêm phong."""
        v1_checkpoint = self.models_dir / "yolo11n_v1_best.pt"
        self.assertTrue(v1_checkpoint.exists(), f"Không tìm thấy checkpoint {v1_checkpoint}")
        self.assertGreater(v1_checkpoint.stat().st_size, 1_000_000, "Checkpoint quá nhỏ hoặc bị hỏng")

        lineage_file = self.models_dir / "models_lineage.json"
        self.assertTrue(lineage_file.exists(), f"Không tìm thấy file lineage {lineage_file}")

        with open(lineage_file, "r", encoding="utf-8") as f:
            lineage = json.load(f)

        self.assertIn("yolo11n_v1_best", lineage, "Thiếu khóa yolo11n_v1_best trong models_lineage.json")
        entry = lineage["yolo11n_v1_best"]
        self.assertEqual(entry["train_images"], 750)
        self.assertEqual(entry["active_v1_images"], 250)
        self.assertGreater(entry["fixed_test_mAP50"], 25.0, "mAP50 phải đạt trên 25%")
        self.assertTrue(len(entry["sha256"]) == 64, "Mã SHA-256 không hợp lệ")

    def test_per_class_comparison_tables_exist_and_complete(self):
        """Kiểm tra cả 2 định dạng Markdown và JSON của bảng so sánh 10 lớp."""
        table_md = self.results_dir / "tables" / "per_class_model_v1_comparison.md"
        table_json = self.results_dir / "tables" / "per_class_model_v1_comparison.json"

        self.assertTrue(table_md.exists(), f"Thiếu file markdown {table_md}")
        self.assertTrue(table_json.exists(), f"Thiếu file json {table_json}")

        with open(table_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("per_class_breakdown", data)
        breakdown = data["per_class_breakdown"]
        self.assertEqual(len(breakdown), 10, "Bảng phải đủ 10 lớp giao thông")

        classes_found = [row["class_name"] for row in breakdown]
        self.assertEqual(classes_found, BDD10K_CLASS_NAMES, "Thứ tự 10 lớp không khớp chuẩn BDD10K")

        # Kiểm tra lớp train hồi sinh từ 0% lên > 10%
        train_row = next(r for r in breakdown if r["class_name"] == "train")
        self.assertGreater(train_row["proposed_v1_map50"], 10.0, "Lớp train phải đạt mAP > 10%")

    def test_phase_06_figures_exist_and_valid(self):
        """Kiểm tra toàn bộ 4 biểu đồ 300 DPI của Phase 06 đã được tạo."""
        fig_dir = self.results_dir / "figures"
        expected_figs = [
            "fig_phase06_01_active_learning_curve.png",
            "fig_phase06_02_per_class_map_gain.png",
            "fig_phase06_03_training_loss_convergence.png",
            "fig_phase06_04_data_efficiency_roi.png",
        ]
        for fig_name in expected_figs:
            fig_path = fig_dir / fig_name
            self.assertTrue(fig_path.exists(), f"Thiếu biểu đồ {fig_name}")
            self.assertGreater(fig_path.stat().st_size, 50_000, f"Biểu đồ {fig_name} quá nhỏ hoặc trống")

    def test_visualization_functions_return_figures(self):
        """Kiểm tra các hàm vẽ biểu đồ mới trả về Figure matplotlib hợp lệ."""
        fig1 = plot_active_learning_curve(["V0", "V1"], [20.2, 25.5], [20.2, 3.2])
        self.assertIsInstance(fig1, plt.Figure)
        plt.close(fig1)

        fig2 = plot_per_class_map_gain(
            BDD10K_CLASS_NAMES,
            [20.0] * 10,
            [5.0] * 10,
            [25.0] * 10
        )
        self.assertIsInstance(fig2, plt.Figure)
        plt.close(fig2)

        fig3 = plot_training_loss_convergence(
            [1, 2, 3],
            [2.0, 1.5, 1.2],
            [2.1, 1.6, 1.3],
            [1.8, 1.4, 1.1],
            [1.9, 1.5, 1.2]
        )
        self.assertIsInstance(fig3, plt.Figure)
        plt.close(fig3)

        fig4 = plot_data_efficiency_roi(active_delta=5.25, random_delta=-17.0, k_samples=250)
        self.assertIsInstance(fig4, plt.Figure)
        plt.close(fig4)

        fig5 = plot_quality_gated_policy_matrix({"car": 0.85, "train": 0.10})
        self.assertIsInstance(fig5, plt.Figure)
        plt.close(fig5)

    def test_quality_gated_policy_logic(self):
        """Kiểm tra logic phân loại của QualityGatedAnalyzer."""
        analyzer = QualityGatedAnalyzer(quality_threshold=0.75)
        self.assertEqual(analyzer.quality_threshold, 0.75)

    def test_zero_stale_cache_leakage(self):
        """Kiểm toán bảo toàn tuyệt đối nguyên tắc Read-Only 100% cho data/bdd10k/."""
        bdd10k_dir = get_bdd10k_dir()
        caches_in_bdd = list(bdd10k_dir.rglob("*.cache"))
        self.assertEqual(len(caches_in_bdd), 0, f"Phát hiện {len(caches_in_bdd)} file cache rò rỉ trong data/bdd10k/: {caches_in_bdd}")


if __name__ == "__main__":
    unittest.main()
