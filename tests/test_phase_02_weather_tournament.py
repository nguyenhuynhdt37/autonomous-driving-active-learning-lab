"""
test_phase_02_weather_tournament.py - Unit Test Bộ Phân Loại Thời Tiết Suy Giảm Phase 02
========================================================================================
Kiểm toán chuyên sâu 5 phương pháp phân loại thời tiết:
- C1: Pure Dark Channel Prior (DCP)
- C2: Normalized Atmospheric Transmission Map (He et al.) kết hợp Sky Region Masking
- C3: Directional Rain-Streak Gradient Ratio (Sobel 75°-90°)
- C4: Color Saturation Suppression & Contrast Degradation
- C5: Multi-Domain Weather Ensemble Quán quân (Clear Day Guard)
- Đo đạc đầy đủ các chỉ số APA/IEEE: Accuracy, Precision, Recall, F1-Score, Jaccard (IoU).
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

from src.profiling import (
    compute_dark_channel_prior,
    compute_atmospheric_transmission_map,
    compute_directional_rain_streak,
    compute_color_contrast_degradation,
    classify_weather_ensemble,
    evaluate_binary_metric,
    run_weather_tournament_benchmark,
    HybridEnsembleProfiler,
)


class TestPhase02WeatherTournament(unittest.TestCase):
    """Kiểm toán toàn diện các hàm phân loại thời tiết suy giảm."""

    def setUp(self):
        # 1. Ảnh giả lập ban ngày trong xanh rực rỡ (Clear Day: Bầu trời xanh, bão hòa cao, tương phản mạnh)
        self.clear_day_img = np.zeros((200, 300, 3), dtype=np.uint8)
        # Nửa trên là bầu trời xanh sáng
        self.clear_day_img[:80, :] = [220, 180, 100]  # BGR
        # Nửa dưới là mặt cỏ xanh và đường sá có độ bão hòa cao
        self.clear_day_img[80:, :] = [40, 160, 40]
        # Thêm vài khối có viền tương phản cao
        cv2.rectangle(self.clear_day_img, (50, 100), (120, 180), (0, 0, 240), -1)

        # 2. Ảnh giả lập sương mù dày đặc (Foggy: độ chói phẳng, độ bão hòa rất thấp, tán xạ mạnh)
        self.foggy_img = np.full((200, 300, 3), 190, dtype=np.uint8)
        # Thêm nhiễu nhẹ
        noise = np.random.normal(0, 5, (200, 300, 3)).astype(np.int16)
        self.foggy_img = np.clip(self.foggy_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 3. Ảnh giả lập trời mưa có vệt mưa dốc đứng (Rain Streaks: các đường kẻ mảnh góc 85°)
        self.rainy_img = np.full((200, 300, 3), 100, dtype=np.uint8)
        for x in range(10, 290, 15):
            cv2.line(self.rainy_img, (x, 10), (x + 5, 190), (230, 230, 230), 1)

    def test_01_compute_dark_channel_prior(self):
        """Kiểm toán C1: Pure Dark Channel Prior."""
        dcp_clear = compute_dark_channel_prior(self.clear_day_img, patch_size=15)
        dcp_fog = compute_dark_channel_prior(self.foggy_img, patch_size=15)

        self.assertIsInstance(dcp_clear, float)
        self.assertGreaterEqual(dcp_clear, 0.0)
        self.assertGreater(dcp_fog, dcp_clear, "Dark Channel của sương mù phải cao hơn nhiều so với ngày quang!")

    def test_02_compute_atmospheric_transmission_map(self):
        """Kiểm toán C2: Bản đồ truyền suốt khí quyển He et al. & Sky Masking."""
        res_fog = compute_atmospheric_transmission_map(self.foggy_img, omega=0.95, patch_size=15)
        
        self.assertIn("atmospheric_light", res_fog)
        self.assertIn("mean_transmission", res_fog)
        self.assertIn("low_transmission_ratio", res_fog)
        self.assertIn("is_degraded", res_fog)

        # Ảnh sương mù phải có transmission thấp hoặc tỷ lệ low_trans cao
        self.assertLessEqual(res_fog["mean_transmission"], 0.70)
        self.assertTrue(res_fog["is_degraded"])

    def test_03_compute_directional_rain_streak(self):
        """Kiểm toán C3: Tỷ lệ vệt mưa dốc đứng Sobel 75°-90°."""
        gray_rain = cv2.cvtColor(self.rainy_img, cv2.COLOR_BGR2GRAY)
        res_rain = compute_directional_rain_streak(gray_rain, min_grad_mag=20.0, angle_min=75.0)

        self.assertIn("rain_streak_ratio", res_rain)
        self.assertIn("vertical_edge_density", res_rain)
        self.assertIn("is_degraded", res_rain)
        self.assertGreater(res_rain["rain_streak_ratio"], 0.35, "Vệt mưa dốc đứng phải có tỷ lệ gradient cao!")

    def test_04_compute_color_contrast_degradation(self):
        """Kiểm toán C4: Suy giảm độ bão hòa màu và độ tương phản RMS."""
        res_fog = compute_color_contrast_degradation(self.foggy_img, sat_thresh=65.0, contrast_thresh=55.0)
        
        self.assertIn("sat_mean", res_fog)
        self.assertIn("rms_contrast", res_fog)
        self.assertIn("is_degraded", res_fog)
        # Sương mù có độ bão hòa màu cực thấp
        self.assertLess(res_fog["sat_mean"], 30.0)

    def test_05_clear_day_guard_mechanism(self):
        """Kiểm toán C5: Cơ chế Clear Day Guard loại trừ 100% ảnh trời quang ban ngày rực rỡ."""
        is_degraded, scores = classify_weather_ensemble(self.clear_day_img)

        self.assertFalse(is_degraded, "Ảnh ban ngày trời trong sắc nét bão hòa cao phải được Clear Day Guard loại trừ!")
        self.assertIn("is_weather_ensemble", scores)
        self.assertIn("weather_ensemble_score", scores)
        self.assertIn("is_weather_c1_dcp", scores)
        self.assertIn("is_weather_c2_trans", scores)
        self.assertIn("is_weather_c3_rain", scores)
        self.assertIn("is_weather_c4_color", scores)

    def test_06_evaluate_binary_metric_jaccard_iou(self):
        """Kiểm toán thước đo nhị phân mở rộng gồm Jaccard (IoU)."""
        y_true = [True, True, False, False, True]
        y_pred = [True, False, False, True, True]

        metrics = evaluate_binary_metric(y_true, y_pred)
        # TP=2, FP=1, FN=1, TN=1
        # Accuracy = 3/5 = 60.0%
        # Precision = 2/3 = 66.67%
        # Recall = 2/3 = 66.67%
        # Jaccard = 2 / (2 + 1 + 1) = 2/4 = 50.0%
        self.assertEqual(metrics["TP"], 2)
        self.assertEqual(metrics["FP"], 1)
        self.assertEqual(metrics["FN"], 1)
        self.assertEqual(metrics["TN"], 1)
        self.assertAlmostEqual(metrics["Accuracy"], 60.0, places=1)
        self.assertAlmostEqual(metrics["Precision"], 66.67, places=1)
        self.assertAlmostEqual(metrics["Recall"], 66.67, places=1)
        self.assertAlmostEqual(metrics["F1_Score"], 66.67, places=1)
        self.assertAlmostEqual(metrics["Jaccard"], 50.0, places=1)

    def test_07_hybrid_ensemble_profiler_weather_slice(self):
        """Kiểm toán HybridEnsembleProfiler gán đúng cờ và scores thời tiết."""
        profiler = HybridEnsembleProfiler()
        # Lưu tạm ảnh ra file scratch
        test_dir = Path("paper_v4_10k/scripts")
        test_file = test_dir / "temp_test_profiler.jpg"
        cv2.imwrite(str(test_file), self.clear_day_img)

        try:
            record = profiler.profile_image(test_file)
            self.assertIn("is_weather_degraded", record)
            scores = record["scores"]
            self.assertIn("dark_channel_prior", scores)
            self.assertIn("atmospheric_transmission", scores)
            self.assertIn("rain_streak_ratio", scores)
            self.assertIn("saturation_mean", scores)
            self.assertIn("rms_contrast", scores)
            self.assertIn("weather_ensemble_score", scores)
        finally:
            if test_file.exists():
                test_file.unlink()


if __name__ == "__main__":
    unittest.main()
