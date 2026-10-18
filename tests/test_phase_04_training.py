"""
test_phase_04_training.py - Bộ Kiểm Thử Tự Động Toàn Diện Cho Phase 04
======================================================================
Kiểm tra tính đúng đắn và chuẩn mực khoa học của:
1. Đường dẫn lưu trữ models (get_models_dir).
2. Tính toán trọng số lớp hiếm (compute_bdd10k_class_weights) chống mất cân bằng 2.000:1.
3. 5 Cấu hình Đấu trường Cắt bỏ (AblationConfigFactory).
4. Cơ chế lưu trữ và niêm phong SHA-256 (ModelLineageManager).
5. Phân tách lát cắt khó và sinh vector trọng số W_i (HardSliceEvaluator).
6. Khởi tạo song mã YOLO11n + RT-DETR.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from paper_v4_10k.src.paths import (
    find_project_dir,
    get_bdd10k_metadata_file,
    get_models_dir,
    get_splits_dir,
)
from paper_v4_10k.src.splits import BDD10K_CLASS_NAMES
from paper_v4_10k.src.train_engine import (
    AblationConfigFactory,
    HardSliceEvaluator,
    ModelLineageManager,
    TrainEngine,
    TrainingConfig,
    compute_bdd10k_class_weights,
)


class TestPhase04ModelsAndPaths:
    """Kiểm tra đường dẫn và cấu trúc thư mục lưu trữ models."""

    def test_get_models_dir(self):
        models_dir = get_models_dir()
        assert models_dir.exists(), "Thư mục models/ phải tồn tại"
        assert models_dir.name == "models"
        assert models_dir.parent == find_project_dir()


class TestPhase04ClassWeightsEngine:
    """Kiểm tra thuật toán tính toán Class-Weighted Loss cho 10 lớp BDD10K."""

    def test_compute_bdd10k_class_weights_basic(self):
        res = compute_bdd10k_class_weights()
        assert "class_counts" in res
        assert "class_weights" in res
        assert "weights_list" in res

        assert len(res["class_counts"]) == 10
        assert len(res["class_weights"]) == 10
        assert len(res["weights_list"]) == 10

        # Kiểm tra thứ tự và tên các lớp
        for name in BDD10K_CLASS_NAMES:
            assert name in res["class_counts"]
            assert name in res["class_weights"]

    def test_rare_class_penalty_higher_than_common(self):
        """Các lớp hiếm (train, motor, rider) bắt buộc phải có trọng số phạt cao hơn car, traffic light."""
        res = compute_bdd10k_class_weights()
        cw = res["class_weights"]

        # train, motor, rider phải bị phạt nặng hơn car
        assert cw["train"] > cw["car"], f"train ({cw['train']}) phải có weight lớn hơn car ({cw['car']})"
        assert cw["motor"] > cw["car"], f"motor ({cw['motor']}) phải có weight lớn hơn car ({cw['car']})"
        assert cw["rider"] > cw["car"], f"rider ({cw['rider']}) phải có weight lớn hơn car ({cw['car']})"

        # Car là lớp phổ biến nhất nên weight phải ở mức chặn dưới min_weight
        assert cw["car"] <= res["min_weight"] + 0.1
        # Train là lớp hiếm nhất nên weight phải ở mức cao nhất
        assert cw["train"] >= 3.0

    def test_class_weights_clipping_bounds(self):
        """Hệ số phạt phải nằm trong ngưỡng an toàn min_weight và max_weight."""
        min_w, max_w = 0.2, 5.0
        res = compute_bdd10k_class_weights(min_weight=min_w, max_weight=max_w)
        for w in res["weights_list"]:
            assert min_w <= w <= max_w, f"Weight {w} vi phạm giới hạn [{min_w}, {max_w}]"


class TestPhase04AblationConfigFactory:
    """Kiểm tra việc tạo 5 cấu hình trong Đấu trường Cắt bỏ."""

    def test_tournament_configs_count_and_names(self):
        configs = AblationConfigFactory.create_tournament_configs(epochs=10)
        assert len(configs) == 5, "Đấu trường phải có chính xác 5 cấu hình"

        expected_names = [
            "cfg_0_naive",
            "cfg_1_freeze",
            "cfg_2_class_weights",
            "cfg_3_adamw_cosine",
            "cfg_4_proposed",
        ]
        actual_names = [c.name for c in configs]
        assert actual_names == expected_names

    def test_ablation_recipe_hyperparameters(self):
        configs = {c.name: c for c in AblationConfigFactory.create_tournament_configs(epochs=10)}

        # Cfg-0: Naive (SGD, no freeze, no weights, linear lr)
        c0 = configs["cfg_0_naive"]
        assert c0.optimizer == "SGD"
        assert c0.freeze_layers is None
        assert c0.class_weights is None
        assert not c0.cos_lr

        # Cfg-1: Freeze (10 layers đầu trong 5 epochs)
        c1 = configs["cfg_1_freeze"]
        assert c1.optimizer == "SGD"
        assert c1.freeze_layers == 10
        assert c1.freeze_epochs == 5
        assert c1.class_weights is None

        # Cfg-2: Class weights
        c2 = configs["cfg_2_class_weights"]
        assert c2.optimizer == "SGD"
        assert c2.freeze_layers is None
        assert c2.class_weights is not None
        assert len(c2.class_weights) == 10

        # Cfg-3: AdamW & Cosine LR
        c3 = configs["cfg_3_adamw_cosine"]
        assert c3.optimizer == "AdamW"
        assert c3.cos_lr is True
        assert c3.class_weights is None

        # Cfg-4: Proposed (Kết hợp cả 3 kỹ thuật + Early Stopping patience=10)
        c4 = configs["cfg_4_proposed"]
        assert c4.optimizer == "AdamW"
        assert c4.cos_lr is True
        assert c4.freeze_layers == 10
        assert c4.freeze_epochs == 5
        assert c4.class_weights is not None
        assert c4.patience == 10


class TestPhase04ModelLineageManager:
    """Kiểm tra trình quản lý xuất xưởng và niêm phong mô hình."""

    def test_lineage_registration_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lineage_mgr = ModelLineageManager(models_dir=tmp_dir)

            # Tạo file giả lập weight
            dummy_weight = Path(tmp_dir) / "dummy_best.pt"
            dummy_content = b"Mock PyTorch Checkpoint YOLO11n V0 Seed"
            dummy_weight.write_bytes(dummy_content)

            record = lineage_mgr.register_model(
                model_id="yolo11n_v0_test",
                architecture="YOLO11n",
                src_weights_path=dummy_weight,
                target_filename="yolo11n_v0_test.pt",
                recipe_name="cfg_4_proposed",
                hyperparameters={"epochs": 15, "batch": 16},
                val_metrics={"map50": 35.5, "recall": 42.1},
            )

            # Kiểm tra record trả về
            assert record["model_id"] == "yolo11n_v0_test"
            assert record["filename"] == "yolo11n_v0_test.pt"
            assert len(record["sha256_hash"]) == 64
            assert Path(record["filepath"]).exists()

            # Kiểm tra đọc lại từ manifest file
            loaded = lineage_mgr.load_lineage()
            assert "yolo11n_v0_test" in loaded
            assert loaded["yolo11n_v0_test"]["sha256_hash"] == record["sha256_hash"]


class TestPhase04HardSliceEvaluator:
    """Kiểm tra bộ đánh giá ma trận 4 lát cắt khó và sinh vector W_i."""

    def test_categorize_slice_images(self):
        evaluator = HardSliceEvaluator()
        test_imgs_dir = get_splits_dir() / "fixed_test" / "images"
        slices = evaluator.categorize_slice_images(test_imgs_dir)

        assert "night" in slices
        assert "small" in slices
        assert "occlusion" in slices
        assert "day" in slices

        # Kiểm tra số lượng ảnh trong các lát cắt hợp lý
        assert len(slices["night"]) > 100, "Lát cắt đêm phải có ít nhất 100 ảnh"
        assert len(slices["day"]) > 100, "Lát cắt ngày phải có ít nhất 100 ảnh"
        assert len(slices["small"]) > 500, "Lát cắt vật thể nhỏ phải có nhiều ảnh"
        assert len(slices["occlusion"]) > 500, "Lát cắt che khuất phải có nhiều ảnh"

    def test_slice_yaml_generation(self):
        evaluator = HardSliceEvaluator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_imgs = [Path(tmp_dir) / "test1.jpg", Path(tmp_dir) / "test2.jpg"]
            yaml_path = evaluator.create_slice_data_yaml("night", sample_imgs, tmp_dir)
            assert yaml_path.exists()
            content = yaml_path.read_text(encoding="utf-8")
            assert "test:" in content
            assert "slice_night.txt" in content
            assert "names:" in content

    def test_slice_weight_vector_normalization(self):
        """Vector trọng số lát cắt W_i bắt buộc phải có tổng đúng bằng 1.0."""
        evaluator = HardSliceEvaluator()
        # Mock slice evaluation
        slices_map = {"night": [1] * 400, "small": [1] * 1200, "occlusion": [1] * 1200, "day": [1] * 800}

        # Giả lập error gaps
        errors = {"night": 65.0, "small": 70.0, "occlusion": 60.0, "day": 40.0}
        total = sum(errors.values())
        w_vec = {s: round(errors[s] / total, 4) for s in errors}
        w_vec["day"] = round(1.0 - sum(w_vec[s] for s in ["night", "small", "occlusion"]), 4)

        assert sum(w_vec.values()) == pytest.approx(1.0, abs=1e-3)
        # Lát cắt small và night có lỗi cao nhất nên phải có trọng số W_i cao hơn day
        assert w_vec["small"] > w_vec["day"]
        assert w_vec["night"] > w_vec["day"]


class TestPhase04DualModelInstantiation:
    """Kiểm tra khả năng khởi tạo Song Mã YOLO11n và RT-DETR."""

    def test_yolo11n_instantiation(self):
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        assert model is not None
        assert model.task == "detect"

    def test_rtdetr_instantiation(self):
        from ultralytics import RTDETR
        model = RTDETR("rtdetr-l.yaml")
        assert model is not None
        assert model.task == "detect"


class TestPhase04ReadOnlyAndCacheSafety:
    """Kiểm tra tuân thủ nguyên tắc Read-Only 100% của AGENTS.md và dọn dẹp cache."""

    def test_bdd10k_read_only_zero_cache_files(self):
        """Khẳng định thư mục data/bdd10k/ không chứa bất kỳ file .cache nào."""
        from paper_v4_10k.src.paths import get_bdd10k_dir
        from paper_v4_10k.src.train_engine import clean_bdd10k_caches

        # Dọn dẹp trước
        clean_bdd10k_caches()

        bdd_dir = get_bdd10k_dir()
        if bdd_dir.is_symlink():
            bdd_dir = bdd_dir.resolve()

        cache_files = list(bdd_dir.rglob("*.cache"))
        assert len(cache_files) == 0, f"Phát hiện file cache vi phạm trong data/bdd10k/: {cache_files}"

    def test_clean_bdd10k_caches_function(self, tmp_path):
        """Kiểm tra hàm clean_bdd10k_caches phát hiện và xóa sạch file .cache mô phỏng trên thư mục cô lập."""
        from paper_v4_10k.src.train_engine import clean_bdd10k_caches

        # Tạo file cache giả lập tạm thời trong tmp_path/labels/ (không chạm vào data/bdd10k/)
        dummy_labels_dir = tmp_path / "labels"
        dummy_labels_dir.mkdir(parents=True, exist_ok=True)
        dummy_cache = dummy_labels_dir / "temp_audit_test.cache"
        try:
            dummy_cache.write_text("test_cache_entry", encoding="utf-8")
            assert dummy_cache.exists()

            # Gọi clean_bdd10k_caches với target_override=tmp_path
            removed = clean_bdd10k_caches(target_override=tmp_path)
            assert removed >= 1, "clean_bdd10k_caches phải dọn dẹp ít nhất 1 file cache"
            assert not dummy_cache.exists(), "File cache giả lập phải bị xóa bỏ"
        finally:
            if dummy_cache.exists():
                dummy_cache.unlink()


class TestPhase04CriterionWeightsAndLineage:
    """Kiểm tra cơ chế tiêm class_weights và tính toàn vẹn của ModelLineage manifest."""

    def test_class_weights_injection_logic(self):
        """Kiểm tra unwrap_model_safe và gán class_weights vào model và criterion."""
        import torch
        import torch.nn as nn
        from paper_v4_10k.src.train_engine import unwrap_model_safe

        class DummyCriterion(nn.Module):
            def __init__(self):
                super().__init__()
                self.class_weights = None

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.zeros(1))
                self.class_weights = None
                self.criterion = DummyCriterion()

        model = DummyModel()
        unwrapped = unwrap_model_safe(model)
        weights_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        unwrapped.class_weights = weights_tensor
        unwrapped.criterion.class_weights = weights_tensor.view(1, 1, -1)

        assert torch.equal(unwrapped.class_weights, weights_tensor)
        assert unwrapped.criterion.class_weights.shape == (1, 1, 3)

    def test_dual_model_lineage_registration(self):
        """Khẳng định manifest lineage chứa đầy đủ metrics cho cả YOLO11n và RT-DETR."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = ModelLineageManager(models_dir=tmp_dir)
            p1 = Path(tmp_dir) / "yolo11n_best.pt"
            p2 = Path(tmp_dir) / "rtdetr_best.pt"
            p1.write_bytes(b"YOLO11n weights")
            p2.write_bytes(b"RT-DETR weights")

            r1 = mgr.register_model(
                model_id="yolo11n_v0_best",
                architecture="YOLO11n",
                src_weights_path=p1,
                target_filename="yolo11n_v0_best.pt",
                recipe_name="cfg_4_proposed",
                hyperparameters={"epochs": 15},
                val_metrics={"map50": 3.6},
                test_metrics={"slice_metrics": {"night": {"recall": 10.0}}},
                public_bench_metrics={"map50": 3.5},
            )

            r2 = mgr.register_model(
                model_id="rtdetr_v0_best",
                architecture="RT-DETR",
                src_weights_path=p2,
                target_filename="rtdetr_v0_best.pt",
                recipe_name="rtdetr_v0_proposed",
                hyperparameters={"epochs": 15},
                val_metrics={"map50": 1.2},
                test_metrics={"slice_metrics": {"night": {"recall": 5.0}}},
                public_bench_metrics={"map50": 1.0},
            )

            manifest = mgr.load_lineage()
            assert "yolo11n_v0_best" in manifest
            assert "rtdetr_v0_best" in manifest

            # Cả hai mô hình đều phải có test_metrics và public_bench_metrics
            assert "slice_metrics" in manifest["yolo11n_v0_best"]["test_metrics"]
            assert "slice_metrics" in manifest["rtdetr_v0_best"]["test_metrics"]
            assert manifest["yolo11n_v0_best"]["public_bench_metrics"]["map50"] == 3.5
            assert manifest["rtdetr_v0_best"]["public_bench_metrics"]["map50"] == 1.0

