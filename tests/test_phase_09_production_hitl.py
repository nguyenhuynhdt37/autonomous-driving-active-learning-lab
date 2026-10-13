# ==============================================================================
# TESTS CHO MODULE 09: PRODUCTION HITL ACTIVE LEARNING PIPELINE
# ==============================================================================

import os
import sys
import zipfile
import tempfile
import shutil
import importlib.util
from pathlib import Path
import pytest

# Load trực tiếp các module của Module 09 bằng importlib để tránh đụng độ namespace với src/ tại root
project_root = Path(__file__).resolve().parent.parent
module_09_src = project_root / "notebooks/09_production_hitl_pipeline/src"

def _load_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Nạp theo thứ tự phụ thuộc
_ingestor_mod = _load_module("universal_zip_ingestor", module_09_src / "universal_zip_ingestor.py")
_packager_mod = _load_module("adaptive_task_packager", module_09_src / "adaptive_task_packager.py")
_checker_mod = _load_module("stopping_checker", module_09_src / "stopping_checker.py")
_pipe_mod = _load_module("production_pipeline", module_09_src / "pipeline.py")

UniversalZipIngestor = _ingestor_mod.UniversalZipIngestor
IngestionReport = _ingestor_mod.IngestionReport
AdaptiveTaskPackager = _packager_mod.AdaptiveTaskPackager
PackagingResult = _packager_mod.PackagingResult
ProductionStoppingChecker = _checker_mod.ProductionStoppingChecker
StoppingVerdict = _checker_mod.StoppingVerdict
ProductionHITLPipeline = _pipe_mod.ProductionHITLPipeline


class TestUniversalZipIngestor:
    """Kiểm thử tính năng giải nén và kiểm toán nhãn đa định dạng."""

    def test_detect_txt_format(self, tmp_path):
        ingestor = UniversalZipIngestor()
        
        # 1. Detection format (5 tokens)
        f_detect = tmp_path / "detect.txt"
        f_detect.write_text("0 0.5 0.5 0.2 0.3\n")
        assert ingestor.detect_txt_format(f_detect) == "yolo_detect"
        
        # 2. Segmentation format (>5 tokens, polygon)
        f_seg = tmp_path / "seg.txt"
        f_seg.write_text("0 0.1 0.1 0.5 0.2 0.3 0.8\n")
        assert ingestor.detect_txt_format(f_seg) == "yolo_segment"
        
        # 3. Pose format (56 tokens: 5 + 17*3)
        f_pose = tmp_path / "pose.txt"
        pose_tokens = ["0", "0.5", "0.5", "0.2", "0.3"] + ["0.1", "0.1", "2"] * 17
        f_pose.write_text(" ".join(pose_tokens) + "\n")
        assert ingestor.detect_txt_format(f_pose) == "yolo_pose"

    def test_zip_extraction_and_ingestion(self, tmp_path):
        ingestor = UniversalZipIngestor()
        
        # Tạo file ZIP chứa nhãn hỗn hợp hợp lệ và lỗi
        zip_path = tmp_path / "test_labels.zip"
        target_dir = tmp_path / "target_labels"
        
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("img_001.txt", "0 0.5 0.5 0.2 0.2\n")
            zf.writestr("img_002.txt", "1 0.3 0.3 0.1 0.1\n")
            zf.writestr("img_bad.txt", "1 2.5 0.3 0.1 0.1\n") # Tọa độ x=2.5 > 1.05 -> Bị loại
            
        report = ingestor.ingest_zip(zip_path, target_dir)
        
        assert isinstance(report, IngestionReport)
        assert report.status == "SUCCESS"
        assert report.detected_format == "yolo_detect"
        assert report.total_label_files == 2
        assert report.invalid_lines_count == 1
        assert (target_dir / "img_001.txt").exists()
        assert (target_dir / "img_002.txt").exists()


class TestStoppingChecker:
    """Kiểm thử 5 tầng rào chắn dừng tự động."""

    def test_target_map_reached(self):
        checker = ProductionStoppingChecker(target_map50=26.85, confirmation_rounds_needed=1)
        
        # Round 1: mAP50 = 24.50 -> Tiếp tục
        res1 = checker.check(current_map50=24.50, previous_map50=20.0)
        assert not res1.should_stop
        assert res1.status == "CONTINUE"
        
        # Round 2: mAP50 = 27.10 -> Dừng vì vượt trần target
        res2 = checker.check(current_map50=27.10, previous_map50=24.50)
        assert res2.should_stop
        assert res2.status == "VICTORY_CONFIRMED"

    def test_knowledge_saturation_plateau(self):
        checker = ProductionStoppingChecker(target_map50=35.0, plateau_delta_threshold=0.30)
        
        # Round 1: tăng mạnh (+3.0) -> Tiếp tục
        res1 = checker.check(current_map50=20.0, previous_map50=17.0)
        assert not res1.should_stop
        
        # Round 2: tăng chậm (+0.2 < 0.3) -> Lần 1
        res2 = checker.check(current_map50=20.2, previous_map50=20.0)
        assert not res2.should_stop
        
        # Round 3: tăng chậm (+0.1 < 0.3) -> Lần 2 liên tiếp -> KNOWLEDGE_SATURATED
        res3 = checker.check(current_map50=20.3, previous_map50=20.2)
        assert res3.should_stop
        assert res3.status == "KNOWLEDGE_SATURATED"

    def test_user_forced_stop(self):
        checker = ProductionStoppingChecker()
        res = checker.check(current_map50=22.0, user_requested_stop=True)
        assert res.should_stop
        assert res.status == "USER_FORCED_STOP"


class TestAdaptiveTaskPackager:
    """Kiểm thử đóng gói task ZIP kèm/không kèm pre-labels."""

    def test_clean_images_packaging(self, tmp_path):
        outbox_dir = tmp_path / "outbox"
        packager = AdaptiveTaskPackager(outbox_dir=outbox_dir)
        
        # Tạo thư mục ảnh giả lập
        images_dir = tmp_path / "task_images"
        images_dir.mkdir()
        (images_dir / "img1.jpg").write_bytes(b"image_bytes_1")
        (images_dir / "img2.jpg").write_bytes(b"image_bytes_2")
        
        res = packager.package_task_zip(
            images_dir=images_dir,
            iteration=1,
            model=None,
            task="detect",
            include_prelabels=False
        )
        
        assert isinstance(res, PackagingResult)
        assert Path(res.zip_path).exists()
        assert res.total_images == 2
        assert not res.prelabels_included
        assert res.total_prelabeled_instances == 0
        
        # Kiểm tra nội dung ZIP
        with zipfile.ZipFile(res.zip_path, "r") as zf:
            namelist = zf.namelist()
            assert "images/img1.jpg" in namelist
            assert "images/img2.jpg" in namelist
            # Không có file nhãn nào
            assert not any(n.startswith("labels/") for n in namelist)


class TestProductionPipelineInit:
    """Kiểm thử khởi tạo đối tượng Pipeline và chuẩn bị thư mục workspace."""

    def test_pipeline_workspace_setup(self, tmp_path):
        ws_dir = tmp_path / "prod_workspace"
        pipe = ProductionHITLPipeline(
            workspace_dir=str(ws_dir),
            model_name="yolo11n.pt",
            task="detect",
            target_map50=26.85
        )
        
        assert (ws_dir / "inbox").exists()
        assert (ws_dir / "outbox").exists()
        assert (ws_dir / "data" / "raw_images").exists()
        assert (ws_dir / "data" / "splits").exists()
        assert (ws_dir / "models").exists()
        assert pipe.model_name == "yolo11n.pt"
        assert pipe.task == "detect"
        assert pipe.target_map50 == 26.85
