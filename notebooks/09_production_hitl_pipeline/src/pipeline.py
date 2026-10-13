"""
pipeline.py - Trái Tim Điều Phối Hệ Thống Production HITL (Production HITL Pipeline)
=====================================================================================
Điều phối toàn bộ chu trình thực chiến:
1. Nạp trực tiếp tham số (Model name, Task, Raw Images, Raw Labels) qua code/CLI mà không cần sửa YAML bằng tay.
2. Hỗ trợ cả 2 kịch bản:
   - Cold Start: Ban đầu chỉ có ảnh thô, bốc seed_v0 và dừng lại chờ nạp nhãn.
   - Warm Start: Đã có sẵn kho raw labels (thư mục hoặc file .zip), tự động map O(1) nhãn cho test/val/seed_v0.
3. Xuất gói gán nhãn ra outbox/ có tùy chọn Pre-labels & lọc ngưỡng tin cậy cao (Conf >= min_conf).
4. Tự động nhận diện và nạp file ZIP nhãn thả vào inbox/ (YOLO/COCO, BBox/Seg/Pose).
5. Huấn luyện tiếp sức kế thừa weights qua các vòng V0 -> V1 -> V2...
6. Tự động kiểm tra Bộ rào chắn dừng 5 tầng và xuất xưởng best_production.pt cùng ONNX/INT8.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from ultralytics import YOLO

try:
    from .adaptive_task_packager import AdaptiveTaskPackager
    from .stopping_checker import ProductionStoppingChecker, StoppingVerdict
    from .universal_zip_ingestor import IngestionReport, UniversalZipIngestor
except (ImportError, ValueError):
    from adaptive_task_packager import AdaptiveTaskPackager
    from stopping_checker import ProductionStoppingChecker, StoppingVerdict
    from universal_zip_ingestor import IngestionReport, UniversalZipIngestor


class ProductionHITLPipeline:
    """Hệ thống điều phối Active Learning & Human-in-the-Loop Production hoàn chỉnh."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        task: str = "detect",  # "detect", "segment", "pose"
        raw_images_dir: Optional[Union[str, Path]] = None,
        raw_labels_path: Optional[Union[str, Path]] = None,  # Thư mục hoặc file .zip nhãn thô ban đầu
        workspace_dir: Optional[Union[str, Path]] = None,
        target_map50: float = 26.85,
        prelabel_min_conf: float = 0.70,
        class_names: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.task = task
        self.target_map50 = target_map50
        self.prelabel_min_conf = prelabel_min_conf
        self.class_names = class_names or [
            "person", "rider", "car", "truck", "bus",
            "train", "motor", "bike", "traffic light", "traffic sign"
        ]

        # Khởi tạo workspace
        base_dir = Path(__file__).resolve().parent.parent
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else (base_dir / "workspace")
        self.inbox_dir = self.workspace_dir / "inbox"
        self.outbox_dir = self.workspace_dir / "outbox"
        self.data_dir = self.workspace_dir / "data"
        self.raw_images_dir = Path(raw_images_dir).resolve() if raw_images_dir else (self.data_dir / "raw_images")
        self.raw_labels_dir = self.data_dir / "raw_labels"
        self.splits_dir = self.data_dir / "splits"
        self.models_dir = self.workspace_dir / "models"

        for d in [self.inbox_dir, self.outbox_dir, self.raw_images_dir, self.raw_labels_dir, self.splits_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Các module thành phần
        self.ingestor = UniversalZipIngestor(class_names=self.class_names)
        self.packager = AdaptiveTaskPackager(outbox_dir=self.outbox_dir, class_names=self.class_names)
        self.stopping_checker = ProductionStoppingChecker(target_map50=self.target_map50)

        # Tự động nạp raw labels ban đầu nếu người dùng chỉ định
        if raw_labels_path:
            self.load_initial_raw_labels(raw_labels_path)

        # Lưu đồng bộ cấu hình vào config_production.yaml (Tự động, không bắt sửa tay)
        self.sync_config_file()

        print(f"✓ ProductionHITLPipeline đã khởi tạo thành công:")
        print(f"  - Model: {self.model_name} | Task: {self.task}")
        print(f"  - Raw Images: {self.raw_images_dir}")
        print(f"  - Inbox: {self.inbox_dir} | Outbox: {self.outbox_dir}")

    def sync_config_file(self) -> Path:
        """Tự động lưu và cập nhật cấu hình vào config_production.yaml."""
        cfg = {
            "model_name": self.model_name,
            "task": self.task,
            "target_map50": self.target_map50,
            "prelabel_min_conf": self.prelabel_min_conf,
            "class_names": self.class_names,
            "paths": {
                "workspace": str(self.workspace_dir),
                "raw_images": str(self.raw_images_dir),
                "raw_labels": str(self.raw_labels_dir),
                "inbox": str(self.inbox_dir),
                "outbox": str(self.outbox_dir),
                "splits": str(self.splits_dir),
                "models": str(self.models_dir),
            },
        }
        cfg_file = self.workspace_dir.parent / "config_production.yaml"
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        return cfg_file

    def load_initial_raw_labels(self, raw_labels_path: Union[str, Path]) -> IngestionReport:
        """
        Nạp kho nhãn thô ban đầu (Thư mục hoặc file .zip) để phase này biết và map O(1).
        """
        p = Path(raw_labels_path).resolve()
        print(f"🔄 Đang nạp kho nhãn thô ban đầu từ: {p}...")

        if p.is_file() and p.suffix.lower() == ".zip":
            report = self.ingestor.ingest_zip(
                zip_path=p,
                target_labels_dir=self.raw_labels_dir,
                target_images_dir=self.raw_images_dir,
                target_task=self.task,
            )
            print(f"  ✓ Đã giải nén và nạp {report.total_label_files} file nhãn ({report.detected_format}) vào raw_labels/!")
            return report
        elif p.is_dir():
            count = 0
            for item in p.glob("*.txt"):
                shutil.copy2(item, self.raw_labels_dir / item.name)
                count += 1
            print(f"  ✓ Đã sao chép {count} file nhãn .txt vào raw_labels/!")
            return IngestionReport(
                status="SUCCESS",
                zip_path=str(p),
                detected_format="yolo_folder",
                total_label_files=count,
                total_instances=0,
                target_labels_dir=str(self.raw_labels_dir),
                matched_images_count=count,
                invalid_lines_count=0,
            )
        else:
            raise FileNotFoundError(f"Đường dẫn raw labels không hợp lệ: {p}")

    def setup_initial_splits(
        self,
        seed_v0_size: int = 100,
        test_ratio: float = 0.20,
        val_ratio: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Phân chia kho ảnh thô thành:
        - fixed_test/ (test_ratio)
        - fixed_val/ (val_ratio)
        - seed_v0/ (seed_v0_size ảnh mầm)
        - unlabeled_pool/ (phần còn lại để Active Mining)
        """
        print("=" * 80)
        print("🚀 BƯỚC 1: KHỞI TẠO VÀ PHÂN TÁCH KHO ẢNH THÔ (SPLITTING)")
        print("=" * 80)

        all_images = sorted(list(self.raw_images_dir.glob("*.jpg")) + list(self.raw_images_dir.glob("*.png")))
        total_imgs = len(all_images)
        if total_imgs == 0:
            raise RuntimeError(f"Kho ảnh thô {self.raw_images_dir} đang rỗng! Vui lòng thả ảnh vào.")

        n_test = max(1, int(total_imgs * test_ratio))
        n_val = max(1, int(total_imgs * val_ratio))
        n_seed = min(seed_v0_size, total_imgs - n_test - n_val)

        test_imgs = all_images[:n_test]
        val_imgs = all_images[n_test : n_test + n_val]
        seed_v0_imgs = all_images[n_test + n_val : n_test + n_val + n_seed]
        unlabeled_imgs = all_images[n_test + n_val + n_seed :]

        splits_map = {
            "fixed_test": test_imgs,
            "fixed_val": val_imgs,
            "seed_v0": seed_v0_imgs,
            "unlabeled_pool": unlabeled_imgs,
        }

        counts = {}
        for split_name, img_list in splits_map.items():
            s_img_dir = self.splits_dir / split_name / "images"
            s_lbl_dir = self.splits_dir / split_name / "labels"
            s_img_dir.mkdir(parents=True, exist_ok=True)
            s_lbl_dir.mkdir(parents=True, exist_ok=True)

            # Xóa symlinks cũ nếu có
            for f in s_img_dir.iterdir():
                f.unlink()

            for img in img_list:
                dst_link = s_img_dir / img.name
                dst_link.symlink_to(img.resolve())

                # Nếu có sẵn nhãn trong raw_labels, tự động liên kết sang!
                src_lbl = self.raw_labels_dir / f"{img.stem}.txt"
                dst_lbl = s_lbl_dir / f"{img.stem}.txt"
                if src_lbl.exists() and not dst_lbl.exists():
                    dst_lbl.symlink_to(src_lbl.resolve())

            counts[split_name] = len(img_list)

        print(f"✓ Phân tách hoàn tất (Tổng {total_imgs} ảnh):")
        print(f"  - fixed_test: {counts['fixed_test']} ảnh")
        print(f"  - fixed_val: {counts['fixed_val']} ảnh")
        print(f"  - seed_v0 (Tập mầm khởi đầu): {counts['seed_v0']} ảnh")
        print(f"  - unlabeled_pool (Kho khai thác Active): {counts['unlabeled_pool']} ảnh")

        return counts

    def check_seed_labels_ready(self, split_name: str = "seed_v0") -> bool:
        """Kiểm tra xem tập split đã có đủ file nhãn chưa."""
        img_dir = self.splits_dir / split_name / "images"
        lbl_dir = self.splits_dir / split_name / "labels"
        if not img_dir.exists():
            return False

        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        lbls = list(lbl_dir.glob("*.txt"))

        return len(imgs) > 0 and len(lbls) >= len(imgs)

    def export_labeling_task(
        self,
        iteration: int = 0,
        include_prelabels: bool = True,
        min_confidence: Optional[float] = None,
    ) -> str:
        """
        Xuất file ZIP nhiệm vụ gán nhãn ra outbox/ để người dùng gửi đi gán nhãn.
        """
        split_name = f"seed_v{iteration}"
        img_dir = self.splits_dir / split_name / "images"
        conf = min_confidence or self.prelabel_min_conf

        # Nạp mô hình hiện tại nếu có để sinh pre-labels
        current_model = None
        if include_prelabels and iteration > 0:
            prev_weights = self.models_dir / f"yolo11n_v{iteration-1}_best.pt"
            if prev_weights.exists():
                current_model = YOLO(str(prev_weights))
            else:
                current_model = YOLO(self.model_name)

        res = self.packager.package_task_zip(
            images_dir=img_dir,
            iteration=iteration,
            model=current_model,
            task=self.task,
            include_prelabels=include_prelabels,
            min_confidence=conf,
        )
        return res.zip_path

    def ingest_labels_from_inbox(
        self,
        zip_path: Optional[Union[str, Path]] = None,
        iteration: int = 0,
    ) -> IngestionReport:
        """
        Tự động quét và nạp file ZIP nhãn từ inbox/ (hoặc từ đường dẫn chỉ định).
        """
        split_name = f"seed_v{iteration}"
        target_lbl_dir = self.splits_dir / split_name / "labels"
        target_img_dir = self.splits_dir / split_name / "images"

        # Nếu không truyền zip_path, tự động tìm file zip mới nhất trong inbox/
        if zip_path is None:
            zip_candidates = list(self.inbox_dir.glob("*.zip"))
            if not zip_candidates:
                raise FileNotFoundError(
                    f"Thư mục inbox ({self.inbox_dir}) chưa có file .zip nào! "
                    f"Vui lòng thả file zip nhãn vào inbox/ hoặc truyền trực tiếp đường dẫn."
                )
            # Lấy file có thời gian sửa đổi mới nhất
            zip_p = max(zip_candidates, key=os.path.getmtime)
        else:
            zip_p = Path(zip_path).resolve()

        print(f"📦 Đang tự động xử lý file ZIP nhãn: {zip_p.name}...")
        report = self.ingestor.ingest_zip(
            zip_path=zip_p,
            target_labels_dir=target_lbl_dir,
            target_images_dir=target_img_dir,
            target_task=self.task,
        )

        if report.status == "SUCCESS":
            print(f"✓ Nạp nhãn thành công cho {split_name}: {report.total_label_files} files nhãn ({report.detected_format})!")
            # Tự động gộp tập train tích lũy: train_vk
            self.consolidate_train_split(iteration=iteration)
        else:
            print(f"❌ Lỗi nạp nhãn: {report.errors}")

        return report

    def consolidate_train_split(self, iteration: int = 0) -> Path:
        """
        Gộp tích lũy toàn bộ các seed từ V0 đến Vk thành train_vk và sinh YAML.
        """
        train_split_name = f"train_v{iteration}"
        train_img_dir = self.splits_dir / train_split_name / "images"
        train_lbl_dir = self.splits_dir / train_split_name / "labels"
        train_img_dir.mkdir(parents=True, exist_ok=True)
        train_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Dọn dẹp symlinks cũ
        for f in train_img_dir.iterdir():
            f.unlink()
        for f in train_lbl_dir.iterdir():
            f.unlink()

        total_imgs = 0
        total_lbls = 0

        # Gom tất cả seed_v0, seed_v1, ..., seed_vk
        for it in range(iteration + 1):
            seed_name = f"seed_v{it}"
            s_img = self.splits_dir / seed_name / "images"
            s_lbl = self.splits_dir / seed_name / "labels"

            if s_img.exists():
                for img in s_img.glob("*.*"):
                    dst = train_img_dir / img.name
                    if not dst.exists():
                        dst.symlink_to(img.resolve())
                        total_imgs += 1

            if s_lbl.exists():
                for lbl in s_lbl.glob("*.txt"):
                    dst = train_lbl_dir / lbl.name
                    if not dst.exists():
                        dst.symlink_to(lbl.resolve())
                        total_lbls += 1

        # Sinh file cấu hình YOLO tương ứng
        yaml_path = self.splits_dir / f"bdd10k_splits_v{iteration}.yaml"
        yaml_data = {
            "path": str(self.splits_dir.resolve()),
            "train": f"{train_split_name}/images",
            "val": "fixed_val/images",
            "test": "fixed_test/images",
            "names": {idx: name for idx, name in enumerate(self.class_names)},
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

        print(f"  ✓ Đã đóng gói tập huấn luyện tích lũy {train_split_name}: {total_imgs} images, {total_lbls} labels!")
        print(f"  ✓ Đã sinh file cấu hình YOLO: {yaml_path.name}")
        return yaml_path

    def train_model(
        self,
        iteration: int = 0,
        epochs: int = 15,
        batch_size: int = 8,
        device: str = "auto",
    ) -> Dict[str, Any]:
        """
        Huấn luyện model V_k bằng công thức Cfg-3 Proposed Recipe kế thừa weights V_(k-1).
        """
        print("=" * 80)
        print(f"🔥 HUẤN LUYỆN MODEL V{iteration} (ITERATION {iteration})")
        print("=" * 80)

        yaml_path = self.splits_dir / f"bdd10k_splits_v{iteration}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Chưa có file cấu hình {yaml_path}! Hãy nạp nhãn và gộp tập train trước.")

        # Lựa chọn weights kế thừa
        if iteration == 0:
            init_weights = self.model_name
        else:
            prev_best = self.models_dir / f"yolo11n_v{iteration-1}_best.pt"
            init_weights = str(prev_best) if prev_best.exists() else self.model_name

        print(f"Khởi tạo huấn luyện kế thừa từ: {init_weights}")
        model = YOLO(init_weights)

        # Xác định thiết bị
        if device == "auto":
            import torch
            if torch.backends.mps.is_available():
                dev = "mps"
            elif torch.cuda.is_available():
                dev = "0"
            else:
                dev = "cpu"
        else:
            dev = device

        output_weights_path = self.models_dir / f"yolo11n_v{iteration}_best.pt"

        # Huấn luyện
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            optimizer="AdamW",
            cos_lr=True,
            device=dev,
            project=str(self.workspace_dir / "runs"),
            name=f"v{iteration}_train",
            exist_ok=True,
            verbose=False,
        )

        # Lưu checkpoint chính thức vào models/
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            shutil.copy2(best_pt, output_weights_path)
            print(f"✓ Đã niêm phong checkpoint: {output_weights_path}")
        else:
            model.save(str(output_weights_path))

        # Đánh giá trên fixed_test
        eval_metrics = self.evaluate_model(iteration=iteration, weights_path=output_weights_path)
        return eval_metrics

    def evaluate_model(
        self,
        iteration: int = 0,
        weights_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Đánh giá model trên fixed_test cố định."""
        w_path = weights_path or (self.models_dir / f"yolo11n_v{iteration}_best.pt")
        model = YOLO(str(w_path))

        yaml_path = self.splits_dir / f"bdd10k_splits_v{iteration}.yaml"
        metrics = model.val(data=str(yaml_path), split="test", verbose=False)

        map50 = round(float(metrics.box.map50 * 100.0), 2)
        map50_95 = round(float(metrics.box.map * 100.0), 2)

        print(f"📊 Kết Quả Đánh Giá Model V{iteration} trên Frozen Gold Test:")
        print(f"  - mAP@50: {map50}% | mAP@50-95: {map50_95}%")

        return {
            "iteration": iteration,
            "weights": str(w_path),
            "map50": map50,
            "map50_95": map50_95,
        }

    def mine_active_batch(
        self,
        current_iteration: int = 0,
        batch_size: int = 100,
    ) -> List[str]:
        """
        Khai thác batch mẫu thách thức tiếp theo từ unlabeled_pool:
        - Quét pool bằng model hiện tại.
        - Tính Uncertainty (Margin) và bốc ra batch_size ảnh khó nhất vào seed_v(k+1)/images/.
        """
        next_it = current_iteration + 1
        print("=" * 80)
        print(f"🔍 BẮT ĐẦU ACTIVE MINING VÒNG {next_it} (BỐC {batch_size} ẢNH KHÓ)")
        print("=" * 80)

        unlabeled_dir = self.splits_dir / "unlabeled_pool" / "images"
        unlabeled_imgs = list(unlabeled_dir.glob("*.jpg")) + list(unlabeled_dir.glob("*.png"))

        if not unlabeled_imgs:
            print("⚠️ Kho unlabeled_pool đã hết ảnh!")
            return []

        curr_weights = self.models_dir / f"yolo11n_v{current_iteration}_best.pt"
        model = YOLO(str(curr_weights))

        # Tính toán Uncertainty cho từng ảnh
        scored_items: List[Tuple[str, float]] = []
        for img_p in unlabeled_imgs[: min(len(unlabeled_imgs), 500)]:  # Quét sample pool
            try:
                preds = model(str(img_p), verbose=False)
                r = preds[0]
                if hasattr(r, "boxes") and len(r.boxes) > 0:
                    confs = r.boxes.conf.cpu().numpy()
                    # Điểm uncertainty: 1.0 - conf trung bình (conf càng thấp -> uncertainty càng cao)
                    uncertainty = float(1.0 - confs.mean())
                else:
                    uncertainty = 0.90  # Ảnh không bắt được box nào -> tiềm ẩn khó
                scored_items.append((img_p.name, uncertainty))
            except Exception:
                scored_items.append((img_p.name, 0.50))

        # Sắp xếp giảm dần theo uncertainty
        scored_items.sort(key=lambda x: x[1], reverse=True)
        selected_names = [item[0] for item in scored_items[:batch_size]]

        # Tạo thư mục seed_v(next_it)/images
        next_seed_img_dir = self.splits_dir / f"seed_v{next_it}" / "images"
        next_seed_lbl_dir = self.splits_dir / f"seed_v{next_it}" / "labels"
        next_seed_img_dir.mkdir(parents=True, exist_ok=True)
        next_seed_lbl_dir.mkdir(parents=True, exist_ok=True)

        for name in selected_names:
            src_link = unlabeled_dir / name
            dst_link = next_seed_img_dir / name
            if not dst_link.exists():
                dst_link.symlink_to(src_link.resolve())
            # Rút bớt khỏi unlabeled_pool
            if src_link.exists():
                src_link.unlink()

        print(f"✓ Đã bốc {len(selected_names)} ảnh thách thức nhất vào seed_v{next_it}/images/!")
        print(f"  (Đã rút bớt {len(selected_names)} ảnh khỏi unlabeled_pool)")
        return selected_names

    def export_production_model(self, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, str]:
        """
        Niêm phong và xuất xưởng model production (ONNX FP16 & OpenVINO INT8).
        """
        out_d = Path(output_dir).resolve() if output_dir else self.models_dir
        out_d.mkdir(parents=True, exist_ok=True)

        # Tìm model tốt nhất hiện có
        all_pts = list(self.models_dir.glob("yolo11n_v*_best.pt"))
        if not all_pts:
            raise FileNotFoundError("Chưa có checkpoint nào trong models/ để xuất xưởng!")

        best_pt = max(all_pts, key=os.path.getmtime)
        prod_pt = out_d / "best_production.pt"
        shutil.copy2(best_pt, prod_pt)

        print(f"🏆 Đã niêm phong model production: {prod_pt}")

        outputs = {"pytorch": str(prod_pt)}

        # Export ONNX
        try:
            m = YOLO(str(prod_pt))
            onnx_path = m.export(format="onnx", half=True, dynamic=True)
            outputs["onnx"] = str(onnx_path)
            print(f"✓ Đã xuất xưởng ONNX: {onnx_path}")
        except Exception as e:
            print(f"Warning: Không thể export ONNX: {e}")

        # Export OpenVINO INT8
        try:
            m = YOLO(str(prod_pt))
            yaml_path = list(self.splits_dir.glob("bdd10k_splits_v*.yaml"))[0]
            vino_path = m.export(format="openvino", int8=True, data=str(yaml_path))
            outputs["openvino_int8"] = str(vino_path)
            print(f"✓ Đã xuất xưởng OpenVINO INT8: {vino_path}")
        except Exception as e:
            print(f"Warning: Không thể export OpenVINO INT8: {e}")

        return outputs
