"""
adaptive_task_packager.py - Đóng Gói Nhiệm Vụ Gán Nhãn Thông Minh (Adaptive Task Packager)
==========================================================================================
Chức năng Production:
1. Đóng gói danh sách ảnh cần gán nhãn thành file ZIP trong workspace/outbox/.
2. Tùy chọn Pre-labeling thông minh (Configurable Pre-labels):
   - Option include_prelabels=False: Chỉ xuất ảnh sạch thuần túy để annotator gán nhãn mù.
   - Option include_prelabels=True: Chạy model hiện tại để dự đoán sẵn bounding boxes / polygons / keypoints.
3. Rào chắn độ tin cậy (Confidence Gatekeeper):
   - Chỉ xuất các dự đoán có Confidence >= min_confidence (ví dụ 0.70).
   - Loại bỏ các dự đoán yếu, không chắc chắn để tránh làm rối mắt và gây nhiễu cho người gán nhãn.
4. Hỗ trợ xuất định dạng YOLO (.txt) hoặc COCO (instances.json).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class PackagingResult:
    """Kết quả đóng gói nhiệm vụ gán nhãn."""
    zip_path: str
    iteration: int
    total_images: int
    prelabels_included: bool
    total_prelabeled_instances: int
    min_confidence_used: float
    manifest_path: str


class AdaptiveTaskPackager:
    """Bộ đóng gói nhiệm vụ gán nhãn thích ứng cho Production."""

    def __init__(
        self,
        outbox_dir: Union[str, Path],
        class_names: Optional[List[str]] = None,
    ):
        self.outbox_dir = Path(outbox_dir).resolve()
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or [
            "person", "rider", "car", "truck", "bus",
            "train", "motor", "bike", "traffic light", "traffic sign"
        ]

    def package_task_zip(
        self,
        images_dir: Union[str, Path],
        iteration: int,
        model: Optional[Any] = None,
        task: str = "detect",  # "detect", "segment", "pose"
        include_prelabels: bool = True,
        min_confidence: float = 0.70,
        export_format: str = "yolo",  # "yolo" hoặc "coco"
    ) -> PackagingResult:
        """
        Đóng gói batch ảnh cần gán nhãn thành 1 file ZIP trong outbox/.
        """
        img_dir = Path(images_dir).resolve()
        image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

        zip_filename = f"batch_v{iteration}_tasks.zip"
        zip_output_path = self.outbox_dir / zip_filename

        temp_dir = Path(tempfile.mkdtemp(prefix=f"pack_v{iteration}_"))
        temp_images_dir = temp_dir / "images"
        temp_labels_dir = temp_dir / "labels"
        temp_images_dir.mkdir(parents=True, exist_ok=True)

        if include_prelabels:
            temp_labels_dir.mkdir(parents=True, exist_ok=True)

        total_prelabeled_instances = 0
        task_manifest_items: List[Dict[str, Any]] = []

        try:
            # 1. Sao chép ảnh vào thư mục đóng gói
            for img_path in image_files:
                dst_img = temp_images_dir / img_path.name
                shutil.copy2(img_path.resolve(), dst_img)

                item_info: Dict[str, Any] = {
                    "image_name": img_path.name,
                    "stem": img_path.stem,
                    "has_prelabels": False,
                    "num_prelabeled_instances": 0,
                }

                # 2. Sinh Pre-labels nếu được yêu cầu và có mô hình
                if include_prelabels and model is not None:
                    try:
                        results = model(str(img_path), verbose=False, conf=min_confidence)
                        r = results[0]

                        label_lines: List[str] = []

                        # Tác vụ Segmentation
                        if task == "segment" and hasattr(r, "masks") and r.masks is not None:
                            polygons = r.masks.xyn  # normalized coordinates
                            classes = r.boxes.cls.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()

                            for cls_idx, conf, poly in zip(classes, confs, polygons):
                                if conf >= min_confidence and len(poly) >= 3:
                                    poly_coords = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in poly])
                                    label_lines.append(f"{int(cls_idx)} {poly_coords}")
                                    total_prelabeled_instances += 1

                        # Tác vụ Pose
                        elif task == "pose" and hasattr(r, "keypoints") and r.keypoints is not None:
                            kpts = r.keypoints.xyn.cpu().numpy()
                            boxes = r.boxes.xywhn.cpu().numpy()
                            classes = r.boxes.cls.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()

                            for cls_idx, conf, box, kpt_set in zip(classes, confs, boxes, kpts):
                                if conf >= min_confidence:
                                    kpt_tokens = []
                                    for k in kpt_set:
                                        # (x, y, visibility)
                                        vis = 2.0 if (k[0] > 0 and k[1] > 0) else 0.0
                                        kpt_tokens.extend([f"{k[0]:.6f}", f"{k[1]:.6f}", f"{vis:.1f}"])
                                    kpt_str = " ".join(kpt_tokens)
                                    label_lines.append(
                                        f"{int(cls_idx)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {kpt_str}"
                                    )
                                    total_prelabeled_instances += 1

                        # Tác vụ Detection chuẩn
                        elif hasattr(r, "boxes") and r.boxes is not None:
                            boxes = r.boxes.xywhn.cpu().numpy()
                            classes = r.boxes.cls.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()

                            for cls_idx, conf, box in zip(classes, confs, boxes):
                                if conf >= min_confidence:
                                    label_lines.append(
                                        f"{int(cls_idx)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                                    )
                                    total_prelabeled_instances += 1

                        if label_lines:
                            txt_path = temp_labels_dir / f"{img_path.stem}.txt"
                            txt_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                            item_info["has_prelabels"] = True
                            item_info["num_prelabeled_instances"] = len(label_lines)

                    except Exception as e:
                        print(f"Warning: Không thể sinh pre-label cho {img_path.name}: {e}")

                task_manifest_items.append(item_info)

            # 3. Ghi file manifest tổng thể của gói
            manifest_data = {
                "iteration": iteration,
                "task": task,
                "total_images": len(image_files),
                "include_prelabels": include_prelabels,
                "min_confidence_filter": min_confidence if include_prelabels else None,
                "total_prelabeled_instances": total_prelabeled_instances,
                "class_names": self.class_names,
                "items": task_manifest_items,
            }
            manifest_file = temp_dir / "task_manifest.json"
            manifest_file.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")

            # 4. Đóng gói toàn bộ thành file ZIP trong outbox/
            if zip_output_path.exists():
                zip_output_path.unlink()

            with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        full_p = Path(root) / file
                        rel_p = full_p.relative_to(temp_dir)
                        zf.write(full_p, rel_p)

            print(f"✓ Đã đóng gói thành công: {zip_output_path} ({len(image_files)} ảnh)")
            if include_prelabels:
                print(f"  (Đã kèm {total_prelabeled_instances} pre-annotations với Conf >= {min_confidence:.2f})")
            else:
                print("  (Gói chỉ chứa ảnh sạch thuần túy, không kèm pre-labels)")

            return PackagingResult(
                zip_path=str(zip_output_path),
                iteration=iteration,
                total_images=len(image_files),
                prelabels_included=include_prelabels,
                total_prelabeled_instances=total_prelabeled_instances,
                min_confidence_used=min_confidence,
                manifest_path=str(manifest_file),
            )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
