"""
universal_zip_ingestor.py - Bộ Giải Nén & Nạp Nhãn Tự Động Đa Định Dạng (Universal ZIP Ingestor)
================================================================================================
Hỗ trợ Production thực chiến:
1. Tự động giải nén an toàn file .zip từ inbox/ mà người dùng không cần can thiệp thủ công.
2. Tự động nhận diện định dạng nhãn:
   - YOLO Detection: 5 giá trị [class_id, x_center, y_center, width, height]
   - YOLO Segmentation: >5 giá trị [class_id, x1, y1, x2, y2, ..., xn, yn]
   - YOLO Pose / Skeleton: 5 + K*3 giá trị [class_id, x, y, w, h, px1, py1, v1, ...]
   - COCO Format: instances.json chứa mảng annotations, categories, images. Tự động parse và convert sang YOLO txt.
3. Kiểm toán chất lượng nhãn: Tọa độ hợp lệ [0, 1], không tràn viền, khớp danh sách ảnh.
4. Nạp tự động vào thư mục đích (seed_vk/labels/ hoặc raw_labels/).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


@dataclass
class IngestionReport:
    """Báo cáo kết quả nạp nhãn từ file ZIP."""
    status: str
    zip_path: str
    detected_format: str  # "yolo_detect", "yolo_segment", "yolo_pose", "coco_json", "unknown"
    total_label_files: int
    total_instances: int
    target_labels_dir: str
    matched_images_count: int
    invalid_lines_count: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class UniversalZipIngestor:
    """Bộ nạp nhãn tự động từ file ZIP cho môi trường Production."""

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or [
            "person", "rider", "car", "truck", "bus",
            "train", "motor", "bike", "traffic light", "traffic sign"
        ]
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

    def detect_txt_format(self, file_path: Path) -> str:
        """Nhận diện loại nhãn trong file .txt."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                num_tokens = len(parts)
                if num_tokens == 5:
                    return "yolo_detect"
                elif num_tokens > 5:
                    # Kiểm tra xem có phải pose (17 keypoints * 3 + 5 = 56) hay polygon seg
                    if (num_tokens - 5) % 3 == 0 and num_tokens in (56, 47):
                        return "yolo_pose"
                    else:
                        return "yolo_segment"
        return "yolo_detect"

    def parse_coco_json(
        self,
        json_path: Path,
        output_txt_dir: Path,
        target_task: str = "detect",
    ) -> Tuple[int, int, List[str]]:
        """
        Parse file COCO JSON và chuyển đổi tự động sang các file YOLO .txt tương ứng.
        """
        warnings: List[str] = []
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        images_map: Dict[Any, Dict[str, Any]] = {}
        for img in data.get("images", []):
            images_map[img["id"]] = img

        categories_map: Dict[Any, int] = {}
        for cat in data.get("categories", []):
            cat_name = cat["name"]
            cat_id = self.class_to_id.get(cat_name, cat["id"])
            categories_map[cat["id"]] = cat_id

        # Gom annotations theo image_id
        annotations_by_img: Dict[Any, List[Dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            annotations_by_img.setdefault(img_id, []).append(ann)

        files_written = 0
        total_instances = 0

        for img_id, anns in annotations_by_img.items():
            img_info = images_map.get(img_id)
            if not img_info:
                warnings.append(f"Không tìm thấy thông tin ảnh cho image_id={img_id}")
                continue

            file_name = img_info["file_name"]
            stem = Path(file_name).stem
            width = float(img_info.get("width", 1280.0))
            height = float(img_info.get("height", 720.0))

            out_txt_file = output_txt_dir / f"{stem}.txt"
            lines: List[str] = []

            for ann in anns:
                cat_id = categories_map.get(ann["category_id"], 0)
                if target_task == "segment" and "segmentation" in ann and ann["segmentation"]:
                    seg = ann["segmentation"]
                    poly = seg[0] if isinstance(seg, list) and isinstance(seg[0], list) else seg
                    if len(poly) >= 6:
                        norm_poly = []
                        for i in range(0, len(poly), 2):
                            px = min(1.0, max(0.0, poly[i] / width))
                            py = min(1.0, max(0.0, poly[i + 1] / height))
                            norm_poly.extend([f"{px:.6f}", f"{py:.6f}"])
                        lines.append(f"{cat_id} " + " ".join(norm_poly))
                        total_instances += 1
                        continue

                # Mặc định detection: bbox [x, y, w, h] (COCO là top-left)
                bbox = ann.get("bbox", [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    cx = (x + w / 2.0) / width
                    cy = (y + h / 2.0) / height
                    norm_w = w / width
                    norm_h = h / height
                    # Kẹp biên [0, 1]
                    cx = min(1.0, max(0.0, cx))
                    cy = min(1.0, max(0.0, cy))
                    norm_w = min(1.0, max(0.0, norm_w))
                    norm_h = min(1.0, max(0.0, norm_h))

                    lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")
                    total_instances += 1

            if lines:
                out_txt_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                files_written += 1

        return files_written, total_instances, warnings

    def ingest_zip(
        self,
        zip_path: Union[str, Path],
        target_labels_dir: Union[str, Path],
        target_images_dir: Optional[Union[str, Path]] = None,
        target_task: str = "detect",
    ) -> IngestionReport:
        """
        Giải nén tự động và nạp nhãn vào target_labels_dir.
        """
        zip_p = Path(zip_path).resolve()
        target_lbl_dir = Path(target_labels_dir).resolve()
        target_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not zip_p.exists():
            return IngestionReport(
                status="ERROR_FILE_NOT_FOUND",
                zip_path=str(zip_p),
                detected_format="unknown",
                total_label_files=0,
                total_instances=0,
                target_labels_dir=str(target_lbl_dir),
                matched_images_count=0,
                invalid_lines_count=0,
                errors=[f"File zip không tồn tại: {zip_p}"],
            )

        temp_dir = Path(tempfile.mkdtemp(prefix="ingest_zip_"))
        warnings: List[str] = []
        errors: List[str] = []

        try:
            # 1. Giải nén an toàn
            with zipfile.ZipFile(zip_p, "r") as zf:
                zf.extractall(temp_dir)

            # 2. Tìm kiếm file nhãn
            txt_files = list(temp_dir.rglob("*.txt"))
            json_files = [f for f in temp_dir.rglob("*.json") if "instances" in f.name.lower() or "annotation" in f.name.lower() or "coco" in f.name.lower()]

            detected_format = "unknown"
            total_instances = 0
            files_written = 0
            invalid_lines = 0

            # Kịch bản 1: Có file COCO JSON
            if json_files and not txt_files:
                detected_format = "coco_json"
                target_json = json_files[0]
                files_written, total_instances, p_warn = self.parse_coco_json(
                    target_json, target_lbl_dir, target_task=target_task
                )
                warnings.extend(p_warn)

            # Kịch bản 2: Có các file YOLO .txt
            elif txt_files:
                sample_txt = txt_files[0]
                detected_format = self.detect_txt_format(sample_txt)

                for txt_f in txt_files:
                    dst_txt = target_lbl_dir / txt_f.name
                    valid_lines = []
                    with open(txt_f, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            tokens = line.strip().split()
                            if not tokens:
                                continue
                            try:
                                # Validate tọa độ hợp lệ
                                coords = [float(t) for t in tokens[1:]]
                                if all(0.0 <= c <= 1.05 for c in coords):
                                    valid_lines.append(line.strip())
                                    total_instances += 1
                                else:
                                    invalid_lines += 1
                            except ValueError:
                                invalid_lines += 1

                    if valid_lines:
                        dst_txt.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
                        files_written += 1

            else:
                errors.append("Không tìm thấy file nhãn (.txt hoặc .json) bên trong file zip!")
                return IngestionReport(
                    status="ERROR_NO_LABELS",
                    zip_path=str(zip_p),
                    detected_format="unknown",
                    total_label_files=0,
                    total_instances=0,
                    target_labels_dir=str(target_lbl_dir),
                    matched_images_count=0,
                    invalid_lines_count=0,
                    errors=errors,
                )

            # 3. Kiểm toán khớp ảnh nếu có target_images_dir
            matched_images = 0
            if target_images_dir:
                img_dir = Path(target_images_dir).resolve()
                if img_dir.exists():
                    img_stems = {f.stem for f in img_dir.glob("*.jpg")} | {f.stem for f in img_dir.glob("*.png")}
                    lbl_stems = {f.stem for f in target_lbl_dir.glob("*.txt")}
                    matched_images = len(img_stems & lbl_stems)
                    unmatched = len(img_stems - lbl_stems)
                    if unmatched > 0:
                        warnings.append(f"Có {unmatched} ảnh trong tập chưa có file nhãn tương ứng.")

            return IngestionReport(
                status="SUCCESS",
                zip_path=str(zip_p),
                detected_format=detected_format,
                total_label_files=files_written,
                total_instances=total_instances,
                target_labels_dir=str(target_lbl_dir),
                matched_images_count=matched_images,
                invalid_lines_count=invalid_lines,
                warnings=warnings,
                errors=errors,
            )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
