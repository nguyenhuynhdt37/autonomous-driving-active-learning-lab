"""
splits.py - Shared Pre-Branching Frozen Test & Val Splits (Tournament Benchmark)
==================================================================================
Module cốt lõi của Phase 03: Đấu trường thực nghiệm song song 6 Chiến lược phân chia
dữ liệu (Candidate Splitting Strategies) trên 6.994 ảnh (4.277 video sequences).

Hỗ trợ đầy đủ:
1. GroupAwareVideoSplitter: Gom cụm theo chuỗi video BDD10K để chống rò rỉ bối cảnh thời gian.
2. 6 Chiến Lược Tuyển Chọn Đối Đầu:
   - C1: RandomGroupSplitter (Baseline ngẫu nhiên truyền thống).
   - C2: DirectMetadataStratifiedSplitter (Lấy trực tiếp theo tags lát cắt Phase 02).
   - C3: StratifiedKCenterSplitter (Pure k-Center Greedy bám biên Minimax).
   - C4: SubmodularFacilityLocationSplitter (Pure Lazy Greedy Facility Location dàn đều mật độ).
   - C5: HybridMixedSplitter 80/20 (80% Facility Location + 20% k-Center Greedy).
   - C6: HybridMixedSplitter 70/30 (70% Facility Location + 30% k-Center Greedy).
3. Dual Ground-Truth Scoring Engine:
   - Trục 1: Chân lý nhãn gốc UC Berkeley BDD10K (bdd10k_metadata.json - Read-Only).
   - Trục 2: Không gian đặc trưng vật lý 11 chiều Phase 02 (data_profiling_manifest.json).
   - Trục 3: Rào chắn an toàn tuyệt đối Gatekeeper (Video ID Overlap == 0, pHash HD < 5 == 0).
4. Composite Benchmark Score (CBS) & Auto-Selection: Tự động chọn ra Winner đạt điểm cao nhất.
5. AntiLeakageAuditor: Kiểm toán rò rỉ video và quét chéo pHash Hamming Distance đa luồng.
6. SymlinkSplitMaterializer: Khởi tạo thư mục và tạo Symlinks 0 KB đĩa (tuân thủ AGENTS.md).
7. SHA256ChecksumSealer: Niêm phong toàn vẹn bất biến bộ dữ liệu.
8. Sinh file cấu hình YOLO bdd10k_splits.yaml linh hoạt.
"""

from __future__ import annotations

import heapq
import hashlib
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

# 10 Lớp Giao Thông Chuẩn BDD10K (0-9 trong file nhãn YOLO)
BDD10K_CLASS_NAMES = [
    "person",        # 0
    "rider",         # 1
    "car",           # 2
    "truck",         # 3
    "bus",           # 4
    "train",         # 5
    "motor",         # 6
    "bike",          # 7
    "traffic light", # 8
    "traffic sign"   # 9
]

# 11 Chiều Điểm Số Liên Tục từ Phase 02 Manifest
CONTINUOUS_FEATURE_KEYS = [
    "brightness_v_mean",
    "brightness_p50",
    "tenengrad_sharpness",
    "laplacian_variance",
    "fft_high_freq_ratio",
    "dark_channel_prior",
    "scale_shannon_entropy",
    "occlusion_pairs",
    "max_occlusion_chain",
    "vehicle_count",
    "total_objects"
]


@dataclass
class VideoSequenceRecord:
    """Đại diện cho một chuỗi video sequence BDD10K bất khả phân ly."""
    video_id: str
    image_names: List[str]
    num_images: int
    raw_feature_mean: np.ndarray
    norm_feature_mean: np.ndarray
    primary_stratum: str
    is_night_ratio: float
    is_small_ratio: float
    is_occluded_ratio: float
    is_day_ratio: float
    class_counts: Dict[str, int]



@dataclass
class SplitResult:
    """Kết quả phân chia của một chiến lược."""
    strategy_name: str
    test_images: List[str]
    val_images: List[str]
    pool_images: List[str]
    test_videos: Set[str]
    val_videos: Set[str]
    pool_videos: Set[str]
    seed_images: List[str] = field(default_factory=list)
    seed_videos: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_test(self) -> int:
        return len(self.test_images)

    @property
    def num_val(self) -> int:
        return len(self.val_images)

    @property
    def num_seed(self) -> int:
        return len(self.seed_images)

    @property
    def num_pool(self) -> int:
        return len(self.pool_images)

    @property
    def total_images(self) -> int:
        return self.num_test + self.num_val + self.num_seed + self.num_pool


class GroupAwareVideoSplitter:
    """
    Quản lý việc gom nhóm 6.994 ảnh sạch thành 4.277 video sequences và
    chuẩn hóa không gian đặc trưng 11 chiều để phục vụ tuyển chọn.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        metadata_path: Union[str, Path],
        random_seed: int = 42
    ):
        self.manifest_path = Path(manifest_path)
        self.metadata_path = Path(metadata_path)
        self.random_seed = random_seed

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Không tìm thấy manifest Phase 02: {self.manifest_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Không tìm thấy metadata gốc BDD10K: {self.metadata_path}")

        self._load_and_preprocess()

    def _load_and_preprocess(self) -> None:
        """Đọc manifest, metadata gốc và xây dựng cấu trúc nhóm video."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest: Dict[str, Any] = json.load(f)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            raw_meta: Union[Dict[str, Any], List[Dict[str, Any]]] = json.load(f)

        # Xây dựng metadata lookup map O(1) theo stem và tên ảnh
        self.metadata_map: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_meta, dict):
            for k, v in raw_meta.items():
                self.metadata_map[k] = v
                img_n = v.get("image_name")
                if img_n:
                    self.metadata_map[img_n] = v
        elif isinstance(raw_meta, list):
            for item in raw_meta:
                name = item.get("name") or item.get("image_name", "")
                stem = Path(name).stem
                self.metadata_map[stem] = item
                self.metadata_map[name] = item

        self.all_images = sorted(list(self.manifest.keys()))
        self.total_images_count = len(self.all_images)

        # 1. Gom nhóm ảnh theo video_id (prefix trước dấu gạch ngang đầu tiên)
        video_to_images: Dict[str, List[str]] = {}
        for img_name in self.all_images:
            vid = img_name.split("-")[0]
            video_to_images.setdefault(vid, []).append(img_name)

        self.unique_video_ids = sorted(list(video_to_images.keys()))
        self.total_videos_count = len(self.unique_video_ids)

        # 2. Trích xuất ma trận đặc trưng 11 chiều cho từng ảnh
        img_features_list = []
        for img_name in self.all_images:
            item = self.manifest[img_name]
            scores = item.get("scores", {})
            vec = [float(scores.get(k, 0.0)) for k in CONTINUOUS_FEATURE_KEYS]
            img_features_list.append(vec)

        self.raw_image_features = np.array(img_features_list, dtype=np.float32)

        # Chuẩn hóa StandardScaler cho toàn bộ không gian 11 chiều
        self.scaler = StandardScaler()
        self.norm_image_features = self.scaler.fit_transform(self.raw_image_features)

        # Ánh xạ nhanh từ image_name -> normalized feature vector
        self.image_to_norm_feat: Dict[str, np.ndarray] = {
            img_name: self.norm_image_features[i]
            for i, img_name in enumerate(self.all_images)
        }
        self.image_to_raw_feat: Dict[str, np.ndarray] = {
            img_name: self.raw_image_features[i]
            for i, img_name in enumerate(self.all_images)
        }

        # 3. Phân vùng 4 Lát Cắt Chuẩn (Mutually Exclusive Strata: Night, Small, Occlusion, Day)
        # Tách các ảnh ban ngày (4.316 ảnh) để phân định rõ ràng giữa Small Objects, Occlusion và Day thường
        day_imgs = [
            img for img in self.all_images
            if not self.manifest[img].get("is_night", False)
        ]

        # Top 1.400 ảnh ban ngày có độ phức tạp quy mô (scale_shannon_entropy) cao nhất đại diện cho lát cắt Small Object
        sorted_by_entropy = sorted(
            day_imgs,
            key=lambda img: float(self.manifest[img].get("scores", {}).get("scale_shannon_entropy", 0.0)),
            reverse=True
        )
        small_set = set(sorted_by_entropy[:1400])

        # Với các ảnh ban ngày còn lại, top 1.400 ảnh có mức độ che khuất (occlusion_pairs) cao nhất đại diện cho Occlusion
        remaining_day = [img for img in day_imgs if img not in small_set]
        sorted_by_occlusion = sorted(
            remaining_day,
            key=lambda img: float(self.manifest[img].get("scores", {}).get("occlusion_pairs", 0)),
            reverse=True
        )
        occl_set = set(sorted_by_occlusion[:1400])

        self.image_stratum_map: Dict[str, str] = {}
        for img_name in self.all_images:
            if self.manifest[img_name].get("is_night", False):
                self.image_stratum_map[img_name] = "night"
            elif img_name in small_set:
                self.image_stratum_map[img_name] = "small"
            elif img_name in occl_set:
                self.image_stratum_map[img_name] = "occlusion"
            else:
                self.image_stratum_map[img_name] = "day"

        # 4. Tổng hợp thông tin cấp độ chuỗi video
        self.video_records: Dict[str, VideoSequenceRecord] = {}
        video_norm_feats = []

        for vid in self.unique_video_ids:
            imgs = video_to_images[vid]
            n_imgs = len(imgs)

            raw_feats = np.array([self.image_to_raw_feat[img] for img in imgs], dtype=np.float32)
            norm_feats = np.array([self.image_to_norm_feat[img] for img in imgs], dtype=np.float32)

            raw_mean = np.mean(raw_feats, axis=0)
            norm_mean = np.mean(norm_feats, axis=0)
            video_norm_feats.append(norm_mean)

            # Xác định stratum chủ đạo của video bằng Majority Vote
            v_strata = [self.image_stratum_map[im] for im in imgs]
            order = ["day", "occlusion", "small", "night"]
            p_stratum = max(set(v_strata), key=lambda s: (v_strata.count(s), order.index(s)))

            # Đếm tỷ lệ lát cắt
            night_cnt = sum(1 for img in imgs if self.manifest[img].get("is_night", False))
            small_cnt = sum(1 for img in imgs if self.manifest[img].get("is_small_object_slice", False))
            occl_cnt = sum(1 for img in imgs if self.manifest[img].get("is_high_occlusion", False))

            # Đếm 10 classes từ metadata Berkeley
            class_cnt: Dict[str, int] = {c: 0 for c in BDD10K_CLASS_NAMES}
            for img in imgs:
                stem = Path(img).stem
                meta_item = self.metadata_map.get(stem) or self.metadata_map.get(img) or {}
                for obj in meta_item.get("objects", []):
                    cat = obj.get("category")
                    if cat in class_cnt:
                        class_cnt[cat] += 1

            rec = VideoSequenceRecord(
                video_id=vid,
                image_names=imgs,
                num_images=n_imgs,
                raw_feature_mean=raw_mean,
                norm_feature_mean=norm_mean,
                primary_stratum=p_stratum,
                is_night_ratio=float(night_cnt / n_imgs),
                is_small_ratio=float(small_cnt / n_imgs),
                is_occluded_ratio=float(occl_cnt / n_imgs),
                is_day_ratio=float((n_imgs - night_cnt) / n_imgs),
                class_counts=class_cnt
            )
            self.video_records[vid] = rec

        self.video_feature_matrix = np.array(video_norm_feats, dtype=np.float32)


        # 4. Tính toán ma trận khoảng cách và tương đồng Gaussian RBF giữa các video
        # Dùng median heuristic cho sigma
        sample_subset = self.video_feature_matrix[:min(500, self.total_videos_count)]
        sample_dists = cdist(sample_subset, sample_subset, metric="euclidean")
        nonzero_dists = sample_dists[sample_dists > 0]
        self.rbf_sigma = float(np.median(nonzero_dists)) if len(nonzero_dists) > 0 else 1.0

        # Lưu ma trận khoảng cách pairwise giữa các video
        self.video_pairwise_dists = cdist(self.video_feature_matrix, self.video_feature_matrix, metric="euclidean").astype(np.float32)
        # Ma trận tương đồng RBF Kernel Sim(i, j) = exp(-dist^2 / (2 * sigma^2))
        self.video_similarity_matrix = np.exp(
            -(self.video_pairwise_dists ** 2) / (2.0 * (self.rbf_sigma ** 2))
        ).astype(np.float32)

        # Trọng số số lượng ảnh của từng video
        self.video_weights = np.array([self.video_records[vid].num_images for vid in self.unique_video_ids], dtype=np.float32)

        # Ánh xạ index <-> video_id
        self.vid_to_idx = {vid: i for i, vid in enumerate(self.unique_video_ids)}
        self.idx_to_vid = {i: vid for i, vid in enumerate(self.unique_video_ids)}

        # Gom nhóm video sequences theo 4 phân tầng lát cắt chuẩn
        self.stratum_to_vids: Dict[str, List[str]] = {
            "night": [],
            "small": [],
            "occlusion": [],
            "day": []
        }
        for vid in self.unique_video_ids:
            self.stratum_to_vids[self.video_records[vid].primary_stratum].append(vid)

    def select_stratified(
        self,
        method_fn: Any,
        target_total: int,
        excluded_vids: Set[str]
    ) -> List[str]:
        """
        Bốc mẫu phân tầng theo 4 Quota chuẩn (35% Đêm, 20% Nhỏ, 20% Che, 25% Ngày).
        Áp dụng thuật toán method_fn(candidate_vids, quota) độc lập trên từng tầng,
        đảm bảo bảo toàn tỷ lệ lát cắt tuyệt đối và tối ưu hóa không gian đồng nhất.
        """
        q_night = int(round(target_total * 0.35))
        q_small = int(round(target_total * 0.20))
        q_occl = int(round(target_total * 0.20))
        q_day = target_total - (q_night + q_small + q_occl)

        quotas = [
            ("night", q_night),
            ("small", q_small),
            ("occlusion", q_occl),
            ("day", q_day)
        ]

        chosen: List[str] = []
        chosen_set = set(excluded_vids)

        for stratum_name, quota in quotas:
            pool = [v for v in self.stratum_to_vids[stratum_name] if v not in chosen_set]
            v_selected = method_fn(pool, quota)
            chosen.extend(v_selected)
            chosen_set.update(v_selected)

        # Tinh chỉnh kiểm soát chặt chẽ tổng số ảnh đạt đúng target_total
        avail_pool = set(self.unique_video_ids) - excluded_vids
        return self.fine_tune_video_selection(chosen, target_total, avail_pool)

    def fine_tune_video_selection(
        self,
        chosen_vids: List[str],
        target_count: int,
        available_pool_vids: Set[str]
    ) -> List[str]:
        """
        Tinh chỉnh tập video sequence đã chọn sao cho TỔNG SỐ LƯỢNG ẢNH
        đạt CHÍNH XÁC mục tiêu target_count (1.400 cho Test hoặc 700 cho Val).
        Tận dụng hàng ngàn video có đúng 1 ảnh để bù trừ hoàn hảo không thừa không thiếu.
        """
        chosen_set = set(chosen_vids)
        current_images = sum(self.video_records[v].num_images for v in chosen_set)

        # Tách danh sách video 1-ảnh còn trống trong pool
        single_frame_pool = [
            v for v in available_pool_vids
            if v not in chosen_set and self.video_records[v].num_images == 1
        ]
        # Tách danh sách video 1-ảnh đã chọn
        single_frame_chosen = [
            v for v in chosen_vids
            if self.video_records[v].num_images == 1
        ]

        if current_images < target_count:
            # Cần thêm (target_count - current_images) ảnh
            diff = target_count - current_images
            if len(single_frame_pool) >= diff:
                to_add = single_frame_pool[:diff]
                chosen_set.update(to_add)
            else:
                # Nếu thiếu, thêm bất kỳ video nào vừa vặn
                for v in sorted(available_pool_vids - chosen_set, key=lambda x: self.video_records[x].num_images):
                    n_v = self.video_records[v].num_images
                    if current_images + n_v <= target_count:
                        chosen_set.add(v)
                        current_images += n_v
                    if current_images == target_count:
                        break
        elif current_images > target_count:
            # Cần loại bỏ (current_images - target_count) ảnh
            diff = current_images - target_count
            if len(single_frame_chosen) >= diff:
                to_remove = single_frame_chosen[-diff:]
                for v in to_remove:
                    chosen_set.remove(v)
            else:
                # Loại bỏ từ cuối danh sách cho đến khi vừa vặn
                for v in reversed(chosen_vids):
                    if current_images - self.video_records[v].num_images >= target_count:
                        chosen_set.remove(v)
                        current_images -= self.video_records[v].num_images
                    if current_images == target_count:
                        break

        # Đảm bảo duy trì thứ tự ban đầu cho các video đã chọn
        final_list = [v for v in chosen_vids if v in chosen_set]
        # Thêm các video mới được bổ sung
        for v in chosen_set:
            if v not in final_list:
                final_list.append(v)

        return final_list


# ==============================================================================
# 2. KHUNG LỚP TRỪU TƯỢNG & 6 CHIẾN LƯỢC PHÂN CHIA (SPLITTING STRATEGIES)
# ==============================================================================

class BaseSplitStrategy(ABC):
    """Lớp trừu tượng đại diện cho một chiến lược phân chia."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def split(
        self,
        splitter: GroupAwareVideoSplitter,
        target_test_images: int = 1400,
        target_val_images: int = 700,
        target_seed_images: int = 500
    ) -> SplitResult:
        pass


class RandomGroupSplitter(BaseSplitStrategy):
    """
    Chiến lược 1: Random Group Split (Baseline ngẫu nhiên truyền thống).
    Phân bổ các video sequence ngẫu nhiên không có tối ưu hóa không gian.
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="1. Random Video Split (Baseline)")
        self.seed = seed

    def split(
        self,
        splitter: GroupAwareVideoSplitter,
        target_test_images: int = 1400,
        target_val_images: int = 700,
        target_seed_images: int = 500
    ) -> SplitResult:
        rng = np.random.RandomState(self.seed)
        shuffled_vids = list(splitter.unique_video_ids)
        rng.shuffle(shuffled_vids)

        test_vids = []
        cur_test_imgs = 0
        idx = 0

        # Lấy video cho test
        while idx < len(shuffled_vids) and cur_test_imgs < target_test_images:
            v = shuffled_vids[idx]
            test_vids.append(v)
            cur_test_imgs += splitter.video_records[v].num_images
            idx += 1

        # Tinh chỉnh chính xác 1.400 ảnh Test
        test_vids = splitter.fine_tune_video_selection(
            test_vids, target_test_images, set(splitter.unique_video_ids)
        )
        test_set = set(test_vids)

        # Lấy video cho val từ tập còn lại
        remaining_vids = [v for v in shuffled_vids if v not in test_set]
        val_vids = []
        cur_val_imgs = 0
        idx = 0
        while idx < len(remaining_vids) and cur_val_imgs < target_val_images:
            v = remaining_vids[idx]
            val_vids.append(v)
            cur_val_imgs += splitter.video_records[v].num_images
            idx += 1

        # Tinh chỉnh chính xác 700 ảnh Val
        val_vids = splitter.fine_tune_video_selection(
            val_vids, target_val_images, set(remaining_vids)
        )
        val_set = set(val_vids)

        # Lấy video cho seed_v0 từ tập còn lại
        pool_candidates = [v for v in remaining_vids if v not in val_set]
        seed_vids = []
        cur_seed_imgs = 0
        idx = 0
        while idx < len(pool_candidates) and cur_seed_imgs < target_seed_images:
            v = pool_candidates[idx]
            seed_vids.append(v)
            cur_seed_imgs += splitter.video_records[v].num_images
            idx += 1

        # Tinh chỉnh chính xác 500 ảnh Seed
        seed_vids = splitter.fine_tune_video_selection(
            seed_vids, target_seed_images, set(pool_candidates)
        )
        seed_set = set(seed_vids)

        # Phần còn lại vào Unlabeled Pool (4.394 ảnh)
        pool_vids = [v for v in pool_candidates if v not in seed_set]
        pool_set = set(pool_vids)

        # Thu thập danh sách ảnh
        test_imgs = [img for v in test_vids for img in splitter.video_records[v].image_names]
        val_imgs = [img for v in val_vids for img in splitter.video_records[v].image_names]
        seed_imgs = [img for v in seed_vids for img in splitter.video_records[v].image_names]
        pool_imgs = [img for v in pool_vids for img in splitter.video_records[v].image_names]

        return SplitResult(
            strategy_name=self.name,
            test_images=test_imgs,
            val_images=val_imgs,
            pool_images=pool_imgs,
            test_videos=test_set,
            val_videos=val_set,
            pool_videos=pool_set,
            seed_images=seed_imgs,
            seed_videos=seed_set,
            metadata={"seed": self.seed}
        )


class DirectMetadataStratifiedSplitter(BaseSplitStrategy):
    """
    Chiến lược 2: Direct Metadata Phase 02 Stratified.
    Bốc mẫu phân tầng ngẫu nhiên TRỰC TIẾP theo các cờ lát cắt của Phase 02:
    - 35% Đêm (490 ảnh Test / 245 ảnh Val / 175 ảnh Seed)
    - 20% Nhỏ (280 ảnh Test / 140 ảnh Val / 100 ảnh Seed)
    - 20% Che (280 ảnh Test / 140 ảnh Val / 100 ảnh Seed)
    - 25% Ngày (350 ảnh Test / 175 ảnh Val / 125 ảnh Seed)
    Không tối ưu hóa khoảng cách không gian đặc trưng.
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="2. Direct Metadata Phase 02 Stratified")
        self.seed = seed

    def _run_random_selection(
        self,
        splitter: GroupAwareVideoSplitter,
        target_images: int,
        candidate_vids: List[str]
    ) -> List[str]:
        rng = np.random.RandomState(self.seed)
        shuffled = list(candidate_vids)
        rng.shuffle(shuffled)
        chosen = []
        cur_imgs = 0
        for v in shuffled:
            if cur_imgs >= target_images:
                break
            chosen.append(v)
            cur_imgs += splitter.video_records[v].num_images
        return splitter.fine_tune_video_selection(chosen, target_images, set(candidate_vids))

    def split(
        self,
        splitter: GroupAwareVideoSplitter,
        target_test_images: int = 1400,
        target_val_images: int = 700,
        target_seed_images: int = 500
    ) -> SplitResult:
        test_vids = splitter.select_stratified(
            lambda cands, q: self._run_random_selection(splitter, q, cands),
            target_test_images,
            set()
        )
        test_set = set(test_vids)

        val_vids = splitter.select_stratified(
            lambda cands, q: self._run_random_selection(splitter, q, cands),
            target_val_images,
            test_set
        )
        val_set = set(val_vids)

        seed_vids = splitter.select_stratified(
            lambda cands, q: self._run_random_selection(splitter, q, cands),
            target_seed_images,
            test_set | val_set
        )
        seed_set = set(seed_vids)

        pool_vids = [
            v for v in splitter.unique_video_ids
            if v not in test_set and v not in val_set and v not in seed_set
        ]
        pool_set = set(pool_vids)

        test_imgs = [img for v in test_vids for img in splitter.video_records[v].image_names]
        val_imgs = [img for v in val_vids for img in splitter.video_records[v].image_names]
        seed_imgs = [img for v in seed_vids for img in splitter.video_records[v].image_names]
        pool_imgs = [img for v in pool_vids for img in splitter.video_records[v].image_names]

        return SplitResult(
            strategy_name=self.name,
            test_images=test_imgs,
            val_images=val_imgs,
            pool_images=pool_imgs,
            test_videos=test_set,
            val_videos=val_set,
            pool_videos=pool_set,
            seed_images=seed_imgs,
            seed_videos=seed_set,
            metadata={"seed": self.seed}
        )


class StratifiedKCenterSplitter(BaseSplitStrategy):
    """
    Chiến lược 3: Pure k-Center Greedy (Bám biên Minimax).
    Cực tiểu hóa bán kính bao phủ lớn nhất: min_S max_x min_{s in S} d(x, s).
    Bắt các ca biên khắc nghiệt trên không gian 11 chiều trong từng phân tầng lát cắt.
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="3. Pure k-Center Greedy")
        self.seed = seed

    def _run_kcenter_greedy(
        self,
        splitter: GroupAwareVideoSplitter,
        target_images: int,
        candidate_vids: List[str],
        init_center_vid: Optional[str] = None
    ) -> List[str]:
        n_candidates = len(candidate_vids)
        if n_candidates == 0 or target_images <= 0:
            return []

        cand_indices = [splitter.vid_to_idx[v] for v in candidate_vids]
        sub_dists = splitter.video_pairwise_dists[cand_indices, :][:, cand_indices]

        # Chọn tâm đầu tiên: nếu không chỉ định, chọn video xa trọng tâm nhất để khởi tạo bám biên
        if init_center_vid is None or init_center_vid not in candidate_vids:
            center_idx_in_sub = int(np.argmax(np.mean(sub_dists, axis=1)))
        else:
            center_idx_in_sub = candidate_vids.index(init_center_vid)

        selected_sub_indices = [center_idx_in_sub]
        min_dists = sub_dists[:, center_idx_in_sub].copy()

        cur_imgs = splitter.video_records[candidate_vids[center_idx_in_sub]].num_images
        selected_set = {center_idx_in_sub}

        while cur_imgs < target_images and len(selected_sub_indices) < n_candidates:
            # Chọn điểm có min_dist lớn nhất (xa tập đã chọn nhất)
            min_dists[list(selected_set)] = -1.0
            best_idx = int(np.argmax(min_dists))

            selected_sub_indices.append(best_idx)
            selected_set.add(best_idx)
            cur_imgs += splitter.video_records[candidate_vids[best_idx]].num_images

            # Cập nhật min_dist
            min_dists = np.minimum(min_dists, sub_dists[:, best_idx])

        chosen_vids = [candidate_vids[i] for i in selected_sub_indices]
        return splitter.fine_tune_video_selection(
            chosen_vids, target_images, set(candidate_vids)
        )

    def split(
        self,
        splitter: GroupAwareVideoSplitter,
        target_test_images: int = 1400,
        target_val_images: int = 700,
        target_seed_images: int = 500
    ) -> SplitResult:
        test_vids = splitter.select_stratified(
            lambda cands, q: self._run_kcenter_greedy(splitter, q, cands),
            target_test_images,
            set()
        )
        test_set = set(test_vids)

        val_vids = splitter.select_stratified(
            lambda cands, q: self._run_kcenter_greedy(splitter, q, cands),
            target_val_images,
            test_set
        )
        val_set = set(val_vids)

        seed_vids = splitter.select_stratified(
            lambda cands, q: self._run_kcenter_greedy(splitter, q, cands),
            target_seed_images,
            test_set | val_set
        )
        seed_set = set(seed_vids)

        pool_vids = [
            v for v in splitter.unique_video_ids
            if v not in test_set and v not in val_set and v not in seed_set
        ]
        pool_set = set(pool_vids)

        test_imgs = [img for v in test_vids for img in splitter.video_records[v].image_names]
        val_imgs = [img for v in val_vids for img in splitter.video_records[v].image_names]
        seed_imgs = [img for v in seed_vids for img in splitter.video_records[v].image_names]
        pool_imgs = [img for v in pool_vids for img in splitter.video_records[v].image_names]

        return SplitResult(
            strategy_name=self.name,
            test_images=test_imgs,
            val_images=val_imgs,
            pool_images=pool_imgs,
            test_videos=test_set,
            val_videos=val_set,
            pool_videos=pool_set,
            seed_images=seed_imgs,
            seed_videos=seed_set,
            metadata={"algorithm": "stratified_k_center_greedy"}
        )


class SubmodularFacilityLocationSplitter(BaseSplitStrategy):
    """
    Chiến lược 4: Pure Submodular Facility Location (Lazy Greedy).
    Cực đại hóa độ tương đồng dàn đều khắp các cụm mật độ tự nhiên:
    max_S sum_x w_x max_{s in S} Sim(x, s) trong từng phân tầng lát cắt.
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="4. Pure Facility Location")
        self.seed = seed

    def _run_lazy_facility_location(
        self,
        splitter: GroupAwareVideoSplitter,
        target_images: int,
        candidate_vids: List[str]
    ) -> List[str]:
        n_cand = len(candidate_vids)
        if n_cand == 0 or target_images <= 0:
            return []

        cand_indices = [splitter.vid_to_idx[v] for v in candidate_vids]
        sub_sim = splitter.video_similarity_matrix[cand_indices, :][:, cand_indices]
        weights = splitter.video_weights[cand_indices]

        max_sim = np.zeros(n_cand, dtype=np.float32)

        # Khởi tạo heap chứa (-gain, node_idx, step_evaluated)
        heap = []
        for i in range(n_cand):
            initial_gain = float(np.sum(sub_sim[:, i] * weights))
            heap.append((-initial_gain, i, 0))
        heapq.heapify(heap)

        selected_sub_indices = []
        cur_imgs = 0
        step = 0

        while cur_imgs < target_images and heap:
            while True:
                neg_gain, node_idx, eval_step = heapq.heappop(heap)
                if eval_step == step:
                    # Gain này đã được cập nhật tại step hiện tại -> tối ưu theo submodular diminishing returns!
                    selected_sub_indices.append(node_idx)
                    vid = candidate_vids[node_idx]
                    cur_imgs += splitter.video_records[vid].num_images
                    max_sim = np.maximum(max_sim, sub_sim[:, node_idx])
                    step += 1
                    break
                else:
                    # Tính lại actual gain
                    actual_gain = float(np.sum(np.maximum(0, sub_sim[:, node_idx] - max_sim) * weights))
                    heapq.heappush(heap, (-actual_gain, node_idx, step))

        chosen_vids = [candidate_vids[i] for i in selected_sub_indices]
        return splitter.fine_tune_video_selection(
            chosen_vids, target_images, set(candidate_vids)
        )

    def split(
        self,
        splitter: GroupAwareVideoSplitter,
        target_test_images: int = 1400,
        target_val_images: int = 700,
        target_seed_images: int = 500
    ) -> SplitResult:
        test_vids = splitter.select_stratified(
            lambda cands, q: self._run_lazy_facility_location(splitter, q, cands),
            target_test_images,
            set()
        )
        test_set = set(test_vids)

        val_vids = splitter.select_stratified(
            lambda cands, q: self._run_lazy_facility_location(splitter, q, cands),
            target_val_images,
            test_set
        )
        val_set = set(val_vids)

        seed_vids = splitter.select_stratified(
            lambda cands, q: self._run_lazy_facility_location(splitter, q, cands),
            target_seed_images,
            test_set | val_set
        )
        seed_set = set(seed_vids)

        pool_vids = [
            v for v in splitter.unique_video_ids
            if v not in test_set and v not in val_set and v not in seed_set
        ]
        pool_set = set(pool_vids)

        test_imgs = [img for v in test_vids for img in splitter.video_records[v].image_names]
        val_imgs = [img for v in val_vids for img in splitter.video_records[v].image_names]
        seed_imgs = [img for v in seed_vids for img in splitter.video_records[v].image_names]
        pool_imgs = [img for v in pool_vids for img in splitter.video_records[v].image_names]

        return SplitResult(
            strategy_name=self.name,
            test_images=test_imgs,
            val_images=val_imgs,
            pool_images=pool_imgs,
            test_videos=test_set,
            val_videos=val_set,
            pool_videos=pool_set,
            seed_images=seed_imgs,
            seed_videos=seed_set,
            metadata={"algorithm": "stratified_submodular_facility_location"}
        )


class HybridMixedSplitter(BaseSplitStrategy):
    """
    Chiến lược 5 & 6: Hybrid Mixed Selector.
    Kết hợp tinh hoa cả 2 trường phái trong từng phân tầng lát cắt:
    - alpha% ảnh: Dàn đều mật độ bằng Submodular Facility Location.
    - (1 - alpha)% ảnh: Săn ca biên cực đoan bằng k-Center Greedy.
    Cân bằng tối thượng giữa độ đại diện trung thực và độ bao phủ góc xa!
    """

    def __init__(self, facility_ratio: float = 0.80, name: Optional[str] = None):
        pct_fac = int(round(facility_ratio * 100))
        pct_kc = 100 - pct_fac
        default_name = f"Hybrid Mix {pct_fac}/{pct_kc} ({pct_fac}% Facility + {pct_kc}% k-Center)"
        super().__init__(name=name or default_name)
        self.facility_ratio = facility_ratio
        self.kcenter_ratio = 1.0 - facility_ratio

    def _run_hybrid_selection(
        self,
        splitter: GroupAwareVideoSplitter,
        target_images: int,
        candidate_vids: List[str]
    ) -> List[str]:
        n_cand = len(candidate_vids)
        if n_cand == 0 or target_images <= 0:
            return []

        target_fac = int(round(target_images * self.facility_ratio))
        cand_indices = [splitter.vid_to_idx[v] for v in candidate_vids]
        sub_sim = splitter.video_similarity_matrix[cand_indices, :][:, cand_indices]
        sub_dists = splitter.video_pairwise_dists[cand_indices, :][:, cand_indices]
        weights = splitter.video_weights[cand_indices]

        # Giai đoạn 1: Facility Location cho target_fac ảnh
        max_sim = np.zeros(n_cand, dtype=np.float32)
        heap = [(-float(np.sum(sub_sim[:, i] * weights)), i, 0) for i in range(n_cand)]
        heapq.heapify(heap)

        chosen_sub_indices = []
        cur_imgs = 0
        step = 0

        while cur_imgs < target_fac and heap:
            while True:
                neg_gain, node_idx, eval_step = heapq.heappop(heap)
                if eval_step == step:
                    chosen_sub_indices.append(node_idx)
                    vid = candidate_vids[node_idx]
                    cur_imgs += splitter.video_records[vid].num_images
                    max_sim = np.maximum(max_sim, sub_sim[:, node_idx])
                    step += 1
                    break
                else:
                    actual_gain = float(np.sum(np.maximum(0, sub_sim[:, node_idx] - max_sim) * weights))
                    heapq.heappush(heap, (-actual_gain, node_idx, step))

        chosen_set = set(chosen_sub_indices)

        # Giai đoạn 2: k-Center Greedy để chọn nốt target_kc ảnh từ các điểm xa nhất
        min_dists = np.min(sub_dists[:, chosen_sub_indices], axis=1) if chosen_sub_indices else np.full(n_cand, 999.0, dtype=np.float32)

        while cur_imgs < target_images and len(chosen_set) < n_cand:
            min_dists[list(chosen_set)] = -1.0
            best_idx = int(np.argmax(min_dists))

            chosen_sub_indices.append(best_idx)
            chosen_set.add(best_idx)
            vid = candidate_vids[best_idx]
            cur_imgs += splitter.video_records[vid].num_images

            # Cập nhật khoảng cách nhỏ nhất
            min_dists = np.minimum(min_dists, sub_dists[:, best_idx])

        chosen_vids = [candidate_vids[i] for i in chosen_sub_indices]
        return splitter.fine_tune_video_selection(
            chosen_vids, target_images, set(candidate_vids)
        )

    def split(
        self,
        splitter: GroupAwareVideoSplitter,
        target_test_images: int = 1400,
        target_val_images: int = 700,
        target_seed_images: int = 500
    ) -> SplitResult:
        test_vids = splitter.select_stratified(
            lambda cands, q: self._run_hybrid_selection(splitter, q, cands),
            target_test_images,
            set()
        )
        test_set = set(test_vids)

        val_vids = splitter.select_stratified(
            lambda cands, q: self._run_hybrid_selection(splitter, q, cands),
            target_val_images,
            test_set
        )
        val_set = set(val_vids)

        seed_vids = splitter.select_stratified(
            lambda cands, q: self._run_hybrid_selection(splitter, q, cands),
            target_seed_images,
            test_set | val_set
        )
        seed_set = set(seed_vids)

        pool_vids = [
            v for v in splitter.unique_video_ids
            if v not in test_set and v not in val_set and v not in seed_set
        ]
        pool_set = set(pool_vids)

        test_imgs = [img for v in test_vids for img in splitter.video_records[v].image_names]
        val_imgs = [img for v in val_vids for img in splitter.video_records[v].image_names]
        seed_imgs = [img for v in seed_vids for img in splitter.video_records[v].image_names]
        pool_imgs = [img for v in pool_vids for img in splitter.video_records[v].image_names]

        return SplitResult(
            strategy_name=self.name,
            test_images=test_imgs,
            val_images=val_imgs,
            pool_images=pool_imgs,
            test_videos=test_set,
            val_videos=val_set,
            pool_videos=pool_set,
            seed_images=seed_imgs,
            seed_videos=seed_set,
            metadata={
                "facility_ratio": self.facility_ratio,
                "kcenter_ratio": self.kcenter_ratio,
                "algorithm": "stratified_hybrid_mixed"
            }
        )


# ==============================================================================
# 3. HỆ THỐNG CHẤM ĐIỂM ĐỘC LẬP KÉP & ĐẤU TRƯỜNG LEADERBOARD
# ==============================================================================

class StrategyTournamentEvaluator:
    """
    Hệ thống Đo lường Kép (Dual Ground-Truth Scoring Engine):
    - Trục 1: Chân lý gốc UC Berkeley BDD10K (bdd10k_metadata.json - Read-Only).
    - Trục 2: Không gian vật lý Phase 02 (data_profiling_manifest.json).
    - Trục 3: Rào chắn an toàn tuyệt đối Gatekeeper.
    Tính Composite Benchmark Score (CBS) và xếp hạng Leaderboard.
    """

    def __init__(self, splitter: GroupAwareVideoSplitter):
        self.splitter = splitter

        # Tính phân bố 10 lớp trên toàn bộ 6.994 ảnh để làm chuẩn đối sánh
        self.full_class_counts: Dict[str, int] = {c: 0 for c in BDD10K_CLASS_NAMES}
        for vid in self.splitter.unique_video_ids:
            rec = self.splitter.video_records[vid]
            for c, cnt in rec.class_counts.items():
                self.full_class_counts[c] += cnt

        total_objs = sum(self.full_class_counts.values()) or 1
        self.full_class_dist = np.array(
            [self.full_class_counts[c] / total_objs for c in BDD10K_CLASS_NAMES],
            dtype=np.float32
        )

        # Tính PCA 3D toàn bộ không gian đặc trưng để đo spread
        self.pca = PCA(n_components=3, random_state=42)
        self.all_pca_coords = self.pca.fit_transform(self.splitter.norm_image_features)
        self.full_pca_variance = float(np.sum(np.var(self.all_pca_coords, axis=0)))

        # Image name -> row index trong norm_image_features
        self.img_to_idx = {name: i for i, name in enumerate(self.splitter.all_images)}

    def evaluate_strategy(self, split_res: SplitResult) -> Dict[str, Any]:
        """Chấm điểm định lượng độc lập một chiến lược phân chia."""
        test_imgs = split_res.test_images
        n_test = len(test_imgs)
        if n_test == 0:
            raise ValueError(f"Chiến lược {split_res.strategy_name} không có ảnh test nào!")

        # ----------------------------------------------------------------------
        # TRỤC 1: CHÂN LÝ NHÃN GỐC UC BERKELEY (bdd10k_metadata.json)
        # ----------------------------------------------------------------------
        st_counts = {"night": 0, "small": 0, "occlusion": 0, "day": 0}
        for img in test_imgs:
            st = self.splitter.image_stratum_map.get(img, "day")
            st_counts[st] += 1

        r_night = st_counts["night"] / n_test
        r_small = st_counts["small"] / n_test
        r_occl = st_counts["occlusion"] / n_test
        r_day = st_counts["day"] / n_test

        # Quota Absolute Error (QAE) theo mục tiêu lát cắt (35% Đêm, 20% Nhỏ, 20% Che, 25% Ngày)
        target_night = 0.35
        target_small = 0.20
        target_occl = 0.20
        target_day = 0.25

        qae = float(
            abs(r_night - target_night) +
            abs(r_small - target_small) +
            abs(r_occl - target_occl) +
            abs(r_day - target_day)
        )
        score_quota = float(max(0.0, min(100.0, 100.0 * (1.0 - 2.5 * qae))))


        # Phân bố 10 lớp giao thông trong tập Test
        test_class_counts: Dict[str, int] = {c: 0 for c in BDD10K_CLASS_NAMES}
        for vid in split_res.test_videos:
            rec = self.splitter.video_records[vid]
            for c, cnt in rec.class_counts.items():
                test_class_counts[c] += cnt

        classes_present = sum(1 for c in BDD10K_CLASS_NAMES if test_class_counts[c] > 0)
        has_rare_train = test_class_counts["train"] > 0
        has_rare_motor = test_class_counts["motor"] > 0
        has_rare_rider = test_class_counts["rider"] > 0

        test_total_objs = sum(test_class_counts.values()) or 1
        test_class_dist = np.array(
            [test_class_counts[c] / test_total_objs for c in BDD10K_CLASS_NAMES],
            dtype=np.float32
        )

        # Cosine similarity giữa phân bố lớp Test và Full
        norm_test = np.linalg.norm(test_class_dist)
        norm_full = np.linalg.norm(self.full_class_dist)
        cos_sim = float(np.dot(test_class_dist, self.full_class_dist) / (norm_test * norm_full + 1e-9))

        # Điểm bảo toàn lớp: 10/10 classes + phạt nếu mất lớp hiếm
        coverage_penalty = 1.0 if classes_present == 10 else (classes_present / 10.0) * 0.7
        score_class = float(max(0.0, min(100.0, 100.0 * cos_sim * coverage_penalty)))

        # ----------------------------------------------------------------------
        # TRỤC 2: KHÔNG GIAN ĐẶC TRƯNG VẬT LÝ 11 CHIỀU (manifest Phase 02)
        # ----------------------------------------------------------------------
        test_indices = [self.img_to_idx[img] for img in test_imgs]
        test_feats = self.splitter.norm_image_features[test_indices]
        all_feats = self.splitter.norm_image_features

        # Khoảng cách từ từng ảnh trong kho dữ liệu đến ảnh gần nhất trong tập Test
        # cdist(all_feats, test_feats): shape (6994, 1400)
        dists_to_test = cdist(all_feats, test_feats, metric="euclidean")
        min_dists = np.min(dists_to_test, axis=1)

        coverage_radius = float(np.max(min_dists))       # Bán kính phủ cực đại (Minimax)
        mean_nn_distance = float(np.mean(min_dists))     # Khoảng cách lân cận trung bình

        # Facility Location Similarity Score F(Test)
        # F(S) = (1 / |V|) * sum_{x in V} max_{s in S} exp(-||x - s||^2 / (2 * sigma^2))
        sims_to_test = np.exp(-(dists_to_test ** 2) / (2.0 * (self.splitter.rbf_sigma ** 2)))
        facility_loc_score = float(np.mean(np.max(sims_to_test, axis=1)))

        # PCA Spatial Spread (Độ phân tán không gian đặc trưng)
        test_pca_coords = self.all_pca_coords[test_indices]
        test_pca_var = float(np.sum(np.var(test_pca_coords, axis=0)))
        pca_spread_ratio = float(test_pca_var / (self.full_pca_variance + 1e-9))

        # ----------------------------------------------------------------------
        # TRỤC 3: RÀO CHẮN AN TOÀN TUYỆT ĐỐI (Gatekeeper)
        # ----------------------------------------------------------------------
        overlap_test_val = len(split_res.test_videos & split_res.val_videos)
        overlap_test_seed = len(split_res.test_videos & split_res.seed_videos)
        overlap_test_pool = len(split_res.test_videos & split_res.pool_videos)
        overlap_val_seed = len(split_res.val_videos & split_res.seed_videos)
        overlap_val_pool = len(split_res.val_videos & split_res.pool_videos)
        overlap_seed_pool = len(split_res.seed_videos & split_res.pool_videos)
        total_video_overlap = (
            overlap_test_val + overlap_test_seed + overlap_test_pool +
            overlap_val_seed + overlap_val_pool + overlap_seed_pool
        )

        gatekeeper_pass = (total_video_overlap == 0) and (classes_present == 10)

        return {
            "strategy_name": split_res.strategy_name,
            "test_images_count": n_test,
            "val_images_count": split_res.num_val,
            "seed_images_count": split_res.num_seed,
            "pool_images_count": split_res.num_pool,
            "quota_absolute_error": qae,
            "score_quota": score_quota,
            "quota_ratios": {
                "night_ratio": r_night,
                "small_ratio": r_small,
                "occlusion_ratio": r_occl,
                "day_ratio": r_day
            },
            "classes_present_count": classes_present,
            "has_rare_train": has_rare_train,
            "has_rare_motor": has_rare_motor,
            "has_rare_rider": has_rare_rider,
            "class_distribution_cosine_sim": cos_sim,
            "score_class": score_class,
            "coverage_radius": coverage_radius,
            "mean_nn_distance": mean_nn_distance,
            "facility_location_score": facility_loc_score,
            "pca_spread_ratio": pca_spread_ratio,
            "video_overlap_count": total_video_overlap,
            "gatekeeper_pass": gatekeeper_pass,
            "test_class_counts": test_class_counts,
        }

    def run_tournament(
        self,
        strategies: Optional[List[BaseSplitStrategy]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, SplitResult], str]:
        """
        Khởi chạy đấu trường thực nghiệm song song cho 6 chiến lược.
        Xếp hạng Leaderboard và tự động chọn ra Winner đạt CBS cao nhất.
        """
        if strategies is None:
            strategies = [
                RandomGroupSplitter(seed=42),
                DirectMetadataStratifiedSplitter(seed=42),
                StratifiedKCenterSplitter(seed=42),
                SubmodularFacilityLocationSplitter(seed=42),
                HybridMixedSplitter(facility_ratio=0.80, name="5. 🏆 Hybrid Mix 80/20 (Proposed Winner)"),
                HybridMixedSplitter(facility_ratio=0.70, name="6. 🥈 Hybrid Mix 70/30 (Stress-Test Runner-up)"),
            ]

        results_map: Dict[str, SplitResult] = {}
        eval_records: List[Dict[str, Any]] = []

        print(f"\n🏟️ KHỞI CHẠY ĐẤU TRƯỜNG THỰC NGHIỆM SONG SONG ({len(strategies)} CHIẾN LƯỢC ĐỐI ĐẦU):")
        for st in strategies:
            t0 = time.time()
            res = st.split(self.splitter, target_test_images=1400, target_val_images=700, target_seed_images=500)
            t_split = time.time() - t0

            ev = self.evaluate_strategy(res)
            ev["elapsed_seconds"] = round(t_split, 3)

            results_map[st.name] = res
            eval_records.append(ev)
            print(f"  ✓ [{st.name}]: Đã chia xong trong {t_split:.2f}s | QAE={ev['quota_absolute_error']*100:.1f}% | R_cov={ev['coverage_radius']:.2f}")

        # Chuẩn hóa tương đối các chỉ số Coverage Radius và Facility Location Score giữa các ứng viên
        cov_vals = [e["coverage_radius"] for e in eval_records]
        fac_vals = [e["facility_location_score"] for e in eval_records]

        cov_min, cov_max = min(cov_vals), max(cov_vals)
        fac_min, fac_max = min(fac_vals), max(fac_vals)

        for ev in eval_records:
            # Coverage Radius: Càng nhỏ càng tốt -> ánh xạ vào [60, 100]
            if cov_max > cov_min:
                score_cov = 100.0 - ((ev["coverage_radius"] - cov_min) / (cov_max - cov_min)) * 35.0
            else:
                score_cov = 90.0

            # Facility Location Score: Càng cao càng tốt -> ánh xạ vào [65, 100]
            if fac_max > fac_min:
                score_fac = 65.0 + ((ev["facility_location_score"] - fac_min) / (fac_max - fac_min)) * 35.0
            else:
                score_fac = 90.0

            ev["score_coverage_radius"] = float(score_cov)
            ev["score_facility_location"] = float(score_fac)

            # Công thức CBS chuẩn theo Action Plan:
            # CBS = 0.35 * Score_Quota + 0.25 * Score_Class + 0.20 * Score_CoverageRadius + 0.20 * Score_FacilityLocation
            cbs = (
                0.35 * ev["score_quota"] +
                0.25 * ev["score_class"] +
                0.20 * score_cov +
                0.20 * score_fac
            )

            # Phạt nếu không pass Gatekeeper
            if not ev["gatekeeper_pass"]:
                cbs *= 0.5

            ev["composite_benchmark_score"] = float(round(cbs, 2))

        # Sắp xếp Leaderboard theo CBS giảm dần
        eval_records.sort(key=lambda x: x["composite_benchmark_score"], reverse=True)

        for rank, ev in enumerate(eval_records, start=1):
            ev["rank"] = rank
            ev["decision"] = "🏆 WINNER (ĐƯỢC CHỌN)" if rank == 1 else "Á Quân (Runner-up)" if rank == 2 else "Loại (Eliminated)"

        winner_name = eval_records[0]["strategy_name"]
        return eval_records, results_map, winner_name


# ==============================================================================
# 4. BỘ KIỂM TOÁN RÒ RỈ KÉP (ANTI-LEAKAGE AUDITOR)
# ==============================================================================

class AntiLeakageAuditor:
    """
    Kiểm toán toàn diện rò rỉ dữ liệu 4 chiều (Test, Val, Seed, Pool):
    1. Video Sequence Overlap Check: Đảm bảo 0 video ID trùng lặp giữa 6 cặp.
    2. pHash Cross-Split Hamming Distance Scan:
       Kiểm tra toàn bộ ~13.45 triệu cặp chéo giữa Test, Val, Seed, Pool.
       Khẳng định 100% không có cặp nào có Hamming Distance < 5!
    """

    def __init__(self, pool_images_dir: Union[str, Path]):
        self.pool_images_dir = Path(pool_images_dir)

    def _compute_image_phash(self, img_path: Path) -> Tuple[str, Any, np.ndarray]:
        """Tính pHash 64-bit và chuyển đổi sang vector nhị phân 64-bit."""
        try:
            with Image.open(img_path) as im:
                h = imagehash.phash(im)
            # Chuyển hash sang uint64 và unpack sang 64-bit numpy array
            h_int = int(str(h), 16)
            arr64 = np.array([h_int], dtype=np.uint64)
            bits = np.unpackbits(arr64.view(np.uint8))
            return img_path.name, str(h), bits
        except Exception as e:
            # Fallback zero hash
            return img_path.name, "0000000000000000", np.zeros(64, dtype=np.uint8)

    def audit_splits(
        self,
        split_result: SplitResult,
        max_workers: int = 8
    ) -> Dict[str, Any]:
        """Thực hiện kiểm toán rò rỉ kép 4 chiều toàn diện."""
        print(f"\n🛡️ BẮT ĐẦU KIỂM TOÁN RÒ RỈ KÉP 4 CHIỀU (ANTI-LEAKAGE AUDIT):")
        t0 = time.time()

        # 1. Kiểm tra rò rỉ video ID giữa 6 cặp
        test_vids = split_result.test_videos
        val_vids = split_result.val_videos
        seed_vids = split_result.seed_videos
        pool_vids = split_result.pool_videos

        leak_tv = list(test_vids & val_vids)
        leak_ts = list(test_vids & seed_vids)
        leak_tp = list(test_vids & pool_vids)
        leak_vs = list(val_vids & seed_vids)
        leak_vp = list(val_vids & pool_vids)
        leak_sp = list(seed_vids & pool_vids)

        video_leakage_detected = (
            (len(leak_tv) > 0) or (len(leak_ts) > 0) or (len(leak_tp) > 0) or
            (len(leak_vs) > 0) or (len(leak_vp) > 0) or (len(leak_sp) > 0)
        )
        print(f"  - Video Overlap Test vs Val  : {len(leak_tv):,} videos")
        print(f"  - Video Overlap Test vs Seed : {len(leak_ts):,} videos")
        print(f"  - Video Overlap Test vs Pool : {len(leak_tp):,} videos")
        print(f"  - Video Overlap Val vs Seed  : {len(leak_vs):,} videos")
        print(f"  - Video Overlap Val vs Pool  : {len(leak_vp):,} videos")
        print(f"  - Video Overlap Seed vs Pool : {len(leak_sp):,} videos")

        # 2. Quét chéo pHash Hamming Distance
        all_imgs = (
            split_result.test_images +
            split_result.val_images +
            split_result.seed_images +
            split_result.pool_images
        )
        print(f"  - Đang tính toán chữ ký pHash cho {len(all_imgs):,} ảnh bằng {max_workers} luồng...")

        img_paths = [self.pool_images_dir / name for name in all_imgs]

        phash_bits_map: Dict[str, np.ndarray] = {}
        phash_str_map: Dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for name, h_str, bits in executor.map(self._compute_image_phash, img_paths):
                phash_bits_map[name] = bits
                phash_str_map[name] = h_str

        # Chuẩn bị ma trận bits cho 4 split
        bits_test = np.array([phash_bits_map[img] for img in split_result.test_images], dtype=np.uint8)
        bits_val = np.array([phash_bits_map[img] for img in split_result.val_images], dtype=np.uint8)
        bits_seed = np.array([phash_bits_map[img] for img in split_result.seed_images], dtype=np.uint8)
        bits_pool = np.array([phash_bits_map[img] for img in split_result.pool_images], dtype=np.uint8)

        # Quét khoảng cách Manhattan (tương đương Hamming distance trên bit vector)
        # 1. Test vs Val (1.400 x 700 = 980.000 cặp)
        d_tv = cdist(bits_test, bits_val, metric="cityblock")
        min_hd_tv = float(np.min(d_tv)) if d_tv.size > 0 else 99.0
        viol_tv = int(np.sum(d_tv < 5))

        # 2. Test vs Seed (1.400 x 500 = 700.000 cặp)
        d_ts = cdist(bits_test, bits_seed, metric="cityblock")
        min_hd_ts = float(np.min(d_ts)) if d_ts.size > 0 else 99.0
        viol_ts = int(np.sum(d_ts < 5))

        # 3. Test vs Pool (1.400 x 4.394 = 6.151.600 cặp)
        d_tp = cdist(bits_test, bits_pool, metric="cityblock")
        min_hd_tp = float(np.min(d_tp)) if d_tp.size > 0 else 99.0
        viol_tp = int(np.sum(d_tp < 5))

        # 4. Val vs Seed (700 x 500 = 350.000 cặp)
        d_vs = cdist(bits_val, bits_seed, metric="cityblock")
        min_hd_vs = float(np.min(d_vs)) if d_vs.size > 0 else 99.0
        viol_vs = int(np.sum(d_vs < 5))

        # 5. Val vs Pool (700 x 4.394 = 3.075.800 cặp)
        d_vp = cdist(bits_val, bits_pool, metric="cityblock")
        min_hd_vp = float(np.min(d_vp)) if d_vp.size > 0 else 99.0
        viol_vp = int(np.sum(d_vp < 5))

        # 6. Seed vs Pool (500 x 4.394 = 2.197.000 cặp)
        d_sp = cdist(bits_seed, bits_pool, metric="cityblock")
        min_hd_sp = float(np.min(d_sp)) if d_sp.size > 0 else 99.0
        viol_sp = int(np.sum(d_sp < 5))

        total_pairs_scanned = d_tv.size + d_ts.size + d_tp.size + d_vs.size + d_vp.size + d_sp.size
        total_violations = viol_tv + viol_ts + viol_tp + viol_vs + viol_vp + viol_sp
        elapsed = time.time() - t0

        print(f"  ✓ Đã quét chéo xong {total_pairs_scanned:,} cặp pHash trong {elapsed:.2f}s:")
        print(f"    * Test vs Val  : min HD = {min_hd_tv:.0f} (Số cặp HD < 5: {viol_tv})")
        print(f"    * Test vs Seed : min HD = {min_hd_ts:.0f} (Số cặp HD < 5: {viol_ts})")
        print(f"    * Test vs Pool : min HD = {min_hd_tp:.0f} (Số cặp HD < 5: {viol_tp})")
        print(f"    * Val vs Seed  : min HD = {min_hd_vs:.0f} (Số cặp HD < 5: {viol_vs})")
        print(f"    * Val vs Pool  : min HD = {min_hd_vp:.0f} (Số cặp HD < 5: {viol_vp})")
        print(f"    * Seed vs Pool : min HD = {min_hd_sp:.0f} (Số cặp HD < 5: {viol_sp})")

        overall_pass = (not video_leakage_detected) and (total_violations == 0)
        status_str = "PASS (HOÀN TOÀN CÁCH LY 4 CHIỀU KHÔNG RÒ RỈ)" if overall_pass else "FAIL (CÓ RÒ RỈ)"
        print(f"  🔒 KẾT QUẢ KIỂM TOÁN TỔNG THỂ: {status_str}")

        return {
            "overall_status": "PASS" if overall_pass else "FAIL",
            "video_sequence_isolation": {
                "test_vs_val_overlap": len(leak_tv),
                "test_vs_seed_overlap": len(leak_ts),
                "test_vs_pool_overlap": len(leak_tp),
                "val_vs_seed_overlap": len(leak_vs),
                "val_vs_pool_overlap": len(leak_vp),
                "seed_vs_pool_overlap": len(leak_sp),
                "status": "PASS" if not video_leakage_detected else "FAIL"
            },
            "phash_cross_hamming_scan": {
                "total_cross_pairs_scanned": total_pairs_scanned,
                "test_vs_val_min_hamming_distance": min_hd_tv,
                "test_vs_val_violations_hd_less_5": viol_tv,
                "test_vs_seed_min_hamming_distance": min_hd_ts,
                "test_vs_seed_violations_hd_less_5": viol_ts,
                "test_vs_pool_min_hamming_distance": min_hd_tp,
                "test_vs_pool_violations_hd_less_5": viol_tp,
                "val_vs_seed_min_hamming_distance": min_hd_vs,
                "val_vs_seed_violations_hd_less_5": viol_vs,
                "val_vs_pool_min_hamming_distance": min_hd_vp,
                "val_vs_pool_violations_hd_less_5": viol_vp,
                "seed_vs_pool_min_hamming_distance": min_hd_sp,
                "seed_vs_pool_violations_hd_less_5": viol_sp,
                "total_violations": total_violations,
                "status": "PASS" if total_violations == 0 else "FAIL"
            },
            "elapsed_seconds": round(elapsed, 2)
        }


# ==============================================================================
# 5. SYMLINK MATERIALIZATION & CHECKSUM SEALER
# ==============================================================================

class SymlinkSplitMaterializer:
    """
    Đổ cấu trúc ảnh và nhãn vào data/splits/ bằng Symbolic Links (Symlink).
    Tuân thủ tuyệt đối AGENTS.md:
    - data/bdd10k/ là Read-Only bất biến 100%.
    - Zero Storage Overhead: Tăng đúng 0 KB dung lượng đĩa thực tế.
    - Tạo 4 tập hoàn chỉnh:
      * fixed_test: 1.400 images + 1.400 labels
      * fixed_val: 700 images + 700 labels
      * seed_v0: 500 images + 500 labels
      * unlabeled_pool: 4.394 images
    - Cung cấp sẵn sàng cho DataLoader của PyTorch / YOLO.
    """

    def __init__(
        self,
        splits_dir: Union[str, Path],
        raw_images_dir: Union[str, Path],
        raw_labels_dir: Optional[Union[str, Path]] = None,
        raw_lbls_dir: Optional[Union[str, Path]] = None
    ):
        self.splits_dir = Path(splits_dir).resolve()
        self.raw_images_dir = Path(raw_images_dir).resolve()
        lbl_candidate = raw_labels_dir if raw_labels_dir is not None else raw_lbls_dir
        if lbl_candidate is None:
            raise ValueError("Cần cung cấp raw_labels_dir hoặc raw_lbls_dir")
        self.raw_labels_dir = Path(lbl_candidate).resolve()

        if not self.raw_images_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh gốc: {self.raw_images_dir}")
        if not self.raw_labels_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục nhãn gốc: {self.raw_labels_dir}")

    def materialize(self, split_result: SplitResult) -> Dict[str, Any]:
        """Tạo các thư mục và khởi tạo toàn bộ symlinks cho 4 tập."""
        print(f"\n📁 BẮT ĐẦU SYMLINK MATERIALIZATION VÀO {self.splits_dir}:")
        t0 = time.time()

        dirs_to_create = [
            self.splits_dir / "fixed_test" / "images",
            self.splits_dir / "fixed_test" / "labels",
            self.splits_dir / "fixed_val" / "images",
            self.splits_dir / "fixed_val" / "labels",
            self.splits_dir / "seed_v0" / "images",
            self.splits_dir / "seed_v0" / "labels",
            self.splits_dir / "unlabeled_pool" / "images",
            self.splits_dir / "reports"
        ]

        # Xóa dọn sạch nếu có unlabeled_pool/labels cũ
        old_pool_lbl = self.splits_dir / "unlabeled_pool" / "labels"
        if old_pool_lbl.exists():
            for item in old_pool_lbl.iterdir():
                if item.is_symlink() or item.is_file():
                    item.unlink()
            try:
                old_pool_lbl.rmdir()
            except Exception:
                pass

        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)
            # Dọn dẹp sạch sẽ toàn bộ symlinks cũ nếu thư mục đã tồn tại để đảm bảo idempotency và chống rò rỉ file thừa
            if d.name in ("images", "labels"):
                for item in d.iterdir():
                    if item.is_symlink() or item.is_file():
                        item.unlink()

        tasks = [
            ("fixed_test", split_result.test_images, True),
            ("fixed_val", split_result.val_images, True),
            ("seed_v0", split_result.seed_images, True),
            ("unlabeled_pool", split_result.pool_images, False)
        ]

        created_symlinks_count = 0
        broken_symlinks_count = 0

        for split_name, img_names, include_labels in tasks:
            img_target_dir = self.splits_dir / split_name / "images"
            lbl_target_dir = self.splits_dir / split_name / "labels" if include_labels else None

            for img_name in img_names:
                stem = Path(img_name).stem
                src_img = self.raw_images_dir / img_name
                dst_img = img_target_dir / img_name

                # Tạo symlink ảnh
                if dst_img.is_symlink() or dst_img.exists():
                    dst_img.unlink()
                dst_img.symlink_to(src_img)
                created_symlinks_count += 1
                if not dst_img.resolve().exists():
                    broken_symlinks_count += 1

                # Tạo symlink nhãn
                if include_labels and lbl_target_dir is not None:
                    txt_name = f"{stem}.txt"
                    src_lbl = self.raw_labels_dir / txt_name
                    dst_lbl = lbl_target_dir / txt_name

                    if src_lbl.exists():
                        if dst_lbl.is_symlink() or dst_lbl.exists():
                            dst_lbl.unlink()
                        dst_lbl.symlink_to(src_lbl)
                        created_symlinks_count += 1
                        if not dst_lbl.resolve().exists():
                            broken_symlinks_count += 1

        elapsed = time.time() - t0
        print(f"  ✓ Đã khởi tạo thành công {created_symlinks_count:,} symlinks trong {elapsed:.2f}s!")
        print(f"  ✓ Số symlink lỗi (Broken symlinks): {broken_symlinks_count}")

        if broken_symlinks_count > 0:
            raise RuntimeError(f"Phát hiện {broken_symlinks_count} symlinks bị gãy đường dẫn!")

        return {
            "total_symlinks": created_symlinks_count,
            "total_symlinks_created": created_symlinks_count,
            "broken_symlinks": broken_symlinks_count,
            "elapsed_seconds": round(elapsed, 3),
            "storage_overhead_bytes": 0
        }


class SHA256ChecksumSealer:
    """
    Niêm phong toàn vẹn bộ dữ liệu bằng mã băm SHA-256.
    Sinh file data/splits/dataset_checksum.sha256 chứa chữ ký mã băm của
    toàn bộ các artifacts phân chia dữ liệu.
    """

    def __init__(self, splits_dir: Union[str, Path]):
        self.splits_dir = Path(splits_dir).resolve()

    def seal_dataset(
        self,
        manifest_data: Dict[str, Any],
        output_file: Optional[Union[str, Path]] = None
    ) -> Path:
        """Tính SHA-256 của manifest và ghi file niêm phong."""
        if output_file is None:
            output_file = self.splits_dir / "dataset_checksum.sha256"
        else:
            output_file = Path(output_file)

        manifest_str = json.dumps(manifest_data, sort_keys=True, indent=2)
        hasher = hashlib.sha256()
        hasher.update(manifest_str.encode("utf-8"))
        checksum_hash = hasher.hexdigest()

        content = (
            f"# SHA-256 DATASET CHECKSUM SEAL — BDD10K ACTIVE LEARNING LAB PHASE 03\n"
            f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
            f"# Target Splits: fixed_test (1,400), fixed_val (700), seed_v0 (500), unlabeled_pool (4,394)\n"
            f"{checksum_hash}  splits_manifest.json\n"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"🔒 Đã niêm phong SHA-256 Seal tại: {output_file}")
        return output_file

    def seal(
        self,
        manifest_input: Union[Dict[str, Any], str, Path],
        output_file: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Niêm phong dữ liệu từ manifest dict hoặc file path.
        Hỗ trợ cả dict lẫn đường dẫn file json.
        """
        if isinstance(manifest_input, (str, Path)):
            with open(manifest_input, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = manifest_input
        return self.seal_dataset(data, output_file=output_file)


def generate_yolo_splits_config(
    splits_dir: Union[str, Path],
    output_yaml: Optional[Union[str, Path]] = None
) -> Path:
    """
    Sinh file cấu hình YOLO bdd10k_splits.yaml linh hoạt tự động.
    """
    splits_path = Path(splits_dir).resolve()
    if output_yaml is None:
        output_yaml = splits_path / "bdd10k_splits.yaml"
    else:
        output_yaml = Path(output_yaml)

    yaml_content = f"""# BDD10K Autonomous Driving Splits - YOLO11n Dataset Configuration
# Phase 03 Frozen Splits: 20% Test (1,400), 10% Val (700), Seed v0 (500), Unlabeled Pool (4,394)
# Generated automatically by SymlinkSplitMaterializer

path: {splits_path.as_posix()}
train: seed_v0/images
val: fixed_val/images
test: fixed_test/images

# 10 Lớp Giao Thông Chuẩn BDD10K
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
"""
    with open(output_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"📄 Đã sinh file cấu hình YOLO tại: {output_yaml}")
    return output_yaml
