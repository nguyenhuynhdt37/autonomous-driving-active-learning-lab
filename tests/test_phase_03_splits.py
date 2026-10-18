"""
test_phase_03_splits.py - Bộ Kiểm Thử Tự Động Toàn Diện Cho Phase 03
====================================================================
Kiểm tra tính đúng đắn toán học, rào chắn an toàn Gatekeeper 4 chiều (Test, Val, Seed, Pool)
và cơ chế Symlink Materialization của Phase 03 (Shared Pre-Branching Frozen Splits).
"""

import json
from pathlib import Path
import pytest

from paper_v4_10k.src.paths import (
    get_processed_data_dir,
    get_bdd10k_metadata_file,
    get_splits_dir,
)
from paper_v4_10k.src.splits import (
    BDD10K_CLASS_NAMES,
    GroupAwareVideoSplitter,
    SplitResult,
    RandomGroupSplitter,
    DirectMetadataStratifiedSplitter,
    StratifiedKCenterSplitter,
    SubmodularFacilityLocationSplitter,
    HybridMixedSplitter,
    StrategyTournamentEvaluator,
)


@pytest.fixture(scope="module")
def splitter():
    processed_dir = get_processed_data_dir()
    manifest_path = processed_dir / "reports" / "data_profiling_manifest.json"
    metadata_path = get_bdd10k_metadata_file()
    return GroupAwareVideoSplitter(manifest_path, metadata_path, random_seed=42)


def test_group_aware_splitter_initialization(splitter):
    """1. Kiểm tra cấu trúc cơ sở: 6.994 ảnh gom thành 4.277 video sequences."""
    assert splitter.total_images_count == 6994
    assert splitter.total_videos_count == 4277
    assert splitter.norm_image_features.shape == (6994, 11)
    assert splitter.rbf_sigma > 0


def test_stratum_proportions_in_pool(splitter):
    """2. Kiểm tra phân vùng 4 lát cắt (Strata) cân đối, không bị triệt tiêu lát cắt day."""
    counts = {"night": 0, "small": 0, "occlusion": 0, "day": 0}
    for im in splitter.all_images:
        st = splitter.image_stratum_map[im]
        counts[st] += 1

    assert counts["night"] == 2678
    assert counts["small"] == 1400
    assert counts["occlusion"] == 1400
    assert counts["day"] == 1516

    # Cấp độ video
    for st in ("night", "small", "occlusion", "day"):
        vids = splitter.stratum_to_vids[st]
        assert len(vids) >= 800, f"Stratum {st} quá ít video: {len(vids)}"


def test_fine_tuning_and_single_frame_availability(splitter):
    """3. Kiểm tra tính sẵn sàng của video 1-khung hình để bù trừ quota chính xác 100%."""
    for st in ("night", "small", "occlusion", "day"):
        vids = splitter.stratum_to_vids[st]
        s1 = sum(1 for v in vids if splitter.video_records[v].num_images == 1)
        assert s1 >= 500, f"Stratum {st} thiếu single-frame videos: {s1}"


def test_split_result_dataclass_structure():
    """4. Kiểm tra cấu trúc SplitResult hỗ trợ đầy đủ 4 tập và thuộc tính tổng."""
    res = SplitResult(
        strategy_name="MockStrategy",
        test_images=["t1", "t2"],
        val_images=["v1"],
        pool_images=["p1", "p2", "p3"],
        test_videos={"vt1"},
        val_videos={"vv1"},
        pool_videos={"vp1"},
        seed_images=["s1"],
        seed_videos={"vs1"},
    )
    assert res.num_test == 2
    assert res.num_val == 1
    assert res.num_seed == 1
    assert res.num_pool == 3
    assert res.total_images == 7


def test_tournament_winner_is_hybrid_80_20(splitter):
    """5. Kiểm tra chiến lược chiến thắng phải là Hybrid Mix 80/20 với CBS cao nhất."""
    evaluator = StrategyTournamentEvaluator(splitter)
    eval_records, results_map, winner_name = evaluator.run_tournament()

    assert "Hybrid Mix 80/20" in winner_name
    winner_ev = eval_records[0]
    assert winner_ev["rank"] == 1
    assert winner_ev["composite_benchmark_score"] >= 90.0
    assert winner_ev["score_quota"] >= 75.0
    assert winner_ev["classes_present_count"] == 10
    assert winner_ev["gatekeeper_pass"] is True


def test_hybrid_80_20_split_counts(splitter):
    """6. Kiểm tra HybridMixedSplitter 80/20 bốc chính xác số lượng cho 4 tập."""
    strat = HybridMixedSplitter(facility_ratio=0.80)
    res = strat.split(splitter, target_test_images=1400, target_val_images=700, target_seed_images=500)
    assert res.num_test == 1400
    assert res.num_val == 700
    assert res.num_seed == 500
    assert res.num_pool == 4394
    assert res.total_images == 6994


def test_random_splitter_counts(splitter):
    """7. Kiểm tra RandomGroupSplitter bốc chính xác số lượng cho 4 tập."""
    strat = RandomGroupSplitter(seed=42)
    res = strat.split(splitter, target_test_images=1400, target_val_images=700, target_seed_images=500)
    assert res.num_test == 1400
    assert res.num_val == 700
    assert res.num_seed == 500
    assert res.num_pool == 4394
    assert res.total_images == 6994


def test_kcenter_splitter_counts(splitter):
    """8. Kiểm tra StratifiedKCenterSplitter bốc chính xác số lượng cho 4 tập."""
    strat = StratifiedKCenterSplitter(seed=42)
    res = strat.split(splitter, target_test_images=1400, target_val_images=700, target_seed_images=500)
    assert res.num_test == 1400
    assert res.num_val == 700
    assert res.num_seed == 500
    assert res.num_pool == 4394
    assert res.total_images == 6994


def test_facility_location_splitter_counts(splitter):
    """9. Kiểm tra SubmodularFacilityLocationSplitter bốc chính xác số lượng cho 4 tập."""
    strat = SubmodularFacilityLocationSplitter(seed=42)
    res = strat.split(splitter, target_test_images=1400, target_val_images=700, target_seed_images=500)
    assert res.num_test == 1400
    assert res.num_val == 700
    assert res.num_seed == 500
    assert res.num_pool == 4394
    assert res.total_images == 6994


def test_direct_metadata_splitter_counts(splitter):
    """10. Kiểm tra DirectMetadataStratifiedSplitter bốc chính xác số lượng cho 4 tập."""
    strat = DirectMetadataStratifiedSplitter(seed=42)
    res = strat.split(splitter, target_test_images=1400, target_val_images=700, target_seed_images=500)
    assert res.num_test == 1400
    assert res.num_val == 700
    assert res.num_seed == 500
    assert res.num_pool == 4394
    assert res.total_images == 6994


def test_seed_quota_proportions(splitter):
    """11. Kiểm tra tập seed_v0 đạt tỷ lệ phân tầng tương đối chuẩn (35% Đêm, 20% Nhỏ, 20% Che, 25% Ngày)."""
    strat = HybridMixedSplitter(facility_ratio=0.80)
    res = strat.split(splitter, target_test_images=1400, target_val_images=700, target_seed_images=500)
    seed_imgs = res.seed_images

    counts = {"night": 0, "small": 0, "occlusion": 0, "day": 0}
    for im in seed_imgs:
        st = splitter.image_stratum_map.get(im, "day")
        counts[st] += 1

    r_night = counts["night"] / len(seed_imgs)
    r_small = counts["small"] / len(seed_imgs)
    r_occl = counts["occlusion"] / len(seed_imgs)
    r_day = counts["day"] / len(seed_imgs)

    # Đảm bảo mỗi lát cắt đều có tối thiểu ít nhất 50 ảnh trong tập seed 500 ảnh
    for st in ("night", "small", "occlusion", "day"):
        assert counts[st] >= 50, f"Lát cắt {st} có quá ít ảnh trong seed_v0: {counts[st]}"

    # Đảm bảo tỷ lệ bám sát quota phân tầng (35% Đêm, 20% Nhỏ, 20% Che, 25% Ngày)
    assert 0.25 <= r_night <= 0.40, f"Night ratio {r_night:.3f} lệch quota 0.35"
    assert 0.10 <= r_small <= 0.25, f"Small ratio {r_small:.3f} lệch quota 0.20"
    assert 0.15 <= r_occl <= 0.30, f"Occlusion ratio {r_occl:.3f} lệch quota 0.20"
    assert 0.20 <= r_day <= 0.38, f"Day ratio {r_day:.3f} lệch quota 0.25"


def test_splits_symlinks_materialization_four_splits():
    """12. Kiểm tra thư mục data/splits/: số lượng symlinks trong 4 tập."""
    splits_dir = get_splits_dir()
    test_imgs = list((splits_dir / "fixed_test" / "images").glob("*.jpg"))
    test_lbls = list((splits_dir / "fixed_test" / "labels").glob("*.txt"))
    val_imgs = list((splits_dir / "fixed_val" / "images").glob("*.jpg"))
    val_lbls = list((splits_dir / "fixed_val" / "labels").glob("*.txt"))
    seed_imgs = list((splits_dir / "seed_v0" / "images").glob("*.jpg"))
    seed_lbls = list((splits_dir / "seed_v0" / "labels").glob("*.txt"))
    pool_imgs = list((splits_dir / "unlabeled_pool" / "images").glob("*.jpg"))

    seed_v1_dir = splits_dir / "seed_v1" / "images"
    active_mined = len(list(seed_v1_dir.glob("*.jpg"))) if seed_v1_dir.exists() else 0

    assert len(test_imgs) == 1400
    assert len(test_lbls) == 1400
    assert len(val_imgs) == 700
    assert len(val_lbls) == 700
    assert len(seed_imgs) == 500
    assert len(seed_lbls) == 500
    # Chấp nhận pool nguyên bản 4394, hoặc sau khi tuyển chọn batch Phase 05 (4144), hoặc Phase 07 (4394 - active_mined)
    assert len(pool_imgs) in (4394, 4144, 4394 - active_mined)


def test_total_symlinks_count_is_9594():
    """13. Kiểm tra tổng số symlinks khởi tạo đúng chuẩn 9.594."""
    splits_dir = get_splits_dir()
    test_imgs = list((splits_dir / "fixed_test" / "images").glob("*.jpg"))
    test_lbls = list((splits_dir / "fixed_test" / "labels").glob("*.txt"))
    val_imgs = list((splits_dir / "fixed_val" / "images").glob("*.jpg"))
    val_lbls = list((splits_dir / "fixed_val" / "labels").glob("*.txt"))
    seed_imgs = list((splits_dir / "seed_v0" / "images").glob("*.jpg"))
    seed_lbls = list((splits_dir / "seed_v0" / "labels").glob("*.txt"))
    pool_imgs = list((splits_dir / "unlabeled_pool" / "images").glob("*.jpg"))

    seed_v1_dir = splits_dir / "seed_v1" / "images"
    active_mined = len(list(seed_v1_dir.glob("*.jpg"))) if seed_v1_dir.exists() else 0

    all_links = test_imgs + test_lbls + val_imgs + val_lbls + seed_imgs + seed_lbls + pool_imgs
    assert len(all_links) in (9594, 9344, 9594 - active_mined)


def test_zero_broken_symlinks():
    """14. Kiểm tra 100% symlinks hợp lệ, 0 broken symlinks."""
    splits_dir = get_splits_dir()
    all_links = []
    for split_sub in [
        splits_dir / "fixed_test" / "images",
        splits_dir / "fixed_test" / "labels",
        splits_dir / "fixed_val" / "images",
        splits_dir / "fixed_val" / "labels",
        splits_dir / "seed_v0" / "images",
        splits_dir / "seed_v0" / "labels",
        splits_dir / "unlabeled_pool" / "images",
    ]:
        all_links.extend([f for f in split_sub.iterdir() if not f.name.startswith(".")])

    seed_v1_dir = splits_dir / "seed_v1" / "images"
    active_mined = len(list(seed_v1_dir.glob("*.jpg"))) if seed_v1_dir.exists() else 0

    assert len(all_links) in (9594, 9344, 9594 - active_mined)
    broken = [lnk for lnk in all_links if not lnk.resolve().exists()]
    assert len(broken) == 0, f"Có {len(broken)} broken symlinks"


def test_symlinks_target_are_bdd10k_originals():
    """15. Kiểm tra các symlinks trỏ chính xác về thư mục gốc bdd10k/ (Read-Only)."""
    splits_dir = get_splits_dir()
    sample_img = next((splits_dir / "seed_v0" / "images").glob("*.jpg"))
    sample_lbl = next((splits_dir / "seed_v0" / "labels").glob("*.txt"))

    assert "data/bdd10k/images/train" in sample_img.resolve().as_posix()
    assert "data/bdd10k/labels/train" in sample_lbl.resolve().as_posix()


def test_splits_isolation_four_way_video_zero_leakage():
    """16. Kiểm tra cách ly 4 chiều: 0 video ID overlap giữa cả 6 cặp phân tập."""
    splits_dir = get_splits_dir()
    manifest_path = splits_dir / "reports" / "splits_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    test_imgs = m["test_images"]
    val_imgs = m["val_images"]
    seed_imgs = m["seed_images"]
    pool_imgs = m["pool_images"]

    test_vids = {im.split("-")[0] for im in test_imgs}
    val_vids = {im.split("-")[0] for im in val_imgs}
    seed_vids = {im.split("-")[0] for im in seed_imgs}
    pool_vids = {im.split("-")[0] for im in pool_imgs}

    assert len(test_vids & val_vids) == 0, "Rò rỉ video Test vs Val!"
    assert len(test_vids & seed_vids) == 0, "Rò rỉ video Test vs Seed!"
    assert len(test_vids & pool_vids) == 0, "Rò rỉ video Test vs Pool!"
    assert len(val_vids & seed_vids) == 0, "Rò rỉ video Val vs Seed!"
    assert len(val_vids & pool_vids) == 0, "Rò rỉ video Val vs Pool!"
    assert len(seed_vids & pool_vids) == 0, "Rò rỉ video Seed vs Pool!"


def test_no_duplicate_images_across_splits():
    """17. Kiểm tra giao nhau giữa danh sách ảnh 4 tập là rỗng, hợp lại đủ 6.994 ảnh."""
    splits_dir = get_splits_dir()
    manifest_path = splits_dir / "reports" / "splits_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    test_set = set(m["test_images"])
    val_set = set(m["val_images"])
    seed_set = set(m["seed_images"])
    pool_set = set(m["pool_images"])

    assert len(test_set) == 1400
    assert len(val_set) == 700
    assert len(seed_set) == 500
    assert len(pool_set) == 4394

    assert len(test_set & val_set) == 0
    assert len(test_set & seed_set) == 0
    assert len(test_set & pool_set) == 0
    assert len(val_set & seed_set) == 0
    assert len(val_set & pool_set) == 0
    assert len(seed_set & pool_set) == 0

    combined = test_set | val_set | seed_set | pool_set
    assert len(combined) == 6994


def test_yolo_yaml_splits_configuration():
    """18. Kiểm tra file cấu hình YOLO bdd10k_splits.yaml trỏ đúng seed_v0 làm train."""
    splits_dir = get_splits_dir()
    yaml_file = splits_dir / "bdd10k_splits.yaml"
    assert yaml_file.exists()

    content = yaml_file.read_text(encoding="utf-8")
    assert "train: seed_v0/images" in content
    assert "val: fixed_val/images" in content
    assert "test: fixed_test/images" in content
    assert "9: traffic sign" in content


def test_dataset_checksum_sha256_seal():
    """19. Kiểm tra file mã băm SHA-256 niêm phong toàn vẹn bất biến."""
    splits_dir = get_splits_dir()
    checksum_file = splits_dir / "dataset_checksum.sha256"
    assert checksum_file.exists()

    content = checksum_file.read_text(encoding="utf-8")
    assert "splits_manifest.json" in content
    assert "seed_v0 (500)" in content
    assert "SHA-256" in content


def test_audit_report_phash_zero_violations():
    """20. Kiểm tra báo cáo audit xác nhận pHash không có cặp nào vi phạm HD < 5."""
    splits_dir = get_splits_dir()
    audit_path = splits_dir / "reports" / "frozen_splits_audit_report.json"
    assert audit_path.exists()

    with open(audit_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["overall_status"] == "PASS"
    ph = rep["phash_cross_hamming_scan"]
    assert ph["total_violations"] == 0
    assert ph["total_cross_pairs_scanned"] == 13454400
    assert ph["test_vs_val_violations_hd_less_5"] == 0
    assert ph["test_vs_seed_violations_hd_less_5"] == 0
    assert ph["test_vs_pool_violations_hd_less_5"] == 0
    assert ph["val_vs_seed_violations_hd_less_5"] == 0
    assert ph["val_vs_pool_violations_hd_less_5"] == 0
    assert ph["seed_vs_pool_violations_hd_less_5"] == 0
