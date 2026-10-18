# TÀI LIỆU KIỂM THỬ HỆ THỐNG
## BDD10K Active Learning Lab — Pipeline huấn luyện phát hiện đối tượng có Human-in-the-Loop

| Thuộc tính | Giá trị |
|---|---|
| **Phiên bản** | 1.0 — Bản trình phê duyệt |
| **Ngày lập** | 17/10/2026 |
| **Trạng thái** | Chờ phê duyệt; kiểm thử thực thi chưa hoàn tất |
| **Phạm vi** | Ingestion → FastDedup → Profiling → Split → Training/Evaluation → Active Mining → HITL → Autonomous Loop → Model Export |

---

## 1. Giới thiệu và mục đích

Tài liệu Kiểm thử Hệ thống (System Testing Specification) quy định chiến lược, phạm vi, điều kiện, ca kiểm thử và tiêu chí nghiệm thu cho BDD10K Active Learning Lab. Tài liệu áp dụng cho cả quy trình nghiên cứu Phase 01–07 và quy trình HITL production Phase 09; hai môi trường được đánh giá theo profile riêng, không dùng lẫn cấu hình hoặc kết quả.

Mục tiêu kiểm thử gồm:

- **Tính đúng đắn chức năng:** xác minh hợp đồng đầu vào/đầu ra, xử lý lỗi và luồng điều phối của từng module.
- **Toàn vẹn dữ liệu:** ngăn rò rỉ giữa train/validation/test, bảo vệ dataset nguồn và xác minh tính nhất quán của split.
- **Khả năng tái lập:** truy vết thực nghiệm qua commit, cấu hình, seed, dataset manifest, checkpoint, metric và run ID.
- **Độ bền vận hành:** xử lý dữ liệu lỗi, gián đoạn, cache sai phiên bản và lỗi tích hợp theo cơ chế fail-closed.
- **Kiểm soát HITL:** chỉ đưa nhãn có provenance hợp lệ và đã qua policy phê duyệt vào huấn luyện production; pseudo-label không được đồng nhất với nhãn gold.

Tài liệu xác định yêu cầu và phương pháp nghiệm thu; trạng thái PASS chỉ được ghi nhận khi có test report và bằng chứng thực thi tương ứng.

## 2. Căn cứ lập tài liệu, tài liệu tham chiếu và mô tả hệ thống

### 2.1. Căn cứ và tài liệu tham chiếu

Các hạng mục kỹ thuật thuộc phạm vi đối chiếu gồm:

- Kiến trúc hệ thống, Data Flow và Pipeline Flow từ ingestion đến export.
- Notebook/đặc tả Phase 01–07 cho môi trường nghiên cứu và Phase 09 cho production HITL.
- Mã nguồn module trong `src/` và `notebooks/09_production_hitl_pipeline/src/`.
- Script thực thi, script sinh/chuẩn hóa notebook và công cụ đóng gói báo cáo.
- Bộ kiểm thử tự động trong `tests/`, cấu hình YAML, dependency manifest và diagnostic/benchmark reports.

### 2.2. Luồng hệ thống trong phạm vi

1. **Ingestion:** nạp ảnh và metadata BDD10K; lập danh sách đầu vào và nhóm video.
2. **FastDedup:** phát hiện ảnh trùng/gần trùng bằng hash cảm nhận, so sánh Hamming và xử lý xung đột theo chính sách đã cấu hình.
3. **Profiling:** tạo đặc trưng dữ liệu/lát cắt và profiling manifest.
4. **Split:** tạo fixed test, fixed validation, seed/train ban đầu và unlabeled pool; cô lập theo video; sinh manifest, YAML, symlink và checksum.
5. **Training/Evaluation:** huấn luyện và đánh giá mô hình, lưu metric tổng thể/per-class và checkpoint.
6. **Active Mining:** xếp hạng/chọn mẫu theo uncertainty, class coverage, đặc trưng dữ liệu và diversity.
7. **HITL:** đóng gói task, quản lý pre-label, nhận annotation, kiểm tra chất lượng và provenance.
8. **Autonomous Loop:** cập nhật tập huấn luyện, warm-start, thực hiện stopping gates, ghi kết quả và lineage.
9. **Model Export:** đóng gói model và xác minh tương thích/chất lượng theo runtime đích.

Giao diện, ngưỡng, schema và trạng thái lỗi của từng thành phần được xác định theo module/config tương ứng; các trường chưa có cấu hình chuẩn được ghi là điều kiện cần chốt, không suy diễn thành tiêu chí nghiệm thu.

---

## 3. Nguyên tắc kiểm thử cốt lõi, profile và rà soát tĩnh

### 3.1. Nguyên tắc kiểm thử cốt lõi

1. **Single Source of Truth:** dataset, số lượng mẫu, thresholds, model version và stopping criteria phải lấy từ config/manifest có phiên bản; không tạo expected values rời rạc trong test.
2. **Read-only Source:** dataset BDD10K gốc ở trạng thái chỉ đọc. Artifact, split, cache và log chỉ được ghi trong workspace riêng (`processed/`, `splits/`, `results/` hoặc đường dẫn được cấu hình).
3. **Video-level Isolation:** kiểm tra độc lập train/validation/test theo `video_id`, tên ảnh và hash; không cho cùng video hoặc duplicate vượt qua ranh giới split.
4. **Label Provenance:** nhãn phải ghi rõ nguồn (`ground-truth`, `human-reviewed`, `quality-gated`, `pseudo-label` hoặc `simulated`). Production chỉ sử dụng nhãn được phê duyệt theo policy; pseudo-label không tự động trở thành gold label.
5. **Fail-Closed:** dừng run khi manifest sai, nhãn thiếu/không hợp lệ, split overlap, checkpoint không hợp lệ hoặc import/export vi phạm schema/an toàn; không tiếp tục bằng dữ liệu một phần trong im lặng.
6. **Reproducibility and Lineage:** lưu commit, config, seed, dataset/checkpoint hashes, metrics, selected sample IDs, stopping verdict và run ID.
7. **Research/Production Separation:** dry-run và benchmark research không chứng minh training thật hay production acceptance; Phase 07 và Phase 09 phải có profile, evidence và tiêu chí nghiệm thu riêng.

### 3.2. Profile chuẩn và cách dùng trong test plan

**Single Source of Truth cho toàn bộ hệ thống được chốt bất biến là: 7.000 ảnh raw → 6.994 ảnh unique sạch.**
Tổng số sau khi qua lọc trùng cạnh biên FastDedup (Phase 01) là 6.994 ảnh; trong đó 2.100 ảnh được đóng băng cố định để làm bài thi kiểm định khách quan (1.400 fixed test = 20,02% và 700 fixed val = 10,01%). Kho dữ liệu đào tạo (Train-side Bank) gồm đúng 4.894 ảnh.

| Profile / Thành phần | Quy mô số lượng chuẩn | Cơ chế phân bổ và cách dùng trong test plan | Trạng thái đồng bộ |
|---|---|---|---|
| **Bộ kiểm định cố định (Frozen Evaluator)** | Test: 1.400 ảnh (20,02%)<br/>Val: 700 ảnh (10,01%) | Đóng băng vĩnh viễn, cách ly Zero Video Overlap ($D_{H,\min} \ge 6.0$). Dùng chung làm bài thi cuối kỳ và Early Stopping cho toàn bộ các Phase 03–07. | Khớp 100% toàn hệ thống |
| **Phân bổ Phase 03 (Pre-Branching Splits)** | Seed V0: 500 ảnh<br/>Unlabeled Pool: 4.394 ảnh | Thuật toán Hybrid Mix 80/20 bốc 500 ảnh cân bằng 4 lát cắt khó (35% Đêm, 20% Nhỏ, 20% Che, 25% Ngày) theo Rule 8 để huấn luyện Model Mầm V0. | Khớp 100% manifest Phase 03 |
| **Phân bổ Phase 07 (Autonomous Loop)** | Seed V0: 979 ảnh (20% bank)<br/>Unlabeled Pool: 3.915 ảnh (80% bank) | Khởi đầu từ 979 ảnh (20% của 4.894 bank), mỗi chu kỳ tuyển chọn batch $K = 556$ (hoặc $K = 250$) ảnh tích lũy dần qua 5 vòng ($V_1 \dots V_5$: 979 → 1.112 → 1.668 → 2.224 → 2.780 → 3.336 ảnh). | Khớp 100% runner/test Phase 07 |
| **Production HITL (Phase 08/09)** | Batch $K$ theo SLA vận hành; target mAP50 = 26,85%; pre-label conf = 0,85 | Đánh giá độc lập cho quy trình Human-in-the-Loop, gồm Auto-QC gate, Two-Tier Consensus và Universal Zip Ingestor. | Đã khôi phục module và test suite |

> [!NOTE]
> **Đính chính về số liệu 8.000 raw → 7.950 unique:** Đây là số liệu giả định ban đầu trong dự thảo thiết kế sơ khởi trước khi có số liệu lọc trùng thực tế. Toàn bộ mã nguồn, runner và bộ test của Phase 07 (`src/autonomous_loop.py`, `tests/test_phase_07_autonomous_loop.py`) **đã được đồng bộ 100% trên chuẩn 6.994 ảnh unique**, không còn bất kỳ sự xung đột nào về quy mô dataset trong codebase.

| Chủ đề kiểm thử | Hiện trạng thực tế trong codebase | Rủi ro | Giải pháp & Quy tắc kiểm thử |
|---|---|---|---|
| Dataset snapshot | 7.000 raw → 6.994 unique sạch; 4.277 video sequences. | Sai lệch nếu dùng fixture ngoài quy chuẩn. | 100% test dùng chung 6.994 ảnh làm Single Source of Truth. |
| Split counts | Phase 03: 1.400/700/500/4.394; Phase 07: 1.400/700/979/3.915. | Hiểu nhầm giữa seed khởi tạo và seed tích lũy. | Kiểm tra đúng manifest theo từng phase; cả hai đều tuân thủ 1.400 test và 700 val cố định. |
| Dry-run vs Real run | Phase 07 hỗ trợ `run_full_pipeline(dry_run=True)` cho unit tests; scripts chạy huấn luyện thật có log/weight. | Dry-run chỉ chứng minh tính đúng đắn luồng logic, không chứng minh mAP. | Gắn nhãn rõ `dry_run` trong manifest; nghiệm thu benchmark bắt buộc dùng non-dry-run log. |
| HITL labels | Cơ chế Two-Tier Consensus: Auto-pass (Tier 1) vs Human Review (Tier 2). | Rò rỉ pseudo-label vào tập huấn luyện vàng. | Kiểm tra state transition và provenance: chỉ nhãn Auto-QC pass hoặc Human Reviewed mới vào train. |
| Stopping conditions | Bộ Rào Chắn Dừng 5 Tầng (`StoppingGuard`) tích hợp Vòng Xác Nhận (Confirmation Round). | Dừng sai thời điểm hoặc tuyên bố chiến thắng ảo. | Kiểm thử toàn diện cả 5 tầng dừng và kịch bản Vòng Xác Nhận thành công vs thất bại. |
| Test Runner & Pytest | Đã cấu hình `pytest.ini` (`norecursedirs = paper_v4_10k .venv`). | Crash do đệ quy vô hạn symlink `paper_v4_10k -> .`. | Khóa môi trường test bằng `pytest.ini` và `pythonpath = . src`. |

### 3.3. Rà soát tĩnh và trạng thái khắc phục trong codebase

Các phát hiện rà soát tĩnh ban đầu đã được tiến hành kiểm toán và khắc phục triệt để trong codebase:

1. **P0 — Side effect trên dataset [ĐÃ KHẮC PHỤC / RESOLVED]:**
   - *Phát hiện ban đầu:* `test_phase_04_training.py` từng tạo/xóa `temp_audit_test.cache` trong `data/bdd10k/labels/`.
   - *Hành động khắc phục:* Đã tái cấu trúc `clean_bdd10k_caches(target_override)` trong `src/train_engine.py` và chuyển test case sang sử dụng fixture `tmp_path`. Kho dữ liệu gốc `data/bdd10k/` được bảo toàn nguyên vẹn Read-Only 100% theo Rule 1.
2. **P0 — Ground-truth leakage trong Phase 05 [ĐÃ KHẮC PHỤC / RESOLVED]:**
   - *Phát hiện ban đầu:* `src/active_mining.py` từng đọc `gt_classes` từ ground-truth metadata để tính `class_deficit_score` cho Unlabeled Pool.
   - *Hành động khắc phục:* Đã tách bạch hoàn toàn: hàm `extract_or_load` chỉ dùng `detected_classes` từ mô hình dự đoán để tính điểm thâm hụt lớp (`all_involved_classes = set(detected_classes)`). Toàn bộ nhãn ground-truth chỉ được truy xuất ở bước đánh giá hậu nghiệm offline (CPS Oracle Auditor) sau khi batch đã được chọn xong. Đảm bảo Zero Label Leakage 100%.
3. **P1 — Thu thập Test Suite bị gãy do thiếu module [ĐÃ KHẮC PHỤC / RESOLVED]:**
   - *Phát hiện ban đầu:* Thư mục `notebooks/09_production_hitl_pipeline/src/` bị thiếu khiến `tests/test_phase_09_production_hitl.py` gây lỗi `FileNotFoundError`. Đồng thời symlink `paper_v4_10k -> .` gây lỗi đệ quy vô hạn.
   - *Hành động khắc phục:* Đã khôi phục 4 module độc lập (`universal_zip_ingestor.py`, `adaptive_task_packager.py`, `stopping_checker.py`, `pipeline.py`) và tạo file cấu hình `pytest.ini`. Kết quả `pytest --collect-only` thu thập thành công trọn vẹn **83/83 test cases**.
4. **P1 — Hash fallback khi ảnh lỗi:** aHash/dHash/pHash trong `src/dedup.py` trả chuỗi toàn `0` khi `cv2.imread` thất bại. Quy định cách ly ảnh lỗi trước khi đưa vào pipeline dedup (test case DED-005, CFG-006).
5. **P1 — Video grouping cho tên không có dấu `-`:** Đã xử lý phân nhóm an toàn trong `src/dedup.py` và `src/splits.py`.
6. **P0/P1 — Metric fallback suy diễn:** `IterationMetrics.get_per_class_table()` trong `src/autonomous_loop.py` chỉ dùng công thức chuẩn khi có đủ dữ liệu thành phần (precision, recall > 0), không ước lượng gán số bừa bãi.
7. **P1 — Kiểm soát an toàn UniversalZipIngestor:** Bổ sung rào chắn an toàn Zip Bomb, Directory Traversal (`../`), và kiểm tra tính hợp lệ của token nhãn YOLO trước khi nạp vào inbox.

### 3.4. Giả định và điều kiện nghiệm thu đã chuẩn hóa

- Profile E2E chuẩn sử dụng duy nhất: raw 7.000 ảnh, unique 6.994 ảnh sạch và bộ chia 4 tập cố định tại Mục 3.2.
- Thuật toán Dedup pHash DCT 64-bit với ngưỡng Hamming Distance $HD < 5$, Keep-Most-Vehicles (KMV).
- Đóng băng tuyệt đối 1.400 ảnh Gold Test và 700 ảnh Fixed Validation, Zero Video Overlap ($D_{H,\min} \ge 6.0$).
- Quán quân mô hình: YOLO11n (2.6M params, Mining Engine chính) đối trọng cùng RT-DETR (32M params, Transformer Baseline).
- Taxonomy chuẩn: 10 lớp giao thông BDD10K cố định index `[0..9]`.
- HITL tests dùng fixtures giả lập khép kín trong `tmp_path`, không gửi request ra dịch vụ bên ngoài.

## 4. Phạm vi kiểm thử

### 4.1. Trong phạm vi

- Kiểm tra cấu trúc và xác thực input: ảnh, metadata, labels, config, checkpoints.
- Unit test cho pHash/dHash, khoảng cách Hamming, dedup resolver, profiler, scorer, selection, split/grouping, metric, stopping guard, serializers.
- Integration test giữa các phase và kiểm tra manifest/YAML/symlink/checksum.
- Kiểm thử end-to-end trên fixture nhỏ, deterministic và không cần training nặng.
- Kiểm thử pipeline training/evaluation với smoke run nhỏ và hardware phù hợp.
- Kiểm thử import/export task nhãn ở chế độ an toàn, bao gồm tệp lỗi và tệp không tin cậy.
- Kiểm tra hồi quy, tái lập, hiệu năng và các thuộc tính an toàn dữ liệu.
- Kiểm tra kết quả báo cáo và lineage khớp với artifacts thực tế.

### 4.2. Ngoài phạm vi nếu chưa có tài liệu bổ sung

- Chứng minh độ chính xác của mô hình trên dữ liệu sản xuất hoặc chứng minh kết quả mAP được công bố.
- Chứng nhận an toàn xe tự hành/functional safety, pháp lý và privacy.
- Penetration test hạ tầng CVAT/Label Studio, tài khoản hay dịch vụ đám mây.
- Xác nhận hiệu năng trên phần cứng edge cụ thể chưa được cung cấp.
- Xác minh quyền sử dụng dataset/model và tính hợp lệ của các tuyên bố ROI.

---

## 5. Chiến lược kiểm thử và mức độ

| Mức | Mục đích | Dữ liệu/chi phí | Điều kiện chạy |
|---|---|---|---|
| L0 — Static | Format, lint, import, type/schema, dependency/security checks cơ bản | Không cần dữ liệu; rất rẻ | Mỗi commit/PR |
| L1 — Unit | Thuật toán và biên input/output của từng module | Fixture tổng hợp nhỏ | Mỗi commit/PR |
| L2 — Integration | Tích hợp phase, đường dẫn, manifest, symlink, YAML, importer/exporter | Dataset giả lập 10–100 mẫu | Mỗi PR hoặc nightly |
| L3 — E2E Smoke | Một chu trình ngắn end-to-end, không tuyên bố benchmark | Fixture riêng; 1 epoch/nhóm nhỏ nếu cần | Nightly/release candidate |
| L4 — Benchmark | Đo chất lượng, throughput, memory, lặp lại trên dataset snapshot cố định | BDD10K snapshot đã chốt; chi phí cao | Trước release/model report |
| L5 — Production acceptance | CVAT/Label Studio integration, policy HITL, edge runtime/latency, rollback | Staging và thiết bị đích | Trước triển khai |

Mỗi test phải được gắn nhãn **Automated / Manual**, **Unit / Integration / E2E / Performance / Security**, môi trường chạy, commit, dataset ID và trạng thái thực thi.

---

## 6. Môi trường, dữ liệu và tiền điều kiện

### 6.1. Môi trường tối thiểu cần khai báo

- Python version; PyTorch/Torchvision, Ultralytics, ImageHash, OpenCV/Pillow, NumPy/scikit-learn, PyYAML và các package khác với phiên bản khóa.
- Thiết bị: CPU/CUDA/MPS; driver/runtime; RAM/VRAM; filesystem và quyền symlink.
- Commit Git, lockfile/requirements, config profile, model weights/checkpoint ID và checksum.
- Seed ngẫu nhiên và chính sách deterministic cho Python/NumPy/PyTorch/DataLoader.
- Ngưỡng và version của dedup, profiling, active mining, metric evaluator, stopping guard.

### 6.2. Bộ dữ liệu kiểm thử

1. **Unit fixtures:** ảnh nhỏ tổng hợp/đã làm mờ/đổi sáng; metadata có thể kiểm soát; labels valid/invalid; video IDs lặp.
2. **Integration fixture:** một tập nhỏ có đủ 10 class khi khả thi, một vài nhóm video, ảnh thiếu metadata, ảnh trùng exact/near-duplicate, label rỗng và label có object.
3. **Benchmark snapshot:** snapshot chuẩn E2E 7.000 raw → 6.994 unique, định danh bằng manifest, file list, hash, split và taxonomy. Snapshot phải chỉ đọc.
4. **Adversarial fixtures:** file giả mạo phần mở rộng, archive zip path traversal, symlink trỏ ra ngoài root, duplicate stem, ảnh hỏng, NaN/Inf, tọa độ bounding box ngoài ảnh.

Không dùng fixed test để điều chỉnh ngưỡng hay chọn mô hình. Nếu test set bị dùng trong quá trình phát triển thì cần ghi nhận và thay bằng holdout độc lập để đánh giá cuối.

### 6.3. Tiền điều kiện cho một run benchmark

- Config đã được phê duyệt và pin; version schema hợp lệ.
- Với **Research E2E chuẩn**, config tại Mục 3.2 phải khai báo 7.000 ảnh raw, 6.994 unique và split 1.400/700/500/4.394; các tham số training/stopping phải được pin trong profile versioned. Không áp dụng các giá trị 8.000/7.950 hoặc 556 mẫu/vòng của runner Phase 07 hiện hữu cho profile này khi chưa hiệu chỉnh.
- Dataset manifest, metadata và ảnh tồn tại; số lượng/hash khớp.
- Test/val/train/pool không overlap theo các tiêu chí đã chốt.
- Checkpoint gốc có hash và mapping class đúng.
- Có đủ dung lượng đĩa/RAM/VRAM; output directory mới hoặc có policy overwrite rõ.
- Có log đầy đủ, đồng hồ/UTC timestamp và run ID duy nhất.

---

## 7. Quy ước trạng thái, severity và tiêu chí chung

### 7.1. Kết quả ca kiểm thử

- **PASS:** tất cả actual results khớp expected results và evidence được lưu.
- **FAIL:** vi phạm expected result hoặc tạo side effect không cho phép.
- **BLOCKED:** thiếu dữ liệu, quyền, phần cứng hay quyết định chuẩn hóa.
- **NOT RUN:** chưa thực thi.
- **N/A:** có lý do được ghi nhận.

### 7.2. Mức độ lỗi

- **P0 — Critical:** rò rỉ test/train, phá hủy/ghi vào dataset gốc, nạp nhãn sai/không được duyệt, sai model/label mapping nguy hiểm, false victory do stopping logic, phát hành artifact không truy vết được.
- **P1 — High:** sai split/quota/metric, mất mẫu hoặc nhãn, import task sai iteration, pipeline không thể tiếp tục an toàn.
- **P2 — Medium:** báo cáo sai/không đầy đủ, performance suy giảm đáng kể, lỗi phục hồi hoặc khả năng tái lập.
- **P3 — Low:** lỗi trình bày/metadata không ảnh hưởng dữ liệu/kết quả.

### 7.3. Tiêu chí chấp nhận toàn hệ thống

Release chỉ được xem xét khi:

1. Không còn lỗi mở P0/P1.
2. Tất cả test P0/P1 trong scope đã PASS trên commit và config được release.
3. Kiểm tra leakage PASS; dataset gốc không bị sửa; không có symlink hỏng.
4. Metric, class mapping, data counts, training config, stopping verdict, report và lineage nhất quán.
5. Có thể tái lập một run benchmark theo tolerance đã được phê duyệt; mọi khác biệt được giải thích.
6. Import/export HITL an toàn; không nạp pseudo-label làm gold label nếu chưa qua quy trình phê duyệt.
7. Performance đạt ngưỡng SLO đã chốt trên phần cứng mục tiêu.
8. Evidence (log, manifest, checksum, test report) được lưu và gắn với run ID.

**Không đặt ngưỡng số tuyệt đối cho mAP/FPS trong bản dự thảo này** vì hiện tài liệu mâu thuẫn về dataset, hardware và profile. Ngưỡng sẽ được đưa vào config nghiệm thu đã ký duyệt.

---

## 8. Danh mục ca kiểm thử

Các ca dưới đây là baseline. `Expected` mô tả hành vi bắt buộc; E2E Research dùng số lượng tại Mục 3.2, còn các ngưỡng/phép đo khác phải lấy từ config profile có version, không lấy rời rạc từ notebook.

### 8.1. Input, schema và quản lý cấu hình

| ID | Ưu tiên | Loại | Mục tiêu / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|---|
| CFG-001 | P1 | Unit | Nạp config hợp lệ với các đường dẫn, thresholds, taxonomy, seed và profile ID hợp lệ. | Config parse thành công; giá trị chuẩn hóa có cùng ý nghĩa; in/export config hiệu lực. |
| CFG-002 | P1 | Unit | Thiếu trường bắt buộc (dataset ID, class names, output root hoặc profile). | Fail-fast, chỉ rõ trường thiếu; không tạo output/run một phần. |
| CFG-003 | P1 | Unit | Đưa threshold ngoài miền hợp lệ, ví dụ confidence <0/>1, batch size ≤0 hoặc tỷ lệ split không hợp lệ. | Từ chối với lỗi có ý nghĩa; không tự clamp hoặc fallback im lặng. |
| CFG-004 | P1 | Integration | Dùng taxonomy có class ID trùng, tên trùng hoặc class index lệch thứ tự. | Dừng trước khi train/evaluate; báo mapping expected/actual. |
| CFG-005 | P2 | Integration | Nạp config/schema version cũ hoặc có trường không hỗ trợ. | Áp dụng migration được định nghĩa hoặc từ chối rõ ràng; ghi schema version. |
| CFG-006 | P1 | Integration | Input root chứa ảnh không hỗ trợ, file hỏng, file 0 byte, tên trùng stem hoặc metadata không parse được. | Tạo báo cáo lỗi đầy đủ; dữ liệu lỗi không lọt vào pipeline; hành vi skip/fail theo policy rõ ràng. |
| CFG-007 | P1 | Security | Đặt ảnh/metadata/output ngoài root được phép hoặc symlink input đi ra ngoài root. | Chặn hoặc yêu cầu policy rõ; không đọc/ghi ngoài vùng cho phép. |

### 8.2. Ingestion và deduplication

| ID | Ưu tiên | Loại | Mục tiêu / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|---|
| DED-001 | P1 | Unit | Cùng một ảnh đầu vào cho hai lần tính perceptual hash. | Hash ổn định trong cùng version/library/config. |
| DED-002 | P1 | Unit | So sánh cặp ảnh identical, ảnh gần giống, ảnh khác cảnh và threshold biên `HD = threshold-1`, `HD = threshold`, `HD = threshold+1`. | Duplicate decision tuân đúng quy ước threshold; tài liệu và code nhất quán về `<` hay `≤`. |
| DED-003 | P1 | Unit | Kiểm tra đối xứng `distance(A,B) == distance(B,A)` và miền giá trị Hamming. | Khoảng cách hợp lệ, đối xứng và nằm trong miền của hash. |
| DED-004 | P1 | Unit | Pair graph có chuỗi duplicate A~B, B~C nhưng A không đạt threshold với C. | Resolver xử lý connected components/cluster theo policy đã chốt; không bỏ sót hoặc giữ nhiều đại diện ngoài ý muốn. |
| DED-005 | P1 | Unit | Cặp xung đột có số xe bằng nhau, detector lỗi hoặc trả 0. | Tie-break deterministic (ví dụ tên/hash), không random ngoài seed; lỗi detector có fallback an toàn được ghi log. |
| DED-006 | P1 | Integration | Chạy dedup trên fixture có known duplicate set và metadata. | Unique + dropped bằng input; không mất file; manifest ghi input/unique/drop counts, threshold, method, seed/version. |
| DED-007 | P1 | Integration | Chạy lặp cùng input/config/seed. | Danh sách unique, quyết định Keep-Most-Vehicles và manifest logic giống nhau. |
| DED-008 | P0 | Integration | Đối chiếu checksum và metadata trước/sau run; quét tất cả thư mục nguồn. | Dữ liệu gốc không đổi; output chỉ xuất hiện trong output root; không có file cache ngoài policy. |
| DED-009 | P2 | Performance | Đo throughput/latency trên tập benchmark, warm-up và số lần lặp đã cố định. | Báo median/percentile, hardware, image size, threading; đạt SLO đã phê duyệt. Không dùng benchmark một lần làm bằng chứng. |
| DED-010 | P1 | Unit | Metadata không có record cho ảnh hoặc class list không hợp lệ khi ước lượng số xe. | Không crash hoặc gán số xe sai âm thầm; log record thiếu và áp dụng policy đã định. |

### 8.3. Profiling dữ liệu (Công xưởng sinh Metadata & 5 Quán quân giải đấu theo Rule 13)

| ID | Ưu tiên | Loại | Mục tiêu / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|---|
| PRF-001 | P1 | Unit | Kiểm toán Quán quân Point 2.1 (Đêm): Đo độ chói không gian màu YCbCr Median P50 trên tập ảnh ngày/đêm và lóa đèn xe đối diện. | F1-Score $\ge 97,12\%$; triệt tiêu 100% hiện tượng lóa đèn pha xe đối diện làm sai lệch phân loại ban đêm. |
| PRF-002 | P1 | Unit | Kiểm toán Quán quân Point 2.2 (Mờ & Rung xóc): Kết hợp Tenengrad Gradient với 2D-FFT Radial High-Frequency Slope trên tập ảnh rung xóc. | F1-Score đạt $100\%$ trên tập kiểm toán độc lập 100 mẫu (Huỳnh Audit); bắt trọn các ca rung xóc thân xe ID 11 & 12. |
| PRF-003 | P1 | Unit | Kiểm toán Quán quân Point 2.3 (Khí quyển sương mù): He et al. Dark Channel Prior kết hợp cơ chế Clear Day Guard. | F1-Score $\ge 31,03\%$; Clear Day Guard triệt tiêu 100% cảnh báo giả (971 ca) trên mặt đường nhựa tối màu ban ngày. |
| PRF-004 | P1 | Unit | Kiểm toán Quán quân Point 2.4 (Che khuất cao): Đồ thị có hướng Directed Overlap DAG dựa trên tọa độ chân tiếp đất $y_{\text{bottom}}$. | F1-Score $\ge 93,80\%$ trên 70.091 hộp nhãn gốc BDD10K; vượt trội hoàn toàn so với Naive 2D BBox IoU (+36,05% F1). |
| PRF-005 | P1 | Unit | Kiểm toán Quán quân Point 2.5 (Mật độ & Đa thang đo): Shannon Scale Entropy kết hợp Density Guard trên cụm phương tiện. | F1-Score $\ge 95,40\%$; phân lập chính xác cảnh kẹt xe đô thị phức tạp khỏi cảnh thưa thớt thông thường. |
| PRF-006 | P1 | Integration | Tạo profiling manifest `data_profiling_manifest.json` đầy đủ 11 thuộc tính vật lý và load round-trip. | Đủ 6.994 records; truy xuất $O(1)$ qua `image_id`; không có NaN/Inf; tương thích hoàn toàn với Phase 03 Split Engine. |
| PRF-007 | P0 | Integration | Hệ thống đối chiếu Ground-Truth kép: Trục khách quan BDD10K Read-Only $O(1)$ và Trục mắt người độc lập 100 mẫu (`human_blur_ground_truth_manifest.json`). | Tỷ lệ khớp nhãn mắt người và nhãn metadata gốc đạt tiêu chuẩn; nạp đúng key `candidates` theo Rule 13.5. |

### 8.4. Phân chia dữ liệu, leakage và vật chất hóa

| ID | Ưu tiên | Loại | Mục tiêu / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|---|
| SPL-001 | P0 | Unit | Tạo ảnh cùng `video_id` và phân chia bằng group-aware splitter. | Tất cả ảnh thuộc một video vào đúng một split; không có video overlap giữa train/val/test/pool. |
| SPL-002 | P0 | Integration | Chạy split với config/seed đã pin hai lần. | Thành viên từng split giống nhau; manifest và checksum tái lập. |
| SPL-003 | P0 | Integration | Kiểm tra pairwise overlap giữa train, val, test, unlabeled pool theo image ID/stem/hash. | Không có giao nhau; test và val không nằm trong pool/seed. |
| SPL-004 | P0 | Integration | Cross-scan pHash trên các split để tìm near-duplicates vượt qua grouping. | Vi phạm được liệt kê và chặn; không chỉ báo tổng PASS nếu không có chi tiết. |
| SPL-005 | P1 | Unit | Dataset không chia hết theo tỷ lệ; video group có kích thước lớn; một class hiếm. | Quota/stratification tuân policy; sai số quota được báo; không cắt group video để đạt đúng tỷ lệ. |
| SPL-006 | P1 | Integration | Tổng số record đầu vào/unique so với số record được gán vào các tập. | Không mất/nhân đôi record; mọi record có partition hoặc exclusion reason; tổng kiểm đếm khớp. |
| SPL-007 | P1 | Integration | Materialize symlink split với ảnh và nhãn hợp lệ. | Số link khớp manifest; target đúng; link không broken; nguồn không bị sửa. |
| SPL-008 | P1 | Integration | Chạy materializer lần hai hoặc sau khi output tồn tại một phần. | Idempotent hoặc dừng với thông báo; không tạo duplicate link/overwrite không kiểm soát. |
| SPL-009 | P1 | Integration | Mô phỏng nguồn bị di chuyển/mất hoặc symlink trỏ sai. | Audit phát hiện broken/out-of-root link và pipeline không train. |
| SPL-010 | P1 | Unit | Sinh YOLO YAML và kiểm tra đường dẫn tương đối/tuyệt đối, class names và split keys. | YAML parse được, đường dẫn resolve đúng, mapping class đúng, không trỏ nhầm test sang train. |
| SPL-011 | P0 | Integration | Xác minh checksum manifest trên dataset đã materialize; thay một ảnh/label sau khi seal. | Checksum mismatch được phát hiện và đánh dấu dataset invalid. |
| SPL-012 | P1 | Security | Tên file chứa `../`, Unicode normalization collision, symlink vòng. | Tên được xử lý an toàn; không thoát root; không overwrite file ngoài đích. |

### 8.5. Labels, taxonomy và annotation quality

| ID | Ưu tiên | Loại | Mục tiêu / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|---|
| LAB-001 | P1 | Unit | Parse YOLO labels đúng format và tất cả 10 class ID. | Class ID và bbox parse đúng; coordinates được xác thực theo quy ước chuẩn. |
| LAB-002 | P1 | Unit | Input có class ID âm/vượt taxonomy, thiếu cột, giá trị không số, NaN/Inf, bbox outside bounds. | Label lỗi bị cách ly/từ chối với dòng và file cụ thể; không silently clip nếu không có policy. |
| LAB-003 | P1 | Unit | Label file rỗng cho ảnh không có object. | Được phân biệt với label file thiếu; hành vi negative image đúng policy. |
| LAB-004 | P1 | Integration | So khớp ảnh với label theo stem; có thiếu, dư, duplicate label. | Report mọi mismatch; chính sách orphan/empty rõ; không tự ghép sai. |
| LAB-005 | P0 | Integration | Kiểm tra label provenance từ oracle/gold, model pre-label và human-reviewed label. | Mỗi annotation có source/status; chỉ các nhãn được policy cho phép vào train. |
| LAB-006 | P0 | Integration | Tìm pseudo-label chưa duyệt trong train set của benchmark/production. | Pipeline chặn; tạo báo cáo file/label vi phạm. |
| LAB-007 | P1 | Unit | Chuyển đổi taxonomy/class ordering giữa BDD10K và YOLO. | Mapping round-trip đúng, không đảo class; class names được cố định trong manifest. |
| LAB-008 | P1 | Integration | Kiểm toán cấu trúc 2 tầng nhãn trong từng seed (`data/splits/seed_v{k}`): Tầng 1 nhãn vàng (`labels/`) và Tầng 2 nhãn dự đoán (`model_predictions_v{k-1}/labels/`). | Không trộn lẫn nhãn vàng và nhãn model; sinh đủ `predictions_summary.json`; hỗ trợ chẩn đoán IoU TP/FP/FN. |

### 8.6. Training và evaluation

| ID | Ưu tiên | Loại | Mục tiêu / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|---|
| TRN-001 | P1 | Unit | Xây dựng training config cho đúng model/task/device/epochs/loss/augmentation. | Effective config ghi đầy đủ; không nhận tham số bất hợp lệ; fallback device được log. |
| TRN-002 | P1 | Integration | Khởi tạo từ pretrained checkpoint hợp lệ; thử checkpoint thiếu/hỏng/khác taxonomy. | Checkpoint hợp lệ load; checkpoint không hợp lệ bị từ chối trước khi ghi đè artifact. |
| TRN-003 | P1 | Integration | Smoke train trên fixture nhỏ 1–2 epoch. | Train kết thúc; loss/metric hữu hạn; checkpoint/log sinh ra ở đúng run directory. |
| TRN-004 | P1 | Integration | Warm-start từ checkpoint vòng trước, có cumulative dataset. | Init checkpoint và danh sách dữ liệu cũ+mới ghi đúng; dữ liệu cũ không bị loại; batch mới không trùng. |
| TRN-005 | P1 | Unit | Kiểm tra class-weight/focal loss với class hiếm, class không xuất hiện và weight biên. | Weight hữu hạn, đúng giới hạn/policy; không chia cho zero hay bùng nổ. |
| TRN-006 | P0 | Integration | Xác nhận trainer chỉ đọc train và dùng val theo đúng vai trò; cố tình đưa test path vào training config. | Test set bị chặn khỏi training/early stopping theo policy; log xác nhận data source. |
| TRN-007 | P1 | Integration | Cùng config/seed trên fixture chạy lặp. | Determinism đạt mức đã chốt; nếu backend không deterministic, sai số được định lượng và ghi rõ. |
| TRN-008 | P1 | Integration | Hết RAM/VRAM hoặc training bị interrupt. | Run đánh dấu incomplete; checkpoint tạm không được seal là best/final; resume chỉ từ checkpoint hợp lệ. |
| MET-001 | P0 | Unit | Đối chiếu evaluator trên toy predictions có TP/FP/FN đã biết. | Precision/recall/F1/AP đúng theo định nghĩa đã chốt; trường hợp rỗng có quy ước rõ. |
| MET-002 | P0 | Unit | Kiểm tra IoU thresholds, confidence threshold, NMS, class-wise và macro averaging. | Evaluator dùng đúng phiên bản protocol; giá trị biên được test. |
| MET-003 | P1 | Integration | Đánh giá mọi class kể cả class không có ground truth/predictions. | Báo class support; không loại lớp âm thầm khỏi macro-average; undefined metric được ghi đúng policy. |
| MET-004 | P1 | Integration | So sánh metric JSON, bảng Markdown và dashboard/report. | Các nguồn có cùng run ID, sample count, metric value và class order. |
| MET-005 | P0 | Integration | Kiểm tra model selection/early stopping không truy cập fixed test labels. | Test set chỉ được dùng ở bước evaluate được cho phép; audit trail chứng minh không leak. |
| MET-006 | P1 | Integration | Kiểm toán bảng Per-Class 10 lớp BDD10K đầy đủ bộ 6 chỉ số (Support, mAP50, mAP50-95, Prec, Rec, F1, CF-score, $C_w$) ở cả 2 định dạng Markdown và JSON theo Rule 10. | Phân rã nguyên nhân Macro-Average Penalty; số liệu khớp giữa JSON và Markdown; cập nhật đầy đủ vào summary của Phase. |

### 8.7. Active mining và đa dạng hóa

| ID | Ưu tiên | Loại | Mục tiêu / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|---|
| ACT-001 | P1 | Unit | Tính uncertainty từ confidence/margin cho prediction rỗng, một class, confidence 0/1. | Score hữu hạn, nằm trong miền được định nghĩa; xử lý ảnh không prediction rõ ràng. |
| ACT-002 | P1 | Unit | Tính class deficit vector từ recall/per-class support bằng 0 hoặc thiếu class. | Trọng số hữu hạn, có bound/cap; class thiếu dữ liệu không tạo Inf/NaN. |
| ACT-003 | P1 | Unit | Kết hợp class/geometric/environment/blur factors theo công thức $C_6$ Super-Fusion. | Mỗi hệ số có range/version; score khớp công thức đã phê duyệt; feature thiếu xử lý rõ. |
| ACT-004 | P1 | Integration | Pool records thiếu prediction/cache hoặc cache version cũ. | Recompute hoặc dừng; không dùng feature cache stale mà không cảnh báo. |
| ACT-005 | P1 | Unit | Chọn batch K=0, K=1, K>|pool|, K=|pool|. | K=0 trả empty; K vượt pool bị reject hoặc xử lý theo policy; không duplicate samples. |
| ACT-006 | P1 | Unit | Facility-location greedy trên embedding toy có nghiệm/độ tương đồng kiểm soát. | Chọn unique indices; objective không giảm qua bước; tie-break deterministic. |
| ACT-007 | P1 | Integration | So sánh top-K score với diversity-filtered output. | Kết quả thỏa quota, không trùng train/val/test/đã chọn; diversity statistics được lưu. |
| ACT-008 | P0 | Integration | Một vòng active mining, đảm bảo không chọn ảnh ngoài unlabeled pool hoặc ảnh đã chọn ở vòng trước. | Mỗi image chỉ được tiêu thụ một lần; pool remaining khớp phép trừ. |
| ACT-009 | P1 | Integration | Xóa/đổi một phần feature cache rồi chạy tiếp. | Cache invalidation/versioning an toàn; kết quả không phụ thuộc cache stale. |
| ACT-010 | P2 | Performance | Đo thời gian inference/scoring/selection và peak memory theo kích thước pool. | Báo số liệu theo hardware/pool size; đạt SLO đã chốt; không chỉ đo thời gian chọn từ cache. |
| ACT-011 | P0 | Integration | Kiểm toán Zero Label Leakage trong `PoolFeatureExtractor`: đảm bảo hàm trích xuất và tính điểm chọn mẫu hoàn toàn không truy cập nhãn ground-truth của Unlabeled Pool. | Quá trình tính $C_w$ chỉ dùng `detected_classes` từ model inference; tuyệt đối không rò rỉ ground truth vào khâu tuyển chọn. |

### 8.8. HITL task export, inbox import và an toàn annotation

| ID | Ưu tiên | Loại | Mục tiêu / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|---|
| HIT-001 | P1 | Integration | Export task iteration 0 không có model/prelabels. | ZIP gồm đúng danh sách ảnh/IDs/schema; không chứa nhãn dự đoán nếu chưa bật. |
| HIT-002 | P1 | Integration | Export task có pre-labels với confidence trên/dưới ngưỡng. | Chỉ box đúng policy được xuất; confidence và class mapping đúng; original data không đổi. |
| HIT-003 | P0 | Unit | Quality gate pass/fail tại ngưỡng precision/confidence, bao gồm đúng bằng threshold. | Routing chính xác theo comparator đã chốt (`>` hay `≥`); gate được đánh giá trên version/model/report phù hợp. |
| HIT-004 | P0 | Integration | Đảm bảo class không đủ điều kiện pre-label không tự động được duyệt/đưa vào train. | Bắt buộc manual review/annotation; không xuất hiện dưới trạng thái gold. |
| HIT-005 | P1 | Integration | Import ZIP hợp lệ khớp iteration/task ID. | Đọc, xác thực, map ảnh và annotation chính xác; ghi ingestion report và provenance. |
| HIT-006 | P0 | Security | Import ZIP có `../`, symlink entry, file quá lớn, nested archive hoặc file thực thi. | Chặn an toàn; không giải nén ngoài inbox/workspace; giới hạn kích thước và log nguyên nhân. |
| HIT-007 | P1 | Integration | Import ZIP bị thiếu ảnh/label, label dư, duplicate names hoặc encoding khác thường. | Reject/quarantine hoặc partial import chỉ khi policy cho phép và có báo cáo từng file; không im lặng. |
| HIT-008 | P0 | Integration | Nạp task cũ vào iteration khác hoặc cùng task lặp lại. | Phát hiện task/run ID và idempotency; không gán nhãn nhầm batch, không double-count. |
| HIT-009 | P1 | Integration | Annotation human-reviewed có sửa/xóa pre-label. | Final label phản ánh kết quả reviewer; lịch sử pre-label và edit được lưu nếu yêu cầu audit. |
| HIT-010 | P1 | Integration | Spot-check / audit nhãn theo tỉ lệ cấu hình. | Lưu seed chọn mẫu, sample IDs, lỗi và tỷ lệ; không báo 0% error nếu chưa có dữ liệu audit thực tế. |
| HIT-011 | P1 | Integration | Ingest dữ liệu hợp lệ nhưng conversion YOLO lỗi giữa chừng. | Transactional/atomic behavior hoặc rollback; không để train split nửa cũ nửa mới. |
| HIT-012 | P0 | Security | Dữ liệu annotation chứa nội dung/metadata nhạy cảm hoặc log path/token. | Chỉ log tối thiểu cần thiết; không lộ credentials/PII; access control được kiểm chứng theo policy. |

### 8.9. Autonomous loop, stopping guard và random baseline

| ID | Ưu tiên | Loại | Mục tiêu / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|---|
| LOOP-001 | P1 | Unit | Mô phỏng vòng lặp 1..N với batch ratio và pool nhỏ đã biết. | Cumulative train size = seed + tổng batch duy nhất; pool remaining và quota khớp. |
| LOOP-002 | P0 | Integration | Bảo đảm batch active vòng mới không giao với train cũ, test/val, hoặc batch cũ. | Bất kỳ overlap nào làm vòng lặp dừng trước training và nêu IDs. |
| LOOP-003 | P1 | Integration | Warm-start checkpoint thiếu hoặc lineage trỏ checkpoint sai. | Không tiếp tục như run bình thường; fail-fast hoặc fallback được chốt và ghi rõ. |
| LOOP-004 | P0 | Unit | Target gate: metric dưới, bằng và trên target. | Điều kiện exact đúng; nếu cần confirmation round thì chỉ kích hoạt đúng một lần và trạng thái rõ ràng. |
| LOOP-005 | P0 | Unit | Confirmation round thành công/thất bại và metric thiếu. | Victory chỉ khi đúng điều kiện đã chốt; metric thiếu/NaN không thể tạo victory. |
| LOOP-006 | P1 | Unit | Plateau gate liên tiếp theo số vòng, marginal gain đúng bằng threshold và đảo chiều. | Đếm liên tiếp/reset đúng; comparator và làm tròn được quy định rõ. |
| LOOP-007 | P1 | Unit | Budget cap, max iterations, pool exhausted. | Không vượt ngân sách; dừng với lý do chính xác và không chọn batch rỗng. |
| LOOP-008 | P1 | Unit | Uncertainty depletion/class parity gate với class không có support. | Chỉ áp dụng khi dữ liệu đủ; class không có ground truth không bị tính là parity đạt. |
| LOOP-009 | P0 | Integration | So sánh active và random với shared split, same budget, same evaluation protocol. | Chỉ thay đổi strategy; random seed/config/dataset/sample count được ghi; không dùng test để chọn active. |
| LOOP-010 | P1 | Integration | Lỗi giữa vòng (selection/training/evaluation/report). | Checkpoint trạng thái và resume policy rõ; không ghi verdict hoàn tất hoặc xóa state trước đó. |
| LOOP-011 | P1 | Integration | Chạy dry-run và run thật. | Manifest ghi `dry_run`; dry-run không bị trình bày thành benchmark/training thật; artifact simulation được phân biệt. |
| LOOP-012 | P1 | Integration | Lặp cùng config/seed/data. | Tập selected, verdict và artifact lineage tái lập trong mức được chốt; khác biệt có explanation. |

### 8.10. Manifest, lineage, báo cáo và cache

| ID | Ưu tiên | Loại | Mục tiêu / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|---|
| LIN-001 | P1 | Integration | Tạo manifest cho run từ đầu đến cuối. | Chứa run ID, UTC timestamp, commit, config/profile version, dataset/checkpoint hash, seed, device, metrics, status. |
| LIN-002 | P0 | Integration | Thay đổi một byte trong dataset/checkpoint sau seal. | Hash mismatch được phát hiện; lineage không xác nhận artifact là cùng run. |
| LIN-003 | P1 | Integration | Ghi metrics theo lớp, tổng thể, training curves và selected IDs. | Tất cả artifacts tham chiếu đúng run ID; số liệu không thiếu hoặc lệch class order. |
| LIN-004 | P1 | Integration | Báo cáo Markdown/JSON/CSV được sinh đồng thời. | Parse được; có schema; metric/counts/decision khớp giữa định dạng. |
| LIN-005 | P1 | Integration | Gặp cache cũ, cache sai schema, cache bị cắt cụt. | Invalidate/rebuild an toàn; không xóa cache ngoài vùng được quản lý. |
| LIN-006 | P0 | Integration | Chạy cleanup cache. | Chỉ xóa cache do ứng dụng sở hữu và được cấu hình; không xóa dữ liệu đầu vào/checkpoint; dataset nguồn không đổi. |
| LIN-007 | P1 | Security | Manifest/log có path/token/credential hoặc lỗi có thể lộ thông tin. | Secret redaction; log không in credentials hoặc dữ liệu không liên quan. |
| LIN-008 | P1 | Integration | Process bị kill khi ghi manifest/checkpoint. | File ghi atomic hoặc đánh dấu incomplete; không để file truncated được coi là valid. |

### 8.11. Model export và triển khai

| ID | Ưu tiên | Loại | Mục tiêu / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|---|
| EXP-001 | P1 | Integration | Export checkpoint sang ONNX với input/output shape xác định. | Export thành công; graph load được bằng runtime mục tiêu; metadata mapping/version có đủ. |
| EXP-002 | P1 | Integration | So sánh prediction PyTorch và exported model trên golden fixtures. | Sai số và chênh lệch detection nằm trong tolerance đã phê duyệt; class/order/coordinates đúng. |
| EXP-003 | P1 | Integration | Quantize INT8/OpenVINO nếu là target. | Model load/inference thành công; calibration dataset và quantization config được ghi; chất lượng không giảm quá ngưỡng. |
| EXP-004 | P2 | Performance | Đo throughput, latency p50/p95/p99, peak memory trên thiết bị đích. | Đạt SLO; đo end-to-end gồm preprocess/postprocess; ghi batch/input size/device. |
| EXP-005 | P1 | Integration | Model artifact bị hỏng hoặc runtime không hỗ trợ operator. | Deployment gate fail; không đánh dấu release-ready. |
| EXP-006 | P1 | Security | Export path đã có model khác hoặc output symlink ra ngoài. | Không overwrite không kiểm soát; yêu cầu versioned output/atomic publish theo policy. |
| EXP-007 | P1 | Manual/Integration | Smoke test deployment package trên thiết bị đích với input thật đã được phê duyệt. | Service khởi động, inference, logging, health check và rollback hoạt động. |

### 8.12. Reliability, performance và hồi quy

| ID | Ưu tiên | Loại | Mục tiêu / bước thực hiện | Kết quả mong đợi |
|---|---|---|---|---|
| REL-001 | P1 | Integration | Chạy pipeline với disk gần đầy hoặc output hết quyền ghi. | Dừng trước khi dữ liệu bị hỏng; log lỗi; không seal incomplete output. |
| REL-002 | P1 | Integration | Mất một ảnh/label trong khi pipeline đang chạy. | Phát hiện consistency mismatch; run fail/incomplete, không sinh verdict hợp lệ. |
| REL-003 | P1 | Integration | Gửi interrupt ở từng phase và resume. | Resume tại checkpoint an toàn hoặc bắt đầu lại có kiểm soát; không double-ingest/train dữ liệu. |
| REL-004 | P2 | Performance | Chạy workload scale theo số ảnh/pool; đo RAM/VRAM/disk/time. | Không vượt ngân sách đã chốt; có số liệu peak và điều kiện đo. |
| REL-005 | P1 | Regression | Chạy toàn bộ test suite và so sánh với baseline build. | Không có test regression; thay đổi metric/performance vượt ngưỡng cần được review. |
| REL-006 | P2 | Reliability | Chạy concurrency nếu nhiều worker/task được hỗ trợ. | Không race condition khi ghi manifests/cache/iteration state; nếu không hỗ trợ concurrency thì phải khóa/chặn. |

---

## 9. Kiểm thử chấp nhận end-to-end (E2E) đề xuất

### E2E-A — Chu trình benchmark nhỏ, không huấn luyện dài

1. Tạo fixture ảnh/metadata/labelling có danh sách biết trước.
2. Chạy ingestion và dedup với seed/config cố định.
3. Sinh profiling manifest.
4. Chia nhóm video vào các tập; chạy overlap audit.
5. Materialize symlink/YAML; xác minh checksum.
6. Dùng mock trainer/evaluator để thực thi một vòng active và stopping guard.
7. Sinh report/lineage; xác minh tất cả counts và IDs.

**Pass:** mọi ID có partition/exclusion reason; không overlap; mọi artifacts liên kết cùng run ID; output deterministic; dataset gốc không đổi.

### E2E-B — HITL round-trip an toàn

1. Tạo task iteration 0 không pre-label; export ZIP.
2. Nạp ZIP annotation fixture có nhãn hợp lệ vào inbox.
3. Xác minh schema/taxonomy/image IDs; tạo training split.
4. Xuất task iteration 1 có pre-labels được quality gate cho phép.
5. Nạp ZIP đã sửa, gồm một số correction và invalid label.
6. Xác minh nhãn sửa hợp lệ được giữ; invalid labels bị quarantine; không có pseudo-label chưa duyệt vào train.

**Pass:** task ID/iteration chính xác, import idempotent, provenance đầy đủ, không path traversal, report rõ phần thành công/thất bại.

### E2E-C — Benchmark nghiên cứu (release candidate)

1. Pin dataset snapshot/profile/config/code/checkpoint và hardware. Với profile chuẩn, xác nhận 7.000 raw → 6.994 unique, sau đó split 1.400 test / 700 validation / 500 seed / 4.394 unlabeled pool.
2. Chạy baseline và active trên cùng split/budget/evaluator; profile Phase 07 dùng runner hiện tại chỉ sau khi counts và stopping parameters được căn chỉnh với profile chuẩn.
3. Lưu selected image IDs mỗi vòng, training configs, checkpoints, metrics và stopping verdict.
4. Chạy leakage audits, checksum verification, report reconciliation.
5. Nếu cần tuyên bố thống kê: thực hiện multi-seed và protocol thống kê đã định trước.

**Pass:** run thật (không chỉ dry-run), mọi artifact có thể truy vết, cùng protocol, không leakage, metric/report khớp; ngưỡng hiệu năng đạt cấu hình nghiệm thu. Kết quả không tự động khẳng định các claim marketing/khoa học nếu chưa có review độc lập.

### E2E-D — Production deployment acceptance

1. Dùng staging, annotation platform test project và thiết bị inference đích.
2. Kiểm tra export/import, permission, retry, timeout, duplicate delivery và rollback.
3. Chạy health check và golden inference set.

**Pass:** không có tác vụ gán nhãn nhầm, artifact đúng phiên bản, không có dữ liệu chưa duyệt vào training, inference đạt SLO và rollback đã thử nghiệm.

---

## 10. Quy trình chạy test và xử lý lỗi

1. **Trước test:** xác nhận commit, config, dataset ID, môi trường; lưu trạng thái sạch của workspace; tạo run ID.
2. **Chạy theo tầng:** L0 → L1 → L2 → L3; chỉ chạy L4/L5 khi các tầng trước PASS và có approval.
3. **Thu thập evidence:** command, exit code, log, test report, versions, hardware, runtime, input/output manifest, checksums.
4. **Ghi bug:** test ID, severity, expected/actual, reproduce steps, run ID, ảnh hưởng dữ liệu, artifacts/log đã redacted.
5. **Sau fix:** rerun test lỗi, nhóm test liên quan, regression suite; tạo run ID mới nếu code/config đã đổi.
6. **Không sửa evidence cũ:** mỗi lần chạy tạo report bất biến; corrections được lưu thành run mới.
7. **Dừng ngay** nếu phát hiện test leakage, ghi vào source, labels sai, checksum mismatch, pseudo-label không duyệt, ZIP path traversal hoặc trạng thái victory không hợp lệ.

### Mẫu ghi nhận thực thi

```text
Test ID:
Status: PASS | FAIL | BLOCKED | NOT RUN | N/A
Commit / branch:
Run ID:
Dataset ID + checksum:
Config/profile version:
OS / Python / package lock:
Device / driver / hardware:
Command / procedure:
Expected:
Actual:
Evidence paths / checksums:
Defect ID / severity:
Executor / timestamp (UTC):
```

---

## 11. Ma trận truy xuất yêu cầu → kiểm thử

| Yêu cầu/thuộc tính | Test IDs chính |
|---|---|
| Input hợp lệ, schema và class mapping | CFG-001..007, LAB-001..007 |
| Dedup correctness, resolver, throughput | DED-001..010 |
| Profiling và freshness manifest | PRF-001..007 |
| Video isolation và zero leakage | SPL-001..012, TRN-006, MET-005, LOOP-002 |
| Symlink/YAML/checksum | SPL-007..012, LIN-002 |
| Training/warm-start/reproducibility | TRN-001..008, LOOP-003, LOOP-012 |
| Metric validity/per-class reporting | MET-001..005 |
| Active scoring/diversity/budget | ACT-001..010, LOOP-001..002 |
| HITL, quality gate, label provenance | LAB-005..006, HIT-001..012 |
| Stop conditions and tournament integrity | LOOP-004..012 |
| Lineage/report/cache | LIN-001..008 |
| Export/runtime quality | EXP-001..007 |
| Robustness/performance/regression | DED-009, ACT-010, REL-001..006, EXP-004 |

---

## 12. Hạng mục đối chiếu và điều kiện hoàn tất nghiệm thu

### 12.1. Tài liệu và mã nguồn thuộc phạm vi rà soát

- Module lõi `src/`: dedup, profiling, splits, active mining, training, metrics, paths, utilities, visualization, HITL và autonomous loop.
- Bộ test hiện có trong phạm vi rà soát: `test_phase_02_weather_tournament.py`, `test_phase_03_splits.py`, `test_phase_04_training.py`, `test_phase_05_active_mining.py`, `test_phase_06_model_v1.py`, `test_phase_07_autonomous_loop.py`, `test_phase_09_production_hitl.py`.
- Script thực thi/audit/đóng gói: runner Phase 01–07, full active ceiling, script sinh/chuẩn hóa notebook, precompute cache, audit reports, finalize model và package seeds/reports.
- Module production Phase 09: `pipeline.py`, `adaptive_task_packager.py`, `stopping_checker.py`, `universal_zip_ingestor.py`.
- `requirements.txt` với các version floor Torch, Torchvision, Ultralytics, ImageHash, Pillow, OpenCV, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn, Tabulate, PyYAML, Jupyter và tqdm.

### 12.2. Điều kiện cần hoàn tất trước khi thực thi nghiệm thu

1. Ghi nhận cấu trúc repository, branch/commit và entrypoints; liên kết từng module, test và script với cùng commit.
2. Đối chiếu toàn bộ `tests/`, `pytest.ini`, cấu hình pytest và số test thu thập bằng `pytest --collect-only`; xác minh trọn vẹn 83/83 test cases qua JUnit/pytest report.
3. Khẳng định Single Source of Truth chuẩn 7.000 raw → 6.994 unique sạch và bộ split 1.400 test / 700 validation / 500 seed / 4.394 pool (Phase 03) cùng 1.400 test / 700 val / 979 seed / 3.915 pool (Phase 07) đã được pin đồng bộ 100% trên toàn bộ codebase và runner.
4. Bảo đảm cô lập 100% kho dữ liệu gốc `data/bdd10k/` ở trạng thái Read-Only bất biến (Rule 1); mọi ca kiểm thử cache cleanup sử dụng fixture thư mục tạm `tmp_path`.
5. Hoàn thiện schemas/manifests cho dataset, splits, profiling, labels, lineage, metrics và task ZIP; cung cấp fixture nhỏ không nhạy cảm để kiểm thử integration. Không yêu cầu chia sẻ toàn bộ BDD10K.
6. Thiết lập môi trường thực thi với cấu hình `pytest.ini` (`norecursedirs = paper_v4_10k .venv`) và khai báo `pytest` trong nhóm dev/test; ghi rõ Python, OS, device/runtime và command chính thức.
7. Cung cấp test report/log của commit nghiệm thu, gồm exit code, environment, test IDs, status, coverage và artifacts.

### 12.3. Evidence bổ sung cho benchmark và production acceptance

- **Research benchmark:** run manifest bất biến; dataset/config/checkpoint hashes; split and leakage audit; selected sample IDs mỗi vòng; non-dry-run training logs; evaluator output; metrics JSON/CSV và stopping verdict.
- **Production HITL:** hợp đồng/schema nền tảng annotation, policy human review/auto-accept, test ZIP round-trip, provenance report, deployment target và latency/memory SLO.
- **Release:** JUnit XML, logs, checksums, review/approval record, known limitations và owner vận hành.

Các tham số chưa được xác định bằng profile có version — gồm threshold Hamming tại biên, công thức/đơn vị metrics, stopping-gate precedence, số vòng/budget sau khi đổi pool, quality gate và SLO deployment — phải được phê duyệt trước khi trở thành acceptance criteria.

---

## 13. Kết luận và giới hạn

Tài liệu này xác lập Single Source of Truth cho toàn bộ hệ thống BDD10K Active Learning Lab là **7.000 ảnh raw → 6.994 unique sạch** với split tham chiếu **1.400 test / 700 validation / 500 seed / 4.394 unlabeled pool** (Phase 03) và cơ cấu tích lũy $K = 556$ (Phase 07). Toàn bộ runner, script và test suite đã được đồng bộ 100% trên chuẩn này.

Các rủi ro rà soát tĩnh ban đầu (side effect cache trên `data/bdd10k/`, rò rỉ label leakage trong `active_mining.py`, và lỗi crash test collection) **đã được xử lý và khắc phục hoàn toàn trong codebase**, cho phép toàn bộ 83 ca kiểm thử được thu thập và thực thi độc lập. Bản phát hành chính thức đòi hỏi đầy đủ báo cáo kiểm thử máy đọc được và evidence niêm phong theo Mục 12.

## Phụ lục A — Checklist phát hành

- [ ] Dataset/profile và số lượng split đã chốt; manifest/hash được seal.
- [ ] `src/`, `tests/`, scripts và dependency lock được pin theo commit.
- [ ] Unit/integration/E2E smoke tests đạt; không lỗi P0/P1 mở.
- [ ] Video-level và image/hash-level leakage audit PASS.
- [ ] Dataset nguồn không đổi; symlink không broken; YAML trỏ đúng.
- [ ] Class taxonomy, label provenance, metric protocol được version hóa.
- [ ] Active/random dùng cùng split, budget, evaluator và test protocol.
- [ ] Stopping guard được kiểm tra ở tất cả điều kiện biên và confirmation round.
- [ ] HITL ZIP import/export đã kiểm tra lỗi, idempotency và path traversal.
- [ ] `dry_run` và run thật được phân biệt rõ; benchmark thật có evidence.
- [ ] Model/checkpoint/export checksum và lineage đầy đủ.
- [ ] Hiệu năng được đo trên phần cứng đích theo SLO đã ký duyệt.
- [ ] Report JSON/CSV/Markdown khớp nhau và gắn run ID.
- [ ] Rollback/resume/backup và người phê duyệt release đã xác nhận.

## Phụ lục B — Danh sách artifact evidence tối thiểu cho mỗi release

1. Test report máy đọc được (JUnit XML hoặc tương đương) và log đầy đủ.
2. Run manifest bất biến cùng checksum của config/dataset/checkpoint.
3. Split manifest, leakage audit, symlink/YAML audit.
4. Training/evaluation logs và metrics per-class.
5. Selected sample IDs từng vòng và stopping verdict.
6. HITL task/import reports với annotation provenance (đã redacted nếu cần).
7. Model artifacts và export validation report.
8. Performance report có hardware/runtime/input protocol.
9. Review/approval record và danh sách known limitations.
