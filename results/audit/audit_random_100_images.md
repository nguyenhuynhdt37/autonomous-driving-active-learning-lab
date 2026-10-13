# Báo Cáo Kiểm Toán Ngẫu Nhiên 100 Ảnh Từ Splits & Ánh Xạ Ngược Metadata

- **Thời điểm kiểm toán:** `2026-10-13T11:29:31Z`
- **Quy mô mẫu:** **100 ảnh ngẫu nhiên** (Seed = 42)
- **Tổng số Bounding Boxes được soi:** **1,927 boxes**

## 1. Bảng Chỉ Số Toàn Vẹn Cốt Lõi

| Chỉ Số Kiểm Toán | Giá Trị Thực Đo | Tiêu Chuẩn Cho Phép | Trạng Thái |
| :--- | :---: | :---: | :---: |
| **Tỷ lệ Symlink hợp lệ (Zero Broken)** | **100/100 (100.0%)** | 100.0% | ✅ ĐẠT TUYỆT ĐỐI |
| **Tỷ lệ ánh xạ ngược Metadata $O(1)$** | **100/100 (100.0%)** | 100.0% | ✅ ĐẠT TUYỆT ĐỐI |
| **Tỷ lệ Bounding Box hợp lệ (Không âm/tràn viền)** | **1927/1927 (100.0%)** | 100.0% | ✅ ĐẠT TUYỆT ĐỐI |
| **Số lượng box lệch tọa độ TXT vs JSON** | **0 / 1927 (0.0%)** | 0 | ✅ KHỚP 100% |
| **Sai số cực đại giữa TXT và JSON (Max Delta)** | **0.000000 pixel** | $< 0.01$ px | ✅ SAI SỐ LÀM TRÒN LÝ TƯỞNG |

## 2. Phân Bổ 100 Ảnh Theo Tập Dữ Liệu

| Tập Dữ Liệu (`Split`) | Số Lượng Mẫu | Tỷ Lệ Mẫu (%) | Ghi Chú |
| :--- | :---: | :---: | :--- |
| `unlabeled_pool` | **56 ảnh** | 56.0% | Kiểm toán Symlink + Metadata |
| `fixed_val` | **14 ảnh** | 14.0% | Kiểm toán Symlink + Metadata |
| `seed_v0` | **7 ảnh** | 7.0% | Kiểm toán Symlink + Metadata |
| `fixed_test` | **23 ảnh** | 23.0% | Kiểm toán Symlink + Metadata |

## 3. Phân Bố 10 Lớp Giao Thông Trong 100 Ảnh

| Lớp Giao Thông (`Class`) | Class ID | Số Lượng Boxes | Tỷ Trọng (%) |
| :--- | :---: | :---: | :---: |
| `person` | `0` | **169** | 8.77% |
| `rider` | `1` | **4** | 0.21% |
| `car` | `2` | **992** | 51.48% |
| `truck` | `3` | **43** | 2.23% |
| `bus` | `4` | **20** | 1.04% |
| `train` | `5` | **0** | 0.00% |
| `motor` | `6` | **3** | 0.16% |
| `bike` | `7` | **16** | 0.83% |
| `traffic light` | `8` | **318** | 16.50% |
| `traffic sign` | `9` | **362** | 18.79% |

## 4. Danh Sách 20 Ảnh Đầu Tiên Trong 100 Mẫu Kiểm Toán

| STT | Tên File (`image_id`) | Tập (`Split`) | Thời Tiết | Thời Điểm | Bối Cảnh | Số Boxes | Max Delta (px) |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| #1 | `0b80f475-55a8484e` | `unlabeled_pool` | undefined | daytime | highway | **6** | `0.0` |
| #2 | `0a4d38c3-8f2ea1a7` | `fixed_val` | clear | night | highway | **6** | `0.0` |
| #3 | `07812b19-1a472b19` | `seed_v0` | overcast | daytime | highway | **13** | `0.0` |
| #4 | `0efe0129-c499802e` | `unlabeled_pool` | rainy | daytime | city street | **16** | `0.0` |
| #5 | `0da41df0-2e70a8a9` | `fixed_test` | clear | daytime | residential | **9** | `0.0` |
| #6 | `0a7eefe7-83d04ff8` | `fixed_test` | clear | daytime | highway | **22** | `0.0` |
| #7 | `080a57f6-4f3ac2f0` | `fixed_test` | undefined | daytime | tunnel | **7** | `0.0` |
| #8 | `10453a95-40759ac3` | `fixed_val` | clear | daytime | highway | **18** | `0.0` |
| #9 | `0ecebc1d-0f060556` | `unlabeled_pool` | clear | daytime | highway | **12** | `0.0` |
| #10 | `08780f20-02182d09` | `fixed_val` | clear | night | highway | **13** | `0.0` |
| #11 | `0cd7b5f1-21b1e324` | `unlabeled_pool` | rainy | night | residential | **22** | `0.0` |
| #12 | `0ef48e46-fd09a3a9` | `unlabeled_pool` | clear | night | city street | **23** | `0.0` |
| #13 | `083b1f31-a10adc37` | `unlabeled_pool` | clear | night | highway | **21** | `0.0` |
| #14 | `05c28cb7-a52fa59d` | `fixed_val` | clear | night | city street | **17** | `0.0` |
| #15 | `09ac21d6-3c3afe65` | `unlabeled_pool` | clear | daytime | city street | **42** | `0.0` |
| #16 | `03b72f80-84574506` | `unlabeled_pool` | overcast | night | highway | **5** | `0.0` |
| #17 | `0a0c3694-24fe9c8d` | `seed_v0` | undefined | daytime | residential | **18** | `0.0` |
| #18 | `0973a4ae-441acd83` | `seed_v0` | clear | night | city street | **30** | `0.0` |
| #19 | `071055c2-651bcc30` | `fixed_val` | clear | night | undefined | **21** | `0.0` |
| #20 | `07576c3b-5c9dccb3` | `fixed_test` | partly cloudy | daytime | residential | **11** | `0.0` |

*(Xem chi tiết 100 bản ghi tại file `audit_random_100_images.json`)*
