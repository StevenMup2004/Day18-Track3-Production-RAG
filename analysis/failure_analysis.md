# Phân Tích Lỗi - Lab 18: Production RAG

**Nhóm:** Solo  
**Ngày cập nhật:** 2026-05-04  
**Số câu test:** 25

## 1. Bảng so sánh metric

| Metric | Naive Baseline | Production | Delta |
|--------|---------------:|-----------:|------:|
| Faithfulness | 1.0000 | 0.8651 | -0.1349 |
| Answer Relevancy | 0.0204 | 0.5029 | +0.4825 |
| Context Precision | 0.0155 | 0.0907 | +0.0752 |
| Context Recall | 0.1049 | 0.8805 | +0.7756 |

## 2. Kết luận nhanh

- Production đã cải thiện mạnh chất lượng retrieval và khả năng bám ý câu hỏi.
- Điểm nghẽn lớn nhất vẫn là `context_precision`: có nhiều câu trả về đúng miền nội dung nhưng chưa đúng đoạn bằng chứng tốt nhất.
- `faithfulness` giảm là hệ quả của bước sinh câu trả lời bằng LLM khi context vẫn còn nhiễu.

## 3. Bottom-5 failures (theo báo cáo mới nhất)

### #1
- **Question:** Chủ thể dữ liệu có quyền phản đối xử lý dữ liệu để làm gì?
- **Expected:** Để ngăn chặn hoặc hạn chế tiết lộ dữ liệu cá nhân hoặc sử dụng cho mục đích quảng cáo, tiếp thị.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Worst metric:** `context_precision` = 0.0647
- **Chẩn đoán:** Context trả về liên quan nhưng chưa tập trung đúng Điều 9 khoản 8.
- **Hướng sửa:** Boost mạnh cụm `phản đối`, `quảng cáo`, `tiếp thị`; ưu tiên chunk đúng điều khoản.

### #2
- **Question:** Dữ liệu cá nhân có được mua bán không?
- **Expected:** Không được mua, bán dưới mọi hình thức, trừ trường hợp luật có quy định khác.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Worst metric:** `context_precision` = 0.0315
- **Chẩn đoán:** Truy vấn phủ định bị chìm trong context nhiễu.
- **Hướng sửa:** Thêm rule boost cho pattern phủ định (`không được`, `cấm`, `mua bán`).

### #3
- **Question:** Dữ liệu cá nhân được lưu trữ trong bao lâu?
- **Expected:** Chỉ được lưu trữ trong khoảng thời gian phù hợp với mục đích xử lý dữ liệu.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Worst metric:** `context_precision` = 0.1016
- **Chẩn đoán:** Retriever lấy đúng chủ đề nhưng chưa nổi bật đúng câu trả lời chuẩn.
- **Hướng sửa:** Tăng trọng số lexical cho cụm `lưu trữ`, `thời gian`, `mục đích xử lý`.

### #4
- **Question:** Sự im lặng của chủ thể dữ liệu có được coi là đồng ý không?
- **Expected:** Không, sự im lặng hoặc không phản hồi không được coi là sự đồng ý.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Worst metric:** `context_precision` = 0.0699
- **Chẩn đoán:** Câu hỏi yes/no nhưng context top-k chưa đủ sắc nét.
- **Hướng sửa:** Áp dụng answer template yes/no và ưu tiên chunk có mẫu phủ định rõ ràng.

### #5
- **Question:** Nếu không thể chỉnh sửa dữ liệu cá nhân thì phải thông báo sau bao lâu?
- **Expected:** Thông báo tới chủ thể dữ liệu sau 72 giờ kể từ khi nhận được yêu cầu chỉnh sửa.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Worst metric:** `context_precision` = 0.1124
- **Chẩn đoán:** Có context liên quan nhưng câu chứa mốc `72 giờ` không ở vị trí ưu tiên.
- **Hướng sửa:** Boost truy vấn chứa mốc thời gian số + đơn vị (`72`, `giờ`) trước rerank.

## 4. Kế hoạch cải tiến ngắn hạn

1. Parse `Điều/Khoản` và áp dụng metadata-aware retrieval.
2. Tách chiến lược truy vấn theo nhóm câu hỏi: định nghĩa, phủ định, yes/no, mốc thời gian.
3. Áp dụng citation-first prompting để tăng faithfulness.
4. Tuning `RERANK_TOP_K` + threshold theo từng nhóm truy vấn thay vì một cấu hình chung.
