# Phân Tích Lỗi - Lab 18: Production RAG

**Nhóm:** Solo  
**Ngày cập nhật:** 2026-05-04  
**Số câu test:** 25

## 1. Bảng so sánh metric

| Metric | Naive Baseline | Production | Delta |
|--------|---------------:|-----------:|------:|
| Faithfulness | 1.0000 | 0.9338 | -0.0662 |
| Answer Relevancy | 0.0204 | 0.4815 | +0.4610 |
| Context Precision | 0.0155 | 0.1425 | +0.1271 |
| Context Recall | 0.1049 | 0.9388 | +0.8339 |

## 2. Kết luận nhanh

- Dữ liệu OCR mới giúp retrieval cải thiện mạnh, đặc biệt là recall và precision.
- Production hiện có độ phủ ngữ cảnh rất cao, nhưng vẫn có một số câu trả lời bị “an toàn quá mức” hoặc chưa bám trúng cụm cần nêu.
- Nhóm lỗi chính vẫn nghiêng về `context_precision` (context còn nhiễu ở vài câu pháp lý cụ thể).

## 3. Bottom-5 failures (theo báo cáo mới nhất)

### #1
- **Question:** Hình ảnh của cá nhân thuộc loại dữ liệu nào?
- **Expected:** Thuộc dữ liệu cá nhân cơ bản.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Worst metric:** `context_precision` = 0.0532
- **Chẩn đoán:** Có chunk liên quan nhưng câu trả lời đúng chưa được đưa lên vị trí nổi bật trong top context.
- **Hướng sửa:** Boost cụm `hình ảnh` + `dữ liệu cá nhân cơ bản`; ưu tiên chunk thuộc Điều 2.

### #2
- **Question:** Một nghĩa vụ của chủ thể dữ liệu theo Điều 10 là gì?
- **Expected:** Cung cấp đầy đủ, chính xác dữ liệu cá nhân khi đồng ý cho phép xử lý dữ liệu cá nhân.
- **Got:** Một nghĩa vụ của chủ thể dữ liệu theo Điều 10 là tự bảo vệ dữ liệu cá nhân của mình.
- **Worst metric:** `context_precision` = 0.0722
- **Chẩn đoán:** Câu trả lời đúng về mặt pháp lý nhưng chưa khớp cụm ground truth mục tiêu.
- **Hướng sửa:** Thêm constraint ưu tiên span có cụm `cung cấp đầy đủ, chính xác` khi query có mẫu `một nghĩa vụ`.

### #3
- **Question:** Nguyên tắc nào yêu cầu dữ liệu phải thu thập phù hợp, giới hạn?
- **Expected:** Dữ liệu cá nhân thu thập phải phù hợp và giới hạn trong phạm vi, mục đích cần xử lý.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Worst metric:** `answer_relevancy` = 0.1875
- **Chẩn đoán:** Đây là lỗi thiên về bước trả lời (generator), không phải retrieval thuần.
- **Hướng sửa:** Tinh chỉnh prompt bắt buộc trích đúng nguyên tắc khi context có cụm `phù hợp`, `giới hạn`, `mục đích`.

### #4
- **Question:** Chủ thể dữ liệu có quyền phản đối xử lý dữ liệu để làm gì?
- **Expected:** Để ngăn chặn hoặc hạn chế tiết lộ dữ liệu cá nhân hoặc sử dụng cho mục đích quảng cáo, tiếp thị.
- **Got:** Chủ thể dữ liệu có quyền phản đối việc xử lý dữ liệu cá nhân của mình nhằm ngăn chặn hoặc hạn chế việc xử lý đó.
- **Worst metric:** `context_precision` = 0.1047
- **Chẩn đoán:** Trả lời gần đúng nhưng thiếu vế mục đích `quảng cáo, tiếp thị` theo ground truth.
- **Hướng sửa:** Query expansion cho `quảng cáo`, `tiếp thị`; ép answer template đủ 2 ý.

### #5
- **Question:** Bên kiểm soát phải thực hiện yêu cầu phản đối xử lý dữ liệu trong bao lâu?
- **Expected:** Trong 72 giờ sau khi nhận được yêu cầu.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Worst metric:** `context_precision` = 0.1047
- **Chẩn đoán:** Mốc thời gian có trong tài liệu nhưng không được ưu tiên ở top context.
- **Hướng sửa:** Boost pattern số + đơn vị thời gian (`72`, `giờ`) trước rerank.

## 4. Kế hoạch cải tiến ngắn hạn

1. Metadata-aware retrieval theo `Điều/Khoản/Điểm` để giảm nhiễu.
2. Rerank theo intent câu hỏi (định nghĩa, nghĩa vụ, mốc thời gian, phủ định).
3. Prompt trả lời theo khung pháp lý ngắn gọn, tránh trả lời vòng vo và thiếu ý chính.
4. Theo dõi riêng các câu có nhãn `72 giờ`, `không được`, `quảng cáo/tiếp thị` để regression test.
