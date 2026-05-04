# Báo Cáo Nhóm - Lab 18: Production RAG

**Nhóm:** Solo (1 người)  
**Ngày cập nhật:** 2026-05-04

## 1. Phân công và trạng thái

| Thành viên | Module | Trạng thái | Kiểm thử |
|------------|--------|------------|----------|
| dangv | M1: Chunking | Hoàn thành | Pass |
| dangv | M2: Hybrid Search | Hoàn thành | Pass |
| dangv | M3: Reranking | Hoàn thành | Pass |
| dangv | M4: Evaluation | Hoàn thành | Pass |
| dangv | M5: Enrichment | Hoàn thành | Pass |

## 2. Kết quả RAGAS (bộ test sạch 25 câu)

| Metric | Naive Baseline | Production | Delta |
|--------|---------------:|-----------:|------:|
| Faithfulness | 1.0000 | 0.8651 | -0.1349 |
| Answer Relevancy | 0.0204 | 0.5029 | +0.4825 |
| Context Precision | 0.0155 | 0.0907 | +0.0752 |
| Context Recall | 0.1049 | 0.8805 | +0.7756 |

## 3. Nhận định chính

1. Production cải thiện rất mạnh ở khả năng thu hồi ngữ cảnh đúng (`Context Recall`) và mức độ trả lời sát câu hỏi (`Answer Relevancy`).
2. `Context Precision` vẫn còn thấp vì context top-k còn nhiễu, đặc biệt ở nhóm câu truy vấn theo Điều/Khoản hoặc câu phủ định.
3. `Faithfulness` thấp hơn baseline là trade-off dễ thấy khi chuyển từ kiểu copy-context sang generate bằng LLM.
4. Với mục tiêu tra cứu thực tế, pipeline hiện tại đã đạt tiến bộ rõ rệt; hướng tối ưu tiếp theo là làm retrieval tập trung hơn để tăng precision và kéo faithfulness lên thêm.

## 4. Đầu ra phục vụ debug/presentation

Đã lưu đầy đủ output từng câu hỏi của production tại:
- `reports/production_outputs.json`

File gồm các trường:
- `question`
- `ground_truth`
- `answer`
- `contexts`
- `article`
- `evidence_span`

## 5. Gợi ý tối ưu tiếp theo

1. Parse pattern `Điều/Khoản` từ query để boost đúng metadata section trước khi rerank.
2. Thêm query expansion cho nhóm từ phủ định và nhóm phân loại dữ liệu.
3. Bổ sung answer template ngắn + trích dẫn câu nguồn để tăng faithfulness.
4. Tinh chỉnh `RERANK_TOP_K` và score-threshold theo từng nhóm câu hỏi.
