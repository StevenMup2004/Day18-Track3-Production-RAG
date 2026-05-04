# Báo Cáo Nhóm - Lab 18: Production RAG

**Nhóm:** Solo (1 người)  
**Ngày cập nhật:** 2026-05-04

## 1. Phân công và trạng thái

| Thành viên | Module | Trạng thái | Kiểm thử |
|------------|--------|------------|----------|
| Luong Tien Dung | M1: Chunking | Hoàn thành | Pass |
| Vu Hai Dang| M2: Hybrid Search | Hoàn thành | Pass |
| Tran Ngoc Son  | M3: Reranking | Hoàn thành | Pass |
| Le Hoang Dat  | M4: Evaluation | Hoàn thành | Pass |
| Tran Quang Huy | M5: Enrichment | Hoàn thành | Pass |

## 2. Kết quả RAGAS (bộ test sạch 25 câu)

| Metric | Naive Baseline | Production | Delta |
|--------|---------------:|-----------:|------:|
| Faithfulness | 1.0000 | 0.9338 | -0.0662 |
| Answer Relevancy | 0.0204 | 0.4815 | +0.4610 |
| Context Precision | 0.0155 | 0.1425 | +0.1271 |
| Context Recall | 0.1049 | 0.9388 | +0.8339 |

## 3. Nhận định chính

1. Sau khi thay dữ liệu bằng bản OCR, chất lượng retrieval tăng rõ rệt, đặc biệt ở `Context Recall` và `Context Precision`.
2. `Faithfulness` vẫn thấp hơn baseline nhưng đã cải thiện đáng kể so với các vòng trước (giảm còn -0.0662).
3. `Answer Relevancy` tăng mạnh so với baseline, cho thấy pipeline production đã trả lời sát câu hỏi hơn đáng kể.
4. Điểm nghẽn còn lại tập trung ở nhóm câu hỏi pháp lý cần trích xuất đúng cụm mục tiêu (Điều/Khoản, mốc 72 giờ, câu phủ định).

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

1. Parse `Điều/Khoản` từ query để boost đúng metadata section trước rerank.
2. Thêm query expansion cho nhóm phủ định (`không`, `cấm`, `không được`) và mốc thời gian (`72 giờ`).
3. Siết prompt trả lời theo khuôn “trả lời ngắn + trích đúng cụm bằng chứng”.
4. Tuning `RERANK_TOP_K` theo nhóm câu hỏi thay vì dùng một cấu hình chung.
