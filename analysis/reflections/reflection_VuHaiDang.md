# Individual Reflection — Lab 18

**Tên:** Vu Hai Dang - 2A202600339
**Module phụ trách:** M2 (Hybrid Search: BM25 + Dense + RRF)

---

## 1. Đóng góp kỹ thuật

- Mình phụ trách toàn bộ luồng retrieval của pipeline production, gồm 3 lớp chính:
  - **Lexical retrieval (BM25)** cho tiếng Việt.
  - **Dense retrieval (Qdrant + OpenAI Embeddings)**.
  - **Fusion (RRF)** để hợp nhất hai danh sách kết quả theo thứ hạng.
- Các hạng mục kỹ thuật đã hoàn thiện trong M2:
  - Tích hợp tách từ tiếng Việt bằng `underthesea` để BM25 xử lý query tốt hơn thay vì split trắng đơn thuần.
  - Dùng `rank_bm25.BM25Okapi` để index và search lexical thực tế.
  - Dùng OpenAI Embeddings API với model `text-embedding-3-large` (3072 chiều) cho dense retrieval.
  - Index vector vào **Qdrant thật** (`QdrantClient`) theo collection chuẩn của lab.
  - Triển khai RRF để giữ ưu điểm của cả lexical match và semantic match.
- Phần hardening production đã xử lý:
  - Chuẩn hóa chạy **real mode** (không mock search, không mock qdrant).
  - Vá tương thích `qdrant-client` mới: chuyển sang `query_points()` khi `search()` không còn khả dụng.
  - Kiểm tra kích thước embedding để tránh lỗi mismatch dimension khi tạo collection.
- Kết quả test và vận hành:
  - Auto-tests: **37/37 pass**.
  - Pipeline chạy end-to-end với Qdrant local + OpenAI embeddings thật.

## 2. Kiến thức học được

- Bài học lớn nhất là retrieval trong production không thể chỉ dựa vào một kỹ thuật:
  - BM25 tốt với từ khóa pháp lý/điều khoản cụ thể.
  - Dense tốt với diễn đạt tự nhiên, paraphrase.
  - RRF giúp cân bằng hai hướng, tăng recall rõ rệt.
- Mình hiểu sâu hơn trade-off giữa các metric RAG:
  - Tăng `context_recall` thường dễ hơn tăng `context_precision`.
  - Precision thấp sẽ làm generator dễ trả lời an toàn hoặc bỏ sót span quan trọng.
- Mình cũng rút ra rằng chuẩn hóa dữ liệu test (ground truth ngắn, rõ span) ảnh hưởng trực tiếp độ tin cậy của đánh giá.

## 3. Khó khăn & Cách giải quyết

- Khó khăn 1: API Qdrant thay đổi theo version (`search` không còn trong client mới).
  - Cách giải quyết: thêm nhánh tương thích `query_points()` và giữ backward compatibility.
- Khó khăn 2: lock thư mục Qdrant local khi chạy nhiều phiên.
  - Cách giải quyết: tách `QDRANT_LOCAL_PATH` theo từng lần run để tránh xung đột lock file.
- Khó khăn 3: phân biệt rõ “đang chạy thật” hay fallback.
  - Cách giải quyết: rà soát code M2 theo hướng real-only, xác thực runtime class/client thực tế trước khi benchmark.
- Thời gian debug tổng cho M2: khoảng **3-4 giờ** (bao gồm fix tương thích thư viện và chạy lại eval nhiều vòng).

## 4. Nếu làm lại

- Mình sẽ thêm metadata-aware retrieval sớm hơn (boost theo `Điều/Khoản`) để cải thiện precision cho domain pháp lý.
- Mình sẽ tách cấu hình benchmark rõ hơn theo profile:
  - Profile tối ưu recall.
  - Profile tối ưu precision.
- Mình cũng muốn thử thêm chiến lược score calibration giữa BM25 và dense trước bước fusion.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |
