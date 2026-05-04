# Individual Reflection — Lab 18

**Tên:** Tran Ngoc Son  
**Module phụ trách:** M3 (Reranking)

---

## 1. Đóng góp kỹ thuật

- Phụ trách tầng rerank sau retrieval để chọn context tốt nhất trước khi generate.
- Triển khai `CrossEncoderReranker` với model `BAAI/bge-reranker-v2-m3`.
- Thiết kế interface rerank thống nhất với pipeline (`query + docs -> top_k`).
- Benchmark độ trễ rerank để đánh đổi giữa chất lượng và tốc độ.

## 2. Kiến thức học được

- Rerank là tầng quan trọng để nâng precision khi top-k retrieval còn nhiễu.
- Cần cân bằng giữa model mạnh và latency trong môi trường production.
- Dữ liệu pháp lý cần rerank ưu tiên câu chứa span trả lời thay vì đoạn mô tả chung.

## 3. Khó khăn & Cách giải quyết

- Khó khăn: phụ thuộc model từ HF, dễ phát sinh lỗi môi trường/mạng.
- Cách giải quyết: chuẩn hóa môi trường và kiểm tra backend rerank trước khi chạy full eval.
- Thời gian debug: ~2.5 giờ.

## 4. Nếu làm lại

- Tạo policy rerank riêng cho nhóm query `Điều/Khoản` và query yes/no.
- Bổ sung ngưỡng điểm để loại bớt context nhiễu trước bước generate.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |
