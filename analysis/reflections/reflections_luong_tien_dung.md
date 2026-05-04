# Individual Reflection — Lab 18

**Tên:** Lương Tiến Dũng 
**Module phụ trách:** M1 (Chunking)

---

## 1. Đóng góp kỹ thuật

- Phụ trách thiết kế và kiểm thử các chiến lược chia đoạn tài liệu:
  - Hierarchical chunking (parent-child).
  - Semantic chunking.
  - Structure-aware chunking theo bố cục điều khoản.
- Chuẩn hóa metadata chunk để các module M2/M3/M4 dùng ổn định.
- Kiểm tra tính đầy đủ dữ liệu đầu vào trước khi index.

## 2. Kiến thức học được

- Chunking ảnh hưởng trực tiếp tới quality retrieval, đặc biệt là `context_recall`.
- Chunk quá ngắn làm mất ngữ cảnh; chunk quá dài làm giảm precision.
- Parent-child chunking hữu ích khi cần vừa giữ ngữ cảnh rộng vừa truy hồi đúng đoạn nhỏ.

## 3. Khó khăn & Cách giải quyết

- Khó khăn: tài liệu pháp lý có nhiều câu dài, nhiều mệnh đề lồng nhau.
- Cách giải quyết: kết hợp chunk theo cấu trúc + giới hạn kích thước để tránh vỡ nghĩa.
- Thời gian debug: ~2 giờ.

## 4. Nếu làm lại

- Bổ sung rule nhận diện mạnh hơn cho mẫu `Điều`, `Khoản`, `Điểm`.
- Tinh chỉnh size chunk theo từng loại câu hỏi (định nghĩa, thời hạn, quyền/nghĩa vụ).

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
