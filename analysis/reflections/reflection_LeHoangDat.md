# Individual Reflection — Lab 18

**Tên:** Le Hoang Dat  
**Module phụ trách:** M4 (Evaluation)

---

## 1. Đóng góp kỹ thuật

- Phụ trách pipeline đánh giá và báo cáo:
  - Chuẩn hóa input cho RAGAS.
  - Tính các metric aggregate.
  - Sinh report JSON và failure analysis.
- Kiểm tra tính nhất quán giữa baseline và production khi so sánh.
- Hỗ trợ làm sạch test set để giảm nhiễu khi chấm điểm.

## 2. Kiến thức học được

- Một metric đơn lẻ không đủ phản ánh chất lượng RAG.
- Cần đọc đồng thời faithfulness, relevancy, precision, recall để hiểu đúng pipeline.
- Chất lượng ground truth ảnh hưởng rất mạnh tới độ tin cậy đánh giá.

## 3. Khó khăn & Cách giải quyết

- Khó khăn: dữ liệu test ban đầu chưa đồng đều về độ dài/độ rõ span.
- Cách giải quyết: tinh chỉnh test set theo hướng câu trả lời ngắn, bám đúng nội dung tài liệu.
- Thời gian debug: ~2 giờ.

## 4. Nếu làm lại

- Bổ sung dashboard mini theo nhóm lỗi để theo dõi drift qua từng lần tuning.
- Xây dựng checklist kiểm thử regression cho từng module.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
