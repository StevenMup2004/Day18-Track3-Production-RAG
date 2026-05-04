# Individual Reflection — Lab 18

**Tên:** Tran Quang Huy  
**Module phụ trách:** M5 (Enrichment)

---

## 1. Đóng góp kỹ thuật

- Phụ trách enrichment cho chunk trước khi index:
  - Contextual enrichment.
  - HYQA-style enrichment.
  - Metadata enrichment.
- Chuẩn hóa format enriched text để M2 index ổn định.
- Kiểm tra khả năng cải thiện recall khi bật enrichment.

## 2. Kiến thức học được

- Enrichment có thể tăng recall tốt nhưng cũng có nguy cơ thêm nhiễu nếu quá dài.
- Metadata enrichment giúp truy vấn theo điều khoản có thêm tín hiệu.
- Cần cân bằng giữa “thêm thông tin” và “giữ câu trả lời đúng trọng tâm”.

## 3. Khó khăn & Cách giải quyết

- Khó khăn: enrichment dễ làm context phình to, ảnh hưởng precision.
- Cách giải quyết: kiểm soát cấu trúc output và giữ metadata rõ ràng để rerank xử lý tốt hơn.
- Thời gian debug: ~1.5 giờ.

## 4. Nếu làm lại

- Tạo profile enrichment nhẹ cho câu hỏi factoid ngắn.
- Tách text dùng để index và text dùng để answer để giảm nhiễu ở bước generate.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
