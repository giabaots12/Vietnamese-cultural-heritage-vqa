# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH BASELINE (VINTERN-1B-V3.5)
*Dự án: Hỏi - Đáp Thị Giác Di Sản Văn Hóa Việt Nam (Vietnamese Cultural Heritage VQA)*  
*Tập dữ liệu kiểm thử: `data/processed/test.json` (1,060 câu hỏi / 212 ảnh độc lập)*

---

## 1. BẢNG HIỆU NĂNG THEO CẤP ĐỘ KHÓ (DIFFICULTY LEVELS)

| Cấp độ khó | Số câu | Token F1 (%) | ROUGE-L (%) | ROUGE-L Rec (%) | BLEU-4 (%) | Keyword Match (%) | Thời gian (s) | LLM Judge (1-5) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Level 1 (Màu sắc/Đặc trưng) | 212 | 21.12 | 22.84 | 78.66 | 6.7 | 65.57 | 1.7 | 4.29 |
| Level 2 (Đếm số lượng) | 212 | 8.32 | 8.84 | 43.92 | 4.31 | 33.96 | 2.02 | 2.64 |
| Level 3 (Nhận diện công trình) | 212 | 17.06 | 17.07 | 48.32 | 6.47 | 24.53 | 2.34 | 2.32 |
| Level 4 (Ý nghĩa lịch sử) | 212 | 19.75 | 16.51 | 42.12 | 5.17 | 14.62 | 3.17 | 2.77 |
| Level 5 (Suy luận tổng hợp) | 212 | 36.16 | 28.13 | 40.09 | 10.05 | 4.25 | 3.16 | 2.58 |
| >> TOÀN BỘ TẬP TEST << | 1060 | 20.48 | 18.68 | 50.62 | 6.54 | 28.58 | 2.48 | 2.98 |

---

## 2. BẢNG HIỆU NĂNG THEO Ý ĐỊNH CÂU HỎI (QUESTION INTENTS)

| Nhóm Ý Định | Số câu | Token F1 (%) | ROUGE-L (%) | Keyword Match (%) | LLM Judge (1-5) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Miêu tả chi tiết / Khác | 447 | 22.08 | 18.83 | 20.13 | 2.83 |
| Màu sắc | 202 | 23.5 | 24.3 | 59.9 | 4.12 |
| Nhân vật / Lịch sử | 5 | 21.95 | 22.16 | 20.0 | 2.33 |
| Số lượng / Đếm | 227 | 10.14 | 10.34 | 31.72 | 2.63 |
| Thời gian / Niên đại | 5 | 21.53 | 20.01 | 0.0 | 1.50 |
| Vị trí / Địa danh | 33 | 36.19 | 31.48 | 6.06 | 2.40 |
| Ý nghĩa / Mục đích | 86 | 26.15 | 20.33 | 8.14 | 3.00 |
| Định danh / Tên gọi | 55 | 20.59 | 20.49 | 18.18 | 1.58 |

---

## 3. NHẬN XÉT ĐẶC TÍNH MÔ HÌNH GỐC (BASELINE INSIGHTS)

1. **Năng lực thị giác bề mặt (Level 1 & Level 2):**
   * Mô hình nhận diện màu sắc rất nhạy bén, đạt Keyword Match cao ở Level 1.
   * Với Level 2 (Đếm), mô hình gặp khó khăn khi đếm các vật thể nhỏ ở cự ly xa (xe cộ, người đi lại).
2. **Năng lực nhận diện công trình di sản (Level 3):**
   * Bắt đầu có sự phân hóa: Các địa danh nổi tiếng (núi đá vôi, vịnh Hạ Long, chùa Cầu) được nhận diện tốt, nhưng các di tích ít phổ biến hơn có hiện tượng nhầm lẫn.
3. **Nhu cầu tri thức ngoài (Level 4 & Level 5):**
   * Ở các câu hỏi về ý nghĩa tôn giáo, niên đại xây dựng, mô hình thường trả lời chung chung hoặc thừa nhận không có đủ thông tin từ bức ảnh.
   * 👉 **Đây chính là minh chứng thuyết phục nhất cho sự cần thiết của Giai đoạn 3 (Fine-tuning LoRA) và Giai đoạn 4 (Multimodal RAG)!**
