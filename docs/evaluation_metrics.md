# KHUNG ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH (VQA EVALUATION METRICS FRAMEWORK)
**Dự án:** Hỏi - Đáp Thị Giác Di Sản Văn Hóa Việt Nam (*Vietnamese Cultural Heritage VQA*)  
**Mục tiêu:** Đo lường và đánh giá toàn diện năng lực của các mô hình VLM (Baseline $\rightarrow$ Fine-tuned LoRA $\rightarrow$ Multimodal RAG) trên tập dữ liệu kiểm thử chuẩn hóa `data/processed/test.json`.

---

## 1. BỐI CẢNH & THÁCH THỨC ĐẶC THÙ CỦA DỮ LIỆU

Từ kết quả rà soát thực tế trên $1.060$ câu hỏi kiểm thử thuộc $212$ ảnh di sản, bài toán VQA tiếng Việt về di sản văn hóa có các đặc tính khác biệt so với các bài toán NLP truyền thống:

1. **Sự bất đối xứng về độ dài (Length Asymmetry):**
   * **Đáp án gốc (Ground Truth):** Thường rất cô đọng, trung bình $15.1$ từ (trung vị $4$ từ). Ở Level 1 (màu sắc) và Level 2 (đếm), đáp án chỉ từ $1 - 3$ từ (ví dụ: *"Màu cam/đỏ"*, *"Một chiếc"*).
   * **Mô hình sinh ra (Model Prediction):** Xu hướng trả lời theo văn phong trợ lý đàm thoại hoàn chỉnh, trung bình $50.0$ từ (ví dụ: *"Mái nhà của ngôi đền trong ảnh có màu đỏ rực rỡ..."*).
2. **Sự thất bại của Exact Match (EM):**
   * Nếu dùng so sánh chuỗi tuyệt đối (EM), câu trả lời đúng bản chất của mô hình sẽ bị chấm **0 điểm** chỉ vì mô hình sinh thêm chủ ngữ hoặc từ đệm.
3. **Đa dạng cấp độ nhận thức (Levels 1 - 5):**
   * Trải dài từ nhận diện thị giác bề mặt (màu sắc, đếm số lượng) đến suy luận lịch sử, địa danh và văn hóa sâu sắc đòi hỏi cơ sở tri thức ngoài.

Do đó, hệ thống đánh giá của dự án được xây dựng theo **mô hình đa tầng (Multi-tier Evaluation Framework)**: Kết hợp giữa **Chỉ số truyền thống (Lexical Metrics)** để đảm bảo chuẩn mực học thuật và **Trọng tài AI (LLM-as-a-Judge)** để phản ánh đúng trí thông minh ngữ nghĩa thực tế.

---

## 2. TẦNG 1: CHỈ SỐ TRUYỀN THỐNG (LEXICAL METRICS)
*Được tính toán tự động offline, chi phí bằng 0, đảm bảo tính khách quan và chuẩn mực khi so sánh với các công bố khoa học.*

### 2.1. Token F1-Score (Trọng Tâm Lexical)
* **Ý nghĩa:** Đo lường độ trùng lặp từ vựng giữa Ground Truth ($GT$) và Prediction ($Pred$) sau khi đã đưa về chữ thường và xóa dấu câu.
* **Công thức toán học:**
  $$\text{Precision} = \frac{|Tokens(Pred) \cap Tokens(GT)|}{|Tokens(Pred)|}$$
  $$\text{Recall} = \frac{|Tokens(Pred) \cap Tokens(GT)|}{|Tokens(GT)|}$$
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **Ứng dụng:** Thước đo công bằng cho Level 1, 2 và 3, giúp ghi nhận điểm khi mô hình bắt trúng từ khóa chính mà không phạt nặng việc nói thêm các từ miêu tả ngữ cảnh.

### 2.2. ROUGE-L (Longest Common Subsequence - LCS)
* **Ý nghĩa:** Đo lường độ dài chuỗi con chung dài nhất mà không yêu cầu các từ phải liền kề nhau liên tục.
* **Công thức toán học:**
  $$R_{LCS} = \frac{LCS(GT, Pred)}{m}, \quad P_{LCS} = \frac{LCS(GT, Pred)}{n}$$
  $$\text{ROUGE-L} = \frac{(1 + \beta^2) R_{LCS} P_{LCS}}{R_{LCS} + \beta^2 P_{LCS}}$$
  *(trong đó $m$ là độ dài $GT$, $n$ là độ dài $Pred$, $\beta$ thường chọn là 1.2).*
* **Ứng dụng:** Cực kỳ hiệu quả cho **Level 4 (Ý nghĩa lịch sử)** và **Level 5 (Suy luận tổng hợp)**. ROUGE-L kiểm tra xem mô hình có diễn giải đúng mạch logic và cấu trúc ý nghĩa của câu chuyện văn hóa hay không.

### 2.3. BLEU-1 & BLEU-4 (Bilingual Evaluation Understudy)
* **Ý nghĩa:** Đo lường độ chính xác của cụm n-gram từ $1$-gram đến $4$-gram, kết hợp hệ số phạt câu ngắn (Brevity Penalty - $BP$).
* **Công thức:**
  $$\text{BLEU} = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$
* **Ứng dụng:** Được báo cáo nhằm phục vụ mục đích so sánh với các bài báo khoa học quốc tế trong lĩnh vực VQA.

### 2.4. Keyword Match / Relaxed Accuracy
* **Ý nghĩa:** Kiểm tra xem từ khóa hoặc thực thể then chốt trong $GT$ có xuất hiện nguyên vẹn trong câu trả lời của mô hình hay không:
  $$\text{Keyword Match} = \mathbb{I}\left(\text{clean}(GT) \subseteq \text{clean}(Pred)\right)$$
* **Ứng dụng:** Đánh giá độ tin cậy nhanh cho các câu hỏi nhận diện màu sắc, tên công trình kiến trúc, hoặc tên triều đại lịch sử.

---

## 3. TẦNG 2: CHỈ SỐ NGỮ NGHĨA (LLM-AS-A-JUDGE)
*Thước đo quan trọng nhất cho bài toán VQA mở, giải quyết triệt để vấn đề đồng nghĩa và ngữ cảnh văn hóa Việt Nam.*

Sử dụng mô hình ngôn ngữ lớn tiên tiến (như **Gemini 2.0 Flash** hoặc **GPT-4o-mini**) đóng vai trò là một chuyên gia khảo thí độc lập, chấm điểm dựa trên độ chính xác ngữ nghĩa và tính trung thực lịch sử.

### 3.1. Thang Đo Điểm Số (Scoring Rubric 1 - 5 Điểm)

| Điểm Số | Phân Loại | Tiêu Chuẩn Đánh Giá | Ví Dụ Thực Tế |
| :---: | :--- | :--- | :--- |
| **5** | **Hoàn toàn chính xác** | Câu trả lời phản ánh chính xác và trọn vẹn sự thật của đáp án gốc. Cho phép diễn đạt bằng các từ ngữ khác nhau hoặc thêm chi tiết mô tả hợp lý. | GT: *"Màu cam/đỏ"*<br>Pred: *"Mái nhà của ngôi đền trong ảnh có màu đỏ."* $\rightarrow$ **5đ** |
| **4** | **Đúng phần lớn** | Trả lời trúng thực thể / ý tưởng cốt lõi, nhưng thiếu một chi tiết phụ nhỏ không ảnh hưởng lớn đến ngữ cảnh. | GT: *"Núi đá vôi và thung lũng lúa"*<br>Pred: *"Những ngọn núi đá vôi nhấp nhô."* $\rightarrow$ **4đ** |
| **3** | **Đúng một phần** | Nhận diện được một nửa thông tin, còn lại mơ hồ hoặc thiếu dữ kiện quan trọng. | GT: *"Chùa Một Cột, xây thời vua Lý Thái Tông"*<br>Pred: *"Đây là di tích Chùa Một Cột."* $\rightarrow$ **3đ** |
| **2** | **Kém / Lạc đề nhẹ** | Câu trả lời chung chung, không đi vào trọng tâm câu hỏi, hoặc đoán mò không có căn cứ. | GT: *"Năm chiếc xe"*<br>Pred: *"Rất khó xác định, có thể có nhiều xe..."* $\rightarrow$ **2đ** |
| **1** | **Sai hoàn toàn / Bịa đặt** | Thông tin sai sự thật, ảo giác địa danh (hallucination), hoặc mô hình thừa nhận không biết. | GT: *"Cờ cầu nguyện ngũ sắc Phật giáo"*<br>Pred: *"Không thể xác định được ý nghĩa..."* $\rightarrow$ **1đ** |

### 3.2. Prompt Chuẩn Cho Trọng Tài LLM-as-a-Judge

```text
Bạn là một chuyên gia thẩm định chất lượng các hệ thống Trí tuệ Nhân tạo trong bài toán Hỏi - Đáp Thị Giác về Di sản Văn hóa Việt Nam.

Nhiệm vụ của bạn là so sánh "Câu trả lời của Mô hình" với "Đáp án chuẩn (Ground Truth)" dựa trên "Câu hỏi", sau đó chấm điểm độ chính xác ngữ nghĩa theo thang điểm từ 1 đến 5.

Quy tắc chấm điểm:
- Điểm 5: Hoàn toàn chính xác về mặt ngữ nghĩa và sự thật lịch sử/văn hóa. Chấp nhận câu trả lời dạng câu đầy đủ dù đáp án chuẩn chỉ là cụm từ ngắn.
- Điểm 4: Đúng ý chính, thiếu sót nhỏ ở chi tiết phụ.
- Điểm 3: Đúng khoảng một nửa thông tin quan trọng.
- Điểm 2: Trả lời mơ hồ, phán đoán chung chung hoặc sai lệch phần lớn.
- Điểm 1: Sai sự thật, bịa đặt thực thể lịch sử, hoặc không trả lời được.

Dữ liệu đánh giá:
- Câu hỏi: {question}
- Đáp án chuẩn (Ground Truth): {ground_truth}
- Câu trả lời của Mô hình: {prediction}

Hãy trả về kết quả dưới định dạng JSON duy nhất như sau:
{
  "score": <Điểm số từ 1 đến 5>,
  "reasoning": "<Giải thích ngắn gọn lý do chấm điểm trong 1-2 câu>"
}
```

---

## 4. TẦNG 3: PHÂN RÃ THEO ĐẶC THÙ (FINE-GRAINED SLICE ANALYSIS)

Để tránh hiện tượng điểm số trung bình làm che giấu các điểm yếu cục bộ, mọi chỉ số đánh giá bắt buộc phải được phân rã theo 2 lát cắt:

### 4.1. Phân rã theo Cấp độ khó (Difficulty Levels)
* **Level 1 (Màu sắc & Đặc trưng thị giác):** Đo năng lực của Vision Encoder.
* **Level 2 (Đếm & Số lượng):** Đo khả năng đếm vật thể nhỏ / mật độ.
* **Level 3 (Nhận diện công trình kiến trúc / Địa danh):** Đo năng lực gán nhãn thực thể di sản.
* **Level 4 (Ý nghĩa lịch sử, văn hóa, niên đại):** Đo sự hiểu biết tri thức lịch sử chuyên sâu.
* **Level 5 (Suy luận tổng hợp & Không gian):** Đo năng lực reasoning đa chiều.

### 4.2. Phân rã theo Ý định câu hỏi (Question Intent)
* **Nhóm Thuần Thị Giác:** `Màu sắc`, `Số lượng/Đếm`.
* **Nhóm Tri Thức Văn Hóa & Địa Danh:** `Vị trí/Địa lý`, `Thời gian/Niên đại`, `Nhân vật/Lịch sử`.
* **Nhóm Suy Luận:** `Ý nghĩa/Mục đích`, `Miêu tả chi tiết`.

---

## 5. MẪU BẢNG BÁO CÁO KẾT QUẢ BENCHMARK (CHUẨN BÀI BÁO KHOA HỌC)

Khi công bố kết quả của từng giai đoạn (Baseline, LoRA Fine-Tuned, Multimodal RAG), bảng số liệu chuẩn sẽ được trình bày như sau:

| Tập Dữ Liệu / Phân Đoạn | Số Câu | Token F1 (%) | ROUGE-L (%) | BLEU-4 (%) | Keyword Match (%) | LLM Judge (1-5) | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Toàn bộ Test Set** | **1,060** | **...** | **...** | **...** | **...** | **...** | **2.48s** |
| - *Level 1 (Màu sắc)* | 212 | ... | ... | ... | ... | ... | ~1.8s |
| - *Level 2 (Đếm)* | 212 | ... | ... | ... | ... | ... | ~2.2s |
| - *Level 3 (Địa danh/Công trình)*| 212 | ... | ... | ... | ... | ... | ~2.5s |
| - *Level 4 (Lịch sử/Văn hóa)* | 212 | ... | ... | ... | ... | ... | ~3.1s |
| - *Level 5 (Suy luận tổng hợp)* | 212 | ... | ... | ... | ... | ... | ~3.4s |
