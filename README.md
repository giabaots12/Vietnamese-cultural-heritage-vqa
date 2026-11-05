# Vietnamese Cultural Heritage VQA (Hỏi - Đáp Thị Giác Di Sản Văn Hóa Việt Nam)

> Hệ thống ứng dụng mô hình đa phương thức ngôn ngữ - thị giác (**Vision-Language Model - VLM**) chuyên biệt phục vụ nhận diện, hỏi đáp và bảo tồn tri thức về di tích lịch sử, danh lam thắng cảnh và kiến trúc văn hóa Việt Nam.

---

## 1. Giới Thiệu Dự Án

Dự án **Vietnamese Cultural Heritage VQA** hướng tới việc giải quyết bài toán thị giác máy tính kết hợp ngôn ngữ tự nhiên trong lĩnh vực di sản văn hóa Việt Nam. Bằng cách tiếp cận mô hình thị giác đa phương thức hiện đại **Vintern-1B-v3.5** (kiến trúc InternVL 2.5), hệ thống cho phép người dùng tải lên hình ảnh một công trình kiến trúc, ngôi chùa, lăng tẩm hoặc hiện vật lịch sử và trò chuyện tương tác tự nhiên bằng tiếng Việt.

### Điểm nổi bật kỹ thuật:
- **Cơ chế thị giác động (Dynamic Aspect-Ratio Patching)**: Tự động phân giải ảnh theo tỉ lệ khía cạnh thực tế mà không bóp méo hay cắt xén hình ảnh di tích.
- **Chiến lược tối ưu bộ nhớ VRAM**: Khóa cứng phần trích xuất thị giác (InternViT) và chỉ tinh chỉnh tham số nhẹ qua **PEFT LoRA** trên mô hình ngôn ngữ (Qwen2 backbone), hoạt động mượt mà trên GPU phổ thông (6GB VRAM như RTX 4050).
- **Bộ chuẩn đánh giá 2 tầng (Two-Tier Evaluation)**: Kết hợp 5 module chẩn đoán cục bộ (100% offline, 0 cost) và giám khảo AI (Gemini LLM-as-a-Judge).

---

## 2. Lộ Trình Phát Triển 3 Giai Đoạn

Dự án được phân kỳ thực hiện theo 3 giai đoạn rõ ràng:

1. **Giai đoạn 1 (Baseline Zero-Shot)**: Đánh giá khả năng hiểu biết văn hóa nguyên bản của mô hình Vintern-1B-v3.5 trên tập 936 câu hỏi kiểm thử `VietTravelVQA`.
2. **Giai đoạn 2 (Fine-Tuning LoRA)**: Tinh chỉnh tham số hiệu quả thích nghi miền văn hóa (Domain Adaptation) trên 5.489 mẫu huấn luyện với cơ chế Early Stopping theo dõi `val_loss`.
3. **Giai đoạn 3 (Multimodal RAG)**: Tích hợp cơ sở tri thức ngoài (Retrieval-Augmented Generation) để cung cấp điển tích, niên đại lịch sử chuẩn xác và triệt tiêu tối đa ảo giác (Hallucination).

---

## 3. Cấu Trúc Thư Mục Dự Án (Directory Structure)

Hệ thống được tổ chức theo chuẩn module hóa công nghiệp (Production-Grade ML Pipeline):

```text
Vietnamese Cultural Heritage VQA/
├── .gitignore                      # Quy chuẩn loại trừ file nặng, checkpoint và cache
├── README.md                       # Tài liệu tổng quan và hướng dẫn dự án (file này)
├── requirements.txt                # Danh sách thư viện phụ thuộc (PyTorch, Transformers, PEFT, Streamlit)
├── app.py                          # Điểm khởi chạy ứng dụng Web Demo (Streamlit)
│
├── configs/                        # Cấu hình siêu tham số tập trung (YAML)
│   ├── default_config.yaml         # Cấu hình nạp mô hình và suy luận cho Web Demo
│   ├── finetune_config.yaml        # Siêu tham số LoRA, batch size, learning rate, epochs
│   └── benchmark_config.yaml       # Cấu hình tập benchmark và các ngưỡng đánh giá
│
├── data/                           # Quản lý tài nguyên dữ liệu
│   ├── raw/                        # Dữ liệu gốc thu thập từ nguồn ngoài
│   ├── processed/                  # Dữ liệu tiền xử lý (train.jsonl, val.jsonl)
│   ├── samples/                    # Ảnh mẫu minh họa tải nhanh trên giao diện web
│   └── benchmark/                  # Tập dữ liệu kiểm thử chuẩn hóa (936 câu hỏi test)
│
├── models/                         # Trọng số mô hình (Không đưa lên Git)
│   ├── Vintern-1B-v3_5/            # Trọng số Base Model cục bộ (~3.75GB)
│   └── adapters/                   # Nơi lưu trữ checkpoint LoRA sau huấn luyện
│
├── src/                            # Mã nguồn chính của dự án
│   ├── core/                       # Quản lý cấu hình Pydantic và session người dùng
│   ├── inference/                  # Engine suy luận VLM và tiền xử lý ảnh cắt lát động
│   ├── training/                   # Module nạp dữ liệu, cấu hình LoRA và bộ vòng lặp Trainer
│   ├── evaluation/                 # Bộ thư viện đo lường metric chẩn đoán
│   ├── ui/                         # Thành phần giao diện Streamlit (chat bubbles, controls)
│   └── utils/                      # Các tiện ích bộ nhớ GPU, logger hệ thống
│
├── scripts/                        # Các kịch bản thực thi qua giao diện dòng lệnh (CLI)
│   ├── prepare_dataset.py          # Tiền xử lý và chia tập dữ liệu train/val theo Image-level
│   ├── run_train.py                # Kích hoạt quá trình huấn luyện LoRA
│   └── run_benchmark.py            # Chạy đánh giá đối đầu (Side-by-Side Benchmark)
│
├── experiments/                    # Lưu trữ kết quả đo đạc thực nghiệm
│   ├── baseline/                   # Kết quả zero-shot baseline (JSON và markdown report)
│   ├── fine_tune/                  # Kết quả sau khi huấn luyện LoRA
│   └── fine_tune_and_RAG/          # Kết quả đánh giá hệ thống kết hợp RAG
│
├── tests/                          # Bộ kịch bản sinh dữ liệu và chấm điểm tự động
│   ├── baseline/                   # Scripts sinh câu trả lời và chấm điểm baseline
│   ├── fine_tune/                  # Scripts đánh giá mô hình LoRA
│   └── fine_tune_and_RAG/          # Scripts kiểm thử hệ thống RAG
│
└── docs/                           # Tài liệu đặc tả kỹ thuật, nghiên cứu và log huấn luyện
    ├── metrics_evaluation_specify.md # Đặc tả chi tiết 5 module chẩn đoán và công thức
    └── training_logs/              # Lịch sử suy giảm hàm mất mát (loss curve)
```

---

## 4. Hướng Dẫn Cài Đặt Nhanh (Quickstart)

### Bước 1: Khởi tạo môi trường ảo Python
Khuyến nghị sử dụng Python 3.10 trên môi trường Conda:

```bash
conda create -n heritage-vqa python=3.10 -y
conda activate heritage-vqa
```

### Bước 2: Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 3: Chuẩn bị mô hình Vintern-1B-v3.5
Tải trọng số mô hình `5CD-AI/Vintern-1B-v3_5` từ Hugging Face và đặt vào thư mục:
```text
models/Vintern-1B-v3_5/
```

### Bước 4: Chạy thử nghiệm ứng dụng Web Demo
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở tại địa chỉ `http://localhost:8501`.

---

## 5. Quy Chuẩn Đánh Giá Năng Lực (Diagnostic Suite)

Hệ thống đánh giá mô hình dựa trên 5 thước đo chẩn đoán độc lập:
1. **Acc_Counting**: Năng lực đếm vật thể chi tiết (số cột đình, số bậc thang, số người).
2. **Acc_Entity**: Mức độ định danh chính xác tên chùa, lăng tẩm, hiện vật lịch sử.
3. **Acc_Refusal**: Khả năng phản biện và chống sập bẫy trước các câu hỏi có tiền đề sai lệch.
4. **ROUGE-L & Verbosity Ratio**: Mức độ bao phủ ý văn hóa và kiểm soát tính dài dòng.
5. **Geo_Hallucination**: Tỷ lệ phát hiện bịa đặt thông tin địa danh hành chính (63 tỉnh thành).

