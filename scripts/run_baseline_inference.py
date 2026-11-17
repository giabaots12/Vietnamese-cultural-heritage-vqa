#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script: run_baseline_inference.py
Mục đích:
    Thực thi quá trình suy luận (inference) của mô hình gốc Vintern-1B-v3.5
    trên tập dữ liệu kiểm thử (data/processed/test.json).
    Kết quả dự đoán được lưu lại vào experiments/baseline/predictions.json
    để phục vụ giai đoạn tính toán Metrics (F1, ROUGE, BLEU, LLM-as-a-judge).
"""

import os
import sys
import time
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from PIL import Image

# 1. Tắt toàn bộ các cảnh báo rác (deprecation, future warnings của thư viện)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Thư viện thanh tiến trình tqdm
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Thêm thư mục gốc dự án vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Cấu hình logging gọn gàng, rõ ràng
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Baseline")


def parse_arguments() -> argparse.Namespace:
    """Định nghĩa và phân tích các tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Chạy suy luận mô hình VLM Baseline trên tập dữ liệu kiểm thử."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "test.json"),
        help="Đường dẫn tới file JSON kiểm thử (mặc định: data/processed/test.json).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "VietTravelVQA" / "VietTravelVQA" / "images"),
        help="Đường dẫn thư mục ảnh gốc (mặc định: data/VietTravelVQA/VietTravelVQA/images).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "baseline" / "predictions.json"),
        help="File lưu kết quả (mặc định: experiments/baseline/predictions.json).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=str(PROJECT_ROOT / "models" / "Vintern-1B-v3_5"),
        help="Thư mục trọng số mô hình (mặc định: models/Vintern-1B-v3_5).",
    )
    parser.add_argument(
        "--lora-weights",
        type=str,
        default=None,
        help="Đường dẫn thư mục LoRA adapter sau fine-tune (để trống nếu chạy baseline gốc).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Thiết bị: 'cuda', 'cpu', hoặc 'auto'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Số lượng mẫu chạy thử (ví dụ: --limit 10).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè lại từ đầu (mặc định: tự động chạy tiếp Resume).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Số lượng token tối đa sinh ra (mặc định: 512).",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, device_arg: str, lora_weights: Optional[str] = None):
    """Nạp mô hình Vintern-1B-v3.5 và Tokenizer với logging im lặng tuyệt đối."""
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    # Tắt thông báo nội bộ của transformers
    transformers.logging.set_verbosity_error()

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("Không tìm thấy GPU tương thích. Tự động chuyển sang CPU.")
        device = "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"⚡ Thiết bị: {gpu_name} ({vram_gb:.1f} GB VRAM) | Kiểu dữ liệu: bfloat16")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        logger.info("⚡ Thiết bị: CPU | Kiểu dữ liệu: float32")
        dtype = torch.float32

    # Nếu có chỉ định LoRA weights thì nạp qua PEFT adapter
    if lora_weights and Path(lora_weights).is_dir():
        from src.training.model_utils import load_lora_for_inference
        logger.info(f"⚡ Đang nạp mô hình kết hợp LoRA Adapter từ: {lora_weights}...")
        model, tokenizer = load_lora_for_inference(model_path, lora_weights, device=device)
        return model, tokenizer, device, dtype

    logger.info("⏳ Đang nạp Tokenizer & Trọng số mô hình Vintern-1B-v3.5...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.model_max_length = 32768

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    if device == "cuda":
        model = model.to("cuda")

    logger.info("✅ Nạp mô hình thành công. Sẵn sàng suy luận!")
    return model, tokenizer, device, dtype


def load_existing_predictions(output_path: Path) -> List[Dict[str, Any]]:
    """Đọc các kết quả đã sinh trước đó để hỗ trợ tính năng Resume."""
    if output_path.is_file():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                logger.info(f"🔄 Tìm thấy {len(data):,} câu đã hoàn thành từ trước. Kích hoạt Resume!")
                return data
        except Exception:
            pass
    return []


def save_predictions(output_path: Path, predictions: List[Dict[str, Any]]) -> None:
    """Lưu danh sách kết quả dự đoán vào file JSON một cách an toàn."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    temp_path.replace(output_path)


def main():
    args = parse_arguments()

    input_path = Path(args.input)
    images_dir = Path(args.images_dir)
    output_path = Path(args.output)
    model_path = Path(args.model_id)

    if not input_path.is_file():
        logger.error(f"❌ Không tìm thấy file: {input_path}")
        sys.exit(1)
    if not images_dir.is_dir():
        logger.error(f"❌ Không tìm thấy thư mục ảnh: {images_dir}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if args.limit and args.limit > 0:
        test_data = test_data[: args.limit]

    # Kiểm tra các câu đã hoàn thành để Resume
    existing_preds = [] if args.overwrite else load_existing_predictions(output_path)
    completed_qids = {x["qid"] for x in existing_preds if "prediction" in x and x["prediction"]}
    
    pending_items = [x for x in test_data if x["qid"] not in completed_qids]
    
    logger.info(f"📊 Tổng số: {len(test_data):,} câu | Đã xong: {len(completed_qids):,} | Còn lại: {len(pending_items):,} câu")

    if not pending_items:
        logger.info("🎉 Toàn bộ tập dữ liệu đã hoàn thành! Không cần chạy thêm.")
        return

    # Gom nhóm theo ảnh (1 ảnh xử lý 1 lần cho cả 5 câu hỏi)
    items_by_image = defaultdict(list)
    for item in pending_items:
        items_by_image[item["image_id"]].append(item)

    # Nạp Model
    model, tokenizer, device, dtype = load_model_and_tokenizer(str(model_path), args.device, args.lora_weights)

    from src.inference.image_processor import prepare_image_tensor
    import torch

    # Cấu hình sinh từ (tắt toàn bộ warning về pad_token / temperature)
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
    }

    system_prompt = (
        "Bạn là một trợ lý trí tuệ nhân tạo am hiểu và trân trọng di sản văn hóa, lịch sử Việt Nam. "
        "Nhiệm vụ của bạn là quan sát kỹ bức ảnh được cung cấp và trả lời câu hỏi của người dùng "
        "bằng tiếng Việt một cách chính xác, ngắn gọn, tôn trọng lịch sử và văn hóa."
    )

    all_predictions = list(existing_preds)
    total_to_process = len(pending_items)
    start_time = time.time()

    print("\n" + "="*70)
    print("🚀 BẮT ĐẦU CHẠY SUY LUẬN (Nhấn Ctrl + C bất cứ lúc nào để dừng và lưu an toàn)")
    print("="*70)

    # Khởi tạo thanh tiến trình tqdm nếu có
    pbar = tqdm(total=total_to_process, desc="Tiến độ", unit="câu") if tqdm else None
    processed_count = 0

    try:
        for img_idx, (image_id, q_list) in enumerate(items_by_image.items(), start=1):
            img_path = images_dir / image_id
            if not img_path.is_file():
                if pbar:
                    pbar.update(len(q_list))
                continue

            # Tiền xử lý ảnh 1 lần duy nhất
            try:
                with Image.open(img_path).convert("RGB") as pil_img:
                    pixel_values = prepare_image_tensor(
                        image=pil_img,
                        input_size=448,
                        max_num_patches=4,
                        device=device,
                        dtype=dtype,
                    )
            except Exception as img_err:
                if pbar:
                    pbar.update(len(q_list))
                continue

            # Sinh câu trả lời cho từng câu hỏi
            for item in q_list:
                qid = item["qid"]
                diff = item.get("difficulty", "?")
                question = item["question"]
                ground_truth = item.get("answer", "")
                prompt = f"{system_prompt}\n\n<image>\n{question}"

                t0 = time.time()
                try:
                    with torch.inference_mode():
                        pred_response = model.chat(
                            tokenizer,
                            pixel_values,
                            prompt,
                            gen_config,
                        )
                    latency = round(time.time() - t0, 2)
                    pred_clean = pred_response.strip()
                except Exception as infer_err:
                    pred_clean = f"[ERROR: {infer_err}]"
                    latency = 0.0

                record = {
                    "qid": qid,
                    "image_id": image_id,
                    "difficulty": str(diff),
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": pred_clean,
                    "latency_s": latency,
                }
                all_predictions.append(record)
                processed_count += 1

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({"Level": diff, "Tốc độ": f"{latency}s"})
                else:
                    if processed_count % 10 == 0 or processed_count == total_to_process:
                        logger.info(f"[{processed_count}/{total_to_process}] QID: {qid} (L{diff}) - {latency}s")

            # Lưu file predictions.json sau mỗi ảnh hoàn thành
            save_predictions(output_path, all_predictions)

    except KeyboardInterrupt:
        if pbar:
            pbar.close()
        print("\n")
        logger.warning("⚠️ Đã nhấn Ctrl + C! Tiến hành lưu ngay dữ liệu đã xong...")
        save_predictions(output_path, all_predictions)
        logger.info(f"✅ Đã lưu an toàn {len(all_predictions)} câu vào: {output_path}")
        sys.exit(0)

    if pbar:
        pbar.close()

    total_time = time.time() - start_time
    avg_speed = total_time / max(processed_count, 1)

    print("\n" + "="*70)
    logger.info("🎉 HOÀN THÀNH TOÀN BỘ SUY LUẬN!")
    logger.info(f"Tổng số câu đã dự đoán    : {len(all_predictions):,} câu")
    logger.info(f"Thời gian chạy             : {total_time/60:.1f} phút")
    logger.info(f"Tốc độ trung bình          : {avg_speed:.2f} giây / câu")
    logger.info(f"File kết quả lưu tại       : {output_path.resolve()}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
