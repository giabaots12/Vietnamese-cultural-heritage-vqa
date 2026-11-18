# -*- coding: utf-8 -*-
"""
================================================================================
KỊCH BẢN HUẤN LUYỆN VINTERN-1B VỚI QLORA (scripts/train_lora.py)
================================================================================
Dự án: Vietnamese Cultural Heritage VQA (Hỏi - Đáp Thị Giác Di Sản Văn Hóa Việt Nam)
Mô hình: Vintern-1B-v3.5 (InternVL-Chat / Qwen2-0.5B + InternViT-300M)
Phương pháp: QLoRA 4-bit (NF4) + Freeze ViT + Train Projector & LLM LoRA
Tối ưu hóa: 6GB VRAM (NVIDIA GeForce RTX 4050 Laptop GPU)

Kịch bản này cung cấp giao diện CLI hoàn chỉnh để:
1. Nạp và kiểm định tập dữ liệu train (4,920 câu) và val (1,050 câu).
2. Thiết lập cấu hình QLoRA 4-bit (BitsAndBytes) và LoRA PEFT.
3. Huấn luyện bằng HuggingFace Trainer với paged_adamw_8bit & Cosine Scheduler.
4. Đánh giá loss định kỳ trên tập Validation và lưu Best Checkpoint.
5. Xuất LoRA Adapter hoàn chỉnh sẵn sàng cho pha Đánh giá kiểm chứng (Benchmark).
================================================================================
"""

import os
import sys
import gc
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import transformers
from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed


class MemoryCleanupCallback(TrainerCallback):
    """Giải phóng bộ nhớ đệm GPU định kỳ và trước/sau mỗi lượt eval để chống tràn VRAM (OOM)."""

    def on_evaluate(self, args, state, control, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class ProjectorSaveCallback(TrainerCallback):
    """Callback tự động lưu trọng số Projector (mlp1) vào mỗi thư mục checkpoint."""

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.is_dir():
            model = kwargs.get("model")
            if model is not None:
                raw_model = model.base_model.model if hasattr(model, "base_model") else model
                if hasattr(raw_model, "mlp1"):
                    torch.save(raw_model.mlp1.state_dict(), checkpoint_dir / "projector_weights.pt")

# Bổ sung thư mục gốc vào PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.dataset import (
    CulturalHeritageVQADataset,
    VinternDataCollator,
    build_vqa_dataset_and_collator,
)
from src.training.model_utils import (
    setup_vintern_qlora,
    save_trained_lora_checkpoint,
    print_trainable_parameters,
)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainLoRA")


def parse_arguments() -> argparse.Namespace:
    """Định nghĩa và phân tích các tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình Vintern-1B-v3.5 bằng QLoRA trên tập dữ liệu Di sản Văn hóa Việt Nam."
    )

    # 1. Đường dẫn Dữ liệu & Mô hình
    parser.add_argument(
        "--train-data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "train.json"),
        help="Đường dẫn file JSON tập huấn luyện (mặc định: data/processed/train.json).",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "val.json"),
        help="Đường dẫn file JSON tập kiểm định (mặc định: data/processed/val.json).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "VietTravelVQA" / "VietTravelVQA" / "images"),
        help="Thư mục chứa ảnh gốc (mặc định: data/VietTravelVQA/VietTravelVQA/images).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=str(PROJECT_ROOT / "models" / "Vintern-1B-v3_5"),
        help="Thư mục mô hình nền Vintern-1B (mặc định: models/Vintern-1B-v3_5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "lora_finetuned"),
        help="Thư mục lưu trữ checkpoint và adapter (mặc định: experiments/lora_finetuned).",
    )

    # 2. Siêu tham số Huấn luyện
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Số epoch huấn luyện (mặc định: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size trên mỗi GPU (mặc định: 1 - an toàn tối đa cho 6GB VRAM).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="Số bước tích lũy gradient (mặc định: 16 -> Effective Batch Size = 16).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Tốc độ học Learning Rate cho LoRA (mặc định: 2e-4).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Tỷ lệ số bước khởi động Warmup (mặc định: 0.05).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Hệ số Weight Decay (mặc định: 0.01).",
    )

    # 3. Cấu hình LoRA & Mô hình
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="Hạng ma trận LoRA rank (mặc định: 16).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="Hệ số tỷ lệ LoRA alpha (mặc định: 32).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (mặc định: 0.05).",
    )
    parser.add_argument(
        "--train-projector",
        action="store_true",
        default=True,
        help="Mở khóa và huấn luyện tầng Projector mlp1 (mặc định: True).",
    )
    parser.add_argument(
        "--no-train-projector",
        dest="train_projector",
        action="store_false",
        help="Đóng băng tầng Projector mlp1.",
    )
    parser.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        default=True,
        help="Không dùng lượng tử hóa 4-bit, huấn luyện ở dạng native BFloat16 LoRA (không cần bitsandbytes).",
    )

    # 4. Cấu hình Kích thước Token & Ảnh
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Độ dài chuỗi token tối đa sau khi đóng gói (mặc định: 2048).",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=4,
        help="Số lượng patch ảnh tối đa cho dynamic tiling (mặc định: 4).",
    )

    # 5. Lưu Checkpoint & Logging
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Khoảng cách số bước để đánh giá trên tập Val (mặc định: 100).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Khoảng cách số bước để lưu checkpoint (mặc định: 100).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Khoảng cách số bước ghi log loss (mặc định: 10).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Giới hạn số mẫu huấn luyện (dùng để chạy thử nghiệm nhanh Sanity Check).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed đảm bảo tính lặp lại (mặc định: 42).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Đường dẫn checkpoint để tiếp tục huấn luyện (nếu có).",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    set_seed(args.seed)

    # Kiểm tra thiết bị phần cứng
    if not torch.cuda.is_available():
        logger.error("❌ Không tìm thấy GPU CUDA! QLoRA yêu cầu GPU NVIDIA có CUDA.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(f"⚡ Thiết bị: {gpu_name} ({vram_gb:.2f} GB VRAM) | Seed: {args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # 1. NẠP MÔ HÌNH NỀN VÀ THIẾT LẬP QLORA 4-BIT
    # ==========================================================================
    logger.info("\n" + "=" * 75)
    logger.info("🛠️  GIAI ĐOẠN 1: KHỞI TẠO MÔ HÌNH VINTERN-1B VỚI QLORA")
    logger.info("=" * 75)

    model, tokenizer = setup_vintern_qlora(
        model_path=args.model_id,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_projector=args.train_projector,
        use_gradient_checkpointing=True,
        load_in_4bit=args.load_in_4bit,
    )

    # ==========================================================================
    # 2. XÂY DỰNG DATASET & DATA COLLATOR
    # ==========================================================================
    logger.info("\n" + "=" * 75)
    logger.info("📦  GIAI ĐOẠN 2: XÂY DỰNG DATA PIPELINE & COLLATOR")
    logger.info("=" * 75)

    train_dataset, val_dataset, data_collator = build_vqa_dataset_and_collator(
        train_path=args.train_data,
        val_path=args.val_data,
        images_dir=args.images_dir,
        tokenizer=tokenizer,
        max_num_patches=args.max_patches,
        input_size=448,
        max_length=args.max_length,
    )

    # Rút gọn số mẫu nếu chạy Sanity Test
    if args.max_train_samples and args.max_train_samples > 0:
        logger.info(f"🧪 Chế độ thử nghiệm nhanh: Giới hạn tập Train còn {args.max_train_samples} mẫu.")
        train_dataset.items = train_dataset.items[: args.max_train_samples]
        if val_dataset is not None:
            val_sample_count = min(len(val_dataset.items), args.max_train_samples)
            logger.info(f"🧪 Chế độ thử nghiệm nhanh: Giới hạn tập Val tương ứng còn {val_sample_count} mẫu.")
            val_dataset.items = val_dataset.items[: val_sample_count]

    logger.info(f"📊 Tập Huấn luyện (Train): {len(train_dataset):,} mẫu.")
    if val_dataset:
        logger.info(f"📊 Tập Kiểm định (Val):     {len(val_dataset):,} mẫu.")

    effective_batch_size = args.batch_size * args.grad_accum
    steps_per_epoch = max(len(train_dataset) // effective_batch_size, 1)
    total_steps = steps_per_epoch * args.epochs
    logger.info(
        f"⚙️  Batch Size: {args.batch_size} x {args.grad_accum} (Accum) = {effective_batch_size} | "
        f"Số steps/epoch: ~{steps_per_epoch} | Tổng số steps: ~{total_steps}"
    )

    # Tự động điều chỉnh save_steps và eval_steps cho các bài chạy nhỏ / Sanity check
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    if total_steps <= save_steps:
        save_steps = max(total_steps // 2, 1)
        eval_steps = save_steps
        logger.info(f"🔄 Điều chỉnh tự động: save_steps={save_steps}, eval_steps={eval_steps} cho bài chạy {total_steps} bước.")

    # ==========================================================================
    # 3. THIẾT LẬP HUGGINGFACE TRAINING ARGUMENTS
    # ==========================================================================
    logger.info("\n" + "=" * 75)
    logger.info("🚀  GIAI ĐOẠN 3: CẤU HÌNH ENGINE HUẤN LUYỆN (HUGGINGFACE TRAINER)")
    logger.info("=" * 75)

    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    # Tương thích giữa các phiên bản transformers mới và cũ
    eval_arg_name = "eval_strategy" if hasattr(TrainingArguments, "eval_strategy") else "evaluation_strategy"
    eval_kwargs = {
        eval_arg_name: "steps" if val_dataset else "no",
    }

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,     # Đánh giá với batch_size=1 để triệt tiêu memory peak
        gradient_accumulation_steps=args.grad_accum,
        eval_accumulation_steps=4,        # Chuyển dần tensor dự đoán về CPU để tránh đọng trong VRAM
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_dir=str(output_dir / "logs"),
        logging_steps=max(min(args.logging_steps, total_steps // 2), 1),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        eval_steps=eval_steps if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        remove_unused_columns=False,      # BẮT BUỘC: Giữ lại pixel_values và image_flags
        dataloader_pin_memory=False,      # Tránh lỗi bộ nhớ trên Windows
        report_to=["tensorboard"] if "tensorboard" in sys.modules else "none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **eval_kwargs,
    )

    trainer_callbacks = [MemoryCleanupCallback()]
    if args.train_projector:
        trainer_callbacks.append(ProjectorSaveCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=trainer_callbacks,
    )

    # Lưu lại toàn bộ tham số cấu hình
    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # ==========================================================================
    # 4. KHỞI CHẠY HUẤN LUYỆN
    # ==========================================================================
    logger.info("🔥 Bắt đầu quá trình huấn luyện QLoRA...")
    start_time = time.time()

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    total_duration = time.time() - start_time
    hours, rem = divmod(total_duration, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info(f"🎉 Huấn luyện hoàn thành sau: {int(hours)}h {int(minutes)}m {int(seconds)}s!")
    logger.info(f"📉 Train Loss cuối cùng: {train_result.training_loss:.4f}")

    # ==========================================================================
    # 5. LƯU BEST CHECKPOINT & ADAPTER HOÀN CHỈNH
    # ==========================================================================
    final_output_dir = output_dir / "final"
    final_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"💾 Đang lưu mô hình tốt nhất (Best Checkpoint) vào: {final_output_dir}...")
    save_trained_lora_checkpoint(
        model=trainer.model,
        tokenizer=tokenizer,
        output_dir=final_output_dir,
        save_projector=args.train_projector,
    )

    # Ghi nhận kết quả tóm tắt
    summary = {
        "model_id": args.model_id,
        "epochs": args.epochs,
        "total_steps": train_result.global_step,
        "train_loss": round(train_result.training_loss, 4),
        "duration_seconds": round(total_duration, 2),
        "duration_formatted": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_projector": args.train_projector,
        "final_adapter_dir": str(final_output_dir),
    }

    if val_dataset:
        logger.info("🧪 Đang đánh giá lần cuối trên toàn bộ tập Validation...")
        eval_metrics = trainer.evaluate()
        logger.info(f"📊 Validation Loss: {eval_metrics.get('eval_loss', 'N/A')}")
        summary["final_eval_loss"] = round(eval_metrics.get("eval_loss", 0.0), 4)

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print("🏆 HUẤN LUYỆN THÀNH CÔNG VÀ ĐÃ XUẤT ADAPTER HOÀN CHỈNH!")
    print(f"📁 Thư mục Adapter LoRA: {final_output_dir}")
    print(f"📉 Train Loss: {summary['train_loss']} | Thời gian: {summary['duration_formatted']}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
