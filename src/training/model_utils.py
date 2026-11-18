# -*- coding: utf-8 -*-
"""
================================================================================
MODULE: CẤU HÌNH MÔ HÌNH VINTERN-1B VỚI QLORA 4-BIT (src/training/model_utils.py)
================================================================================
Dự án: Vietnamese Cultural Heritage VQA (Hỏi - Đáp Thị Giác Di Sản Văn Hóa Việt Nam)
Kiến trúc: Vintern-1B-v3.5 (InternVL-Chat / Qwen2-0.5B + InternViT-300M)

Module này chịu trách nhiệm:
1. Nạp mô hình Vintern-1B ở định dạng lượng tử hóa 4-bit (NF4 - QLoRA) để tiết kiệm VRAM.
2. Đóng băng 100% Vision Encoder (InternViT-300M) để bảo toàn năng lực thị giác và tránh tràn VRAM.
3. Cấu hình LoRA (PEFT) lên toàn bộ các tầng Attention & MLP của LLM Backbone (Qwen2).
4. Quản lý Projector (mlp1): Hỗ trợ mở khóa (unfreeze) để học ánh xạ văn hóa Việt.
5. Kích hoạt Gradient Checkpointing và tối ưu hóa bộ nhớ cho GPU 6GB VRAM (RTX 4050).
6. Thống kê và in bảng tham số huấn luyện (Trainable Parameters).
7. Hỗ trợ lưu và nạp adapter LoRA phục vụ suy luận / đánh giá kiểm chứng.
================================================================================
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("ModelUtils")


def get_bnb_config(compute_dtype: torch.dtype = torch.bfloat16) -> Any:
    """
    Tạo cấu hình lượng tử hóa BitsAndBytes 4-bit (NF4) tối ưu cho QLoRA.
    
    Args:
        compute_dtype: Kiểu dữ liệu tính toán (mặc định bfloat16 cho RTX 4050/3060).
        
    Returns:
        BitsAndBytesConfig
    """
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",               # Phân phối chuẩn NormalFloat4 tối ưu cho weights
        bnb_4bit_use_double_quant=True,          # Lượng tử hóa 2 lần (tiết kiệm thêm ~0.4 bit/param)
        bnb_4bit_compute_dtype=compute_dtype,    # Tính toán gradient ở bfloat16
    )
    return bnb_config


def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Đếm số lượng tham số huấn luyện được và tổng số tham số của mô hình.

    Returns:
        Tuple[trainable_params, all_params, percentage]
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    percentage = (100.0 * trainable_params / all_params) if all_params > 0 else 0.0
    return trainable_params, all_params, percentage


def print_trainable_parameters(model: nn.Module, model_label: str = "Vintern-1B QLoRA") -> None:
    """
    In bảng thống kê chi tiết các khối tham số được mở khóa / đóng băng.
    """
    trainable_params, all_params, percentage = count_parameters(model)
    
    # Ước tính bộ nhớ VRAM cho tham số (bfloat16 / 4-bit xấp xỉ)
    approx_mem_mb = (all_params * 0.5) / (1024 ** 2)  # 4-bit ~ 0.5 byte/param

    print("\n" + "=" * 75)
    print(f"📊 BẢNG TỔNG KẾT THAM SỐ MÔ HÌNH [{model_label.upper()}]")
    print("=" * 75)
    print(f"• Tổng số tham số (Total Parameters):        {all_params:,}")
    print(f"• Tham số huấn luyện (Trainable Parameters):  {trainable_params:,}")
    print(f"• Tỷ lệ cập nhật (Trainable Ratio):           {percentage:.3f}%")
    print(f"• Bộ nhớ mô hình nền ước tính (4-bit Base):    ~{approx_mem_mb:.1f} MB VRAM")
    print("=" * 75 + "\n")


def setup_vintern_qlora(
    model_path: Union[str, Path],
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    train_projector: bool = True,
    use_gradient_checkpointing: bool = True,
    device_map: str = "auto",
    load_in_4bit: bool = True,
) -> Tuple[Any, Any]:
    """
    Nạp mô hình Vintern-1B-v3.5, đóng băng Vision Encoder và gắn LoRA vào LLM Backbone.

    Args:
        model_path: Đường dẫn tới thư mục mô hình Vintern-1B (local hoặc HuggingFace ID).
        lora_r: Hạng của ma trận LoRA (mặc định 16).
        lora_alpha: Hệ số tỷ lệ LoRA scaling factor (mặc định 32).
        lora_dropout: Dropout cho các tầng LoRA (mặc định 0.05).
        train_projector: Mở khóa Projector (mlp1) để học ánh xạ văn hóa Việt.
        use_gradient_checkpointing: Bật gradient checkpointing để tiết kiệm VRAM.
        device_map: Chiến lược phân bổ thiết bị (mặc định 'auto').
        load_in_4bit: Nén 4-bit NF4 qua bitsandbytes (nếu False sẽ train dạng BFloat16 LoRA).

    Returns:
        Tuple[model, tokenizer]
    """
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    model_path = str(model_path)
    logger.info(f"⏳ Đang nạp Tokenizer từ: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.model_max_length = 2048
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|im_end|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # Xác định kiểu dữ liệu phần cứng hỗ trợ
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if load_in_4bit:
        bnb_config = get_bnb_config(compute_dtype=compute_dtype)
        logger.info(f"⏳ Đang nạp mô hình nền Vintern-1B ở chế độ 4-bit NF4 (compute_dtype={compute_dtype})...")
        base_model = AutoModel.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
    else:
        logger.info(f"⏳ Đang nạp mô hình nền Vintern-1B ở chế độ Native {compute_dtype} (BFloat16 LoRA)...")
        base_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )

    # 1. BƯỚC 1: ĐÓNG BĂNG 100% VISION ENCODER (InternViT-300M)
    if hasattr(base_model, "vision_model") and base_model.vision_model is not None:
        logger.info("🔒 Đóng băng toàn bộ trọng số của Vision Encoder (InternViT-300M)...")
        base_model.vision_model.eval()
        for param in base_model.vision_model.parameters():
            param.requires_grad = False

    # 2. BƯỚC 2: GIA CỐ INTERFACE CHO HUGGINGFACE / PEFT TRÊN INTERNVL
    # Ánh xạ get/set input embeddings từ language_model
    if hasattr(base_model, "language_model"):
        base_model.get_input_embeddings = lambda: base_model.language_model.get_input_embeddings()
        base_model.set_input_embeddings = lambda val: base_model.language_model.set_input_embeddings(val)

    # Cài đặt cờ & ủy quyền Gradient Checkpointing sang language_model
    type(base_model).supports_gradient_checkpointing = True
    base_model.supports_gradient_checkpointing = True

    def _enable_gc(gradient_checkpointing_kwargs=None):
        if hasattr(base_model, "language_model"):
            base_model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
    def _disable_gc():
        if hasattr(base_model, "language_model"):
            base_model.language_model.gradient_checkpointing_disable()

    base_model.gradient_checkpointing_enable = _enable_gc
    base_model.gradient_checkpointing_disable = _disable_gc

    def _enable_input_require_grads():
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        if hasattr(base_model, "language_model"):
            base_model.language_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    base_model.enable_input_require_grads = _enable_input_require_grads

    # Cài đặt token ID ngữ cảnh hình ảnh bắt buộc cho forward pass
    img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    base_model.img_context_token_id = img_context_token_id

    # Wrapper bảo vệ: tự động ép kiểu pixel_values & lọc bỏ tham số thừa từ PEFT (như inputs_embeds)
    import inspect
    orig_forward = base_model.forward
    valid_forward_params = set(inspect.signature(orig_forward).parameters.keys())

    def safe_forward(*args, **kwargs):
        if "pixel_values" in kwargs and kwargs["pixel_values"] is not None:
            target_dtype = getattr(base_model.vision_model, "dtype", compute_dtype)
            if kwargs["pixel_values"].dtype != target_dtype:
                kwargs["pixel_values"] = kwargs["pixel_values"].to(target_dtype)
        if "image_flags" in kwargs and kwargs["image_flags"] is not None:
            if kwargs["image_flags"].dtype != torch.long:
                kwargs["image_flags"] = kwargs["image_flags"].to(torch.long)
        # Chỉ giữ lại các tham số mà InternVLChatModel.forward thực sự hỗ trợ
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_forward_params}
        return orig_forward(*args, **filtered_kwargs)
    base_model.forward = safe_forward

    # 3. BƯỚC 3: CHUẨN BỊ MÔ HÌNH NỀN CHO K-BIT TRAINING
    if load_in_4bit:
        from peft import prepare_model_for_kbit_training
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=False,
        )

    # Vô hiệu hóa cache trong quá trình huấn luyện
    if hasattr(base_model, "config"):
        if hasattr(base_model.config, "llm_config"):
            base_model.config.llm_config.use_cache = False
        base_model.config.use_cache = False

    # 3. BƯỚC 3: CẤU HÌNH LORA CHO LLM BACKBONE (Qwen2)
    # Target modules nhắm vào toàn bộ attention projections và MLP gates
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    logger.info(f"⚡ Đang gắn LoRA adapter vào LLM Backbone (r={lora_r}, alpha={lora_alpha})...")
    model = get_peft_model(base_model, lora_config)

    # 4. BƯỚC 4: QUẢN LÝ PROJECTOR (mlp1)
    if train_projector and hasattr(model.base_model.model, "mlp1"):
        if not load_in_4bit:
            logger.info("🔓 Mở khóa Projector (mlp1) để tối ưu hóa bộ dịch thị giác -> văn hóa Việt...")
            projector = model.base_model.model.mlp1
            for param in projector.parameters():
                if param.is_floating_point():
                    param.requires_grad = True
                    param.data = param.data.to(compute_dtype)
        else:
            logger.info("🔒 Trong chế độ 4-bit QLoRA, Projector nén 4-bit cùng base model; LLM LoRA sẽ đảm nhận việc thích ứng ngữ nghĩa.")

    # 5. KÍCH HOẠT GRADIENT CHECKPOINTING
    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.base_model.model, "language_model"):
            model.base_model.model.language_model.gradient_checkpointing_enable()
        logger.info("🛡️ Đã kích hoạt Gradient Checkpointing (Giảm ~60% bộ nhớ activation).")

    # In thống kê
    print_trainable_parameters(model, model_label="Vintern-1B QLoRA (LLM + Projector)")

    return model, tokenizer


def save_trained_lora_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Union[str, Path],
    save_projector: bool = True,
) -> None:
    """
    Lưu checkpoint LoRA adapter và trọng số Projector một cách hoàn chỉnh.

    Args:
        model: Mô hình đã gắn LoRA sau khi huấn luyện.
        tokenizer: Tokenizer đi kèm.
        output_dir: Thư mục lưu checkpoint.
        save_projector: Có lưu thêm file trọng số projector riêng biệt hay không.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"💾 Đang lưu LoRA Adapter vào: {output_dir}...")
    # Lưu weights LoRA qua PEFT (adapter_model.safetensors, adapter_config.json)
    model.save_pretrained(str(output_dir))

    # Lưu Tokenizer
    tokenizer.save_pretrained(str(output_dir))

    # Lưu Projector nếu được unfreeze
    if save_projector:
        raw_model = model.base_model.model if hasattr(model, "base_model") else model
        if hasattr(raw_model, "mlp1"):
            projector_file = output_dir / "projector_weights.pt"
            logger.info(f"💾 Đang lưu trọng số Projector riêng biệt vào: {projector_file}...")
            torch.save(raw_model.mlp1.state_dict(), projector_file)

    logger.info("✅ Lưu checkpoint thành công!")


def load_lora_for_inference(
    base_model_path: Union[str, Path],
    lora_dir: Union[str, Path],
    device: str = "cuda",
) -> Tuple[Any, Any]:
    """
    Nạp mô hình gốc và ghép nối (merge / attach) LoRA adapter phục vụ suy luận / benchmark.

    Args:
        base_model_path: Đường dẫn mô hình Vintern-1B gốc.
        lora_dir: Thư mục chứa LoRA adapter đã huấn luyện.
        device: Thiết bị chạy ('cuda' hoặc 'cpu').

    Returns:
        Tuple[model, tokenizer]
    """
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel

    base_model_path = str(base_model_path)
    lora_dir = Path(lora_dir)

    logger.info(f"⏳ Nạp Tokenizer từ: {lora_dir} (fallback: {base_model_path})...")
    tokenizer_path = lora_dir if (lora_dir / "tokenizer_config.json").is_file() else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        use_fast=False,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    logger.info(f"⏳ Nạp Base Model từ: {base_model_path} ({dtype})...")
    base_model = AutoModel.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Nạp lại trọng số projector nếu có
    projector_file = lora_dir / "projector_weights.pt"
    if projector_file.is_file() and hasattr(base_model, "mlp1"):
        logger.info(f"🔄 Nạp lại trọng số Projector từ: {projector_file}...")
        projector_weights = torch.load(projector_file, map_location="cpu")
        base_model.mlp1.load_state_dict(projector_weights)

    logger.info(f"⚡ Gắn LoRA Adapter từ: {lora_dir}...")
    model = PeftModel.from_pretrained(
        base_model,
        str(lora_dir),
        torch_dtype=dtype,
    )

    model = model.to(device)
    model.eval()

    logger.info("✅ Nạp mô hình LoRA cho suy luận THÀNH CÔNG!")
    return model, tokenizer
