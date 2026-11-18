# -*- coding: utf-8 -*-
"""
================================================================================
MODULE: DATASET & DATA COLLATOR CHO HUẤN LUYỆN VINTERN-1B (QLoRA SFT)
================================================================================
Dự án: Vietnamese Cultural Heritage VQA (Hỏi - Đáp Thị Giác Di Sản Văn Hóa Việt Nam)
Kiến trúc: Vintern-1B-v3.5 (InternVL-Chat / Qwen2 + InternViT-300M)

Module này chịu trách nhiệm:
1. Nạp tập dữ liệu train/val dạng JSON và ảnh từ thư mục di sản.
2. Tiền xử lý ảnh thích ứng động (Dynamic Patch Tiling: 1-4 patches, 448x448).
3. Định dạng hội thoại chuẩn ChatML (Hermes-2 template của Vintern).
4. Áp dụng kỹ thuật Label Masking (Completion-only Loss):
   - Mask out toàn bộ prompt, token ảnh và câu hỏi với nhãn -100.
   - Chỉ tính gradient loss trên câu trả lời Ground Truth để triệt tiêu nói dài dòng.
5. VinternDataCollator ghép batch an toàn cho huấn luyện phân tán / GPU cá nhân.
================================================================================
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image

# Tái sử dụng bộ tiền xử lý ảnh động tối ưu từ src/inference
from src.inference.image_processor import (
    build_transform,
    dynamic_preprocess,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

logger = logging.getLogger("TrainingDataset")

# Các hằng số Token đặc biệt của Vintern-1B / InternVL
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
NUM_IMAGE_TOKENS_PER_PATCH = 256  # (448/14)^2 * (0.5^2) = 32^2 * 0.25 = 256

DEFAULT_SYSTEM_PROMPT = (
    "Bạn là một chuyên gia về di sản văn hóa, lịch sử và kiến trúc Việt Nam. "
    "Dựa vào hình ảnh được cung cấp, hãy quan sát kỹ và trả lời câu hỏi của người dùng "
    "bằng tiếng Việt một cách chính xác, ngắn gọn, đi thẳng vào trọng tâm câu hỏi."
)


class CulturalHeritageVQADataset(Dataset):
    """
    Dataset nạp các cặp (Ảnh, Câu hỏi, Câu trả lời) phục vụ Supervised Fine-Tuning (SFT)
    cho mô hình đa phương thức Vintern-1B-v3.5.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        images_dir: Union[str, Path],
        tokenizer: Any,
        max_num_patches: int = 4,
        input_size: int = 448,
        max_length: int = 2048,
        system_prompt: Optional[str] = None,
        is_train: bool = True,
    ):
        """
        Khởi tạo Dataset.

        Args:
            data_path: Đường dẫn file JSON chứa danh sách câu hỏi (train.json hoặc val.json).
            images_dir: Thư mục chứa các file ảnh gốc.
            tokenizer: Tokenizer của mô hình Vintern-1B.
            max_num_patches: Số lượng patch tối đa cắt từ ảnh động (mặc định: 4 để đồng nhất với baseline).
            input_size: Kích thước pixel mỗi patch (mặc định: 448).
            max_length: Độ dài chuỗi token tối đa sau khi đóng gói.
            system_prompt: Thông điệp hệ thống định hướng phong cách trả lời.
            is_train: Cờ xác định tập huấn luyện hay tập kiểm định.
        """
        self.data_path = Path(data_path)
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.max_num_patches = max_num_patches
        self.input_size = input_size
        self.max_length = max_length
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.is_train = is_train

        # Kiểm tra sự tồn tại của file dữ liệu và thư mục ảnh
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {self.data_path}")
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {self.images_dir}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.items: List[Dict[str, Any]] = json.load(f)

        logger.info(
            f"📦 Đã nạp {len(self.items):,} mẫu từ '{self.data_path.name}' "
            f"({'Train' if is_train else 'Validation'} Mode)."
        )

        # Xây dựng pipeline biến đổi tensor ảnh chuẩn
        self.image_transform = build_transform(input_size=self.input_size)

        # Token ID đặc biệt
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.im_start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, image_id: str) -> Image.Image:
        """Đọc và chuyển đổi ảnh RGB an toàn, hỗ trợ fallback khi ảnh lỗi."""
        img_path = self.images_dir / image_id
        if not img_path.is_file():
            # Tạo ảnh trắng dự phòng nếu file ảnh thất lạc
            logger.warning(f"Không tìm thấy ảnh: {img_path}, sử dụng ảnh trắng thay thế.")
            return Image.new("RGB", (self.input_size, self.input_size), color=(255, 255, 255))

        try:
            with Image.open(img_path) as img:
                return img.convert("RGB")
        except Exception as e:
            logger.warning(f"Lỗi đọc ảnh {img_path}: {e}, sử dụng ảnh trắng thay thế.")
            return Image.new("RGB", (self.input_size, self.input_size), color=(255, 255, 255))

    def _process_image_patches(self, pil_img: Image.Image) -> Tuple[torch.Tensor, int]:
        """
        Cắt ảnh theo tỷ lệ động (Dynamic Preprocessing) và chuẩn hóa thành Tensor.

        Returns:
            Tuple[torch.Tensor, int]: Tensor kích thước (num_patches, 3, 448, 448) và số lượng patch.
        """
        patches = dynamic_preprocess(
            pil_img,
            min_num=1,
            max_num=self.max_num_patches,
            image_size=self.input_size,
            use_thumbnail=True,
        )
        tensors = [self.image_transform(patch) for patch in patches]
        pixel_values = torch.stack(tensors, dim=0)  # Shape: (num_patches, 3, 448, 448)
        return pixel_values, len(patches)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Xử lý 1 mẫu huấn luyện hoàn chỉnh gồm:
        1. Ảnh -> Dynamic Patches Tensor.
        2. Prompt + Ground Truth -> ChatML Tokens.
        3. Labels có mặt nạ -100 trên Prompt (Completion-only Loss).
        """
        item = self.items[idx]
        image_id = item["image_id"]
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()

        # 1. Đọc và xử lý các patch ảnh
        pil_img = self._load_image(image_id)
        pixel_values, num_patches = self._process_image_patches(pil_img)

        # 2. Xây dựng chuỗi token đại diện cho ảnh
        # Mỗi patch tương ứng 256 token <IMG_CONTEXT> nằm giữa <img> và </img>
        total_image_tokens = num_patches * NUM_IMAGE_TOKENS_PER_PATCH
        image_tokens_str = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * total_image_tokens}{IMG_END_TOKEN}"

        # 3. Tạo cấu trúc hội thoại chuẩn ChatML (Hermes-2 style của Vintern)
        # Phần Prompt (Người dùng hỏi + Ảnh + System)
        prompt_text = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{image_tokens_str}\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Phần Phản hồi đầy đủ (Prompt + Đáp án + <|im_end|>)
        full_conversation = f"{prompt_text}{answer}<|im_end|>"

        # 4. Tokenize
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_conversation, add_special_tokens=False)

        # Cắt bớt nếu vượt quá max_length
        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]

        # 5. Tạo Labels với kỹ thuật Label Masking (Completion-only)
        # Toàn bộ vị trí trước câu trả lời (Prompt + Image tokens) gán nhãn -100
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        # Đảm bảo độ dài labels bằng full_ids
        if len(labels) < len(full_ids):
            labels.extend([-100] * (len(full_ids) - len(labels)))
        elif len(labels) > len(full_ids):
            labels = labels[: len(full_ids)]

        input_ids_tensor = torch.tensor(full_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Cờ đánh dấu các patch ảnh hợp lệ (toàn 1)
        image_flags = torch.ones(num_patches, 1, dtype=torch.long)

        return {
            "qid": item.get("qid", f"idx_{idx}"),
            "image_id": image_id,
            "pixel_values": pixel_values,      # (num_patches, 3, 448, 448)
            "image_flags": image_flags,        # (num_patches, 1)
            "input_ids": input_ids_tensor,     # (seq_len,)
            "labels": labels_tensor,           # (seq_len,)
            "num_patches": num_patches,
        }


class VinternDataCollator:
    """
    Data Collator ghép các mẫu đơn lẻ thành một Batch hợp lệ cho Vintern-1B:
    - Nối (concatenate) pixel_values của các ảnh dọc theo chiều patch.
    - Đệm (pad) input_ids với pad_token_id và labels với -100.
    - Sinh attention_mask chính xác.
    """

    def __init__(self, tokenizer: Any, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Tập hợp danh sách các sample thành 1 Batch dictionary nạp thẳng vào model.forward().
        """
        # 1. Ghép toàn bộ pixel_values và image_flags
        # InternVL xử lý ảnh batch bằng cách gom toàn bộ patch của các mẫu lại với nhau
        all_pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
        all_image_flags = torch.cat([item["image_flags"] for item in batch], dim=0)

        # 2. Tìm độ dài chuỗi lớn nhất trong batch này
        batch_input_ids = [item["input_ids"] for item in batch]
        batch_labels = [item["labels"] for item in batch]

        max_seq_len = max(len(seq) for seq in batch_input_ids)
        max_seq_len = min(max_seq_len, self.max_length)

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for input_ids, labels in zip(batch_input_ids, batch_labels):
            seq_len = len(input_ids)
            if seq_len > max_seq_len:
                # Cắt ngắn nếu vượt ngưỡng
                curr_ids = input_ids[:max_seq_len]
                curr_labels = labels[:max_seq_len]
                curr_mask = torch.ones(max_seq_len, dtype=torch.long)
            else:
                pad_len = max_seq_len - seq_len
                # Đệm phải (Right padding)
                curr_ids = torch.cat([
                    input_ids,
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
                ], dim=0)
                
                # Với labels: vị trí đệm PHẢI là -100
                curr_labels = torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=torch.long)
                ], dim=0)
                
                # Attention mask: 1 cho token thật, 0 cho token đệm
                curr_mask = torch.cat([
                    torch.ones(seq_len, dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long)
                ], dim=0)

            padded_input_ids.append(curr_ids)
            padded_labels.append(curr_labels)
            attention_masks.append(curr_mask)

        return {
            "pixel_values": all_pixel_values,                           # (total_patches_in_batch, 3, 448, 448)
            "image_flags": all_image_flags,                             # (total_patches_in_batch, 1)
            "input_ids": torch.stack(padded_input_ids, dim=0),          # (B, max_seq_len)
            "attention_mask": torch.stack(attention_masks, dim=0),      # (B, max_seq_len)
            "labels": torch.stack(padded_labels, dim=0),                # (B, max_seq_len)
        }


def build_vqa_dataset_and_collator(
    train_path: Union[str, Path],
    val_path: Optional[Union[str, Path]],
    images_dir: Union[str, Path],
    tokenizer: Any,
    max_num_patches: int = 4,
    input_size: int = 448,
    max_length: int = 2048,
    system_prompt: Optional[str] = None,
) -> Tuple[CulturalHeritageVQADataset, Optional[CulturalHeritageVQADataset], VinternDataCollator]:
    """
    Hàm tiện ích (Factory function) khởi tạo toàn bộ Datasets và Data Collator trong 1 lệnh duy nhất.
    """
    train_dataset = CulturalHeritageVQADataset(
        data_path=train_path,
        images_dir=images_dir,
        tokenizer=tokenizer,
        max_num_patches=max_num_patches,
        input_size=input_size,
        max_length=max_length,
        system_prompt=system_prompt,
        is_train=True,
    )

    val_dataset = None
    if val_path and Path(val_path).is_file():
        val_dataset = CulturalHeritageVQADataset(
            data_path=val_path,
            images_dir=images_dir,
            tokenizer=tokenizer,
            max_num_patches=max_num_patches,
            input_size=input_size,
            max_length=max_length,
            system_prompt=system_prompt,
            is_train=False,
        )

    collator = VinternDataCollator(tokenizer=tokenizer, max_length=max_length)
    return train_dataset, val_dataset, collator
