import logging
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

from src.core.config import ProjectConfig, load_config
from src.inference.image_processor import prepare_image_tensor

logger = logging.getLogger(__name__)


class VLMEngine:
    """Manages the lifecycle and inference pipeline for Vintern-1B-v3.5."""

    def __init__(self, config: Optional[ProjectConfig] = None) -> None:
        """Initialize the VLM engine with configuration.

        Args:
            config: Optional ProjectConfig instance. If omitted, loads default_config.yaml.
        """
        self.config = config or load_config()
        self.model = None
        self.tokenizer = None
        self.device = self._resolve_device(self.config.model.device)
        self.dtype = self._resolve_dtype(self.config.model.torch_dtype, self.device)

    def _resolve_device(self, requested_device: str) -> str:
        """Verify device availability with safe fallback to CPU."""
        if requested_device in ("cuda", "auto"):
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return "cpu"

    def _resolve_dtype(self, requested_dtype: str, device: str) -> torch.dtype:
        """Resolve torch dtype with device compatibility guards."""
        if device == "cpu":
            # BFloat16/Float16 can be unstable or unsupported on standard CPU backends
            return torch.float32

        dtype_map = {
            "bfloat16": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(requested_dtype.lower(), torch.float16)

    def load_model(self) -> None:
        """Load tokenizer and model weights into memory if not already initialized."""
        if self.model is not None and self.tokenizer is not None:
            return

        model_id = self.config.model.model_id
        # If model_id points to a local directory relative to project root, resolve absolute path
        project_root = Path(__file__).resolve().parent.parent.parent
        local_candidate = project_root / model_id
        if local_candidate.is_dir():
            model_id = str(local_candidate)

        logger.info(f"Loading tokenizer from {model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=self.config.model.trust_remote_code,
            use_fast=False,
        )

        logger.info(f"Loading model {model_id} on device '{self.device}' with dtype '{self.dtype}'...")

        model_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.config.model.trust_remote_code,
        }

        # Optional Flash Attention configuration
        if self.config.model.use_flash_attn and self.device == "cuda":
            model_kwargs["use_flash_attn"] = True

        # Optional low-VRAM quantization
        if self.config.model.load_in_4bit and self.device == "cuda":
            model_kwargs["load_in_4bit"] = True
        elif self.config.model.load_in_8bit and self.device == "cuda":
            model_kwargs["load_in_8bit"] = True

        self.model = AutoModel.from_pretrained(model_id, **model_kwargs).eval()

        # Place on CUDA if not using bitsandbytes device auto-mapping
        if not (self.config.model.load_in_4bit or self.config.model.load_in_8bit):
            self.model = self.model.to(self.device)

        logger.info("Vintern-1B-v3.5 loaded successfully.")

    def generate_response(
        self,
        image: Image.Image,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Generate a contextual Vietnamese response given an image and dialogue history.

        Args:
            image: PIL Image depicting the cultural heritage item.
            question: User's natural language question in Vietnamese.
            history: List of previous (user_query, assistant_response) tuples.

        Returns:
            Tuple[str, List[Tuple[str, str]]]: Generated response string and updated history tuples.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # Prepare normalized pixel values tensor
        pixel_values = prepare_image_tensor(
            image=image,
            input_size=448,
            max_num_patches=6,  # 6 patches balance detail and VRAM
            device=self.device,
            dtype=self.dtype,
        )

        # Assemble prompt and multi-turn history:
        # InternVL requires that the very first turn in history contains the '<image>' token.
        # If history exists, format turn 0 with '<image>' and ensure subsequent turns do not repeat it.
        is_first_turn = history is None or len(history) == 0
        sys_prompt = self.config.context.system_prompt.strip()

        if is_first_turn:
            prompt = f"{sys_prompt}\n\n<image>\n{question}" if sys_prompt else f"<image>\n{question}"
            model_history = None
        else:
            formatted_history = []
            for idx, (old_q, old_a) in enumerate(history):
                if idx == 0:
                    # Turn 0 must contain <image> for InternVL vision-context alignment
                    if "<image>" not in old_q:
                        first_q = f"{sys_prompt}\n\n<image>\n{old_q}" if sys_prompt else f"<image>\n{old_q}"
                    else:
                        first_q = old_q
                    formatted_history.append((first_q, old_a))
                else:
                    clean_q = old_q.replace("<image>\n", "").replace("<image>", "").strip()
                    formatted_history.append((clean_q, old_a))

            # Current turn in a multi-turn conversation does not repeat <image>
            prompt = question.replace("<image>\n", "").replace("<image>", "").strip()
            model_history = formatted_history

        # Configure decoding hyperparameters
        generation_config = {
            "max_new_tokens": self.config.generation.max_new_tokens,
            "do_sample": self.config.generation.do_sample,
            "temperature": self.config.generation.temperature,
            "top_p": self.config.generation.top_p,
            "repetition_penalty": self.config.generation.repetition_penalty,
        }

        try:
            with torch.inference_mode():
                response, updated_history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    prompt,
                    generation_config,
                    history=model_history,
                    return_history=True,
                )
            return response.strip(), updated_history
        except torch.cuda.OutOfMemoryError as e:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.error("CUDA Out-of-Memory during VLM generation.", exc_info=True)
            raise RuntimeError(
                "Mô hình bị quá tải bộ nhớ VRAM của GPU. Hãy thử hỏi lại hoặc bấm 'Làm mới phiên' để giải phóng bộ nhớ."
            ) from e
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            raise
