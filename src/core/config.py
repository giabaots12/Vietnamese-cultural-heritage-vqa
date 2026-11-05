from pathlib import Path
from typing import List, Optional, Union
import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Application-level meta configuration."""
    project_name: str = Field(default="Vietnamese Cultural Heritage VQA")
    version: str = Field(default="0.1.0-mvp")
    log_level: str = Field(default="INFO")
    sample_dir: str = Field(default="data/samples")


class ModelConfig(BaseModel):
    """Vision-Language Model and runtime compute settings."""
    model_id: str = Field(default="models/Vintern-1B-v3_5")
    device: str = Field(default="cuda")
    torch_dtype: str = Field(default="bfloat16")
    trust_remote_code: bool = Field(default=True)
    load_in_4bit: bool = Field(default=False)
    load_in_8bit: bool = Field(default=False)
    use_flash_attn: bool = Field(default=False)


class GenerationConfig(BaseModel):
    """Text generation and sampling hyperparameters."""
    max_new_tokens: int = Field(default=512, ge=64, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)
    do_sample: bool = Field(default=True)


class ContextConfig(BaseModel):
    """Dialogue context and multimodal input constraints."""
    max_history_turns: int = Field(default=4, ge=1, le=20)
    image_token: str = Field(default="<image>")
    max_image_dim: int = Field(default=1024, ge=256, le=4096)
    system_prompt: str = Field(
        default=(
            "Bạn là một trợ lý trí tuệ nhân tạo am hiểu và trân trọng di sản văn hóa, lịch sử Việt Nam. "
            "Nhiệm vụ của bạn là quan sát kỹ bức ảnh được cung cấp và trả lời câu hỏi của người dùng "
            "bằng tiếng Việt một cách khách quan, mạch lạc, tôn trọng lịch sử và văn hóa."
        )
    )


class UIConfig(BaseModel):
    """User interface display and server parameters."""
    title: str = Field(default="Vietnamese Cultural Heritage VQA")
    description: str = Field(
        default="Khám phá và tìm hiểu di sản văn hóa, di tích lịch sử và kiến trúc truyền thống Việt Nam qua hình ảnh."
    )
    disclaimer: str = Field(
        default="Lưu ý: Câu trả lời được sinh tự động bởi mô hình VLM baseline phục vụ mục đích học tập và thử nghiệm. Vui lòng kiểm chứng lại các thông tin lịch sử quan trọng qua tài liệu chính thống."
    )
    server_name: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8501)
    share: bool = Field(default=False)
    max_upload_size_mb: int = Field(default=10, ge=1, le=100)
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"]
    )


class ProjectConfig(BaseModel):
    """Master configuration schema combining all subsystems."""
    app: AppConfig = Field(default_factory=AppConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    ui: UIConfig = Field(default_factory=UIConfig)


def get_default_config_path() -> Path:
    """Resolve the default configuration file location relative to project root."""
    # Assuming this file is located at <project_root>/src/core/config.py
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "configs" / "default_config.yaml"


def load_config(config_path: Optional[Union[str, Path]] = None) -> ProjectConfig:
    """Load and validate project configuration from a YAML file.
    
    Args:
        config_path: Optional custom path to YAML config. If omitted,
                     falls back to configs/default_config.yaml.

    Returns:
        ProjectConfig: Validated strongly-typed configuration instance.
    """
    path = Path(config_path) if config_path else get_default_config_path()
    
    if not path.is_file():
        # Fall back to default schema values if file does not exist
        return ProjectConfig()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return ProjectConfig(**data)

