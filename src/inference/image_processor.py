from typing import List, Tuple
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448) -> T.Compose:
    """Build standard PyTorch image normalization transform pipeline."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    orig_width: int,
    orig_height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Find the grid ratio from candidates that best matches the image aspect ratio."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            current_diff = abs(orig_width * orig_height - image_size**2 * ratio[0] * ratio[1])
            best_diff = abs(orig_width * orig_height - image_size**2 * best_ratio[0] * best_ratio[1])
            if current_diff < best_diff:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """Slice high-resolution image into an optimal grid of fixed-size patches."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Candidate aspect ratios
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    sorted_target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, sorted_target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
    processed_images: List[Image.Image] = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        processed_images.append(thumbnail_img)

    return processed_images


def prepare_image_tensor(
    image: Image.Image,
    input_size: int = 448,
    max_num_patches: int = 6,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Preprocess a PIL image into normalized pixel values tensor.

    Args:
        image: Source PIL image.
        input_size: Dimension of each square patch (default: 448).
        max_num_patches: Maximum number of visual slices to construct (default: 6 for VRAM balance).
        device: Target torch device ('cuda' or 'cpu').
        dtype: Target torch data type (e.g., torch.bfloat16 or torch.float16).

    Returns:
        torch.Tensor: Tensor of shape (num_patches, 3, input_size, input_size) on target device.
    """
    image_rgb = image.convert("RGB")
    transform = build_transform(input_size=input_size)
    patches = dynamic_preprocess(
        image_rgb,
        min_num=1,
        max_num=max_num_patches,
        image_size=input_size,
        use_thumbnail=True,
    )
    tensors = [transform(p) for p in patches]
    pixel_values = torch.stack(tensors).to(dtype=dtype, device=device)
    return pixel_values

