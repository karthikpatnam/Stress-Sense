"""Image loading and validation utilities for Agriculture-Vision dataset.

Supports RGB (3-channel) and NIR (Near-Infrared, 1-channel) images,
as well as stacked 4-channel (RGB + NIR) arrays.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image


DEFAULT_EXPECTED_SHAPE: Tuple[int, int] = (512, 512)


def validate_image_dimensions(
    image: np.ndarray,
    expected_hw: Tuple[int, int] = DEFAULT_EXPECTED_SHAPE,
) -> bool:
    """Validate that the given image array matches expected height and width.

    Args:
        image: NumPy array representing an image (H, W) or (H, W, C).
        expected_hw: Tuple of (expected_height, expected_width). Default (512, 512).

    Returns:
        True if dimensions match expected_hw, False otherwise.
    """
    if not isinstance(image, np.ndarray):
        return False
    if image.ndim < 2:
        return False
    return (image.shape[0], image.shape[1]) == expected_hw


def load_rgb_image(
    image_path: Union[str, Path],
    expected_hw: Optional[Tuple[int, int]] = DEFAULT_EXPECTED_SHAPE,
    normalize: bool = False,
) -> np.ndarray:
    """Load an RGB image from disk and return a NumPy array of shape (H, W, 3).

    Uses OpenCV and converts from BGR to RGB color space.

    Args:
        image_path: Path to the image file (.jpg or .png).
        expected_hw: Expected (height, width). If provided and mismatch occurs,
            raises ValueError.
        normalize: If True, converts uint8 [0, 255] to float32 [0.0, 1.0].

    Returns:
        np.ndarray of shape (H, W, 3) in RGB order.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If file cannot be decoded or dimensions mismatch.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"RGB image file not found: {path}")

    # Read image using OpenCV
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        # Fallback to Pillow in case OpenCV codec issue occurs
        try:
            with Image.open(path) as pil_img:
                img_rgb = np.array(pil_img.convert("RGB"))
        except Exception as exc:
            raise ValueError(f"Corrupted or unreadable image file: {path} ({exc})") from exc
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if expected_hw is not None and not validate_image_dimensions(img_rgb, expected_hw):
        raise ValueError(
            f"Image dimensions {img_rgb.shape[:2]} do not match expected {expected_hw} in {path}"
        )

    if normalize:
        return (img_rgb.astype(np.float32) / 255.0)

    return img_rgb


def load_nir_image(
    image_path: Union[str, Path],
    expected_hw: Optional[Tuple[int, int]] = DEFAULT_EXPECTED_SHAPE,
    keep_dim: bool = False,
    normalize: bool = False,
) -> np.ndarray:
    """Load a Near-Infrared (NIR) single-channel image.

    Args:
        image_path: Path to the NIR image file (.jpg or .png).
        expected_hw: Expected (height, width).
        keep_dim: If True, returns shape (H, W, 1); otherwise (H, W).
        normalize: If True, converts uint8 [0, 255] to float32 [0.0, 1.0].

    Returns:
        np.ndarray of shape (H, W) or (H, W, 1).

    Raises:
        FileNotFoundError: If the NIR file does not exist.
        ValueError: If file cannot be decoded or dimensions mismatch.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"NIR image file not found: {path}")

    img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        try:
            with Image.open(path) as pil_img:
                img_gray = np.array(pil_img.convert("L"))
        except Exception as exc:
            raise ValueError(f"Corrupted or unreadable NIR image: {path} ({exc})") from exc

    if expected_hw is not None and not validate_image_dimensions(img_gray, expected_hw):
        raise ValueError(
            f"NIR dimensions {img_gray.shape} do not match expected {expected_hw} in {path}"
        )

    if normalize:
        img_out = img_gray.astype(np.float32) / 255.0
    else:
        img_out = img_gray

    if keep_dim:
        return np.expand_dims(img_out, axis=-1)
    return img_out


def load_stacked_rgb_nir(
    rgb_path: Union[str, Path],
    nir_path: Union[str, Path],
    expected_hw: Optional[Tuple[int, int]] = DEFAULT_EXPECTED_SHAPE,
    normalize: bool = False,
) -> np.ndarray:
    """Load RGB and NIR images and stack them into a 4-channel array (H, W, 4).

    Order of channels: [Red, Green, Blue, NIR].

    Args:
        rgb_path: Path to RGB image.
        nir_path: Path to NIR image.
        expected_hw: Expected (height, width).
        normalize: If True, returns float32 in [0.0, 1.0], else uint8.

    Returns:
        np.ndarray of shape (H, W, 4).
    """
    rgb = load_rgb_image(rgb_path, expected_hw=expected_hw, normalize=normalize)
    nir = load_nir_image(nir_path, expected_hw=expected_hw, keep_dim=True, normalize=normalize)

    if rgb.shape[:2] != nir.shape[:2]:
        raise ValueError(
            f"Dimension mismatch between RGB {rgb.shape[:2]} and NIR {nir.shape[:2]}"
        )

    return np.concatenate([rgb, nir], axis=-1)
