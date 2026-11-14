"""Image Normalization and Radiometric Preprocessing Module for StressSense.

Supports configurable normalization methods for:
- 3-channel RGB aerial imagery
- 1-channel Near-Infrared (NIR) aerial imagery
- 4-channel stacked [RGB + NIR] imagery
- Normalized Difference Vegetation Index (NDVI) computation

Preserves original image resolution and avoids lossy transformations.
"""

from enum import Enum
from typing import Dict, Optional, Tuple, Union
import numpy as np


class NormalizationMethod(str, Enum):
    """Supported normalization strategies."""
    NONE = "none"                  # Raw uint8 [0, 255] or unscaled float
    MINMAX = "minmax"              # Linearly scaled to [0.0, 1.0]
    ZSCORE = "zscore"              # Standardized: (x - mean) / std
    CENTERED = "centered"          # Scaled to [-1.0, 1.0]


# Standard aerial/multispectral channel statistics (RGB + NIR)
DEFAULT_CHANNEL_MEANS: Dict[str, Tuple[float, ...]] = {
    "rgb": (0.485, 0.456, 0.406),          # ImageNet-standard RGB means
    "nir": (0.500,),                       # Standard NIR mean
    "rgbn": (0.485, 0.456, 0.406, 0.500),  # 4-channel stack
}

DEFAULT_CHANNEL_STDS: Dict[str, Tuple[float, ...]] = {
    "rgb": (0.229, 0.224, 0.225),          # ImageNet-standard RGB stds
    "nir": (0.250,),                       # Standard NIR std
    "rgbn": (0.229, 0.224, 0.225, 0.250),  # 4-channel stack
}


class ImageNormalizer:
    """Configurable image normalizer for multi-spectral agricultural imagery."""

    def __init__(
        self,
        method: Union[str, NormalizationMethod] = NormalizationMethod.MINMAX,
        means: Optional[Tuple[float, ...]] = None,
        stds: Optional[Tuple[float, ...]] = None,
        eps: float = 1e-7,
    ) -> None:
        """Initialize normalizer with specified method and channel parameters.

        Args:
            method: Normalization strategy ('none', 'minmax', 'zscore', 'centered').
            means: Channel means for z-score standardization.
            stds: Channel standard deviations for z-score standardization.
            eps: Small epsilon to prevent division by zero.
        """
        if isinstance(method, str):
            method = NormalizationMethod(method.lower())
        self.method = method
        self.means = means
        self.stds = stds
        self.eps = eps

    def normalize(
        self,
        image: np.ndarray,
        modality: str = "rgb",
    ) -> np.ndarray:
        """Normalize an input image array according to configured method.

        Args:
            image: NumPy array of shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4).
            modality: Modality tag ('rgb', 'nir', or 'rgbn') used for default stats.

        Returns:
            Normalized NumPy array (float32 if transformed, or original uint8 if 'none').
        """
        if self.method == NormalizationMethod.NONE:
            return image.copy()

        # Convert uint8 [0, 255] to float32 [0.0, 1.0] as base
        if np.issubdtype(image.dtype, np.integer):
            arr = image.astype(np.float32) / 255.0
        else:
            arr = image.astype(np.float32).copy()

        if self.method == NormalizationMethod.MINMAX:
            return arr

        if self.method == NormalizationMethod.CENTERED:
            # Scale [0.0, 1.0] -> [-1.0, 1.0]
            return arr * 2.0 - 1.0

        if self.method == NormalizationMethod.ZSCORE:
            means = self.means or DEFAULT_CHANNEL_MEANS.get(modality.lower())
            stds = self.stds or DEFAULT_CHANNEL_STDS.get(modality.lower())

            if means is None or stds is None:
                # Compute sample-wise mean and std across spatial axes (0, 1)
                axis = (0, 1) if arr.ndim >= 3 else None
                mean_val = arr.mean(axis=axis, keepdims=True)
                std_val = arr.std(axis=axis, keepdims=True)
                return (arr - mean_val) / (std_val + self.eps)

            # Broadcast channel parameters
            means_arr = np.array(means, dtype=np.float32)
            stds_arr = np.array(stds, dtype=np.float32)

            if arr.ndim == 2:
                # Grayscale / 1-channel (H, W)
                return (arr - means_arr[0]) / (stds_arr[0] + self.eps)
            elif arr.ndim == 3 and arr.shape[2] == len(means_arr):
                return (arr - means_arr) / (stds_arr + self.eps)
            else:
                raise ValueError(
                    f"Array shape {arr.shape} incompatible with normalization stats length {len(means_arr)}"
                )

        raise ValueError(f"Unsupported normalization method: {self.method}")

    def denormalize(
        self,
        image: np.ndarray,
        modality: str = "rgb",
    ) -> np.ndarray:
        """Inverse transformation to recover uint8 [0, 255] image for visualization.

        Args:
            image: Normalized float32 NumPy array.
            modality: Modality tag ('rgb', 'nir', or 'rgbn').

        Returns:
            NumPy array of dtype uint8 with values clipped to [0, 255].
        """
        if self.method == NormalizationMethod.NONE:
            return np.clip(image, 0, 255).astype(np.uint8)

        arr = image.astype(np.float32).copy()

        if self.method == NormalizationMethod.ZSCORE:
            means = self.means or DEFAULT_CHANNEL_MEANS.get(modality.lower())
            stds = self.stds or DEFAULT_CHANNEL_STDS.get(modality.lower())
            if means is not None and stds is not None:
                means_arr = np.array(means, dtype=np.float32)
                stds_arr = np.array(stds, dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr * (stds_arr[0] + self.eps) + means_arr[0]
                elif arr.ndim == 3 and arr.shape[2] == len(means_arr):
                    arr = arr * (stds_arr + self.eps) + means_arr

        elif self.method == NormalizationMethod.CENTERED:
            # Inverse of arr * 2.0 - 1.0 -> (arr + 1.0) / 2.0
            arr = (arr + 1.0) / 2.0

        # Rescale [0.0, 1.0] -> [0, 255]
        arr = arr * 255.0
        return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def compute_ndvi(
    rgb_image: np.ndarray,
    nir_image: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index (NDVI) from RGB and NIR.

    Formula:
        NDVI = (NIR - Red) / (NIR + Red + eps)

    Args:
        rgb_image: RGB array (H, W, 3), Red channel at index 0.
        nir_image: NIR array (H, W) or (H, W, 1).
        eps: Small epsilon to avoid division by zero.

    Returns:
        np.ndarray of shape (H, W) with float32 values in [-1.0, 1.0].
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {rgb_image.shape}")

    red = rgb_image[:, :, 0].astype(np.float32)
    nir = (nir_image.squeeze() if nir_image.ndim == 3 else nir_image).astype(np.float32)

    if red.shape != nir.shape:
        raise ValueError(f"Shape mismatch between Red channel {red.shape} and NIR {nir.shape}")

    numerator = nir - red
    denominator = nir + red + eps
    ndvi = numerator / denominator
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)
