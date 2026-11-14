"""Patch Extraction and Reconstruction Engine for StressSense.

Extracts fixed-size spatial patches from 2D (masks, NIR) and 3D (RGB, multi-channel)
aerial imagery with configurable patch dimensions, strides, and boundary handling.
"""

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ExtractedPatch:
    """Container for an individual extracted patch and its spatial coordinates."""
    patch_index: int
    y: int               # Top row coordinate in parent image
    x: int               # Left col coordinate in parent image
    height: int          # Patch height
    width: int           # Patch width
    data: np.ndarray     # Patch array


class PatchExtractor:
    """Configurable patch extractor with boundary padding and reconstruction support."""

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = (128, 128),
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        pad_if_needed: bool = True,
        pad_mode: str = "reflect",
    ) -> None:
        """Initialize extractor.

        Args:
            patch_size: Size of patch as int (square) or (height, width). Default (128, 128).
            stride: Step size as int or (step_y, step_x). Defaults to patch_size (non-overlapping).
            pad_if_needed: If True, pads image if dimensions are not multiples of stride/size.
            pad_mode: NumPy pad mode ('reflect', 'constant', 'edge', etc.). Default 'reflect'.
        """
        if isinstance(patch_size, int):
            self.patch_h, self.patch_w = patch_size, patch_size
        else:
            self.patch_h, self.patch_w = patch_size

        if stride is None:
            self.stride_y, self.stride_x = self.patch_h, self.patch_w
        elif isinstance(stride, int):
            self.stride_y, self.stride_x = stride, stride
        else:
            self.stride_y, self.stride_x = stride

        self.pad_if_needed = pad_if_needed
        self.pad_mode = pad_mode

    def compute_grid_coordinates(
        self,
        image_h: int,
        image_w: int,
    ) -> Tuple[List[int], List[int], Tuple[int, int]]:
        """Compute the top-left (y, x) coordinates for the patch grid.

        Returns:
            (y_coords, x_coords, padded_shape)
        """
        target_h, target_w = image_h, image_w

        if self.pad_if_needed:
            # If dimensions exceed one patch and are not covered evenly
            if (target_h - self.patch_h) % self.stride_y != 0:
                steps_y = int(np.ceil((target_h - self.patch_h) / self.stride_y))
                target_h = self.patch_h + steps_y * self.stride_y
            if (target_w - self.patch_w) % self.stride_x != 0:
                steps_x = int(np.ceil((target_w - self.patch_w) / self.stride_x))
                target_w = self.patch_w + steps_x * self.stride_x

        y_coords = list(range(0, target_h - self.patch_h + 1, self.stride_y))
        x_coords = list(range(0, target_w - self.patch_w + 1, self.stride_x))

        return y_coords, x_coords, (target_h, target_w)

    def pad_image(
        self,
        image: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Pad an image array to target (H, W) using configured pad_mode."""
        cur_h, cur_w = image.shape[:2]
        pad_bottom = target_shape[0] - cur_h
        pad_right = target_shape[1] - cur_w

        if pad_bottom == 0 and pad_right == 0:
            return image

        if image.ndim == 2:
            pad_width = ((0, pad_bottom), (0, pad_right))
        elif image.ndim == 3:
            pad_width = ((0, pad_bottom), (0, pad_right), (0, 0))
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")

        if self.pad_mode == "constant":
            return np.pad(image, pad_width, mode="constant", constant_values=0)
        return np.pad(image, pad_width, mode=self.pad_mode)

    def extract_patches(
        self,
        image: np.ndarray,
    ) -> List[ExtractedPatch]:
        """Extract patches from an image array.

        Args:
            image: NumPy array of shape (H, W) or (H, W, C).

        Returns:
            List of ExtractedPatch instances.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected NumPy array, got {type(image)}")
        if image.ndim < 2:
            raise ValueError(f"Array must have at least 2 dimensions, got {image.ndim}")

        cur_h, cur_w = image.shape[:2]
        y_coords, x_coords, target_shape = self.compute_grid_coordinates(cur_h, cur_w)

        # Apply padding if needed
        working_img = self.pad_image(image, target_shape) if target_shape != (cur_h, cur_w) else image

        patches: List[ExtractedPatch] = []
        idx = 0
        for y in y_coords:
            for x in x_coords:
                patch_data = working_img[y : y + self.patch_h, x : x + self.patch_w].copy()
                patches.append(
                    ExtractedPatch(
                        patch_index=idx,
                        y=y,
                        x=x,
                        height=self.patch_h,
                        width=self.patch_w,
                        data=patch_data,
                    )
                )
                idx += 1

        return patches

    def reconstruct_from_patches(
        self,
        patches: List[ExtractedPatch],
        original_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct the original image array from a list of non-overlapping patches.

        Args:
            patches: List of ExtractedPatch instances.
            original_shape: Target output shape e.g. (H, W) or (H, W, C).

        Returns:
            Reconstructed NumPy array of shape original_shape.
        """
        reconstructed = np.zeros(original_shape, dtype=patches[0].data.dtype)

        for patch in patches:
            y1 = patch.y
            y2 = min(patch.y + patch.height, original_shape[0])
            x1 = patch.x
            x2 = min(patch.x + patch.width, original_shape[1])

            h_slice = y2 - y1
            w_slice = x2 - x1

            if h_slice > 0 and w_slice > 0:
                if reconstructed.ndim == 2:
                    reconstructed[y1:y2, x1:x2] = patch.data[:h_slice, :w_slice]
                else:
                    reconstructed[y1:y2, x1:x2, :] = patch.data[:h_slice, :w_slice, :]

        return reconstructed
