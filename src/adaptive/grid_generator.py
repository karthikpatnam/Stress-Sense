"""Grid generation and spatial region partitioning for StressSense adaptive sampling.

This module provides the GridGenerator and GridRegion data structures used
to divide aerial agricultural orthomosaics and multi-spectral patches into
spatial sub-regions for localized stress scoring and adaptive sampling.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class GridRegion:
    """Represents a spatial grid region within an aerial crop image.

    Attributes:
        region_id: Globally unique deterministic identifier (e.g. 'image01_r02_c03').
        row: Row index in the 2D spatial grid (0-indexed).
        col: Column index in the 2D spatial grid (0-indexed).
        x_min: Minimum x (column) pixel coordinate, inclusive.
        y_min: Minimum y (row) pixel coordinate, inclusive.
        x_max: Maximum x (column) pixel coordinate, exclusive.
        y_max: Maximum y (row) pixel coordinate, exclusive.
        width: Region width in pixels.
        height: Region height in pixels.
        center_x: Horizontal center coordinate in pixel space.
        center_y: Vertical center coordinate in pixel space.
        area: Total pixel area (width * height).
        image_id: Identifier of the parent image/tile, if provided.
        valid_fraction: Fraction of pixels within sensor flight validity mask [0.0, 1.0].
        boundary_fraction: Fraction of pixels within agricultural boundary mask [0.0, 1.0].
        is_valid: Whether the region has sufficient valid coverage for stress scoring.
    """

    region_id: str
    row: int
    col: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int
    center_x: float
    center_y: float
    area: int
    image_id: Optional[str] = None
    valid_fraction: float = 1.0
    boundary_fraction: float = 1.0
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert GridRegion metadata to dictionary format."""
        return {
            "region_id": self.region_id,
            "image_id": self.image_id,
            "row": self.row,
            "col": self.col,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "width": self.width,
            "height": self.height,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "area": self.area,
            "valid_fraction": round(self.valid_fraction, 5),
            "boundary_fraction": round(self.boundary_fraction, 5),
            "is_valid": self.is_valid,
        }

    def contains_point(self, x: float, y: float) -> bool:
        """Check whether the given coordinate point (x, y) falls inside this region."""
        return self.x_min <= x < self.x_max and self.y_min <= y < self.y_max


class GridGenerator:
    """Generates structured spatial grids over aerial crop images.

    Supports configurable cell sizes (e.g. 64x64 or 32x32), non-overlapping or
    strided step sizes, and defensive boundary handling for arbitrary image sizes.
    """

    def __init__(
        self,
        grid_size: Union[int, Tuple[int, int]] = 64,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        min_valid_fraction: float = 0.05,
    ) -> None:
        """Initialize the GridGenerator.

        Args:
            grid_size: Dimension of grid cells. If int, square cells (grid_size, grid_size).
                       If tuple, (height, width).
            stride: Step size between adjacent grid cells. Defaults to grid_size
                    (non-overlapping grid). If tuple, (stride_y, stride_x).
            min_valid_fraction: Minimum ratio of valid sensor coverage required for
                                a region to be marked is_valid=True. Default 0.05.
        """
        if isinstance(grid_size, int):
            if grid_size <= 0:
                raise ValueError(f"grid_size must be positive, got {grid_size}")
            self.cell_h: int = grid_size
            self.cell_w: int = grid_size
        elif isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
            self.cell_h, self.cell_w = int(grid_size[0]), int(grid_size[1])
            if self.cell_h <= 0 or self.cell_w <= 0:
                raise ValueError(f"grid_size dimensions must be positive, got {grid_size}")
        else:
            raise TypeError(f"Invalid grid_size type: {type(grid_size)}")

        if stride is None:
            self.stride_y: int = self.cell_h
            self.stride_x: int = self.cell_w
        elif isinstance(stride, int):
            if stride <= 0:
                raise ValueError(f"stride must be positive, got {stride}")
            self.stride_y = stride
            self.stride_x = stride
        elif isinstance(stride, (tuple, list)) and len(stride) == 2:
            self.stride_y, self.stride_x = int(stride[0]), int(stride[1])
            if self.stride_y <= 0 or self.stride_x <= 0:
                raise ValueError(f"stride dimensions must be positive, got {stride}")
        else:
            raise TypeError(f"Invalid stride type: {type(stride)}")

        self.min_valid_fraction = float(min_valid_fraction)

    def generate_grid(
        self,
        image_shape: Tuple[int, ...],
        image_id: Optional[str] = None,
        validity_mask: Optional[np.ndarray] = None,
        boundary_mask: Optional[np.ndarray] = None,
    ) -> List[GridRegion]:
        """Divide the image spatial dimensions into discrete grid regions.

        Args:
            image_shape: Shape tuple of the image array. Expects at least (height, width).
            image_id: Optional identifier string for the parent image.
            validity_mask: Optional 2D binary array (H, W) where >0 represents valid pixels.
            boundary_mask: Optional 2D binary array (H, W) where >0 represents field boundary.

        Returns:
            List of GridRegion objects ordered in row-major order.
        """
        if len(image_shape) < 2:
            raise ValueError(f"image_shape must have at least 2 dimensions (H, W), got {image_shape}")

        height, width = int(image_shape[0]), int(image_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Image dimensions must be positive, got {(height, width)}")

        # Validate mask shapes if provided
        if validity_mask is not None:
            if validity_mask.shape[:2] != (height, width):
                raise ValueError(
                    f"validity_mask shape {validity_mask.shape[:2]} does not match image shape {(height, width)}"
                )
        if boundary_mask is not None:
            if boundary_mask.shape[:2] != (height, width):
                raise ValueError(
                    f"boundary_mask shape {boundary_mask.shape[:2]} does not match image shape {(height, width)}"
                )

        prefix = f"{image_id}_" if image_id else "region_"
        regions: List[GridRegion] = []

        # Determine y offsets
        y_starts: List[int] = list(range(0, height, self.stride_y))
        x_starts: List[int] = list(range(0, width, self.stride_x))

        for r_idx, y_min in enumerate(y_starts):
            y_max = min(y_min + self.cell_h, height)
            actual_h = y_max - y_min

            for c_idx, x_min in enumerate(x_starts):
                x_max = min(x_min + self.cell_w, width)
                actual_w = x_max - x_min
                actual_area = actual_w * actual_h

                region_id = f"{prefix}r{r_idx:02d}_c{c_idx:02d}"
                center_x = (x_min + x_max) / 2.0
                center_y = (y_min + y_max) / 2.0

                # Compute validity and boundary coverage fractions if masks available
                valid_fraction = 1.0
                boundary_fraction = 1.0

                if validity_mask is not None:
                    mask_sub = validity_mask[y_min:y_max, x_min:x_max]
                    valid_pixels = int(np.count_nonzero(mask_sub > 0))
                    valid_fraction = float(valid_pixels / actual_area) if actual_area > 0 else 0.0

                if boundary_mask is not None:
                    boundary_sub = boundary_mask[y_min:y_max, x_min:x_max]
                    boundary_pixels = int(np.count_nonzero(boundary_sub > 0))
                    boundary_fraction = float(boundary_pixels / actual_area) if actual_area > 0 else 0.0

                is_valid = valid_fraction >= self.min_valid_fraction

                region = GridRegion(
                    region_id=region_id,
                    row=r_idx,
                    col=c_idx,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    width=actual_w,
                    height=actual_h,
                    center_x=center_x,
                    center_y=center_y,
                    area=actual_area,
                    image_id=image_id,
                    valid_fraction=valid_fraction,
                    boundary_fraction=boundary_fraction,
                    is_valid=is_valid,
                )
                regions.append(region)

        return regions

    def extract_region(self, image: np.ndarray, region: GridRegion) -> np.ndarray:
        """Extract the sub-array slice corresponding to a grid region.

        Args:
            image: NumPy array of shape (H, W) or (H, W, C).
            region: GridRegion specifying bounding coordinates.

        Returns:
            Sub-array slice of shape (height, width) or (height, width, C).
        """
        if image.shape[0] < region.y_max or image.shape[1] < region.x_max:
            raise ValueError(
                f"Image shape {image.shape[:2]} cannot accommodate region bounds "
                f"y:[{region.y_min}:{region.y_max}], x:[{region.x_min}:{region.x_max}]"
            )
        return image[region.y_min : region.y_max, region.x_min : region.x_max]

    def extract_all_regions(
        self,
        image: np.ndarray,
        regions: List[GridRegion],
    ) -> List[np.ndarray]:
        """Extract sub-arrays for all specified grid regions.

        Args:
            image: NumPy array of shape (H, W) or (H, W, C).
            regions: List of GridRegion objects.

        Returns:
            List of sub-arrays in the same order as regions.
        """
        return [self.extract_region(image, reg) for reg in regions]

    def get_grid_dimensions(
        self,
        image_shape: Tuple[int, ...],
    ) -> Tuple[int, int]:
        """Calculate the total rows and columns of grid regions for a given image shape.

        Args:
            image_shape: Shape tuple of the image array (H, W, ...).

        Returns:
            Tuple of (num_rows, num_cols).
        """
        height, width = int(image_shape[0]), int(image_shape[1])
        num_rows = len(list(range(0, height, self.stride_y)))
        num_cols = len(list(range(0, width, self.stride_x)))
        return (num_rows, num_cols)
