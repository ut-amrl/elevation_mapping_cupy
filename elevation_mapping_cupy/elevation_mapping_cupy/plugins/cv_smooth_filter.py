#
# Fast cv2-based spatial smoothing for elevation maps.
# Replaces the slow cupyx.scipy.ndimage uniform_filter with OpenCV's
# highly-optimized median/Gaussian/box filters (NEON-accelerated on Jetson).
#
import cupy as cp
import cv2 as cv
import numpy as np
from typing import List

from .plugin_manager import PluginBase


class CvSmoothFilter(PluginBase):
    """Fast spatial smoothing using OpenCV (CPU).

    For small maps (50×50–200×200) the GPU→CPU→GPU round-trip is negligible
    compared to the speedup from OpenCV's SIMD-optimized filter kernels vs
    the generic cupyx.scipy.ndimage path.

    Supports three modes matching common use-cases:
      - ``median``  — edge-preserving spike removal (like FastDEM)
      - ``gaussian`` — smooth Gaussian blur
      - ``box``     — fast uniform/mean filter

    Args:
        cell_n: Grid width/height.
        input_layer_name: Source layer (default ``"elevation"``).
        method: One of ``"median"``, ``"gaussian"``, ``"box"``.
        kernel_size: Filter kernel size (must be odd). Default 3.
        min_valid_neighbors: For median mode, skip cells with fewer valid
            neighbors in the kernel window (like FastDEM). 0 = disabled.
    """

    def __init__(
        self,
        cell_n: int = 100,
        input_layer_name: str = "elevation",
        method: str = "median",
        kernel_size: int = 3,
        min_valid_neighbors: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.input_layer_name = input_layer_name
        self.method = method.lower().strip()
        self.kernel_size = max(3, int(kernel_size))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.min_valid_neighbors = int(min_valid_neighbors)

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        # Resolve input layer
        if self.input_layer_name in layer_names:
            idx = layer_names.index(self.input_layer_name)
            layer = elevation_map[idx].copy()
            if self.input_layer_name == "elevation":
                valid = elevation_map[2] > 0.5
                layer = cp.where(valid & cp.isfinite(layer), layer, cp.nan)
        elif self.input_layer_name in plugin_layer_names:
            idx = plugin_layer_names.index(self.input_layer_name)
            layer = plugin_layers[idx].copy()
        else:
            layer = elevation_map[0].copy()
            valid = elevation_map[2] > 0.5
            layer = cp.where(valid & cp.isfinite(layer), layer, cp.nan)

        # GPU → CPU
        h_np = cp.asnumpy(layer).astype(np.float32)
        finite_mask = np.isfinite(h_np)

        if not finite_mask.any():
            return layer

        # Replace NaN with 0 for cv2 (it can't handle NaN)
        h_fill = np.where(finite_mask, h_np, 0.0).astype(np.float32)

        # Apply the selected filter
        if self.method == "median":
            # cv2.medianBlur requires uint8 or float32;
            # for float32 only kernel_size <= 5 is supported,
            # so for larger kernels we quantize to uint8.
            if self.kernel_size <= 5:
                smoothed = cv.medianBlur(h_fill, self.kernel_size)
            else:
                h_min, h_max = float(h_np[finite_mask].min()), float(h_np[finite_mask].max())
                denom = h_max - h_min
                if denom < 1e-6:
                    return layer
                u8 = np.clip((h_fill - h_min) * 255.0 / denom, 0, 255).astype(np.uint8)
                u8_smooth = cv.medianBlur(u8, self.kernel_size)
                smoothed = u8_smooth.astype(np.float32) * denom / 255.0 + h_min
        elif self.method == "gaussian":
            smoothed = cv.GaussianBlur(h_fill, (self.kernel_size, self.kernel_size), 0)
        else:  # box / uniform
            smoothed = cv.blur(h_fill, (self.kernel_size, self.kernel_size))

        # Only update valid cells; if min_valid_neighbors is set, skip
        # cells with insufficient valid neighbors in the kernel window.
        output = h_np.copy()
        update_mask = finite_mask

        if self.min_valid_neighbors > 0:
            # Count valid neighbors per cell using a box filter on the mask
            valid_count = cv.blur(
                finite_mask.astype(np.float32),
                (self.kernel_size, self.kernel_size),
            ) * (self.kernel_size * self.kernel_size)
            update_mask = finite_mask & (valid_count >= self.min_valid_neighbors)

        output[update_mask] = smoothed[update_mask]

        # CPU → GPU
        return cp.asarray(output, dtype=cp.float32)
