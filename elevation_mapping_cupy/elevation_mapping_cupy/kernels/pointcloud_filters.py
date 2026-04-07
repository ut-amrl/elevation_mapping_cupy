#
# GPU pointcloud pre-filters: azimuth masking and voxel downsampling.
# All operations run entirely on the GPU via CuPy.
#
import cupy as cp

# ── Azimuth filter ────────────────────────────────────────────────────────────
# ElementwiseKernel compiles to a real CUDA kernel the first time it is called
# and is cached for every subsequent call.
_azimuth_mask_kernel = cp.ElementwiseKernel(
    # inputs
    "float32 x, float32 y, float32 az_min, float32 az_max, bool wrap",
    # output
    "bool mask",
    # CUDA C body (one thread per point)
    """
    float az = atan2f(y, x);
    if (wrap) {
        // wrap-around: keep points outside the (az_max, az_min) gap
        mask = (az >= az_min) || (az <= az_max);
    } else {
        mask = (az >= az_min) && (az <= az_max);
    }
    """,
    "azimuth_mask_kernel",
)


def azimuth_filter_gpu(pts: cp.ndarray, az_min: float, az_max: float) -> cp.ndarray:
    """Filter points by bearing angle atan2(y, x) computed in the current frame.

    Wrap-around supported: if az_min > az_max the complement sector is kept
    (e.g. az_min=2.5, az_max=-2.5 keeps the narrow rear wedge).

    Args:
        pts: (N, 3+) float32 cupy array — x,y,z in columns 0-2.
        az_min: minimum azimuth [rad], in (-pi, pi].
        az_max: maximum azimuth [rad], in (-pi, pi].

    Returns:
        Filtered (M, 3+) cupy array.
    """
    if pts.shape[0] == 0:
        return pts
    wrap = bool(az_min > az_max)
    mask = _azimuth_mask_kernel(
        pts[:, 0].astype(cp.float32),
        pts[:, 1].astype(cp.float32),
        cp.float32(az_min),
        cp.float32(az_max),
        wrap,
    )
    return pts[mask]


# ── Voxel downsampling ────────────────────────────────────────────────────────
def voxel_downsample_gpu(pts: cp.ndarray, leaf_size: float) -> cp.ndarray:
    """Voxel grid downsampling: retain one point per leaf_size³ voxel.

    Strategy (all on GPU):
      1. Quantize xyz to integer voxel indices.
      2. Pack the 3-D index into a single int64 key.
      3. Argsort by key (CUB radix sort under the hood).
      4. Mark the first occurrence of each unique key via diff.
      5. Gather the selected rows.

    Args:
        pts: (N, 3+) float32 cupy array — x,y,z in columns 0-2.
        leaf_size: voxel side length [m].  <= 0 disables the filter.

    Returns:
        Filtered (M, 3+) cupy array.
    """
    if pts.shape[0] == 0 or leaf_size <= 0.0:
        return pts

    inv = cp.float32(1.0 / leaf_size)
    # Quantize xyz → integer voxel coordinates
    vox = cp.floor(pts[:, :3] * inv).astype(cp.int32)

    # Encode [ix, iy, iz] into one int64 so a single argsort handles all three
    # axes.  An offset of 2^20 (≈1M voxels/axis) covers ±100 km at 0.1 m leaf.
    OFF = cp.int64(1 << 20)
    STRIDE = cp.int64(1 << 21)  # 2 * OFF — stride between axes
    enc = (
        (vox[:, 0].astype(cp.int64) + OFF) * STRIDE * STRIDE
        + (vox[:, 1].astype(cp.int64) + OFF) * STRIDE
        + (vox[:, 2].astype(cp.int64) + OFF)
    )

    # GPU radix sort (via CuPy → CUB)
    sort_idx = cp.argsort(enc)
    sorted_enc = enc[sort_idx]

    # First occurrence of each unique key: True at every transition
    keep = cp.empty(pts.shape[0], dtype=cp.bool_)
    keep[0] = True
    keep[1:] = sorted_enc[1:] != sorted_enc[:-1]

    return pts[sort_idx[keep]]
