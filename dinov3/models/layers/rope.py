from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class RopePositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for 2D inputs.

    Takes an input image with HxW patches and returns sin and cos positional embeddings.
    First, normalize coordinates of each patch to range [0,1], then convert to [-1,+1] range.
    Then, apply random augmentations (shift, jitter, rescale) to the coordinates during training.
    Finally, compute sin and cos embeddings using either a fixed base or a range of periods.

    Args:
        embed_dim (int): Embedding dimension. Must be divisible by 4 * num_heads
        num_heads (int): Number of attention heads.
        base (float | None): Base for computing periods. Either `base` or `min_period`+`max_period` must be provided.
        min_period (float | None): Minimum period for computing periods.
        max_period (float | None): Maximum period for computing periods.
        normalize_coords (str): How to normalize coordinates. One of "min", "max", "separate".
            "min": normalize by min(H,W), "max": normalize by max(H,W), "separate": normalize H and W separately.
        shift_coords (float | None): If provided, during training, shift coordinates by a random uniform value in [-shift, shift].
        jitter_coords (float | None): If provided, during training, multiply coordinates by a random log-uniform value in [1/jitter, jitter].
        rescale_coords (float | None): If provided, during training, multiply coordinates by a random log-uniform value in [1/rescale, rescale].
        dtype (mx.Dtype | None): Data type for the computations.
    """
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: mx.Dtype | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.dtype = dtype

        # Initialize periods
        if base is not None:
            self.periods = base ** (2 * mx.arange(0, D_head // 4, dtype=dtype) / (D_head // 2))  # (D_head//4,)
        else:
            base = max_period / min_period
            exponents = mx.linspace(0, 1, D_head // 4, dtype=dtype)  # (D_head//4,)
            self.periods: mx.array = base ** exponents / base * max_period  # (D_head//4,)

    def __call__(self, *, H: int, W: int) -> tuple[mx.array, mx.array]:
        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = mx.arange(0.5, H, dtype=self.dtype) / max_HW  # (H,)
            coords_w = mx.arange(0.5, W, dtype=self.dtype) / max_HW  # (W,)
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = mx.arange(0.5, H, dtype=self.dtype) / min_HW  # (H,)
            coords_w = mx.arange(0.5, W, dtype=self.dtype) / min_HW  # (W,)
        elif self.normalize_coords == "separate":
            coords_h = mx.arange(0.5, H, dtype=self.dtype) / H  # (H,)
            coords_w = mx.arange(0.5, W, dtype=self.dtype) / W  # (W,)
        else:
            raise ValueError(f"Unsupported normalize_coords: {self.normalize_coords}")

        coords = mx.stack(mx.meshgrid(coords_h, coords_w, indexing="ij"), axis=-1)  # (H,W,2)
        coords = coords.flatten(0, 1)  # (H*W,2)
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a random uniform value for H and W in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = mx.random.uniform(-self.shift_coords, self.shift_coords, shape=(2,), dtype=self.dtype)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = mx.random.uniform(jitter_min, jitter_max, shape=(2,), dtype=self.dtype).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = mx.random.uniform(rescale_min, rescale_max, shape=(1,), dtype=self.dtype).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = np.pi * coords[:, :, None] / self.periods[None, None, :]  # (H*W,2,D//4)
        angles: mx.array = angles.flatten(1, 2)  # (H*W,D//2)
        angles = mx.tile(angles, (1, 2))  # (H*W,D)
        cos_angles = mx.cos(angles)  # (H*W,D)
        sin_angles = mx.sin(angles)  # (H*W,D)

        return sin_angles, cos_angles
