import mlx.core as mx
import mlx.nn as nn


class LayerScale(nn.Module):
    """LayerScale module.

    Args:
        dim (int): Dimension of the input tensor.
        init_values (float): Initial value for the learnable scaling parameter.
    """
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
    ):
        super().__init__()
        self.gamma = mx.full((dim,), init_values)
        self.init_values = init_values

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma
