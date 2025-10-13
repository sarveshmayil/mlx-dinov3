from typing import Callable

import mlx.core as mx
import mlx.nn as nn


class PatchEmbed(nn.Module):
    """Converts 2D images into patch embeddings.

    (B,C,H,W) -> (B,N,D)

    Args:
        img_size (int,int): Image size.
        patch_size (int,int): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Module): Normalization layer.
        flatten_embedding (bool): If True, flatten the output to (B, N, D), otherwise (B, H', W', D).
    """
    def __init__(
        self,
        img_size: tuple[int,int] = (224,224),
        patch_size: tuple[int,int] = (16,16),
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable[..., nn.Module] | None = None,
        flatten_embedding: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        _, _, H, W = x.shape
        assert H % self.img_size[0] == 0, f"Input image height {H} is not a multiple of patch height {self.img_size[0]}"
        assert W % self.img_size[1] == 0, f"Input image width {W} is not a multiple of patch width: {self.img_size[1]}"

        x = self.proj(x)  # B C H W
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # B C H W -> B C HW -> B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x
