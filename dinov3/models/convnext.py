from typing import Sequence

import mlx.core as mx
import mlx.nn as nn


SIZE_DICT = {
    "tiny": dict(
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
    ),
    "small": dict(
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
    ),
    "base": dict(
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
    ),
    "large": dict(
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
    ),
}


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def __call__(self, x: mx.array):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = mx.random.uniform(shape=shape, device=x.device)
        random_tensor = mx.floor(random_tensor + keep_prob)
        return mx.divide(x, keep_prob) * random_tensor


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Args:
        normalized_shape (int): The feature dimension of the input to normalize over
        eps (float): A small additive constant for numerical stability
        data_format (str): The data format of the input tensor, either "channels_last" or "channels_first"
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = mx.ones((normalized_shape,))
        self.bias = mx.zeros((normalized_shape,))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape, )

    def __call__(self, x: mx.array):
        if self.data_format == "channels_last":
            return mx.fast.layer_norm(x, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(3, keepdims=True)
            s = mx.power(x - u, 2).mean(3, keepdims=True)
            x = (x - u) / mx.sqrt(s + self.eps)
            x = self.weight[None, None, :] * x + self.bias[None, None, :]
            return x
        

class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4*dim)  # pointwise/1x1 convs implemented using Linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4*dim, dim)
        self.gamma = mx.ones((dim,)) * layer_scale_init_value if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def __call__(self, x: mx.array):
        in_x = x
        x = self.dwconv(x) # (B, H, W, C)
        x = self.norm(x)
        x = self.act(self.pwconv1(x))
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return in_x + self.drop_path(x)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        patch_size: int | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            print(f"Warning: Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        # Stem + 3 intermediate downsampling conv layers
        self.downsample_layers = []
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(len(dims)-1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = []
        dp_rates = [x for x in mx.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur_depth = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path=dp_rates[cur_depth + j], layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur_depth += depths[i]

        assert len(self.downsample_layers) == len(self.stages), "Number of downsample layers must match number of stages"

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        self.head = nn.Identity()  # remove classification head
        self.embed_dim = dims[-1]
        self.embed_dims = dims  # per-layer dimensions
        self.n_blocks = len(self.downsample_layers)
        self.chunked_blocks = False
        self.n_storage_tokens = 0  # no registers

        self.norms = [nn.Identity()] * (self.n_blocks - 1)
        self.norms.append(self.norm)

        self.patch_size = patch_size
        self.input_pad_size = 4  # for first conv with kernel_size=4 and stride=4

    def forward_features_list(self, x_list: list[mx.array], masks_list: list[mx.array | None]) -> list[dict[str, mx.array]]:
        output = []
        for x, masks in zip(x_list, masks_list):
            for i in range(self.n_blocks):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            x_pool = x.mean((1,2))  # global avg pool (B, H, W, C) -> (B, C)
            x = x.flatten(1,2)  # (B, H, W, C) -> (B, H*W, C)

            # Concat [CLS] and patch tokens as (B, H*W+1, C), then normalize
            x_norm = self.norm(mx.concatenate([x_pool[:, None, :], x], axis=1))  # (B, 1, C) + (B, H*W, C) -> (B, H*W+1, C)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_storage_tokens": x_norm[:, 1:self.n_storage_tokens+1],
                    "x_norm_patchtokens": x_norm[:, self.n_storage_tokens+1:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )

        return output
    
    def forward_features(self, x: mx.array | list[mx.array], masks: mx.array | None = None) -> dict[str, mx.array] | list[dict[str, mx.array]]:
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        return self.forward_features_list([x], [masks])[0]
    
    def __call__(self, *args, is_training: bool = False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

    def _get_intermediate_layers(self, x: mx.array, n: int | Sequence[int] = 1) -> list[tuple[mx.array, mx.array]]:
        h, w = x.shape[1:3]
        output = []
        blocks_to_take = range(self.n_blocks - n, self.n_blocks) if isinstance(n, int) else n
        for i in range(self.n_blocks):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in blocks_to_take:
                x_pool = x.mean((1,2))
                x_patches = x
                if self.patch_size is not None:
                    # Resize output feature map to ViT patch size
                    x_patches = nn.layers.upsample.upsample_linear(
                        x,
                        scale_factor=(self.patch_size / h, self.patch_size / w),
                        align_corners=True,
                    )
                output.append(
                    [
                        x_pool,  # [CLS] (B, C)
                        x_patches,  # (B, H', W', C)
                    ]
                )
        assert len(output) == len(blocks_to_take), f"Only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(self, x: mx.array, n: int | Sequence[int] = 1, reshape: bool = False, return_class_token: bool = False, norm: bool = True) -> tuple[mx.array] | list[tuple[mx.array, mx.array]]:
        outputs = self._get_intermediate_layers(x, n)

        if norm:
            bhwc_shapes = [out[-1].shape for out in outputs]
            if isinstance(n, int):
                norms = self.norms[-n:]
            else:
                norms = [self.norms[i] for i in n]
            outputs = [
                # ((B, C), (B, H*W, C))
                (norm(cls_token), norm(patches.flatten(1, 2)))
                for (cls_token, patches), norm in zip(outputs, norms)
            ]
            if reshape:
                outputs = [
                    (cls_token, patches.reshape(*bhwc))
                    for (cls_token, patches), bhwc in zip(outputs, bhwc_shapes)
                ]
        elif not reshape:
            # Force (B, N, C) format for patch tokens
            outputs = [(cls_token, patches.flatten(1, 2)) for cls_token, patches in outputs]

        class_tokens = [out[0] for out in outputs]
        outputs = [out[1] for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)
    
def get_convnext(size: str = "base", **kwargs) -> ConvNeXt:
    if size not in SIZE_DICT:
        raise ValueError(f"Unsupported ConvNeXt size: {size}. Supported sizes: {list(SIZE_DICT.keys())}")
    return ConvNeXt(**SIZE_DICT[size], **kwargs)