# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>2.5",
#   "mlx>=0.28.0",
#   "numpy>=2.3.2",
# ]
# ///

from pathlib import Path
import argparse
from typing import Any, Tuple


def traverse_dict_recursive(d, verbose: bool = False, level: int = 0):
    """Debug helper to print a nested (dict/list) structure with shapes."""
    if isinstance(d, list):
        for i, item in enumerate(d):
            if isinstance(item, (dict, list)):
                if verbose:
                    print(f"{'  ' * level}Index {i}")
                traverse_dict_recursive(item, verbose, level + 1)
            else:
                if verbose:
                    print(f"{'  ' * level}Index {i}")
    elif isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, (dict, list)):
                if verbose:
                    print(f"{'  ' * level}{key}")
                traverse_dict_recursive(value, verbose, level + 1)
            else:
                if verbose:
                    try:
                        shape = value.shape
                    except Exception:
                        shape = "?"
                    print(f"{'  ' * level}{key} - {shape}")


def traverse_recursive_paired(pt_d: Any, mlx_d: Any, verbose: bool = False, level: int = 0):
    """Align / transpose tensors in nested PyTorch->MLX weight structure.

    Handles differences in layout for 4D tensors (NCHW -> NHWC) when shapes match
    after permutation. Returns a structure mirroring the MLX parameter tree.
    """
    if isinstance(pt_d, list) and isinstance(mlx_d, list):
        for i, (pt_item, mlx_item) in enumerate(zip(pt_d, mlx_d)):
            if isinstance(mlx_item, (dict, list)):
                if verbose:
                    print(f"{'  ' * level}Index {i}")
                pt_d[i] = traverse_recursive_paired(pt_item, mlx_item, verbose, level + 1)
            else:
                if verbose:
                    print(f"{'  ' * level}Index {i}")
    elif isinstance(pt_d, dict) and isinstance(mlx_d, dict):
        if not set(pt_d.keys()).issubset(set(mlx_d.keys())):
            missing = set(pt_d.keys()) - set(mlx_d.keys())
            raise KeyError(f"Mismatched keys; extra in PT weights: {missing}")
        for key in list(pt_d.keys()):
            if isinstance(mlx_d[key], (dict, list)) and isinstance(pt_d[key], (dict, list)):
                if verbose:
                    print(f"{'  ' * level}{key}")
                pt_d[key] = traverse_recursive_paired(pt_d[key], mlx_d[key], verbose, level + 1)
            else:
                # NCHW -> NHWC adjustment (mlx Conv2d uses NHWC)
                if (
                    hasattr(pt_d[key], "shape")
                    and hasattr(mlx_d[key], "shape")
                    and getattr(pt_d[key], "ndim", -1) == 4
                    and getattr(mlx_d[key], "ndim", -1) == 4
                ):
                    pt_shape = pt_d[key].shape
                    mlx_shape = mlx_d[key].shape
                    if (pt_shape[0], pt_shape[2], pt_shape[3], pt_shape[1]) == mlx_shape:
                        pt_d[key] = pt_d[key].transpose(0, 2, 3, 1)  # NCHW -> NHWC
                if verbose:
                    try:
                        print(f"{'  ' * level}{key} - PT {pt_d[key].shape} | MLX {mlx_d[key].shape}")
                    except Exception:
                        print(f"{'  ' * level}{key}")
    elif isinstance(pt_d, list) and isinstance(mlx_d, dict):
        # MLX Sequential modules wrap list under a single dict key (e.g. {"layers": [...]})
        if list(mlx_d.keys()) == ["layers"]:
            pt_d = {"layers": pt_d}
            return traverse_recursive_paired(pt_d, mlx_d, verbose, level)
    else:
        raise TypeError("Mismatched container types between PyTorch and MLX structures.")
    return pt_d


def infer_model_from_filename(path: Path) -> Tuple[str, str]:
    name = path.name.lower()
    if "convnext" in name:
        for size in ["tiny", "small", "base", "large"]:
            if size in name:
                return "convnext", size
    elif "vit" in name:
        for size in ["vits", "vitb", "vitl", "vith"]:
            if size in name:
                return "vit", size
    raise ValueError("Could not infer model type/size from filename; specify --model <type> --size <size>.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a PyTorch DINOv3 checkpoint to MLX safetensors.")
    p.add_argument("checkpoint", type=Path, help="Path to the PyTorch .pth checkpoint.")
    p.add_argument("--out-dir", type=Path, default=None, help="Optional directory to write the output .safetensors file.")
    p.add_argument(
        "--model",
        choices=["convnext", "vit"],
        help="Model type. If omitted will attempt to infer from filename.",
        default=None,
    )
    p.add_argument(
        "--size",
        help="Model size. If omitted will attempt to infer from filename.",
        default=None,
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose structure + shape logging.")
    return p.parse_args()


def convert(checkpoint_path: Path, out_dir: Path | None, model_type: str, model_size: str, verbose: bool = False) -> Path:
    import torch
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from dinov3.models import get_convnext

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if verbose:
        print(f"Loading PyTorch checkpoint: {checkpoint_path}")
    weights = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(weights, dict):
        raise ValueError("Expected a state_dict-like object (dict).")

    # Convert tensors to MLX arrays
    for k, v in weights.items():
        if isinstance(v, torch.Tensor):
            weights[k] = mx.array(v.detach().cpu().numpy())
        else:
            weights[k] = v

    # Rebuild nested tree structure expected by model
    weights = tree_unflatten(weights)

    if model_type == "convnext":
        model = get_convnext(model_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Align shapes / transpose where needed
    weights = traverse_recursive_paired(weights, model.parameters(), verbose=verbose)

    # Load into model and flatten again for saving
    model.update(weights)
    params_dict = tree_flatten(model.parameters(), destination={})

    # Remove non-weight meta keys
    for drop_key in ("head", "embed_dims"):
        params_dict.pop(drop_key, None)

    # Determine output path
    if out_dir is None:
        output_path = checkpoint_path.with_suffix(".safetensors")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / checkpoint_path.with_suffix(".safetensors").name

    if verbose:
        print(f"Writing MLX safetensors: {output_path}")
    mx.save_safetensors(str(output_path), params_dict)
    return output_path


if __name__ == "__main__":  # pragma: no cover
    args = parse_args()
    model_type, model_size = args.model or infer_model_from_filename(args.checkpoint)
    output = convert(args.checkpoint, args.out_dir, model_type, model_size, verbose=args.verbose)

