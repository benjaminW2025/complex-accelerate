"""Lazy compilation keeps CUDA setup out of imports and CLI help."""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_ops():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if not torch.cuda.is_available() or CUDA_HOME is None:
        raise RuntimeError("A CUDA-enabled PyTorch installation, NVIDIA GPU, and CUDA toolkit are required.")
    return load(
        name="complex_accelerate_ops",
        sources=[str(Path(__file__).with_name("ops.cu"))],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--fmad=false"],
    )
