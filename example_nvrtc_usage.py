#!/usr/bin/env python3

import torch
from flashinfer.jit.core import gen_jit_spec

def example_nvrtc_usage():
    """Example showing how to use NVRTC backend instead of ninja."""
    
    # Create a JitSpec as usual
    spec = gen_jit_spec(
        name="my_kernels",
        sources=["csrc/activation.cu"], 
        extra_cuda_cflags=["-O3", "--use_fast_math"]
    )
    
    # Traditional way (ninja-based, slower)
    # module = spec.build_and_load()
    
    # New way (NVRTC-based, faster) - specify which kernels you need
    kernels = spec.build_and_load_with_nvrtc(["silu_and_mul_kernel", "gelu_activation_kernel"])
    
    # kernels is now a dict mapping kernel names to callable functions
    print(f"Compiled kernels: {list(kernels.keys())}")
    
    # Use kernels directly with grid/block/args like PyTorch _compile_kernel
    if "silu_and_mul_kernel" in kernels:
        kernel_fn = kernels["silu_and_mul_kernel"]
        # kernel_fn(grid=(grid_x, grid_y, grid_z), block=(block_x, block_y, block_z), args=[tensors])

if __name__ == "__main__":
    example_nvrtc_usage()