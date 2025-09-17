#!/usr/bin/env python3

import time
import torch
from flashinfer.jit.core import gen_jit_spec
from pathlib import Path

def benchmark_compilation():
    """
    Benchmark ninja vs NVRTC compilation on actual FlashInfer kernels.
    Run this on a machine with CUDA GPU.
    """
    
    # Find actual FlashInfer CUDA source files
    csrc_dir = Path("csrc")
    if not csrc_dir.exists():
        print("Run from flashinfer root directory")
        return
    
    # Use a real kernel file
    test_files = [
        csrc_dir / "activation.cu",
        csrc_dir / "norm.cu", 
        csrc_dir / "rope.cu"
    ]
    
    existing_files = [f for f in test_files if f.exists()]
    if not existing_files:
        print("No test files found")
        return
    
    test_file = existing_files[0]
    print(f"Testing with: {test_file}")
    
    # Test ninja compilation
    print("Ninja compilation...")
    start = time.time()
    spec_ninja = gen_jit_spec(
        name="test_ninja",
        sources=[test_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"]
    )
    spec_ninja.build(verbose=False)
    ninja_time = time.time() - start
    
    # Test NVRTC compilation  
    print("NVRTC compilation...")
    
    # You need to specify actual kernel names from the source file
    # For example, from activation.cu: ["silu_and_mul_kernel", "gelu_kernel"]
    kernel_names = ["your_actual_kernel_names_here"]  # TODO: Replace with real kernel names
    
    start = time.time()
    spec_nvrtc = gen_jit_spec(
        name="test_nvrtc", 
        sources=[test_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"]
    )
    kernels = spec_nvrtc.build_with_nvrtc(kernel_names)
    nvrtc_time = time.time() - start
    
    print(f"\nResults:")
    print(f"Ninja: {ninja_time:.3f}s")
    print(f"NVRTC: {nvrtc_time:.3f}s") 
    print(f"Speedup: {ninja_time/nvrtc_time:.1f}x")
    print(f"NVRTC compiled {len(kernels)} kernels: {list(kernels.keys())}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
    else:
        benchmark_compilation()