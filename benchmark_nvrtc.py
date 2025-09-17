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
    
    # Use a simple kernel file that doesn't need complex headers
    test_file = csrc_dir / "nv_internal/tensorrt_llm/kernels/delayStream.cu"
    kernel_names = ["delayStreamKernel"]
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
        
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