import torch
import time
from flashinfer.jit.core import gen_jit_spec
from pathlib import Path

print(f"CUDA available: {torch.cuda.is_available()}")

# Test with real FlashInfer activation kernels
print("=== Testing FlashInfer Activation Kernels ===")

try:
    spec = gen_jit_spec(
        name="test_activation_nvrtc",
        sources=["csrc/activation.cu"],
        extra_cuda_cflags=[]
    )
    
    # These kernels are templated, but let's see what happens
    kernel_names = ["act_and_mul_kernel"]  # From the grep output above
    
    print("Testing NVRTC compilation...")
    start = time.time()
    kernels = spec.build_with_nvrtc(kernel_names)
    nvrtc_time = time.time() - start
    print(f"NVRTC: {nvrtc_time:.3f}s, kernels: {list(kernels.keys())}")
    
except Exception as e:
    print(f"Activation kernel failed with NVRTC: {e}")

# Test with quantization kernels (simpler)
print("\n=== Testing FlashInfer Quantization Kernels ===")

try:
    spec = gen_jit_spec(
        name="test_quant_nvrtc", 
        sources=["csrc/quantization.cu"],
        extra_cuda_cflags=[]
    )
    
    kernel_names = ["PackBitsKernel", "SegmentPackBitsKernel"]
    
    print("Testing NVRTC compilation...")
    start = time.time()
    kernels = spec.build_with_nvrtc(kernel_names)
    nvrtc_time = time.time() - start
    print(f"NVRTC: {nvrtc_time:.3f}s, kernels: {list(kernels.keys())}")
    
    print("Testing ninja compilation...")
    start = time.time()
    spec.build(verbose=False)
    ninja_time = time.time() - start
    print(f"Ninja: {ninja_time:.3f}s")
    print(f"Speedup: {ninja_time/nvrtc_time:.1f}x")
    
except Exception as e:
    print(f"Quantization kernel failed: {e}")

# Test with norm kernels 
print("\n=== Testing FlashInfer Norm Kernels ===")

try:
    spec = gen_jit_spec(
        name="test_norm_nvrtc",
        sources=["csrc/norm.cu"], 
        extra_cuda_cflags=[]
    )
    
    # Look for actual kernel names in norm
    with open("csrc/norm.cu") as f:
        content = f.read()
        print("Norm.cu preview:", content[:500])
    
except Exception as e:
    print(f"Norm kernel test failed: {e}")