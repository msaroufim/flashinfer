import torch
import time
from flashinfer.jit.core import gen_jit_spec

print(f"CUDA available: {torch.cuda.is_available()}")

spec = gen_jit_spec(
    name="test_nvrtc",
    sources=["csrc/nv_internal/tensorrt_llm/kernels/delayStream.cu"],
    extra_cuda_cflags=["-O3"]
)

kernel_names = ["delayStreamKernel"]

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