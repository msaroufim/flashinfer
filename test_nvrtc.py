import torch
import time
from flashinfer.jit.core import gen_jit_spec
from pathlib import Path

print(f"CUDA available: {torch.cuda.is_available()}")

# Create a simple standalone kernel file that doesn't need external headers
simple_kernel = Path("/tmp/simple_test_kernel.cu")
simple_kernel.write_text("""
extern "C" {
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_mul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}
}
""")

spec = gen_jit_spec(
    name="test_nvrtc",
    sources=[simple_kernel],
    extra_cuda_cflags=["-O3"]
)

kernel_names = ["vector_add", "vector_mul"]

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

# Test that a kernel actually works
if "vector_add" in kernels:
    print("Testing kernel execution...")
    n = 1024
    a = torch.randn(n, device='cuda')
    b = torch.randn(n, device='cuda')
    c = torch.zeros(n, device='cuda')
    
    kernel = kernels["vector_add"]
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    kernel(grid=(blocks_per_grid, 1, 1), block=(threads_per_block, 1, 1), args=[a, b, c, n])
    
    expected = a + b
    torch.testing.assert_close(c, expected, rtol=1e-5, atol=1e-5)
    print("Kernel execution test passed!")