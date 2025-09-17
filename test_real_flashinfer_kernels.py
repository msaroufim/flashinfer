import torch
import time
from flashinfer.jit.core import gen_jit_spec
from pathlib import Path

print(f"CUDA available: {torch.cuda.is_available()}")

def create_minimal_kernel_test():
    """Create standalone versions of FlashInfer kernels without dependencies"""
    
    # Test 1: Simplified Page kernel (memory copy pattern)
    page_kernel = Path("/tmp/page_kernel.cu")
    page_kernel.write_text("""
#include <cuda_runtime.h>

template<typename DType, typename IdType>
__global__ void AppendPagedKVCacheDecodeKernel_simple(
    DType* __restrict__ paged_data,
    DType* __restrict__ key, 
    DType* __restrict__ value,
    IdType batch_size,
    IdType head_dim) {
    
    uint32_t tx = threadIdx.x;
    uint32_t batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size && tx < head_dim) {
        uint32_t offset = batch_idx * head_dim + tx;
        paged_data[offset] = key[offset] + value[offset];
    }
}

// Explicit instantiation for common types
template __global__ void AppendPagedKVCacheDecodeKernel_simple<float, int>(
    float*, float*, float*, int, int);
""")
    
    # Test 2: Simplified activation kernel 
    activation_kernel = Path("/tmp/activation_kernel.cu")
    activation_kernel.write_text("""
#include <cuda_runtime.h>

template<typename T>
__device__ float silu_activation(const float& x) {
    return x / (1.0f + __expf(-x));
}

template<typename T>
__global__ void act_and_mul_kernel_simple(
    T* __restrict__ out, 
    const T* __restrict__ input, 
    const int d) {
    
    const int token_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int stride = blockDim.x;
    const int offset = token_idx * 2 * d;
    
    for (int idx = thread_idx; idx < d; idx += stride) {
        float x = float(input[offset + idx]);
        float y = float(input[offset + d + idx]);
        out[token_idx * d + idx] = T(silu_activation<T>(x) * y);
    }
}

// Explicit instantiation
template __global__ void act_and_mul_kernel_simple<float>(float*, const float*, const int);
""")
    
    # Test 3: Simple quantization kernel
    quant_kernel = Path("/tmp/quant_kernel.cu") 
    quant_kernel.write_text("""
#include <cuda_runtime.h>

__global__ void PackBitsKernel_simple(
    bool* input, 
    uint8_t* output, 
    int64_t num_elements) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t byte_idx = idx / 8;
    int bit_pos = idx % 8;
    
    if (idx < num_elements) {
        if (input[idx]) {
            atomicOr(&output[byte_idx], (1 << bit_pos));
        }
    }
}
""")
    
    return [
        (page_kernel, ["AppendPagedKVCacheDecodeKernel_simple"]),
        (activation_kernel, ["act_and_mul_kernel_simple"]), 
        (quant_kernel, ["PackBitsKernel_simple"])
    ]

def test_flashinfer_style_kernels():
    """Test real FlashInfer-style kernels with NVRTC"""
    
    kernel_tests = create_minimal_kernel_test()
    
    for i, (kernel_file, kernel_names) in enumerate(kernel_tests):
        kernel_type = ["Page", "Activation", "Quantization"][i]
        print(f"\n=== Testing {kernel_type} Kernel ===")
        
        try:
            spec = gen_jit_spec(
                name=f"test_{kernel_type.lower()}_nvrtc",
                sources=[kernel_file],
                extra_cuda_cflags=[]
            )
            
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
            
            # Test kernel execution for page kernel
            if kernel_type == "Page" and "AppendPagedKVCacheDecodeKernel_simple" in kernels:
                test_page_kernel_execution(kernels["AppendPagedKVCacheDecodeKernel_simple"])
                
        except Exception as e:
            print(f"{kernel_type} kernel failed: {e}")

def test_page_kernel_execution(kernel_fn):
    """Test that the page kernel actually works"""
    print("Testing kernel execution...")
    
    batch_size, head_dim = 4, 128
    paged_data = torch.zeros(batch_size * head_dim, device='cuda')
    key = torch.randn(batch_size * head_dim, device='cuda')
    value = torch.randn(batch_size * head_dim, device='cuda')
    
    kernel_fn(
        grid=(batch_size, 1, 1),
        block=(head_dim, 1, 1), 
        args=[paged_data, key, value, batch_size, head_dim]
    )
    
    expected = key + value
    torch.testing.assert_close(paged_data, expected, rtol=1e-5, atol=1e-5)
    print("Kernel execution test passed!")

if __name__ == "__main__":
    test_flashinfer_style_kernels()