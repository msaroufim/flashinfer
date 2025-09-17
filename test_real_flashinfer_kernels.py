import torch
import time
from flashinfer.jit.core import gen_jit_spec
from pathlib import Path

print(f"CUDA available: {torch.cuda.is_available()}")

def create_minimal_kernel_test():
    """Get FlashInfer-style kernel files for testing"""
    
    return [
        (Path("kernels/page_kernel.cu"), ["AppendPagedKVCacheDecodeKernel_simple"]),
        (Path("kernels/activation_kernel.cu"), ["act_and_mul_kernel_simple"]), 
        (Path("kernels/quant_kernel.cu"), ["PackBitsKernel_simple"])
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