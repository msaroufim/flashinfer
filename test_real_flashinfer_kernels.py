import torch
import time
from flashinfer.jit.core import gen_jit_spec
from pathlib import Path

print(f"CUDA available: {torch.cuda.is_available()}")

def create_minimal_kernel_test():
    """Get real FlashInfer template kernels directly"""
    
    return [
        (Path("include/flashinfer/quantization.cuh"), ["PackBitsKernel_nvrtc"])
    ]

def test_flashinfer_style_kernels():
    """Test real FlashInfer-style kernels with NVRTC"""
    
    kernel_tests = create_minimal_kernel_test()
    
    for i, (kernel_file, kernel_names) in enumerate(kernel_tests):
        kernel_type = ["Quantization"][i]
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
            
            # Test kernel execution for activation kernel
            if kernel_type == "Activation" and "act_and_mul_kernel_float_silu" in kernels:
                test_activation_kernel_execution(kernels["act_and_mul_kernel_float_silu"])
                
        except Exception as e:
            print(f"{kernel_type} kernel failed: {e}")

def test_activation_kernel_execution(kernel_fn):
    """Test that the activation kernel actually works"""
    print("Testing kernel execution...")
    
    batch_size, d = 4, 128
    input_data = torch.randn(batch_size, 2 * d, device='cuda')
    output_data = torch.zeros(batch_size, d, device='cuda')
    
    kernel_fn(
        grid=(batch_size, 1, 1),
        block=(d, 1, 1), 
        args=[output_data, input_data, d]
    )
    
    # Verify SiLU activation: x / (1 + exp(-x)) * y
    x = input_data[:, :d]
    y = input_data[:, d:]
    expected = (x / (1 + torch.exp(-x))) * y
    torch.testing.assert_close(output_data, expected, rtol=1e-4, atol=1e-4)
    print("Kernel execution test passed!")

if __name__ == "__main__":
    test_flashinfer_style_kernels()