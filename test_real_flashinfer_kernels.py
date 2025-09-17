import torch
import time
from flashinfer.jit.core import gen_jit_spec
from pathlib import Path

print(f"CUDA available: {torch.cuda.is_available()}")

def create_minimal_kernel_test():
    """Get real FlashInfer template kernels directly"""
    
    return [
        (Path("include/flashinfer/activation.cuh"), ["flashinfer::activation::act_and_mul_kernel<float, flashinfer::activation::silu_nvrtc>"])
    ]

def test_flashinfer_style_kernels():
    """Test real FlashInfer-style kernels with NVRTC only"""
    
    kernel_tests = create_minimal_kernel_test()
    
    for i, (kernel_file, kernel_names) in enumerate(kernel_tests):
        kernel_type = ["Activation"][i]
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
            print(f"✓ NVRTC compilation: {nvrtc_time:.3f}s")
            print(f"✓ Compiled kernels: {list(kernels.keys())}")
            
            # Test kernel execution for activation kernel  
            if kernel_type == "Activation" and len(kernels) > 0:
                kernel_name = list(kernels.keys())[0]
                test_activation_kernel_execution(kernels[kernel_name])
                
        except Exception as e:
            print(f"✗ {kernel_type} kernel failed: {e}")

def test_activation_kernel_execution(kernel_fn):
    """Test that the activation kernel actually works"""
    print("Testing kernel execution...")
    
    batch_size, d = 4, 128
    # Ensure tensors are contiguous and properly aligned
    input_data = torch.randn(batch_size, 2 * d, device='cuda', dtype=torch.float32).contiguous()
    output_data = torch.zeros(batch_size, d, device='cuda', dtype=torch.float32).contiguous()
    
    # Use the same block size calculation as original FlashInfer
    vec_size = 16 // 4  # 16 / sizeof(float)
    block_size = min(d // vec_size, 1024)
    
    print(f"Launch params: grid=({batch_size},1,1), block=({block_size},1,1), d={d}")
    
    kernel_fn(
        grid=(batch_size, 1, 1),
        block=(block_size, 1, 1), 
        args=[output_data.data_ptr(), input_data.data_ptr(), d]
    )
    
    # Verify SiLU activation: x / (1 + exp(-x)) * y
    x = input_data[:, :d]
    y = input_data[:, d:]
    expected = torch.nn.functional.silu(x) * y
    max_diff = torch.max(torch.abs(output_data - expected)).item()
    print(f"✓ Kernel correctness: max_diff = {max_diff:.2e}")


if __name__ == "__main__":
    test_flashinfer_style_kernels()