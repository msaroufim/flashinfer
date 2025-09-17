#!/usr/bin/env python3
"""Simple test to debug the NVRTC kernel issue"""
import torch
import os
from torch.cuda import _compile_kernel

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Very simple kernel without templates first
SIMPLE_KERNEL = """
extern "C" __global__ void simple_silu_kernel(
    float* __restrict__ out,
    const float* __restrict__ input,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float y = input[idx + n];
        out[idx] = (x / (1.0f + __expf(-x))) * y;
    }
}
"""

def test_simple_kernel():
    print("Testing simple NVRTC kernel...")
    
    # Compile simple kernel
    kernel_fn = _compile_kernel(SIMPLE_KERNEL, "simple_silu_kernel")
    print("✓ Simple kernel compiled")
    
    # Test with minimal data
    n = 32
    input_data = torch.randn(2 * n, device='cuda', dtype=torch.float32)
    output_data = torch.zeros(n, device='cuda', dtype=torch.float32)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output_data.shape}")
    
    try:
        kernel_fn(
            grid=(1, 1, 1),
            block=(n, 1, 1),
            args=[output_data.data_ptr(), input_data.data_ptr(), n]
        )
        torch.cuda.synchronize()
        print("✓ Simple kernel executed successfully")
        
        # Verify
        x = input_data[:n]
        y = input_data[n:]
        expected = torch.nn.functional.silu(x) * y
        max_diff = torch.max(torch.abs(output_data - expected)).item()
        print(f"✓ Correctness: max_diff = {max_diff:.2e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Simple kernel failed: {e}")
        return False

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_simple_kernel()
    else:
        print("CUDA not available")