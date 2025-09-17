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
        out[idx] = input[idx] * 2.0f;  // Just multiply by 2 to test basic memory access
    }
}
"""

# Even simpler test kernel
MINIMAL_KERNEL = """
extern "C" __global__ void minimal_kernel(float* out, float* in, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}
"""

def test_minimal_kernel():
    print("Testing minimal kernel...")
    
    try:
        kernel_fn = _compile_kernel(MINIMAL_KERNEL, "minimal_kernel")
        print("✓ Minimal kernel compiled")
        
        n = 8
        input_data = torch.ones(n, device='cuda', dtype=torch.float32)
        output_data = torch.zeros(n, device='cuda', dtype=torch.float32)
        
        print(f"Before: input={input_data}, output={output_data}")
        
        kernel_fn(
            grid=(1, 1, 1),
            block=(n, 1, 1),
            args=[output_data.data_ptr(), input_data.data_ptr(), n]
        )
        torch.cuda.synchronize()
        
        print(f"After: output={output_data}")
        print("✓ Minimal kernel worked!")
        return True
        
    except Exception as e:
        print(f"✗ Minimal kernel failed: {e}")
        return False

def test_simple_kernel():
    print("Testing simple NVRTC kernel...")
    
    # Compile simple kernel
    kernel_fn = _compile_kernel(SIMPLE_KERNEL, "simple_silu_kernel")
    print("✓ Simple kernel compiled")
    
    # Test with minimal data
    n = 8
    input_data = torch.ones(n, device='cuda', dtype=torch.float32)
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
        
        expected = input_data * 2.0
        max_diff = torch.max(torch.abs(output_data - expected)).item()
        print(f"✓ Correctness: max_diff = {max_diff:.2e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Simple kernel failed: {e}")
        return False

if __name__ == "__main__":
    if torch.cuda.is_available():
        if test_minimal_kernel():
            test_simple_kernel()
        else:
            print("Basic NVRTC setup has issues")
    else:
        print("CUDA not available")