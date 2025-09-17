#include <cuda/std/cstdint>

extern "C" {
__global__ void PackBitsKernel_simple(
    bool* input, 
    unsigned char* output, 
    cuda::std::int64_t num_elements) {
    
    cuda::std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    cuda::std::int64_t byte_idx = idx / 8;
    int bit_pos = idx % 8;
    
    if (idx < num_elements) {
        if (input[idx]) {
            atomicOr((unsigned int*)&output[byte_idx], (unsigned int)(1 << bit_pos));
        }
    }
}
}