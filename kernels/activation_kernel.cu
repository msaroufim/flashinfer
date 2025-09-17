extern "C" {
__device__ float silu_activation(const float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void act_and_mul_kernel_simple(
    float* __restrict__ out, 
    const float* __restrict__ input, 
    const int d) {
    
    const int token_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int stride = blockDim.x;
    const int offset = token_idx * 2 * d;
    
    for (int idx = thread_idx; idx < d; idx += stride) {
        float x = input[offset + idx];
        float y = input[offset + d + idx];
        out[token_idx * d + idx] = silu_activation(x) * y;
    }
}
}