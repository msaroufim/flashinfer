extern "C" {
__global__ void AppendPagedKVCacheDecodeKernel_simple(
    float* __restrict__ paged_data,
    float* __restrict__ key, 
    float* __restrict__ value,
    int batch_size,
    int head_dim) {
    
    int tx = threadIdx.x;
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size && tx < head_dim) {
        int offset = batch_idx * head_dim + tx;
        paged_data[offset] = key[offset] + value[offset];
    }
}
}