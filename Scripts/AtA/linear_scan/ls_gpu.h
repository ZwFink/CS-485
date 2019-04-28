#ifndef LS_GPU_INCLUDED
#define LS_GPU_INCLUDED

void warm_up_gpu( int device );
cudaError_t create_streams( cudaStream_t *streams, const int num_streams );
__global__ void kernel_max( unsigned long long int *data, unsigned long long int *max_location, unsigned long long int *batch_size );
#endif // LS_GPU_INCLUDED
