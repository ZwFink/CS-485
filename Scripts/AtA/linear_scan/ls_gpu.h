#ifndef LS_GPU_INCLUDED
#define LS_GPU_INCLUDED

void warm_up_gpu( int device );
cudaError_t create_streams( cudaStream_t *streams, const int num_streams );
#endif // LS_GPU_INCLUDED
