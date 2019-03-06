#ifndef MM_GPU_HH_INCLUDED
#define MM_GPU_HH_INCLUDED 

void warm_up_gpu( int device );
cudaError_t create_streams( cudaStream_t *streams, const int num_streams );

#endif // MM_GPU_HH_INCLUDED

