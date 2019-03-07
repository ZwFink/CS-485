#ifndef MM_GPU_HH_INCLUDED
#define MM_GPU_HH_INCLUDED 

void warm_up_gpu( int device );
cudaError_t create_streams( cudaStream_t *streams, const int num_streams );
void copy_to_device_buffer( uint64_t *input, uint64_t *pinned_host,
                            uint64_t *device_ptr, cudaStream_t stream,
                            uint64_t start_index,
                            uint64_t end_index, uint64_t BATCH_SIZE,
                            int thread_id, int stream_id
                          );

#endif // MM_GPU_HH_INCLUDED

