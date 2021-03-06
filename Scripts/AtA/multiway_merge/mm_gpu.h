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
uint64_t get_gpu_output_index( const std::vector<std::vector<uint64_t>> *end_vectors,
                               const uint64_t numCPUBatches, const int num_threads
                               );
void copy_to_device_buffer( uint64_t *pinned, uint64_t *dev_ptr,
                            cudaStream_t stream, uint64_t to_transfer, int stream_id,
                            uint64_t batch_size
                          );

uint64_t copy_to_pinned_buffer( uint64_t *input, uint64_t *pinned_host,
                                uint64_t start_index, uint64_t to_copy,
                                uint64_t stream_id, uint64_t BATCH_SIZE
                                );
void copy_from_device_buffer( uint64_t *output_buffer,
                              uint64_t *pinned_buff, 
                              uint64_t *dev_ptr,
                              cudaStream_t stream,
                              uint64_t BATCH_SIZE,
                              int thread_id, int stream_id,
                              uint64_t num_merged
                              );
#endif // MM_GPU_HH_INCLUDED

