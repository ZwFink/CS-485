#ifndef MULTIWAY_UTILITY_H_INCLUDED 
#define MULTIWAY_UTILITY_H_INCLUDED 

void compute_batches( uint64_t N, uint64_t *input, uint64_t *search, std::vector<uint64_t> *batch_offsets, std::vector<uint64_t> *search_offsets, uint64_t inputBatchSize, uint64_t *max_input_batch_size, uint64_t *max_search_batch_size );
void warm_up_gpu( int device );

#endif // MULTIWAY_UTILITY_H_INCLUDED 
