#ifndef MULTIWAY_UTILITY_H_INCLUDED 
#define MULTIWAY_UTILITY_H_INCLUDED 

void generate_k_sorted_sublists( uint64_t *base_ptr, uint64_t total_elements, unsigned int seed, uint16_t k );
void compute_batches( uint64_t N, uint64_t *input, std::vector<uint64_t> *batch_offsets, uint64_t inputBatchSize )
void compute_offsets( uint64_t *input, std::vector<uint64_t> *batch_offsets, 
                                std::vector<uint64_t> *offset_list, uint64_t batch_index, 
                                                            uint16_t k, uint64_t sublist_size );
void set_beginning_of_offsets( std::vector<uint64_t> begin_offset_list, uint64_t sublist_size, uint16_t k );
void get_offset_beginning( std::vector<uint64_t> offset_list, std::vector<uint64_t> begin_list );

#endif // MULTIWAY_UTILITY_H_INCLUDED 
