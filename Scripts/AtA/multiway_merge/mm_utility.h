#ifndef MULTIWAY_UTILITY_H_INCLUDED 
#define MULTIWAY_UTILITY_H_INCLUDED 

void find_pivot_vectors( uint64_t *input, 
						 std::vector<std::vector<uint64_t>> *start_vectors, 
						 std::vector<std::vector<uint64_t>> *end_vectors, 
						 std::vector<uint64_t> *first_sublist_ends, 
						 std::vector<uint64_t *> *list_begin_ptrs, 
						 uint64_t sublist_size );
std::vector<uint64_t*> *generate_k_sorted_sublists( uint64_t *base_ptr, uint64_t total_elements, unsigned int seed, uint16_t k );
void compute_batches( uint64_t N, uint64_t *input, std::vector<uint64_t> *batch_offsets, uint64_t inputBatchSize, uint64_t sublist_size );

#endif // MULTIWAY_UTILITY_H_INCLUDED 
