#ifndef MULTIWAY_UTILITY_H_INCLUDED 
#define MULTIWAY_UTILITY_H_INCLUDED 

void generate_k_sorted_sublists( uint64_t *base_ptr, uint64_t total_elements, unsigned int seed, uint16_t k );
void find_list_breakpoints( uint64_t *input, uint64_t num_elements, uint64_t **breakpoints, uint16_t k, uint64_t batch_size );
uint64_t find_breakpoint( uint64_t *input, uint64_t start_index, uint64_t sublist_size, uint64_t pivot_val );
#endif // MULTIWAY_UTILITY_H_INCLUDED 
