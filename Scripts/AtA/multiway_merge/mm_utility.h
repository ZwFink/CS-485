#ifndef MULTIWAY_UTILITY_H_INCLUDED 
#define MULTIWAY_UTILITY_H_INCLUDED 

void warm_up_gpu( int device );
void generate_k_sorted_sublists( uint64_t *base_ptr, uint64_t total_elements, unsigned int seed, uint8_t k );

#endif // MULTIWAY_UTILITY_H_INCLUDED 
