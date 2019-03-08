#include <parallel/algorithm>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <random>
#include <algorithm> 
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <queue>
#include <iomanip>
#include <set>
#include <thread>
#include <utility>

// // thrust inclusions
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h> //for streams for thrust (added with Thrust v1.8)
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include "omp.h"
#include "mm_utility.h"
	

void find_pivot_vectors( uint64_t *input, 
						 std::vector<std::vector<uint64_t>> *start_vectors, 
						 std::vector<std::vector<uint64_t>> *end_vectors, 
						 std::vector<uint64_t> *first_sublist_ends, 
						 std::vector<uint64_t *> *list_begin_ptrs, 
						 uint64_t sublist_size )
{
	uint64_t index, piv_index, curr_end_index;
	uint64_t *temp_ptr = nullptr;
	uint64_t pivot_val;
	std::vector<uint64_t> temp_start;
	std::vector<uint64_t> temp_end;	

	for( index = 1; index < list_begin_ptrs->size(); ++index )
    {
		temp_start.clear();
		temp_end.clear();		

        for( piv_index = 0; piv_index < first_sublist_ends->size(); ++piv_index )
        {
            pivot_val = (*first_sublist_ends)[ piv_index ];
    
            temp_ptr = std::upper_bound( 
                                         input + *((*list_begin_ptrs)[ index ]), 
                                         input + *((*list_begin_ptrs)[ index ]) + sublist_size, 
                                         pivot_val 
                                       );

            curr_end_index = thrust::distance( input, temp_ptr );

            temp_end[ piv_index ] = curr_end_index - 1;

            if( piv_index == 0 )
            {
                temp_start[ piv_index ] = *((*list_begin_ptrs)[ index ]);
            }

            else
            {
                temp_start[ piv_index ] = temp_end[ piv_index - 1 ] + 1;
            }
        }

        start_vectors->push_back( temp_start );
        end_vectors->push_back( temp_end );
    }
}


void compute_batches( uint64_t N, uint64_t *input, std::vector<uint64_t> *batch_offsets, uint64_t inputBatchSize )
{
    uint64_t index = 0;

	 uint64_t numBatches = ceil( N * 1.0 / inputBatchSize * 1.0 );
	 //given the input batch size and N, recompute the batch size (apporximate)
	 uint64_t batchSizeApprox = N / numBatches;

	 printf( "\nNum batches: %lu, Approx. batch size: %lu", numBatches, batchSizeApprox );

	 //split the input array based on the approximate batch size

	 // the first offset is index 0
	 batch_offsets->push_back( index );

	 // -1 because the last pivot is the end of the array N-1
	 for( index = batchSizeApprox - 1; index < numBatches - 1 ; index = index + batchSizeApprox )
     {
        batch_offsets->push_back( index );
	 }

	batch_offsets->push_back( N - 1 );
}


std::vector<uint64_t*> *generate_k_sorted_sublists( uint64_t *base_ptr, uint64_t total_elements, unsigned int seed, uint16_t k )
{
    uint64_t outer_index  = 0;
    uint64_t batch_size   = 0;
    uint64_t batch_index  = 0;
    uint64_t total_index  = 0;

    uint64_t elements_per_list = total_elements / k;

    std::vector<uint64_t*> *list_ptrs = new std::vector<uint64_t*>;
    list_ptrs->reserve( k );

	//rng for the keys
	std::mt19937 gen( seed ); 
	//transform the randomly generated numbers into uniform distribution of ints
	std::uniform_int_distribution<uint64_t> dis(0, total_elements );

	for( outer_index = 0; outer_index < total_elements; ++outer_index )
	{
        base_ptr[ outer_index ] = dis( gen );
    }

    for( batch_index = 0; batch_index < k; ++batch_index )
        {
            __gnu_parallel::sort( base_ptr + ( batch_index * elements_per_list ),
                                  base_ptr + ( batch_index * elements_per_list ) + elements_per_list - 1
                                );
            list_ptrs->push_back( base_ptr + ( batch_index * elements_per_list ) );
        }
    return list_ptrs;
}



