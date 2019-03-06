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

void compute_batches( uint64_t N, uint64_t *input, std::vector<uint64_t> *batch_offsets, uint64_t inputBatchSize )
{
    uint64_t index = 0, offset_index = 0;
    uint64_t *val  = nullptr;

	 uint64_t numBatches = ceil( N * 1.0 / inputBatchSize * 1.0 );
	 //given the input batch size and N, recompute the batch size (apporximate)
	 uint64_t batchSizeApprox = N / numBatches;

	 printf( "\nNum batches: %lu, Approx. batch size: %lu", numBatches, batchSizeApprox );

	 //split the input array based on the approximate batch size

	 // the first offset is index 0
	 batch_offsets->push_back( offset_index );

	 // -1 because the last offset is the end of the array N-1
	 for( index = 0; index < numBatches - 1 ; index++ )
    {
	     val = std::upper_bound( input, input + N, input[ ( index + 1 ) * batchSizeApprox ] );
		
        offset_index  = std::distance( input, val );	
		
        batch_offsets->push_back( offset_index );
	 }

	batch_offsets->push_back( N );
}

// compute_offsets( input, first_sublist_offsets, &offset_list_cpu, cpu_index, K, sublist_size ); 

void compute_offsets( uint64_t *input, std::vector<uint64_t> *batch_offsets, 
                                std::vector<uint64_t> *offset_list, uint64_t batch_index, 
                                                            uint16_t k, uint64_t sublist_size )
{
    uint64_t index = 0, offset_index = 0, start_index = 0;
    uint64_t *pivot_val = nullptr;
    uint64_t *val       = nullptr;


    offset_list->push_back( (*batch_offsets)[ batch_index ] ); 

    // find the pivot value
    pivot_val = &input[ (*offset_list)[ 0 ] ];

    // Now find the remaining offsets in each sublist
    // starting from the second sublist since we already
    // found the offset for the first sublist
    for( index = 1; index < k; index++ )
    {
        start_index = index * sublist_size;
        
        val = std::upper_bound( 
                                input + start_index, 
                                input + ( start_index + (sublist_size - 1) ), 
                                *pivot_val 
                              );

	    offset_index = thrust::distance( input, val );

  	    offset_list->push_back( offset_index );
        
        printf("\nInput: %lu, offset_index (upper bound): %lu", 
                                    *val, offset_index );
    }
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


// set_beginning_of_offsets( &offset_begin_cpu, sublist_size, K );
void set_beginning_of_offsets( std::vector<uint64_t> *begin_offset_list, uint64_t sublist_size, uint16_t k )
{
    uint64_t index = 0;

    for( index = 0; index < k; index++ )
    {
        begin_offset_list->push_back( index * sublist_size );
    }
}


// get_offset_beginning( offset_list_cpu, &offset_begin_gpu );
void get_offset_beginning( std::vector<uint64_t> *offset_list, std::vector<uint64_t> *begin_list )
{
    uint64_t index;

    for( index = 0; index < offset_list->size(); index++ )
    {
        begin_list->push_back( (*offset_list)[ index ] );
    }
} 


// start_index = get_start_index( offset_list_cpu, K );
uint64_t get_start_index( std::vector<uint64_t> in_list, uint64_t k, uint64_t sublist_size )
{
    uint64_t size = 0, index = 0;

    for( index = 0; index < k; index++ )
    {   
        size = size + ( in_list[ index ] - ( index * sublist_size ) );
    }

    return size + k;
}




