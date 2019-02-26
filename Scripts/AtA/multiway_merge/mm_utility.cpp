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

// void compute_batches( uint64_t N, uint64_t *input, std::vector<uint64_t> *batch_offsets, uint64_t inputBatchSize, uint64_t *max_input_batch_size );
// {



// 	//search the middle element
// 	// uint64_t *val=std::upper_bound(input, input+N, input[N/2]);
// 	// uint64_t distance=thrust::distance(input,val);	
// 	// uint64_t idx=distance;
// 	// printf("\nInput/2: %lu, idx: %lu",input[N/2], idx);

	
// 	uint64_t numBatches=ceil(N*1.0/inputBatchSize*1.0);
// 	//given the input batch size and N, recompute the batch size (apporximate)
// 	uint64_t batchSizeApprox=N/numBatches;

// 	printf("\nNum batches: %lu, Approx. batch size: %lu",numBatches, batchSizeApprox);

// 	//split the input array based on the approximate batch size



// 	//the first offset is index 0
// 	batch_offsets->push_back(0);
// 	//-1 because the last offset is the end of the array N-1
// 	for (uint64_t i=0; i<numBatches-1; i++){
// 		uint64_t *val=std::upper_bound(input, input+N, input[(i+1)*batchSizeApprox]);
// 		uint64_t distance=thrust::distance(input,val);	
// 		uint64_t idx=distance;
// 		printf("\nInput: %lu, idx (upper bound): %lu",input[(i+1)*batchSizeApprox], idx);
// 		batch_offsets->push_back(idx);
// 	}

// 	batch_offsets->push_back(N);

// 	//split the search array based on the values in the input array
// 	//the search array is further split later into batches
// 	search_offsets->push_back(0);
	

// 	for (uint64_t i=0; i<numBatches-1; i++){
// 		uint64_t *val=std::upper_bound(search, search+N, input[(i+1)*batchSizeApprox]);
// 		uint64_t distance=thrust::distance(search,val);	
// 		uint64_t idx=distance;
// 		printf("\nInput: %lu, search: %lu, idx (upper bound): %lu",input[(i+1)*batchSizeApprox], search[idx], idx);
// 		search_offsets->push_back(idx);
// 	}	

// 	search_offsets->push_back(N);


// 	//compute max batch sizes for input and search arrays to be allocated
// 	uint64_t max_input=0;
// 	uint64_t max_search=0;

// 	for (uint64_t i=0; i<batch_offsets->size()-1; i++){
// 		printf("\nSize input batch: val: %lu", (*batch_offsets)[i+1]-(*batch_offsets)[i]);
// 		max_input=std::max(max_input,(*batch_offsets)[i+1]-(*batch_offsets)[i]);

// 		printf("\nSize search batch: %lu", (*search_offsets)[i+1]-(*search_offsets)[i]);
// 		max_search= std::max(max_search,(*search_offsets)[i+1]-(*search_offsets)[i]);
// 	}

// 	printf("\nMax input batch size: %lu, Max search batch size: %lu", max_input, max_search);

// 	*max_input_batch_size=max_input;
// 	*max_search_batch_size=max_search;




// }

void generate_k_sorted_sublists( uint64_t *base_ptr, uint64_t total_elements, unsigned int seed, uint8_t k )
{
    uint64_t batch_index  = 0;
    uint64_t num_sublists = 0;
    uint64_t inner_index  = 0;
    uint64_t batch_size   = 0;

	//rng for the keys
	std::mt19937 gen(seed); 
	//transform the randomly generated numbers into uniform distribution of ints
	std::uniform_int_distribution<uint64_t> dis(0, total_elements);

    num_sublists = k;
    batch_size   = total_elements / num_sublists;

	for( batch_index = 0; batch_index < num_sublists; ++batch_index )
	{
        for( inner_index = 0; inner_index < batch_size; ++inner_index )
            {
                base_ptr[ ( batch_index * batch_size ) + inner_index ] = dis( gen );
            }
	}

    for( batch_index = 0; batch_index < num_sublists; ++batch_index )
        {
            __gnu_parallel::sort( base_ptr + ( batch_index * batch_size ),
                                  base_ptr + ( batch_index * batch_size ) + batch_size
                                );
        }
}
