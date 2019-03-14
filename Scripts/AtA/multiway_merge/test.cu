// c++ inclusions
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <string.h>
#include <fstream>
#include <math.h>
#include <iostream>
#include <string>
#include <queue>
#include <iomanip>
#include <set>
#include <algorithm> 
#include <thread>
#include <cstdint>
#include <utility>
#include <vector>

// thrust inclusions
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h> //for streams for thrust (added with Thrust v1.8)
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

// custom inclusions
#include "omp.h"
#include "mm_cpu.h"
#include "mm_gpu.h"
#include "mm_utility.h"

int test_copy_to_pinned_buffer( uint64_t *input, uint64_t N, uint64_t K )
{
    uint64_t BATCH_SIZE = 1000000;
    uint64_t *pinned_buff = nullptr;
    cudaMallocHost( (void**) &pinned_buff, sizeof( uint64_t ) * BATCH_SIZE * 8 );

    uint64_t index = 0;
    for( index = 0; index < K; index++ )
        {

        }
}
copied_this_round += copy_to_pinned_buffer( input,
                                                                              input_to_gpu_pinned,
                                                                              start_index_gpu,
                                                                              end_index_gpu,
                                                                              thread_id,
                                                                              BATCH_SIZE
                                                                            );

int main( void )
{

    uint64_t N = 1000000000;
    int seed   = 41;
    uint64_t K = 31250;
       

    uint64_t *input      = ( uint64_t * ) malloc( sizeof( uint64_t ) * N );
    uint64_t *output_arr = (uint64_t *) malloc( sizeof( uint64_t ) * N );


    list_begin_ptrs = *generate_k_sorted_sublists( input, N, seed, K );

    assert( test_copy_to_pinned_buffer( input, N, K ) != 0 );

    free( input );
    free( output_arr );
    return EXIT_SUCCESS;
}
