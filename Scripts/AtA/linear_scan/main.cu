
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
#include "ls_gpu.h"
#include "ls_cpu.h"
#include "ls_utility.h"

int main( int argc, char **argv )
{
    omp_set_num_threads( NTHREADS );
    omp_set_nested(1);

    args commandline_args;

    int args_success = parse_args( &commandline_args, argc, &argv );

    if( !args_success )
        {
            report_args_failure();

            return EXIT_FAILURE;
        }

    uint64_t total_num_batches = commandline_args.N / commandline_args.batch_size;
    uint64_t num_cpu_batches   = total_num_batches * commandline_args.cpu_frac;
    uint64_t num_gpu_batches   = total_num_batches - num_cpu_batches;

    std::vector<uint64_t> batch_indices;
    batch_indices.reserve( total_num_batches );

	////////////////
	//Turn on gpu
	printf("\nTurning on the GPU...\n");
	warm_up_gpu( 0 );

    uint64_t *data = (uint64_t*) malloc( sizeof( uint64_t ) * commandline_args.N );


    free( data );
    return EXIT_SUCCESS;
}