
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

    time_data data_creation;
    time_data cpu_only;
    time_data gpu_only;
    time_data total_time;

    std::vector<uint64_t> batch_indices;
    batch_indices.reserve( total_num_batches );

	////////////////
	//Turn on gpu
	printf("\nTurning on the GPU...\n");
	warm_up_gpu( 0 );

    uint64_t *data = (uint64_t*) malloc( sizeof( uint64_t ) * commandline_args.N );

    // report data
	printf( "\nSeed for random number generator: %d", commandline_args.seed );
	printf( "\nInput size: %lu", commandline_args.N );
	printf( "\nBatch size: %lu", commandline_args.batch_size );
    printf( "\nTotal number of batches: %lu\n", total_num_batches );
    printf( "\nFraction of batches sent to the CPU: %.2f\n", commandline_args.cpu_frac );
    printf( "Number of CPU Batches: %lu\n", num_cpu_batches );
    printf( "Number of GPU Batches: %lu\n", num_gpu_batches );

    assert( num_cpu_batches + num_gpu_batches == total_num_batches );

    data_creation.start = omp_get_wtime();
    generate_dataset( data, commandline_args.N, commandline_args.seed );
    data_creation.end = omp_get_wtime();

    printf( "Time to create dataset: %f\n", get_elapsed( &data_creation ) );


    free( data );
    return EXIT_SUCCESS;
}