
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

    cpu_only.start   = omp_get_wtime();
    gpu_only.start   = omp_get_wtime();
    total_time.start = omp_get_wtime();

    // 0'th item is maximum for CPU, each consecutive is for the maximum for each stream
    uint64_t maximums[ STREAMSPERGPU + 1 ] = { 0 };
    uint64_t global_max = 0;

    uint64_t *device_data     = nullptr;
    uint64_t *pinned_host     = nullptr;
    #pragma omp parallel sections
    {

        // cpu section
        #pragma omp section
        {
            uint64_t cpu_index = 0;
            uint64_t my_max    = 0;

            if( num_cpu_batches > 0 )
                {
                    #pragma omp parallel for private( cpu_index ) reduction( max:my_max )
                    for( cpu_index = 0; cpu_index < commandline_args.batch_size * num_cpu_batches; ++cpu_index )
                        {
                            if( data[ cpu_index ] > my_max )
                                {
                                    my_max = data[ cpu_index ];
                                }
                        }
                    maximums[ 0 ] = my_max;

                    cpu_only.end = omp_get_wtime();
                }
        }

        // gpu section
        #pragma omp section
        {
            uint64_t gpu_index = 0;

            if( num_gpu_batches > 0 )
                {
                    cudaError_t result = cudaSuccess;
                    cudaStream_t streams[ STREAMSPERGPU ];
                    uint64_t *device_maximums = nullptr;
                    uint64_t iterations_per_thread = num_gpu_batches / STREAMSPERGPU;

                    uint64_t batch_size = commandline_args.batch_size;
                    uint64_t transferred_so_far = 0;
                    const uint64_t total_blocks = ceil( batch_size * 1.0 / 1024.0 );
                    uint64_t *dev_batch_size = nullptr;

                    result = create_streams( streams, STREAMSPERGPU );
                    assert( result == cudaSuccess );

                    // allocate enough STREAMSPERGPU batches + STREAMSPERGPU maximums, one max for each stream + one element to store batch_size
                    result = cudaMalloc( &device_data, sizeof( uint64_t ) * ( ( batch_size * STREAMSPERGPU ) + STREAMSPERGPU + 1 ) );
                    assert( result == cudaSuccess );

                    device_maximums = device_data + ( batch_size * STREAMSPERGPU );

                    result = cudaMallocHost( &pinned_host, sizeof( uint64_t ) * PINNEDBUFFER * STREAMSPERGPU );
                    assert( result == cudaSuccess );

                    // copy the batch size to device
                    std::memcpy( pinned_host, &batch_size, sizeof( uint64_t ) );
                    result = cudaMemcpyAsync( device_data + ( batch_size * STREAMSPERGPU ) + STREAMSPERGPU,
                                              pinned_host,
                                              sizeof( uint64_t ),
                                              cudaMemcpyHostToDevice,
                                              streams[ 0 ]
                                              );

                    // synchronize and handle any errors 
                    cudaStreamSynchronize( streams[ 0 ] );
                    assert( result == cudaSuccess );                        

                    dev_batch_size = device_data + ( batch_size * STREAMSPERGPU ) + STREAMSPERGPU;
                               
      
                    #pragma omp parallel for num_threads( STREAMSPERGPU ) shared( pinned_host, device_data, streams, device_maximums ) \
                                             private( result, gpu_index, transferred_so_far )
                    for( gpu_index = num_cpu_batches; gpu_index < total_num_batches; ++gpu_index )
                        {
                            int thread_id = omp_get_thread_num();
                            int stream_id = thread_id;
                            uint64_t left_to_copy = batch_size;                    
                    
                            // device (start/end) pointers for a stream's batch
                            uint64_t *batch_start_ptr = device_data + ( stream_id * batch_size );                        
                            uint64_t *batch_end_ptr   = device_data + ( stream_id * batch_size ) + batch_size - 1;

                            // copy batch to pinned buffer in pinned_buffer_size chunks
                            // note: batch size may exceed size of pinned buffer, i.e., when N >= 3 x 10^9
                            
                            uint64_t size_to_transfer = 0;
                            transferred_so_far  = 0;

                            while( left_to_copy > 0 )
                            { 
                                // left_to_copy initially starting at batch_size
                                size_to_transfer = std::min( (uint64_t) PINNEDBUFFER, left_to_copy );

                                // copy to pinned buffer
                                std::memcpy( pinned_host + ( stream_id * PINNEDBUFFER ),
                                             data + ( gpu_index * batch_size ) + transferred_so_far,
                                             size_to_transfer * sizeof( uint64_t )
                                           ); 
                                
                                // copy to device
                                result = cudaMemcpyAsync( device_data + ( stream_id * batch_size ) + transferred_so_far,
                                                          pinned_host + ( stream_id * PINNEDBUFFER ),
                                                          size_to_transfer * sizeof( uint64_t ),
                                                          cudaMemcpyHostToDevice,
                                                          streams[ stream_id ]
                                                        );

                                // synchronize and handle any errors 
                                cudaStreamSynchronize( streams[ stream_id ] );
                                assert( result == cudaSuccess );                        
                               
                                left_to_copy -= size_to_transfer;
                                transferred_so_far += size_to_transfer;
                            }

                            // now, find the max element for my batch
                            kernel_max<<<total_blocks, 1024, 0, streams[ stream_id ]>>>( (unsigned long long int*) device_data + ( stream_id * batch_size ),
                                                                                         (unsigned long long int*) device_maximums + stream_id, (unsigned long long int*) dev_batch_size
                                                                                       );
                        }

                        // let stream 0 (default) transfer all maximums over        
                        result = cudaMemcpyAsync( pinned_host,
                                                  device_maximums,
                                                  STREAMSPERGPU * sizeof( uint64_t ),
                                                  cudaMemcpyDeviceToHost,
                                                  // stream 0 by default
                                                  streams[ 0 ]
                                                );

                        // synchronize and handle any errors
                        cudaStreamSynchronize( 0 );
                        assert( result == cudaSuccess );
                            
                        std::memcpy( maximums + 1, // first element is from the CPU
                                     pinned_host,
                                     STREAMSPERGPU * sizeof( uint64_t )
                                   );
                    }
            gpu_only.end = omp_get_wtime();
            }        

    }

    total_time.end = omp_get_wtime();

    uint64_t max_index = 0;
    for( max_index = 0; max_index < STREAMSPERGPU + 1; ++max_index )
        {
            if( maximums[ max_index ] > global_max )
                {
                    global_max = maximums[ max_index ];
                }
        }
        
    printf( "Max: %lu\n", global_max );
    printf( "CPU only time:  %f\n", get_elapsed( &cpu_only ) );
    printf( "GPU only time:  %f\n", get_elapsed( &gpu_only ) );
    printf( "Total time:     %f\n", get_elapsed( &total_time ) );
    printf( "Load imbalance: %f\n", abs( get_elapsed( &cpu_only ) - get_elapsed( &gpu_only ) ) / get_elapsed( &total_time ) );

    free( data );
    cudaFree( device_data );
    cudaFreeHost( pinned_host );

    return EXIT_SUCCESS;
}
