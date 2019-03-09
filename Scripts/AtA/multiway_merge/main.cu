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

int main( int argc, char **argv )
{
    uint64_t index, cpu_index, gpu_index;
    unsigned int numCPUBatches, numGPUBatches;
	
    omp_set_num_threads(NTHREADS);
	omp_set_nested(1);

	////////////////
	//Turn on gpu
	printf("\nTurning on the GPU...\n");
	warm_up_gpu( 0 );
	
	/////////////////////////
	// Get information from command line
	//1) the seed for random number generator
	/////////////////////////
	
	//Read in parameters from file:
	//dataset filename and cluster instance file
	if ( argc != 5 )
	{
		printf( "\n\nIncorrect number of input parameters.  \nShould include a seed for the random number generator, "
				"the number of elements, N, the batch size, and the number of lists, K\n"
		      );
		return 0;
	}
	
	//copy parameters from commandline:
	//char inputFname[]="data/test_data_removed_nan.txt";	
	
	char inputseed[ 500 ];
	strcpy( inputseed, argv[ 1 ] );

	char inputN[ 500 ];
	strcpy( inputN, argv[ 2 ] );

	char inputBatchSize[ 500 ];
	strcpy( inputBatchSize, argv[ 3 ] );
	
	unsigned int seed = atoi( inputseed );
	
	// uint64_t N=atoi(inputN);
	uint64_t N = strtoull( inputN, NULL, 0 );

	uint64_t BATCH_SIZE = strtoull( inputBatchSize, NULL, 0 );

	uint16_t K = strtoul( argv[ 4 ], NULL, 0 );

    uint64_t sublist_size = N / K;
	
    // helper vectors
    std::vector<uint64_t> first_sublist_starts;
    std::vector<uint64_t> first_sublist_ends;
    std::vector<uint64_t *> list_begin_ptrs; 
    std::vector<uint64_t> temp_start;
	std::vector<uint64_t> temp_end;
    // start and end vectors containing start and end
    // pivot vectors for each sublist
    std::vector<std::vector<uint64_t>> start_vectors;
    std::vector<std::vector<uint64_t>> end_vectors;
    
    // initialize array of integers
    uint64_t *input      = ( uint64_t * ) malloc( sizeof( uint64_t ) * N );
    uint64_t *output_arr = (uint64_t *) malloc( sizeof( uint64_t ) * N );

	printf( "\nSeed for random number generator: %d", seed );
	printf( "\nInput size: %lu", N );
	printf( "\nBatch size: %lu\n", BATCH_SIZE );
    printf( "K (number of sublists): %u\n", K );

    printf( "\nTotal size of input sorted array (MiB): %f", ((double) N * (sizeof(uint64_t)))/(1024.0*1024.0) );

    // Generate sorted sublists 
	double tstartsort = omp_get_wtime();
    list_begin_ptrs = *generate_k_sorted_sublists( input, N, seed, K );
	double tendsort = omp_get_wtime();

	printf( "\nTime to create K sorted sublists (not part of performance measurements): %f\n", tendsort - tstartsort );
	
	//start hybrid CPU + GPU total time timer
	double tstarthybrid = omp_get_wtime();
   
    // compute the number of batches
	compute_batches( sublist_size, input, &first_sublist_ends, BATCH_SIZE, sublist_size );
	
    // split the data between CPU and GPU for hybrid searches
	numCPUBatches = ( first_sublist_ends.size() - 1 ) * CPUFRAC;
	numGPUBatches = ( first_sublist_ends.size() - 1 ) - numCPUBatches;

    printf( "\nNumber of CPU batches: %u, Number of GPU batches: %u", numCPUBatches, numGPUBatches );
    assert( (numCPUBatches + numGPUBatches) == (first_sublist_ends.size() - 1) );

    // first_sublist_ends includes index 0 as first element which should be erased
    // note that this was only needed for calculation of num cpu/gpu batches.
    first_sublist_ends.erase( first_sublist_ends.begin() );
    
    // find start pivots for first sublist
    for( index = 0; index < sublist_size; index = index + BATCH_SIZE )
    {
        first_sublist_starts.push_back( index );
    }    
    
    start_vectors.push_back( first_sublist_starts );
    end_vectors.push_back( first_sublist_ends );

    
	// find remaining start and end pivot vectors for each sublist
	find_pivot_vectors( input, &start_vectors, &end_vectors, &first_sublist_ends, &list_begin_ptrs, sublist_size );

//	for( index = 0; index < end_vectors[1].size(); index++ )
//	{
//		printf( "\nbatch index: %lu\n", start_vectors[ 5 ][ index ] );
//	}
	
    #pragma omp parallel sections
    {
        
      // BEGIN CPU SECTION        
      #pragma omp section
      {

        for( cpu_index = 0; cpu_index < numCPUBatches; ++cpu_index )
        {
            // merge this round of batches
            multiwayMerge( &input, &output_arr, cpu_index, sublist_size, K, start_vectors, end_vectors );
        }

      }
            
      // BEGIN GPU SECTION
      #pragma omp section
      {
          cudaStream_t streams[ STREAMSPERGPU ];
          const int NUM_THREADS_SEARCH = 4;
          cudaError_t result = cudaSuccess;
          std::vector<uint64_t> gpu_start_ptrs;
          std::vector<uint64_t> gpu_end_ptrs;
          uint64_t result_size = BATCH_SIZE * K * 2;
          uint64_t stream_size = BATCH_SIZE * K;
          uint64_t *output = nullptr;
          uint64_t *stream_dev_ptrs         = nullptr;
          uint64_t *input_to_gpu_pinned = nullptr;
          uint64_t *result_from_batches_pinned = nullptr;

          uint64_t gpu_output_index = get_gpu_output_index( &end_vectors, numCPUBatches, NUM_THREADS_SEARCH );

          gpu_start_ptrs.reserve( K );
          gpu_end_ptrs.reserve( K );

          result = create_streams( streams, STREAMSPERGPU );
          assert( result == cudaSuccess );

          result = cudaMalloc( (void**) &output, sizeof( uint64_t ) * result_size * 2 ); // 2 because we merge out of place
          assert( result == cudaSuccess );

          result = cudaMalloc( (void**) &stream_dev_ptrs, sizeof( uint64_t ) * stream_size );
          assert( result == cudaSuccess );

          result = cudaMallocHost( (void**) &input_to_gpu_pinned, sizeof( uint64_t ) * BATCH_SIZE * STREAMSPERGPU );
          assert( result == cudaSuccess );

          result = cudaMallocHost( (void**) &result_from_batches_pinned, sizeof( uint64_t ) * BATCH_SIZE * STREAMSPERGPU );
          assert( result == cudaSuccess );

        for( gpu_index = numCPUBatches + 1; gpu_index <= numGPUBatches + numCPUBatches; ++gpu_index )
        {

            int thread_id = omp_get_thread_num();
            int stream_id = thread_id % STREAMSPERGPU;

            uint64_t start_index_gpu             = 0;
            uint64_t end_index_gpu               = 0;

            /*#pragma omp parallel for num_threads( STREAMSPERGPU ) schedule( static ) private( index, thread_id, stream_id, start_index_gpu, \
                        end_index_gpu, start_vectors, end_vectors ) \
                        shared ( K, gpu_index, numGPUBatches, numCPUBatches, result_from_batches_pinned, \
                                 input_to_gpu_pinned, stream_dev_ptrs, output, input \
                               ) */
            for( index = 0; index < K; index++ )
            {

                thread_id = omp_get_thread_num();
                stream_id = gpu_index % STREAMSPERGPU;

                start_index_gpu = 0;
                end_index_gpu   = 0;

                // copy data in BATCH_SIZE chunks from host memory to pinned memory
                start_index_gpu = start_vectors[ index ][ gpu_index ];
                end_index_gpu   = end_vectors[ index ][ gpu_index ];

                // calculate relative start
                gpu_start_ptrs[ index ] = gpu_index == numCPUBatches + 1 ? \
                                          0 : \
                                          start_vectors[ index ][ gpu_index ] - start_vectors[ index ][ gpu_index - 1 ];

                // calculate relative end index
                gpu_end_ptrs[ index ]   = gpu_index == numCPUBatches + 1 ? \
                                          start_vectors[ index ][ gpu_index ] - 1 : \
                                          end_vectors[ index ][ gpu_index ] - end_vectors[ index ][ gpu_index - 1 ];


                copy_to_device_buffer( input,
                                       input_to_gpu_pinned, stream_dev_ptrs,
                                       streams[ stream_id ],
                                       start_index_gpu, end_index_gpu,
                                       BATCH_SIZE, thread_id, stream_id
                                     );
                gpu_output_index += gpu_end_ptrs[ index ] - gpu_start_ptrs[ index ];
            }
                // do pairwise merging of sublists
            // merge the first two sublists, after the first merge we alternate
            // between output buffers
            thrust::merge( thrust::device, stream_dev_ptrs + gpu_start_ptrs[ 0 ],
                           stream_dev_ptrs + gpu_end_ptrs[ 0 ],
                           stream_dev_ptrs + gpu_start_ptrs[ 1 ],
                           stream_dev_ptrs + gpu_end_ptrs[ 1 ],
                           output
                         );

            #pragma omp parallel for num_threads( STREAMSPERGPU ) schedule( static ) private( index, thread_id, stream_id, start_index_gpu, \
                        end_index_gpu, start_vectors, end_vectors ) \
                        shared ( K, gpu_index, numGPUBatches, numCPUBatches, result_from_batches_pinned, \
                                 input_to_gpu_pinned, stream_dev_ptrs, output_arr, input, gpu_output_index, gpu_end_ptrs, gpu_start_ptrs \
                               )
            for( index = 0; index < K; index++ )
                {
                    thread_id = omp_get_thread_num();
                    stream_id = thread_id % STREAMSPERGPU;

                    // copy data in BATCH_SIZE chunks from device to host 
                    copy_from_device_buffer( output_arr + gpu_output_index,
                                             result_from_batches_pinned,
                                             output,
                                             streams[ stream_id ],
                                             BATCH_SIZE, thread_id, stream_id,
                                             &gpu_start_ptrs,
                                             &gpu_end_ptrs
                                             );
                }
        }

      }
    }

    // end hybrid CPU + GPU total time timer
	double tendhybrid = omp_get_wtime();
    free( input );
    free( output_arr );

	return EXIT_SUCCESS;

}
