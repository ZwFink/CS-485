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
    uint64_t index, cpu_index, gpu_index, start_index = 0;

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

	printf( "\nSeed for random number generator: %d", seed );
	printf( "\nInput size: %lu", N );
	printf( "\nBatch size: %lu\n", BATCH_SIZE );
    printf( "K (number of sublists): %u\n", K );

    // offset vectors
	std::vector<uint64_t> first_sublist_offsets;
    std::vector<uint64_t> offset_list_cpu;
    std::vector<uint64_t> offset_list_gpu;
    std::vector<uint64_t> offset_begin_cpu;
    std::vector<uint64_t> offset_begin_gpu;


    uint64_t *input = ( uint64_t * ) malloc( sizeof( uint64_t ) * N );
    uint64_t *tempBuff = ( uint64_t * ) malloc( sizeof( uint64_t ) * N );


    printf( "\nTotal size of input sorted array (MiB): %f", ((double) N * (sizeof(uint64_t)))/(1024.0*1024.0) );

    // Generate sorted sublists 
	double tstartsort = omp_get_wtime();
    generate_k_sorted_sublists( input, N, seed, K );
	double tendsort = omp_get_wtime();

	printf( "\nTime to create K sorted sublists (not part of performance measurements): %f\n", tendsort - tstartsort );
	
    //============================================================
    //========== Begin Hybrid CPU/GPU multiway merge =============
    //============================================================

	//start hybrid CPU + GPU total time timer
	double tstarthybrid = omp_get_wtime();
    
    // compute the number of batches
	// The number of batches should ensure that the input dataset is split at one point
	// The input batch size is thus an approximation

	compute_batches( sublist_size, input, &first_sublist_offsets, BATCH_SIZE );
	
    // split the data between CPU and GPU for hybrid searches
	unsigned int numCPUBatches = ( first_sublist_offsets.size() - 1 ) * CPUFRAC;
	unsigned int numGPUBatches = ( first_sublist_offsets.size() - 1 ) - numCPUBatches;

    printf( "\nNumber of CPU batches: %u, Number of GPU batches: %u", numCPUBatches, numGPUBatches );
    assert( (numCPUBatches + numGPUBatches) == (first_sublist_offsets.size() - 1) );
	

    #pragma omp parallel sections
    {
        
      // BEGIN CPU SECTION        
      #pragma omp section
      {
        for( cpu_index = 1; cpu_index <= numCPUBatches; ++cpu_index )
        {
            if( offset_list_cpu.size() == 0 )
            {
                set_beginning_of_offsets( &offset_begin_cpu, sublist_size, K );
            }

            else // copy over indices from offset_list to offset_begin
            {
                get_offset_beginning( &offset_list_cpu, &offset_begin_cpu );
                
                offset_list_cpu.clear();
            }

            // find offset_list_cpu 
            compute_offsets( input, &first_sublist_offsets, &offset_list_cpu, cpu_index, K, sublist_size ); 
    
            // merge this round of batches
            multiwayMerge( &input, &tempBuff, start_index, sublist_size, K, offset_begin_cpu, offset_list_cpu );
            
            // find start_index
            start_index = get_start_index( offset_list_cpu, K, sublist_size );

            // clear offset_list and offset_begin
            offset_begin_cpu.clear();
        }

      }
            
      // BEGIN GPU SECTION
      #pragma omp section
      {
          cudaStream_t streams[ STREAMSPERGPU ];
          cudaError_t result = cudaSuccess;
          uint64_t result_size = BATCH_SIZE * K * numGPUBatches;
          uint64_t stream_size = BATCH_SIZE * STREAMSPERGPU;
          uint64_t *output = nullptr;
          uint64_t *stream_dev_ptrs         = nullptr;
          uint64_t *input_to_gpu_pinned = nullptr;
          uint64_t *result_from_batches_pinned = nullptr;

          result = create_streams( streams, STREAMSPERGPU );
          assert( result == cudaSuccess );

          result = cudaMalloc( (void**) &output, sizeof( uint64_t ) * result_size );
          assert( result == cudaSuccess );

          result = cudaMalloc( (void**) &stream_dev_ptrs, sizeof( uint64_t ) * stream_size );
          assert( result == cudaSuccess );

          result = cudaMallocHost( (void**) &input_to_gpu_pinned, sizeof( uint64_t ) * BATCH_SIZE * STREAMSPERGPU );
          assert( result == cudaSuccess );

          result = cudaMallocHost( (void**) &result_from_batches_pinned, sizeof( uint64_t * ) * BATCH_SIZE * STREAMSPERGPU );
          assert( result == cudaSuccess );

        #pragma omp parallel for num_threads( STREAMSPERGPU ) ordered schedule( static,1 )
        for( gpu_index = numCPUBatches + 1 ; gpu_index <= numGPUBatches + numCPUBatches; ++gpu_index )
        {
            int stream_id = 0;
            int thread_num = omp_get_thread_num();


            #pragma omp single
            {
                if( offset_list_gpu.size() == 0 )
                    {
                        set_beginning_of_offsets( &offset_begin_gpu, sublist_size, K );
                    }

                else // copy over indices from offset_list to offset_begin
                    {
                        get_offset_beginning( &offset_list_gpu, &offset_begin_gpu );
                
                        offset_list_gpu.clear();
                    }
            }

            for( index = 0; index < K; index++ )
            {
                continue;
            }
        }

      }

    }



    // end hybrid CPU + GPU total time timer
	double tendhybrid = omp_get_wtime();

    free( input );

	return EXIT_SUCCESS;





}
