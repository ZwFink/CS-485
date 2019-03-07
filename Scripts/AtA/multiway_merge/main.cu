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
    uint64_t index, cpu_index, gpu_index, curr_end_index = 0, piv_index, pivot_val;
    uint64_t *temp_ptr = nullptr;
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

	printf( "\nSeed for random number generator: %d", seed );
	printf( "\nInput size: %lu", N );
	printf( "\nBatch size: %lu\n", BATCH_SIZE );
    printf( "K (number of sublists): %u\n", K );

    // offset vectors  (NEEDS TO BE DELETED)
	//std::vector<uint64_t> first_sublist_pivots;
    //std::vector<uint64_t> offset_list_cpu;
    std::vector<uint64_t> offset_list_gpu;
    //std::vector<uint64_t> offset_begin_cpu;
    std::vector<uint64_t> offset_begin_gpu;
    
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
    uint64_t *tempBuff   = ( uint64_t * ) malloc( sizeof( uint64_t ) * N );


    printf( "\nTotal size of input sorted array (MiB): %f", ((double) N * (sizeof(uint64_t)))/(1024.0*1024.0) );

    // Generate sorted sublists 
	double tstartsort = omp_get_wtime();
    list_begin_ptrs = *generate_k_sorted_sublists( input, N, seed, K );
	double tendsort = omp_get_wtime();

	printf( "\nTime to create K sorted sublists (not part of performance measurements): %f\n", tendsort - tstartsort );
	

	//start hybrid CPU + GPU total time timer
	double tstarthybrid = omp_get_wtime();
    
    // compute the number of batches
	// The number of batches should ensure that the input dataset is split at one point
	// The input batch size is thus an approximation
	compute_batches( sublist_size, input, &first_sublist_ends, BATCH_SIZE );
	
    // split the data between CPU and GPU for hybrid searches
	numCPUBatches = ( first_sublist_ends.size() - 1 ) * CPUFRAC;
	numGPUBatches = ( first_sublist_ends.size() - 1 ) - numCPUBatches;

    printf( "\nNumber of CPU batches: %u, Number of GPU batches: %u", numCPUBatches, numGPUBatches );
    assert( (numCPUBatches + numGPUBatches) == (first_sublist_ends.size() - 1) );

    // first_sublist_ends includes index 0 as first element which should be erased
    // note that this was only needed for calculation of num cpu/gpu batches.
    first_sublist_ends.erase( first_sublist_ends.begin() );
    
    // find start pivots for first sublist
	uint64_t iter = 0;
    for( index = 0; index < N; index = index + BATCH_SIZE )
    {
        first_sublist_starts[ iter ] = index;
		iter++;
    }    
    
    start_vectors[0] = first_sublist_starts;
    end_vectors[0] = first_sublist_ends;

    // find remaining start and end pivot vectors for each sublist
    // TO DO: create function find_pivot_vectors() for task below
    
	find_pivot_vectors( input, &start_vectors, &end_vectors, &first_sublist_ends, &list_begin_ptrs, sublist_size );

	//for( index = 0; index < list_begin_ptrs.size(); ++index )
    //{
    //    // create sublist pivot starts and pivot ends
    //    //temp_start = new std::vector<uint64_t>;
    //    //temp_end = new std::vector<uint64_t>;
	//	temp_start.clear();
	//	temp_end.clear();		


    //    for( piv_index = 0; piv_index < first_sublist_starts.size(); ++piv_index )
    //    {
    //        pivot_val = first_sublist_ends[ piv_index ];
    //
    //        temp_ptr = std::upper_bound( 
    //                                list_begin_ptrs + index, 
    //                                list_begin_ptrs + index + (sublist_size - 1), 
    //                                pivot_val 
    //                              );

    //        curr_end_index = thrust::distance( list_begin_ptrs, temp_ptr );

    //        temp_end->push_back( curr_end_index );

    //        if( piv_index == 0 )
    //        {
    //            temp_start->push_back( (*list_begin_ptrs)[ index ] );
    //        }

    //        else
    //        {
    //            temp_start->push_back( temp_end[ piv_index - 1 ] );
    //        }
    //    }

    //    start_vectors[ index ] = temp_start;
    //    end_vectors[ index ] = temp_end;
    //}





	

    #pragma omp parallel sections
    {
        
      // BEGIN CPU SECTION        
      #pragma omp section
      {
        //for( cpu_index = 1; cpu_index <= numCPUBatches; ++cpu_index )
        //{
        //    if( offset_list_cpu.size() == 0 )
        //    {
        //        set_beginning_of_offsets( &offset_begin_cpu, sublist_size, K );
        //    }

        //    else // copy over indices from offset_list to offset_begin
        //    {
        //        get_offset_beginning( &offset_list_cpu, &offset_begin_cpu );
        //        
        //        offset_list_cpu.clear();
        //    }

        //    // find offset_list_cpu 
        //    compute_offsets( input, &first_sublist_offsets, &offset_list_cpu, cpu_index, K, sublist_size ); 
    
        //    // merge this round of batches
        //    multiwayMerge( &input, &tempBuff, start_index, sublist_size, K, offset_begin_cpu, offset_list_cpu );
        //    
        //    // find start_index
        //    start_index = get_start_index( offset_list_cpu, K, sublist_size );

        //    // clear offset_list and offset_begin
        //    offset_begin_cpu.clear();
        //}

      }
            
      // BEGIN GPU SECTION
      #pragma omp section
      {
          cudaStream_t streams[ STREAMSPERGPU ];
          cudaError_t result = cudaSuccess;
          uint64_t result_size = BATCH_SIZE * K * numGPUBatches;
          uint64_t stream_size = BATCH_SIZE * K;
          uint64_t *output = nullptr;
          uint64_t *stream_dev_ptrs         = nullptr;
          uint64_t *input_to_gpu_pinned = nullptr;
          uint64_t *result_from_batches_pinned = nullptr;

          result = create_streams( streams, STREAMSPERGPU );
          assert( result == cudaSuccess );

          result = cudaMalloc( (void**) &output, sizeof( uint64_t ) * result_size * 2 ); // 2 because we merge out of place
          assert( result == cudaSuccess );

          result = cudaMalloc( (void**) &stream_dev_ptrs, sizeof( uint64_t ) * stream_size );
          assert( result == cudaSuccess );

          result = cudaMallocHost( (void**) &input_to_gpu_pinned, sizeof( uint64_t ) * BATCH_SIZE * STREAMSPERGPU );
          assert( result == cudaSuccess );

          result = cudaMallocHost( (void**) &result_from_batches_pinned, sizeof( uint64_t * ) * BATCH_SIZE * STREAMSPERGPU );
          assert( result == cudaSuccess );

        for( gpu_index = numCPUBatches + 1 ; gpu_index <= numGPUBatches + numCPUBatches; ++gpu_index )
        {

            int thread_id = omp_get_thread_num();
            int stream_id = thread_id % STREAMSPERGPU;

            uint64_t start_index_gpu = 0;
            uint64_t end_index_gpu   = 0;

            #pragma omp parallel for num_threads( STREAMSPERGPU ) schedule( static ) private( index, thread_id, stream_id, start_index_gpu, \
                        end_index_gpu, start_vectors, end_vectors ) \
                        shared ( K, gpu_index, numGPUBatches, numCPUBatches, result_from_batches_pinned, \
                                 input_to_gpu_pinned, stream_dev_ptrs, output, input \
                               )
            for( index = 0; index < K; index++ )
            {

                thread_id = omp_get_thread_num();
                stream_id = thread_id % STREAMSPERGPU;

                start_index_gpu = 0;
                end_index_gpu   = 0;

                // copy data in BATCH_SIZE chunks from host memory to pinned memory
                start_index_gpu = start_vectors[ index ][ gpu_index ];
                end_index_gpu   = end_vectors[ index ][ gpu_index ];

                copy_to_device_buffer( list_begin_ptrs[ index ],
                                       input_to_gpu_pinned, stream_dev_ptrs,
                                       streams[ stream_id ],
                                       start_index_gpu, end_index_gpu,
                                       BATCH_SIZE, thread_id, stream_id
                                     );
                // copy data in BATCH_SIZE chunks from pinned data to gpu
                // do pairwise merging of sublists

                // copy data in BATCH_SIZE chunks from device to host 
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
