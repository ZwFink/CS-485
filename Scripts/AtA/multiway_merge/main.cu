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
	
	char inputseed[500];
	strcpy(inputseed,argv[1]);

	char inputN[500];
	strcpy(inputN,argv[2]);

	char inputBatchSize[500];
	strcpy(inputBatchSize,argv[3]);
	
	unsigned int seed=atoi(inputseed);
	
	// uint64_t N=atoi(inputN);
	uint64_t N = strtoull(inputN, NULL, 0);

	// uint64_t BATCHSIZE=atoi(inputBatchSize);
	uint64_t BATCHSIZE = strtoull(inputBatchSize, NULL, 0);

	uint16_t K = strtoul( argv[ 4 ], NULL, 0 );

	printf("\nSeed for random number generator: %d",seed);
	printf("\nInput size: %lu",N);
	printf("\nBatch size: %lu\n",BATCHSIZE);
    printf( "K (number of sublists): %u\n", K );

    uint64_t *input = (uint64_t*) malloc( sizeof( uint64_t ) * N );

    printf("\nTotal size of input sorted array (MiB): %f",((double) N * (sizeof(uint64_t)))/(1024.0*1024.0));

	//sort input array in parallel
	double tstartsort=omp_get_wtime();
    generate_k_sorted_sublists( input, N, seed, K );
	double tendsort=omp_get_wtime();

	printf("\nTime to create K sorted sublists (not part of performance measurements): %f\n",tendsort - tstartsort);
	
	//start hybrid CPU+GPU total time timer
	double tstarthybrid=omp_get_wtime();
	
	//compute the number of batches	
	//The number of batches should ensure that the input dataset is split at one point
	//The input batch size is thus an approximation
	std::vector<uint64_t> input_offsets;
	uint64_t max_input_batch_size=0;

    // commented out for now

	//compute_batches( N, input, &input_offsets, BATCHSIZE, &max_input_batch_size );
	/**split the data between CPU and GPU for hybrid searches
	 unsigned int numCPUBatches=(input_offsets.size()-1)*CPUFRAC;
	 unsigned int numGPUBatches=(input_offsets.size()-1)-numCPUBatches;

     printf("\nNumber of CPU batches: %u, Number of GPU batches: %u", numCPUBatches, numGPUBatches);
     assert((numCPUBatches+numGPUBatches)==(input_offsets.size()-1));
    **/
	
    free( input );
	return EXIT_SUCCESS;
}
