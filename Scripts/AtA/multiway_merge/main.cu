// c++ inclusions
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <algorithm> 
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
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
#include "mm_cpu.cpp"
#include "mm_gpu.cu"
#include "mm_utility.cu"

int main( int argc, char **argv )
{

	omp_set_num_threads(NTHREADS);
	omp_set_nested(1);

	////////////////
	//Turn on gpu
	printf("\nTurning on the GPU...\n");
	warmUpGPU(0);
	
	/////////////////////////
	// Get information from command line
	//1) the seed for random number generator
	/////////////////////////
	
	//Read in parameters from file:
	//dataset filename and cluster instance file
	if (argc!=4)
	{
	cout <<"\n\nIncorrect number of input parameters.  \nShould include a seed for the random number generator, the number of elements, N, and the batch size\n";
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

	printf("\nSeed for random number generator: %d",seed);
	printf("\nInput size: %lu",N);
	printf("\nBatch size: %lu",BATCHSIZE);

	
	
		

	//we set the range to be 2*N so that we don't get many duplicates, and this increases the chances that
	//a search traverses to the lowest level of the tree
	
	//rng for the keys
	std::mt19937 gen(seed); 
	//rng for the searched values
	std::mt19937 gen2(seed+74439); 
	//transform the randomly generated numbers into uniform distribution of ints
	std::uniform_int_distribution<uint64_t> dis(0, N);
	

	
	//input to search
	uint64_t * input;
	input=new uint64_t[N];

	// search values:
	uint64_t * search;
	search=new uint64_t[N];

	// result array:
	uint64_t * result;
	result=new uint64_t[N];	
	
	// uint64_t * result_idx;
	// result_idx=new uint64_t[N];

	for (uint64_t i=0; i<N; i++){
		input[i]=dis(gen);
		search[i]=dis(gen2);
		// totalInput+=(int)array[i];
	}

	printf("\nTotal size of input sorted array (MiB): %f",((double)N*(sizeof(uint64_t)))/(1024.0*1024.0));
	printf("\nTotal size of input search array (MiB): %f",((double)N*(sizeof(uint64_t)))/(1024.0*1024.0));
	printf("\nTotal size of result set array (MiB): %f",((double)N*(sizeof(uint64_t)))/(1024.0*1024.0));

	

	//sort input array in parallel
	double tstartsort=omp_get_wtime();
	sortInputDataParallel(input, N);
	sortInputDataParallel(search, N);
	double tendsort=omp_get_wtime();
	printf("\nTime to sort the input and search arrays in parallel (not part of performance measurements): %f",tendsort - tstartsort);
	
	//start hybrid CPU+GPU total time timer
	double tstarthybrid=omp_get_wtime();
	
	//compute the number of batches	
	//The number of batches should ensure that the input dataset is split at one point
	//The input batch size is thus an approximation
	std::vector<uint64_t> input_offsets;
	std::vector<uint64_t> search_offsets;
	uint64_t max_input_batch_size=0;
	uint64_t max_search_batch_size=0;
	compute_batches(N, input, search, &input_offsets, &search_offsets, BATCHSIZE, &max_input_batch_size, &max_search_batch_size);

	//split the data between CPU and GPU for hybrid searches
	unsigned int numCPUBatches=(input_offsets.size()-1)*CPUFRAC;
	unsigned int numGPUBatches=(input_offsets.size()-1)-numCPUBatches;

	printf("\nNumber of CPU batches: %u, Number of GPU batches: %u", numCPUBatches, numGPUBatches);
	assert((numCPUBatches+numGPUBatches)==(input_offsets.size()-1));
	

	return EXIT_SUCCESS;
}
