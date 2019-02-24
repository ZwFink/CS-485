
void computeBatches(uint64_t N, uint64_t * input, uint64_t * search, std::vector<uint64_t> * batch_offsets, std::vector<uint64_t> * search_offsets, uint64_t inputBatchSize, uint64_t * max_input_batch_size, uint64_t * max_search_batch_size)
{



	//search the middle element
	// uint64_t *val=std::upper_bound(input, input+N, input[N/2]);
	// uint64_t distance=thrust::distance(input,val);	
	// uint64_t idx=distance;
	// printf("\nInput/2: %lu, idx: %lu",input[N/2], idx);

	
	uint64_t numBatches=ceil(N*1.0/inputBatchSize*1.0);
	//given the input batch size and N, recompute the batch size (apporximate)
	uint64_t batchSizeApprox=N/numBatches;

	printf("\nNum batches: %lu, Approx. batch size: %lu",numBatches, batchSizeApprox);

	//split the input array based on the approximate batch size



	//the first offset is index 0
	batch_offsets->push_back(0);
	//-1 because the last offset is the end of the array N-1
	for (uint64_t i=0; i<numBatches-1; i++){
		uint64_t *val=std::upper_bound(input, input+N, input[(i+1)*batchSizeApprox]);
		uint64_t distance=thrust::distance(input,val);	
		uint64_t idx=distance;
		printf("\nInput: %lu, idx (upper bound): %lu",input[(i+1)*batchSizeApprox], idx);
		batch_offsets->push_back(idx);
	}

	batch_offsets->push_back(N);

	//split the search array based on the values in the input array
	//the search array is further split later into batches
	search_offsets->push_back(0);
	

	for (uint64_t i=0; i<numBatches-1; i++){
		uint64_t *val=std::upper_bound(search, search+N, input[(i+1)*batchSizeApprox]);
		uint64_t distance=thrust::distance(search,val);	
		uint64_t idx=distance;
		printf("\nInput: %lu, search: %lu, idx (upper bound): %lu",input[(i+1)*batchSizeApprox], search[idx], idx);
		search_offsets->push_back(idx);
	}	

	search_offsets->push_back(N);


	//compute max batch sizes for input and search arrays to be allocated
	uint64_t max_input=0;
	uint64_t max_search=0;

	for (uint64_t i=0; i<batch_offsets->size()-1; i++){
		printf("\nSize input batch: val: %lu", (*batch_offsets)[i+1]-(*batch_offsets)[i]);
		max_input=max(max_input,(*batch_offsets)[i+1]-(*batch_offsets)[i]);

		printf("\nSize search batch: %lu", (*search_offsets)[i+1]-(*search_offsets)[i]);
		max_search=max(max_search,(*search_offsets)[i+1]-(*search_offsets)[i]);
	}

	printf("\nMax input batch size: %lu, Max search batch size: %lu", max_input, max_search);

	*max_input_batch_size=max_input;
	*max_search_batch_size=max_search;




}

void warm_up_gpu( int device )
{
    cudaSetDevice( device ); 		
    // initialize all ten integers of a device_vector to 1 
    thrust::device_vector<int> D(10, 1); 
    // set the first seven elements of a vector to 9 
    thrust::fill(D.begin(), D.begin() + 7, 9); 
    // initialize a host_vector with the first five elements of D 
    thrust::host_vector<int> H(D.begin(), D.begin() + 5); 
    // set the elements of H to 0, 1, 2, 3, ... 
    thrust::sequence(H.begin(), H.end()); // copy all of H back to the beginning of D 
    thrust::copy(H.begin(), H.end(), D.begin()); 
    // print D 

    printf("\nDevice: %d\n",device);

    for(int i = 0; i < D.size(); i++) 
        std::cout << " D[" << i << "] = " << D[i]; 


    // empty the vector
    D.clear();

    // deallocate any capacity which may currently be associated with vec
    D.shrink_to_fit();

    printf("\n");

    return;
}