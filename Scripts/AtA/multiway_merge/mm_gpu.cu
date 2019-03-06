#include <stdint.h>
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


#include "mm_gpu.h"
#include "omp.h"

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

cudaError_t create_streams( cudaStream_t *streams, const int num_streams )
{
    int index = 0;
    cudaError_t error_code = cudaSuccess;
    cudaError_t result     = cudaSuccess;


    for( index = 0; index < num_streams; ++index )
        {
            error_code = cudaStreamCreate( &streams[ index ] );
            if(  error_code != cudaSuccess )
                {
                    result = error_code;
                }
        }
    return result;
}



