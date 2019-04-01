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



void copy_to_device_buffer( uint64_t *pinned, uint64_t *dev_ptr,
                            cudaStream_t stream, uint64_t to_transfer, int stream_id,
                            uint64_t batch_size
                          )
{
    uint64_t to_copy         = to_transfer;
    uint64_t copy_this_round = 0;
    cudaError_t result = cudaSuccess;

            
    copy_this_round = std::min(  to_copy,
                                 batch_size
                              );

    result = cudaMemcpyAsync( dev_ptr,
                              pinned,
                              copy_this_round * sizeof( uint64_t ),
                              cudaMemcpyHostToDevice, stream
                              );


    cudaStreamSynchronize( stream );
    assert( result == cudaSuccess );
    

}
uint64_t copy_to_pinned_buffer( uint64_t *input, uint64_t *pinned_host,
                                uint64_t start_index, uint64_t to_copy,
                                uint64_t stream_id, uint64_t BATCH_SIZE
                              )
{
    uint64_t data_copied       = 0;
    uint64_t left_to_copy       = to_copy;

    // want to make sure that we don't copy extra data

    data_copied = std::min( left_to_copy,
                            BATCH_SIZE 
                            );

    std::memcpy( pinned_host + ( stream_id * BATCH_SIZE ),
                 input + start_index,
                 data_copied * sizeof( uint64_t )
                 );


    return data_copied;
}

uint64_t get_gpu_output_index( const std::vector<std::vector<uint64_t>> *end_vectors,
                               const uint64_t numCPUBatches, const int num_threads
                             )
{

    uint64_t index       = 0;
    uint64_t out_val     = 0;
    for( index = 0; index < end_vectors->size(); ++index )
        {

            if( numCPUBatches > 0 )
                {
                    out_val += (*end_vectors)[ index ][ numCPUBatches - 1 ] - (*end_vectors)[ index ][ 0 ];
                }

        }
    // out_val now contains the location of the last CPU item,
    return numCPUBatches == 0 ? 0 : out_val + 1;

}

void copy_from_device_buffer( uint64_t *output_buffer,
                              uint64_t *pinned_buff, 
                              uint64_t *dev_ptr,
                              cudaStream_t stream,
                              uint64_t BATCH_SIZE,
                              int thread_id, int stream_id,
                              uint64_t num_merged
                              )
{


    uint64_t to_transfer = num_merged;
    uint64_t transferred = 0;
    uint64_t transfer_this_round = 0;

    cudaError_t result = cudaSuccess;

    while( to_transfer > 0 )
        {

            transfer_this_round = std::min( BATCH_SIZE, to_transfer );

            result = cudaMemcpyAsync(  pinned_buff, dev_ptr + transferred, transfer_this_round * sizeof( uint64_t ), cudaMemcpyDeviceToHost, stream );

            cudaStreamSynchronize( stream );
            assert( result == cudaSuccess );

            std::memcpy( output_buffer + transferred, pinned_buff, transfer_this_round * sizeof( uint64_t ) );

            to_transfer -= transfer_this_round;
            transferred += transfer_this_round;
        }
}
