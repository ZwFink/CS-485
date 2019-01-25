#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>

// REMINDER: unsigned int: 4 Bytes, NUM_CHUNKS * (N * 4) = 1024^2 = 1 MiB

// function declaration
void warmUpGPU();
__global__ void warmUp( unsigned int *tmp );

using namespace std;

int main( int argc, char **argv )
{
   // set lowest chunk size to 1
   unsigned int LOWEST_CHUNK_SIZE = 1;

   // set the number of trials to 10
   unsigned int NUM_TRIALS = 10;
   
   // set the number of bytes of unsigned int
   unsigned int NUM_BYTES = 4;
   
   // grab arguments from command line
   int numChunks = atoi( argv[ 1 ] );
   int bytesToTransfer = atoi( argv[ 2 ] ); 
   
   // set iterators
   int index, trialIndex;
 
	// CUDA error code:
	cudaError_t errCode = cudaSuccess;
   
   // warm up GPU for time trialing
   warmUpGPU();

   // loop until chunk size is 1, decrementing number of chunks by half each iteration
   while( numChunks >= LOWEST_CHUNK_SIZE )
   {
      for( trialIndex = 0; trialIndex < NUM_TRIALS; trialIndex++ )
      { 
         // data to transfer in the form of an array of unsigned integers
         // and device data to allocate on GPU
         unsigned int * dataArray[ numChunks ];
         unsigned int * deviceData = NULL;
         
         // chunk size variable: N
         int N = bytesToTransfer / ( numChunks * NUM_BYTES );
         
         // Loop through the array and allocate memory for each element of dataArray
         for( index = 0; index < numChunks; index++ )
         {
            dataArray[ index ] = ( unsigned int * )malloc( sizeof(unsigned int) * N );
         }
         
         // allocate on the device: deviceData
         errCode = cudaMalloc( (unsigned int**) &deviceData, sizeof(unsigned int) * N );	
         
         if( errCode != cudaSuccess ) 
         {
            cout << "\nError: A error with code " << errCode << endl; 
         }

         /*
         // Print the size of data to transfer
         printf( "\nSize of transferred data (Bytes): %lu\n\n", sizeof(unsigned int) * N * numChunks );
         */

         // Loop through array and copy each element in dataArray from Host to Device
         // Do this NUM_CHUNKS amount of times	
         for( index = 0; index < numChunks; index++ )
         {
            errCode = cudaMemcpy( deviceData, dataArray[ index ], sizeof(unsigned int) * N, cudaMemcpyHostToDevice );
         
            if( errCode != cudaSuccess ) 
            {
               cout << "\nError: A memcpy error with code " << errCode << endl; 
            }

            cudaDeviceSynchronize();
         }
         
         // free all data on host and device
         for( index = 0; index < numChunks; index++ )
         {
            free(	dataArray[ index ] );
         }
         
         cudaFree( deviceData );
      }

      // decrement chunk size by half
      numChunks = numChunks / 2;
   }
}


__global__ void warmUp( unsigned int *tmp )
{
   if( threadIdx.x == 0 )
   {
      *tmp = 555;
   }
   
   return;
}

void warmUpGPU()
{
   printf( "Warming up GPU for time trialing...\n\n" );

   unsigned int *devTmp;
   unsigned int *tmp;

   cudaError_t errCode = cudaSuccess;

   tmp = (unsigned int *) malloc( sizeof(unsigned int) );
   errCode = cudaMalloc( (unsigned int **) &devTmp, sizeof(unsigned int) );

   if( errCode != cudaSuccess )
   {
      cout << "Error: devTmp error with code " << errCode << endl;
   }

   warmUp<<1,256>>(devTmp);

   errCode = cudaMemcpy( tmp, devTmp, sizeof(unsigned int), cudaMemcpyDeviceToHost );

   if( errCode != cudaSuccess )
   {
      cout << "Error: getting tmp result from GPU error with code " << errCode << endl;
   }

   cudaDeviceSynchronize();

   printf( "tmp (changed to 555 on GPU): &d\n\n", *tmp );

   cudaFree( devTmp );

   return;

}
