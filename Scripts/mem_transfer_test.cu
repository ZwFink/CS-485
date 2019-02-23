#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

using namespace std;
const unsigned long int EXTRA_SPACE = 100000; 
const unsigned int STEP_SIZE        = 64;
void warmUpGPU();

int main( int argc, char **argv )
{
	cudaError_t error_code;
	unsigned long int num_items = atoi( argv[ 1 ] );
	unsigned long int upper_bound = atoi( argv[ 2 ] );
	char *host_A = NULL;
	char *staging_mem = NULL;
	double start_time = 0;
	double end_time   = 0;

	error_code = cudaMallocHost( (char**) &host_A, sizeof( char ) * num_items );

	warmUpGPU();
	const int num_trials = 3;
	int outer_index = 0;

		while( num_items < upper_bound )
		{

			for( outer_index = 0; outer_index < num_trials; outer_index++ )
			{
				char *dev_A = NULL;

				staging_mem = malloc( sizeof( char ) * num_items + 1 );
				
				int index = 0;
				for( index = 0; index < num_items - 1; index++ )
				{
					staging_mem[ index ] = 'A';
				}

				staging_mem[ num_items - 1 ] = '\0';

				memcpy( host_A, staging_mem, num_items * sizeof( char ) )

				error_code = cudaMalloc( (char **) &dev_A, sizeof( char ) * num_items + 1 );

				if( error_code != cudaSuccess )
				{
					cout << "Error allocating on device" << endl;
				}

				start = omp_get_wtime();
				error_code = cudaMemcpy( dev_A, host_A, sizeof( char ) * num_items, cudaMemcpyHostToDevice );

				cudaDeviceSynchronize();
				end = omp_get_wtime();
				if( error_code != cudaSuccess )
				{
					cout << "Error transferring to device" << endl;
				}

				free( staging_mem );

				cudaFree( dev_A );

				printf( "%ul\t%f\n", num_items, end - start );
			}

			num_items += STEP_SIZE;

		}
		cudaFreeHost( host_A );


	return EXIT_SUCCESS;
}

__global__ void warmup( unsigned int *tmp )
{
    if( threadIdx.x == 0 )
        {
            *tmp = 555;
        }
    return;
}

void warmUpGPU()
{
    printf( "Warming up GPU for time trialing...\n" );

    unsigned int *dev_tmp;
    unsigned int *tmp;

    cudaError_t errCode = cudaSuccess;


    tmp = (unsigned int *) malloc( sizeof( unsigned int ) );
    errCode = cudaMalloc( (unsigned int **) &dev_tmp, sizeof( unsigned int ) );

    if( errCode != cudaSuccess )
        {
            cout << "Error: dev_tmp error with code " << errCode << endl;
        }

    warmup<<<1,256>>>(dev_tmp);

    //copy data from device to host 
	errCode=cudaMemcpy( tmp, dev_tmp, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess)
    {
        cout << "Error: getting tmp result form GPU error with code " << errCode << endl; 
	}

	cudaDeviceSynchronize();

	printf("tmp (changed to 555 on GPU): %d\n",*tmp);

    cudaFree(dev_tmp);

    return;

}
