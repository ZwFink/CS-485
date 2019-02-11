#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/merge.h>

using namespace std;

const unsigned int DEVICE_CAPACITY_GB = 16;
const unsigned int DEFAULT_NUM_ARGS   = 3; // 2 + 1, first is prog name

typedef struct command_args
{
    unsigned int total_data_size;
    unsigned int block_size;
};

int main( int argc, char **argv )
{
    


    return EXIT_SUCCESS;
}

