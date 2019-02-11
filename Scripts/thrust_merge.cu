#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <inttypes.h>
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
    uint64_t total_data_size;
    unsigned int block_size;
} command_args_t;


bool parse_args( int argc, char **argv, command_args_t *dest );
uint64_t calc_num_items( uint64_t max_size_gb );


int main( int argc, char **argv )
{
    bool correct_args = false;
    command_args_t args;

    uint64_t num_items = 0;

    correct_args = parse_args( argc, argv, &args );

    if( !correct_args )
        {
            printf( "USAGE: thrust_merge total_size block_size\n"
                    "total_size and block_size are in GB\n"
                  );
            return EXIT_FAILURE;
        }

    num_items = calc_num_items( args.total_data_size );

    printf( "Num of items: %" PRIu64 "\n", num_items );

    return EXIT_SUCCESS;
}

bool parse_args( int argc, char **argv, command_args_t *dest )
{
    if( (unsigned int) argc != DEFAULT_NUM_ARGS )
        {
            return false;
        }
    dest->total_data_size = (uint64_t) atoll( argv[ 1 ] );
    dest->block_size      = atoi( argv[ 2 ] );

    return true;
}

uint64_t calc_num_items( uint64_t max_size_gb )
{
    uint64_t out_items = 0;

    out_items = ( ( 1ULL << 30 ) * max_size_gb ) / sizeof( uint64_t );

    return out_items;
}
