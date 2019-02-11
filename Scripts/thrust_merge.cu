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
const unsigned int SEED_RAND          = 42;

typedef struct command_args
{
    unsigned int total_data_size;
    unsigned int block_size;
} command_args_t;


bool parse_args( int argc, char **argv, command_args_t *dest );
int compare_ints( const void *a, const void *b );
uint64_t calc_num_items( unsigned int max_size_gb );
void create_sorted_data( unsigned int **data_loc, uint64_t num_items );


int main( int argc, char **argv )
{
    bool correct_args = false;
    command_args_t args;
    srand( SEED_RAND );

    unsigned int **data = NULL;

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

    create_sorted_data( data, num_items );

    printf( "Num of items: %" PRIu64 "\n", num_items );

    free( *data );

    return EXIT_SUCCESS;
}

bool parse_args( int argc, char **argv, command_args_t *dest )
{
    if( (unsigned int) argc != DEFAULT_NUM_ARGS )
        {
            return false;
        }
    dest->total_data_size = (unsigned int) atoll( argv[ 1 ] );
    dest->block_size      = atoi( argv[ 2 ] );

    return true;
}

uint64_t calc_num_items( unsigned int max_size_gb )
{
    const uint64_t GIGABYTE_EXPONENT = 30;
    uint64_t out_items = 0;

    out_items = ( ( (uint64_t) 1 << GIGABYTE_EXPONENT ) * max_size_gb ) /
                sizeof( unsigned int );

    return out_items;
}


void create_sorted_data( unsigned int **data_loc, uint64_t num_items )
{
    int *data_ptr = NULL;
    uint64_t index = 0;

    data_ptr = malloc( sizeof( unsigned int) * num_items );

    for( index = 0; index < num_items; index++ )
        {
            data_ptr[ index ] = (unsigned int) rand();
        }

    qsort( data_ptr, num_items, sizeof( unsigned int ), compare_ints );

    *data_loc = data_ptr;
}

int compare_ints( const void *a, const void *b )
{
    return *(int*)a - *(int*)b;
}
