#include <parallel/algorithm>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <random>
#include <algorithm> 
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <queue>
#include <iomanip>
#include <set>
#include <thread>
#include <utility>

// // thrust inclusions
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h> //for streams for thrust (added with Thrust v1.8)
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include "omp.h"
#include "ls_utility.h"

const int NUM_ARGS = 5;

int parse_args( args *dest, int argc, char ***argv )
{
    char **local_argv    = *argv;

    if( argc != NUM_ARGS )
        {
            return 0;
        }

    dest->N          = strtoull( local_argv[ 2 ], NULL, 0 );
    dest->batch_size = strtoull( local_argv[ 3 ], NULL, 0 );
    dest->cpu_frac   = atof( local_argv[ 4 ] );
    dest->seed       = atoi( local_argv[ 1 ] );

    return 1;
}
