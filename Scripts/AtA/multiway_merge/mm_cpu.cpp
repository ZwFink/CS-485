#include <vector>
#include <utility>
#include <cstdint>
#include <stdlib.h>
#include <functional>
#include <parallel/algorithm>
#include <algorithm>
#include <cstring>
#include <iterator>
#include "mm_cpu.h"

// this merges all batches of every sublist at once 
// (requires one call from main where loc = numCPUBatches - 1)
void multiwayMerge( uint64_t **inputArr, 
                    uint64_t **output_arr, uint64_t loc, 
                    uint64_t sublist_size, uint64_t k, 
                    std::vector< std::vector<uint64_t> > starts, 
                    std::vector< std::vector<uint64_t> > ends )
{
    uint64_t index;
    std::vector< std::pair<uint64_t *, uint64_t*> > seqs;
    
    for( index = 0; index < k; ++index )
    {
        seqs.push_back( std::make_pair< uint64_t *, uint64_t * >( *inputArr + starts[index][0], 
                                                                    *inputArr + ends[index][loc] + 1) );
    }

    __gnu_parallel::multiway_merge( seqs.begin(), 
                                    seqs.end(), 
                                    *output_arr, 
                                    sublist_size * k, 
                                    std::less<uint64_t>(), 
                                    __gnu_parallel::parallel_tag() 
                                  );

    return;
}


// this merges particular batches determined 
// by the location (loc) of the batch in the sublist
void multiwayMergeBySplits( uint64_t **inputArr, 
                            uint64_t **output_arr, uint64_t loc, 
                            uint64_t sublist_size, uint64_t k, 
                            std::vector< std::vector<uint64_t> > starts, 
                            std::vector< std::vector<uint64_t> > ends )
{
    uint64_t index, start_position = 0;
    std::vector< std::pair<uint64_t *, uint64_t*> > seqs;
    
    // find where to start placing merged batches
    if( loc > 0 ) // otherwise loc = 0 and we merge at beginning
    {
        for( index = 0; index < k; ++index )
        {
            start_position = start_position + ( ends[index][loc - 1] - starts[index][0] ) + 1;
        }
    }
 
    for( index = 0; index < k; ++index )
    {
        seqs.push_back( std::make_pair< uint64_t *, uint64_t * >( *inputArr + starts[index][loc], 
                                                                    *inputArr + ends[index][loc] ) );
    }

    __gnu_parallel::multiway_merge( seqs.begin(), 
                                    seqs.end(), 
                                    *output_arr + start_position, 
                                    sublist_size * k, 
                                    std::less<uint64_t>(), 
                                    __gnu_parallel::parallel_tag() 
                                  );

    return;
}



