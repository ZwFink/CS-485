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


//void testMultiwayMerge()
//{
//    //http://manpages.ubuntu.com/manpages/zesty/man3/__gnu_parallel.3cxx.html
// 
//    //sections on:
//    //template<typename _RAIterPairIterator , typename _RAIterOut , typename
//           // _DifferenceTp , typename _Compare > _RAIterOut multiway_merge
//           // (_RAIterPairIterator __seqs_begin, _RAIterPairIterator __seqs_end,
//           // _RAIterOut __target, _DifferenceTp __length, _Compare __comp,
//           // parallel_tag __tag=parallel_tag(0))
//
//
//	// int * test=new int[5];
//
//
//// printf("XXX print from multiway");
//
//    uint64_t NUM_ELEM=700000000;
//    unsigned int NUM_LISTS=7;
//
//
//
//	// int sequences[NUM_ELEM][NUM_ELEM];
//
// //    sequences[0][0]=1;
//
//
//    printf("\nAllocating memory for multiway merging (GiB): %f",(NUM_ELEM*NUM_LISTS*sizeof(uint64_t))/(1024*1024*1024.0));
//   // return;
//   uint64_t ** sequences =new uint64_t *[NUM_LISTS];
//   for (int i=0; i<NUM_LISTS; i++)
//   {
//        sequences[i]=new uint64_t [NUM_ELEM];
//   }
//
//    for (int i=0; i<NUM_LISTS; i++)
//    {    
//        for (uint64_t j=0; j<NUM_ELEM; j++)
//        {
//        sequences[i][j]=uint64_t(j);
//        }
//    }
//
//
//    //print sequences
//    // for (int i=0; i<NUM_ELEM; i++)
//    // {
//    //     printf("\nsequence:");    
//    //     for (int j=0; j<NUM_ELEM; j++)
//    //     {
//    //     printf("%f, ",sequences[i][j]);
//    //     }
//    // }
//
//
//    std::vector<uint64_t> out_vect;
//    out_vect.reserve(NUM_ELEM*NUM_LISTS);
//    std::vector<std::pair<uint64_t *, uint64_t*> > seqs;
//
//    for (int i=0; i<NUM_LISTS; i++)
//    {
//        seqs.push_back(std::make_pair<uint64_t*,uint64_t* >(sequences[i]+0,sequences[i]+NUM_ELEM));
//    }
//
//
//    uint64_t tstart=omp_get_wtime();
//    __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), out_vect.begin(), NUM_ELEM*NUM_LISTS, std::less<uint64_t>(), __gnu_parallel::parallel_tag());
//    uint64_t tend=omp_get_wtime();
//
//    printf("\nTime multiway: %f",tend-tstart);
//
//
//    //print output
//    printf("\nOutput multiway (last 30 elems):");
//    for (uint64_t i=(NUM_ELEM*NUM_LISTS)-30; i<(NUM_ELEM*NUM_LISTS); i++)
//    {
//    printf("%f, ",out_vect[i]);
//    }
//
///* 
//    int out[33];
//    std::vector<int>out_vect;
//    std::vector<std::pair<int*, int *> > seqs;
//    for (int i = 0; i < 10; ++i)
//     { 
//      	seqs.push_back(std::make_pair<int*, int* >(sequences[i], sequences[i] + 10)); 
//     }
//
//    __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), out_vect.begin(), 33, std::less<int>(), __gnu_parallel::parallel_tag());
//*/
//
//return;
//}
