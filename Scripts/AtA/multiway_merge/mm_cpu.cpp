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


// multiwayMerge( &input, &output_arr, cpu_index, sublist_size, K, start_vectors, end_vectors );
// this multiway merges all batches at the end
void multiwayMerge( uint64_t **inputArr, 
                    uint64_t **output_arr, uint64_t loc, 
                    uint64_t sublist_size, uint64_t k, 
                    std::vector< std::vector<uint64_t> > starts, 
                    std::vector< std::vector<uint64_t> > ends )
{
    uint64_t index, start_position = 0;
    std::vector< std::pair<uint64_t *, uint64_t*> > seqs;
    
    // find position of result array (where to start placing merged batches)
    // by summing up all start indices of k batches
    for( index = 0; index < k; index++ )
    {
		if( index == 0 )
		{
        	start_position = start_position + starts[index][loc];
		}
	
		else
		{	
			start_position = start_position + ( starts[index][loc] - (sublist_size * index) );
		}
		
	//	printf( "\nstarts[%d][%lu] = %lu\n", index, loc, starts[index][loc] );
    }
 
    // (start_position - k) since every start position is 1 ahead of end position
    // from previous and there are k start positions. Then add 1 to get position to begin.
    if( start_position > 0 )
    {
        start_position = start_position - k + 1;
    }

    for( index = 0; index < k; index++ )
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

    //old with local variable
    //std::copy(tmp,tmp+(BATCHSIZE*(uint64_t)NUMBATCHES),resultsFromBatches);

    //new copy with the buffer
    // std::copy(*tmpBuffer,*tmpBuffer+(BATCHSIZE*(uint64_t)NUMBATCHES),*resultsFromBatches);

    
    //swap pointers -- avoid copying
    //uint64_t *a = *inputArr;
    // *inputArr + start_index = *tmpBuffer + start_index;

    //delete memory at end
    // delete a;

    return;
}


//Deprecated: better to use the normal merge for pairs of batches

//this one merges give sets of ranges of the batches, e.g., [50,100) [100,150)
// void mergeConsumerMultiwayWithRanges(uint64_t * resultsFromBatches, uint64_t lower1, uint64_t upper1,uint64_t lower2, uint64_t upper2 )

void mergeConsumerMultiwayWithRanges(uint64_t ** resultsFromBatches, uint64_t lower1, uint64_t upper1,uint64_t lower2, uint64_t upper2)
{
    

    
    // if (upper1!=lower2)
    // fprintf(stderr,"\nerror, the two ranges do not have the same value between the upper1/lower2: %d,%d ",upper1,lower2);   

    
    
    //original in place:
    // std::inplace_merge(resultsFromBatches+lower1,resultsFromBatches+(upper1),resultsFromBatches+(upper2));

    
    //tmp vector:
    //not in place:
    
    uint64_t * tmpBuffer;

    uint64_t distance = upper2 - lower1;

    std::vector<std::pair<uint64_t *, uint64_t*> > seqs;


    seqs.push_back(std::make_pair<uint64_t*,uint64_t* >(*resultsFromBatches+(lower1),*resultsFromBatches+(upper1)));
    seqs.push_back(std::make_pair<uint64_t*,uint64_t* >(*resultsFromBatches+(lower2),*resultsFromBatches+(upper2)));
    

    tmpBuffer=new uint64_t[distance];

    printf("\nDistance in merge consumer multiway: %lu",distance);

    // __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), *tmpBuffer, distance, std::less<uint64_t>(), __gnu_parallel::parallel_tag());
    __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), tmpBuffer, distance, std::less<uint64_t>(), __gnu_parallel::parallel_tag());


    std::copy(tmpBuffer,tmpBuffer+ distance,*resultsFromBatches+lower1);

    delete [] tmpBuffer;
    

     // std::vector<std::pair<uint64_t *, uint64_t*> > seqs;
    
    // for (uint64_t i=0; i<2; i++){
        // seqs.push_back(std::make_pair<uint64_t*,uint64_t* >(resultsFromBatches+(i*BATCHSIZE),resultsFromBatches+((i+1)*BATCHSIZE)));
        // seqs.push_back(std::make_pair<uint64_t*,uint64_t* >(*resultsFromBatches+(i*BATCHSIZE),*resultsFromBatches+((i+1)*BATCHSIZE)));
    // }

    
 
    

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
