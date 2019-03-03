#ifndef MM_CPU_HH_INCLUDED
#define MM_CPU_HH_INCLUDED

void testMultiwayMerge();
void multiwayMerge( uint64_t **inputArr, 
                    uint64_t **tmpBuffer, uint64_t start_index, 
                    uint64_t sublist_size, uint64_t k, 
                    std::vector<uint64_t> offset_begin, std::vector<uint64_t> offset_end );
void mergeConsumerMultiwayWithRanges(double ** resultsFromBatches, uint64_t lower1, uint64_t upper1, uint64_t lower2, uint64_t upper2);

#endif // MM_CPU_HH_INCLUDED
