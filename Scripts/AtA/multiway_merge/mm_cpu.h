#ifndef MM_CPU_HH_INCLUDED
#define MM_CPU_HH_INCLUDED

void testMultiwayMerge();

void multiwayMergeBatches(uint64_t BATCHSIZE, int NUMBATCHES, double ** resultsFromBatches, double ** tmpBuffer);
void mergeConsumerMultiwayWithRanges(double ** resultsFromBatches, uint64_t lower1, uint64_t upper1, uint64_t lower2, uint64_t upper2);
void sortInputDataParallel( uint64_t *array, uint64_t N );
#endif // MM_CPU_HH_INCLUDED
