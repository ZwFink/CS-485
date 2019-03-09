#ifndef MM_CPU_HH_INCLUDED
#define MM_CPU_HH_INCLUDED

void testMultiwayMerge();
void multiwayMerge( uint64_t **inputArr, 
                    uint64_t **tmpBuffer, uint64_t loc, 
                    uint64_t sublist_size, uint64_t k, 
                    std::vector< std::vector<uint64_t> > starts, 
                    std::vector< std::vector<uint64_t> > ends );

#endif // MM_CPU_HH_INCLUDED
