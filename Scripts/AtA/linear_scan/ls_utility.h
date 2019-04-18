#ifndef LS_UTILITY_INCLUDED
#define LS_UTILITY_INCLUDED

extern const int NUM_ARGS;
typedef struct args
{
    uint64_t N;
    uint64_t batch_size;
    float cpu_frac;
    int seed;
} args;

int parse_args( args *dest, int argc, char ***argv );

#endif // LS_UTILITY_INCLUDED
