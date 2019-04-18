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
void report_args_failure( void );
void generate_dataset( uint64_t *data, uint64_t num_items, int seed );

#endif // LS_UTILITY_INCLUDED
