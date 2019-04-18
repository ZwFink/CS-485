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

typedef struct time_data
{
    double start;
    double end;
} time_data;

int parse_args( args *dest, int argc, char ***argv );
void report_args_failure( void );
void generate_dataset( uint64_t *data, uint64_t num_items, int seed );
double get_elapsed( time_data *data );

#endif // LS_UTILITY_INCLUDED
