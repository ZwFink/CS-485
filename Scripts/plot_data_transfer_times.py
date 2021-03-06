#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as pyplot  # For creating chart
import sys

def main():
    arg = argparse.ArgumentParser()

    arg.add_argument( '-i', '--input', help = "Tab-delimited input file to parse" )
    arg.add_argument( '--restrict_domain', help = "Comma-separated upper and lower bound "
                                                "for domain"
                  )

    args = arg.parse_args()

    infile = args.input

    infile_data = parse_input( infile )

    for trans_size, times in infile_data.items():
        avg_time = get_average_time( times )
        infile_data[ trans_size ]  = avg_time
    sorted_keys = sorted( infile_data.keys() )

    if args.restrict_domain:
        x_start, x_end = args.restrict_domain.split( ',' )
        x_start = int( x_start )
        x_end   = int( x_end )

        x_axis = list( infile_data.keys() )[ x_start // 64 : x_end // 64 ]
    else:
        x_axis = list( infile_data.keys() )

    y_axis = [ infile_data[ item ] for item in x_axis ]

    ax = pyplot.subplot()
    ax.plot()
    pyplot.xlabel( "Size of Transfer (in MB)")
    pyplot.ylabel( "Time to Transfer (MS)")
    pyplot.title( "Size of Transfer vs. Time to Transfer")
    ax.scatter( x_axis, y_axis )
    pyplot.show()
    
def get_average_time( times_list ):
    length = len( times_list )
    fixed_times = list()
    
    for current_time in times_list:
        us = current_time.strip().split( 'us' )[ 0 ]
        us = float( us ) 
        fixed_times.append( us )
    sum_times = sum( fixed_times )
    return sum_times / length


        
def parse_input( in_filename ):
    out_dict = {}

    with open( in_filename, 'r' ) as open_file:
        
        for line in open_file:
            size, time = line.strip().split( '\t' )
            size = int( size )

            if size not in out_dict:
                out_dict[ int( size ) ] = list()
            out_dict[ size ].append( time + 'us' )
    return out_dict
                


def parse_input_( in_filename ):
    out_dict = {}

    with open( in_filename, 'r' ) as open_file:
        for line in open_file:
            if 'HtoD' in line:
                split_line = line.split()

                byte_size     = to_megabytes( split_line[ 7 ] )
                #transfer_time = split_line[ 1 ]
                throughput = str( to_megabytes( split_line[ 8 ] ) ) + 'us'

                #if byte_size >= 3:
                #   break

                if byte_size not in out_dict:
                    out_dict[ byte_size ] = list()

                out_dict[ byte_size ].append( throughput )
    return out_dict


def to_megabytes( string_value ):
    if 'KB' in string_value:
        value = float( string_value.split( 'KB' )[ 0 ] )
        return value / 1000
    elif 'MB' in string_value:
        value = float( string_value.split( 'MB' )[ 0 ] )
        return value
    elif 'GB' in string_value:
        value = float( string_value.split( 'GB' )[ 0 ] ) * 1000
        return value

    elif 'B' in string_value:
        value = float( string_value.split( 'B' )[ 0 ] )
        return value / ( 1000 * 1000 )

if __name__ == '__main__':
    main()
