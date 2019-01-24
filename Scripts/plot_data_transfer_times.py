#!/usr/bin/env python3

import matplotlib.pyplot as pyplot  # For creating chart
import sys

def main():
    if len( sys.argv ) != 3:
        print( "USAGE: plot_data_transfer_times.py infile outfile" )
        sys.exit( 1 )

    infile = sys.argv[ 1 ]
    outfile = sys.argv[ 2 ]


    infile_data = parse_input( sys.argv[ 1 ] )


def parse_input( in_filename ):
    out_dict = {}

    with open( in_filename, 'r' ) as open_file:
        for line in open_file:
            if 'HtoD' in line:
                split_line = line.split()

                byte_size     = to_megabytes( split_line[ 7 ] )
                transfer_time = split_line[ 1 ]

                if byte_size not in out_dict:
                    out_dict[ byte_size ] = list()

                out_dict[ byte_size ].append( transfer_time )
    return out_dict


def to_megabytes( string_value ):
    if 'KB' in string_value:
        value = float( string_value.split( 'KB' )[ 0 ] )
        return value / 1000
    elif 'MB' in string_value:
        value = float( string_value.split( 'MB' )[ 0 ] )
        return value
    elif 'B' in string_value:
        value = float( string_value.split( 'B' )[ 0 ] )
        return value / ( 1000 * 1000 )

if __name__ == '__main__':
    main()
