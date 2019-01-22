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

    with open( in_filename, 'r' ) open_file:
        for line in open_file:
            if 'DtoH' in line:
                split_line = line.split()

                byte_size     = to_megabytes( split_line[ 7 ] )
                transfer_time = to_ms( split_line[ 7 ])

                if byte_size not in out_dict:
                    out_dict[ byte_size ] = list()

                byte_size[ out_dict ].append( transfer_time )
    return out_dict

def to_ms( string_value ):
    pass

def to_megabytes( string_value ):
    pass


if __name__ == '__main__':
    main()
