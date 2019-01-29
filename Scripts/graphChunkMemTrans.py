#!/usr/bin/env python

import matplotlib.pyplot as pyplot  # For creating chart
import sys

def main():
	if len( sys.argv ) != 3:
		print( "USAGE: plot_data_transfer_times.py infile outfile" )
		sys.exit( 1 )

	infile = sys.argv[ 1 ]
	outfile = sys.argv[ 2 ]

	infile_data = parse_input( sys.argv[ 1 ] )
	plot_dict = {}
	
	with open( outfile, 'w' ) as out_file:
		for chunk_size, times in infile_data.items():
			print( chunk_size )
			print( times )
			sum_time = get_sum( times )
			infile_data[ chunk_size ]  = sum_time

			plot_dict[ len( times ) ] = sum_time
			sort_keys = sorted( plot_dict.keys() )

			sorted_keys = sorted( infile_data.keys() )

		for key in sorted_keys:
			out_file.write( '%f\t%f\n' % ( key, infile_data[ key ]))
		
		out_file.write( '\n\nNumber of Chunks vs Total Time\n' )
		
		for key in sort_keys:
			out_file.write( '%f\t%f\n' % ( key, plot_dict[ key ]))

#	x_axis = list( infile_data.keys() )
#	y_axis = [ infile_data[ item ] for item in x_axis ]
	x_axis = list( plot_dict.keys() )
	y_axis = [ plot_dict[ item ] for item in x_axis ]

	ax = pyplot.subplot()
	ax.plot()
	pyplot.xlabel( "Number of Chunks of 1 MiB")
	pyplot.ylabel( "Total Time to Transfer 1 MiB (in us)")
	pyplot.title( "Number of Chunks vs Time")
	ax.scatter( x_axis, y_axis )
	pyplot.show()
    
def get_sum( times_list ):
	fixed_times = list()
	
	for current_time in times_list:
		us = convert_time_to_us( current_time )	
		us = us.strip().split( 'us' )[ 0 ]
		if 'ns' in us:
			print( us )
		us = float( us )
		fixed_times.append( us )
	
	sum_times = sum( fixed_times )
	return sum_times		
	
def convert_time_to_us( time_string ):
	if 'ns' in time_string:
		split_string = time_string.strip().split( 'ns' )
		ns = float( split_string[ 0 ] )
		us = ns / 1000
		out_str = str( us ) + 'us' 
		return out_str
	elif 'ms' in time_string:
		split_string = time_string.strip().split( 'ms' )
		ms = float( split_string[ 0 ] )
		us = ms / 1000
		out_str = str( us ) + 'us' 
		return out_str

	return time_string

def parse_input( in_filename ):
	out_dict = {}

	with open( in_filename, 'r' ) as open_file:
		for line in open_file:

			if 'HtoD' in line:
				split_line = line.split()
				byte_size = to_megabytes( split_line[ 7 ] )
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
	elif 'GB' in string_value:
		value = float( string_value.split( 'GB' )[ 0 ] )
		return value * 1000
	elif 'B' in string_value:
		value = float( string_value.split( 'B' )[ 0 ] )
		return value / ( 1000 * 1000 )

if __name__ == '__main__':
    main()
