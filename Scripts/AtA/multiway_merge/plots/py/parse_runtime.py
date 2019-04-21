#!/usr/bin/env python3
import re

def five(obj):
    return obj.k

def main():
    filename = '../gpu_only_out.txt'

    parsed = parse( filename )
    parsed.set_hash( five )
    parsed.set_hash( lambda x : hash( getattr( x, 'num_batches' ) ) )

    for item in parsed.get():
        print( hash( item ) )
    print( parsed.get_attr( 'num_batches' ) )
    # print( parsed.get()[ 0 ].__dict__ )
    print( parsed.as_dict( 'total_time' ) )

    data = parse( '../data/outfile_400b.txt' )

    diction = data.as_dict( 'k' )

class ScriptRun:
    def __init__( self, seed = None,
                  input_size = None, batch_size = None,
                  k = None, total_size = None, num_batches = None,
                  num_cpu_batches = None, num_gpu_batches = None,
                  total_time = None,
                  time_cpu_only = None, time_gpu_only = None,
                  load_imbalance = None
                ):
        self.seed            = seed
        self.input_size      = input_size
        self.batch_size      = batch_size
        self.k               = k
        self.total_size      = total_size
        self.num_batches     = num_batches
        self.num_cpu_batches = num_cpu_batches
        self.num_gpu_batches = num_gpu_batches
        self.total_time      = total_time
        self.time_cpu_only   = time_cpu_only
        self.time_gpu_only   = time_gpu_only
        self.load_imbalance  = load_imbalance

    def self_hash( self ):
        return hash( seed )

    def __hash__( self ):
        return self.self_hash( self )

    def is_complete( self ):
        # We couldnt' parse all data for a run, maybe it crashed or something
        return not( None in self.__dict__.values() )
        
class ScriptRunCollection:
    def __init__( self, runs = list() ):
        self.script_runs = runs
    def add( self, new_run ):
        self.script_runs.append( new_run )
    def set_hash( self, hash_fn ):
        for run in self.script_runs:
            run.self_hash = hash_fn
    def get( self ):
        out_list = list()
        for item in self.script_runs:
            if item.is_complete():
                out_list.append( item )
        return out_list

    def get_attr( self, attr ):
        out_data = list()
        for run in self.script_runs:
            if run.is_complete():
                out_data.append( self.get_attr_single( run, attr ) )
        return out_data

    def get_attr_single( self, item, attr ):
        return getattr( item, attr )

    def as_dict( self, key_attr ):
        out_dict = {}
        for run in self.script_runs:
            if run.is_complete():
                attr = self.get_attr_single( run, key_attr )
                if attr not in out_dict:
                    out_dict[ attr ] = ScriptRunCollection( runs = list() )
                out_dict[ attr ].add( run )
        return out_dict

    def apply( self, what, to_what ):
        items = self.get_attr( to_what )
        return what( items )

def parse( filename ):
    out_recs = ScriptRunCollection()
    num_batch_re     = re.compile( 'Number of CPU batches: (\d+), Number of GPU batches: (\d+)' )
    time_total       = re.compile( "Time CPU and GPU \(total time\): (\d+\.\d+)" )
    time_cpu_only_re = re.compile( "Time CPU Only: (\d+\.\d+)" )
    time_gpu_only_re = re.compile( "Time GPU Only: (\d+\.\d+)" )

    with open( filename, 'r' ) as open_file:
        for line in open_file:
            if line:
                if 'Seed' in line:
                    current = ScriptRun()
                    out_recs.add( current )
                    current.seed = int( line.strip().split()[ 5 ] )
                elif 'Input size:' in line:
                    current.input_size = int( line.strip().split()[ 2 ] )
                elif 'Batch size:' in line:
                    current.batch_size = int( line.strip().split()[ 2 ] )
                elif 'K (number of sublists):' in line:
                    current.k = int( line.strip().split()[ 4 ] )
                elif 'Total size of input sorted array (MiB):' in line:
                    current.total_size = float( line.strip().split( ':' )[ 1 ].strip() )
                elif 'Num batches: ' in line:
                    current.num_batches = int( line.strip().split()[ 2 ].replace( ',', '' ) )
                elif num_batch_re.search( line ):
                    found_pat = num_batch_re.search( line )
                    current.num_cpu_batches = int( found_pat.group( 1 ) )
                    current.num_gpu_batches = int( found_pat.group( 2 ) )
                elif time_total.search( line ):
                    found_pat  = time_total.search( line )
                    current.total_time = float( found_pat.group( 1 ) )
                elif time_cpu_only_re.search( line ):
                    found_pat = time_cpu_only_re.search( line )
                    current.time_cpu_only = float( found_pat.group( 1 ) )
                elif time_gpu_only_re.search( line ):
                    found_pat = time_gpu_only_re.search( line )
                    current.time_gpu_only = float( found_pat.group( 1 ) )
                elif "Load imbalance: " in line:
                    current.load_imbalance = abs( float( line.strip().split()[ 2 ] ) )
        return out_recs
        
if __name__ == '__main__':
    main()
