#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from matplotlib import rc
rc('text', **{'usetex':True})
base_dir = '../'

def five(obj):
    return obj.k

def main():

    input_gpu = base_dir + 'outfile_cpu_frac_0.txt'
    input_cpu = base_dir + 'outfile_cpu_frac_1.txt'

    gpu_size_dict = parse( input_gpu ).as_dict( 'input_size' )
    gpu_k_dict = gpu_size_dict[ 4000000000 ].as_dict( 'k' )

    cpu_size_dict = parse( input_cpu ).as_dict( 'input_size' )
    print( cpu_size_dict )
    cpu_k_dict = cpu_size_dict[ 4000000000 ].as_dict( 'k' )

    cpu_times =  [ cpu_k_dict[ k ].apply( np.mean, 'time_cpu_only' ) for k in cpu_k_dict.keys() ]
    gpu_times = [ gpu_k_dict[ k ].apply( np.mean, 'time_gpu_only' ) for k in gpu_k_dict.keys() ]

    print( cpu_times )
    print( gpu_times )

    x = np.arange( 5 )
    fig = plt.figure( figsize = ( 5, 4 ) )
    ax = fig.add_subplot( 111 )
    ax.set_xticks( x )
    ax.set_xticklabels( [ 2, 4, 8, 16, 32 ] )
    
    ax.set_xlabel( 'Number of Sublists ($k$)', fontsize = 20)
    ax.set_ylabel( 'Time (s)', fontsize = 20)
    ax.set_ylim( 1, 10 )
    lab1 = ax.plot( x, gpu_times, ls = 'dashed', c = 'black', marker = 's', 
                    markersize = 7, linewidth = 2, label = "$T^{GPU}$")
    lab2 = ax.plot( x, cpu_times, ls = 'dotted', c = 'black', marker = 'd',
                    markersize = 7, linewidth = 2, label = "$T^{CPU}$")

    lns = lab1 + lab2
    labs = [ l.get_label() for l in lns ]
    l = ax.legend( lns, labs, fontsize = 15, loc = 'upper left', fancybox = False, framealpha = 1,
                   handlelength = 2.5, ncol = 1)

    plt.tight_layout()
    fig.savefig( 'multiway_merge_cpu_vs_gpu_times_k_3.pdf', bbox_inches = 'tight' )

class ScriptRun:
    def __init__( self, seed = None,
                  input_size = None, batch_size = None,
                  k = None, total_size = None, num_batches = None,
                  num_cpu_batches = None, num_gpu_batches = None,
                  total_time = None,
                  time_cpu_only = None, time_gpu_only = None,
                  load_imbalance = None,
                  cache_misses  = None,
                  cache_references = None
                  #mu = None
                  #cache_misses = None
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
        #self.mu = mu
        #self.cache_misses    = cache_misses
        #self.cache_references    = cache_references
        # self.cache_misses    = cache_misses

    def self_hash( self ):
        return hash( seed )

    def __hash__( self ):
        return self.self_hash( self )

    def is_complete( self ):
        # We couldnt' parse all data for a run, maybe it crashed or something
        return not( None in self.__dict__.values() ) #or not( self.cache_references is None  and \
                                                     #        self.cache_misses is None )
        
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
            else:
                print( run.__dict__ )
                    
        return out_dict

    def apply( self, what, to_what ):
        items = self.get_attr( to_what )
        return what( items )

def parse( filename ):
    out_recs = ScriptRunCollection( runs = list() )
    num_batch_re     = re.compile( 'Number of CPU batches: (\d+), Number of GPU batches: (\d+)' )
    time_total       = re.compile( "Time CPU and GPU \(total time\): (\d+\.\d+)" )
    time_cpu_only_re = re.compile( "Time CPU Only: [+-]?(\d+\.\d+)" )
    time_gpu_only_re = re.compile( "Time GPU Only: [+-]?(\d+\.\d+)" )

    with open( filename, 'r' ) as open_file:
        for line in open_file:
            if line:
                #if 'mu\t' in line:
                #    current = ScriptRun()
                #    out_recs.add( current )

                #    current.mu = float( line.strip().split( '\t')[ 1 ] )
                if 'Seed' in line:
                    current = ScriptRun()
                    out_recs.add( current )
                    current.seed = int( line.strip().split()[ 5 ] )
                elif 'Seed' in line:
                    #current = ScriptRun()
                    #out_recs.add( current )
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
                    current.time_gpu_only = abs( float( found_pat.group( 1 ) ) )
                elif "Load imbalance: " in line:
                    current.load_imbalance = abs( float( line.strip().split()[ 2 ] ) )
                #elif 'cache-references' in line:
                #    current.cache_references = int( line.strip().split()[ 0 ].replace( ',', '' ) )
                #elif 'cache-misses' in line:
                #    current.cache_misses = int( line.strip().split()[ 0 ].replace( ',', '' ) )
                #elif 'L1 Cache Misses' in line:
                #    current.cache_misses = int( line.strip().split()[ 3 ] )
        return out_recs
        
if __name__ == '__main__':
    main()
