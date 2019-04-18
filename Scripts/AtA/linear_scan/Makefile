#GPU.cu 
SOURCES = main.cu ls_cpu.cpp ls_gpu.cu ls_cpu.h ls_gpu.h ls_utility.h
OBJECTS = ls_cpu.o ls_utility.o
CUDAOBJECTS = ls_gpu.o main.o 
CC = nvcc
EXECUTABLE = main



#fpermissive because of tbb library on UHHPC
FLAGS =  -std=c++11 -O3 -Xcompiler -fopenmp -arch=compute_60 -code=sm_60 -lcuda -lineinfo -Xcompiler -fpermissive
# FLAGS =  -std=c++11 -O3 -Xcompiler -fopenmp -Xcompiler -fpermissive -ltbb -arch=compute_60 -code=sm_60 -lcuda -lineinfo
NTHREADS = -DNTHREADS=16 
CFLAGS = -c 


PARAMS= -DNUM_TRIALS=1 -DBLOCKSIZE=256 -DNUMGPU=1 -DCPUFRAC=1.0 -DPINNEDBUFFER=1000000 -DSTREAMSPERGPU=8 -DBLOCKING=0 

all: $(EXECUTABLE)

# merge.o:
# 	$(CC) -c -std=c++11 -O3 -fopenmp -fpermissive -ltbb merge.cpp

# import_dataset.o:
# 	$(CC) -c -std=c++11 -O3 -fopenmp -fpermissive -ltbb import_dataset.cpp	

.cpp.o:
	$(CC) $(CFLAGS) $(FLAGS) $(NTHREADS) $(PARAMS) $<

main.o: main.cu
	$(CC) $(FLAGS) $(CFLAGS) $(NTHREADS) $(PARAMS) main.cu 
ls_gpu.o: ls_gpu.cu ls_gpu.h
	$(CC) $(FLAGS) $(CFLAGS) $(NTHREADS) $(PARAMS) ls_gpu.cu

$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@



.PHONY: debug clean 
debug: clean
debug: FLAGS =-std=c++11 -O0 -g -G -Xcompiler -fopenmp -arch=compute_60 -code=sm_60 -lcuda -lineinfo -Xcompiler -fpermissive
debug: main

clean:
	rm -f $(OBJECTS)
	rm -f $(CUDAOBJECTS)
	rm -f main



# all:
# 	$(CC) $(FLAGS) -c $(NTHREADS) $(SEARCHMODE) $(PARAMS) import_dataset.cpp globals.cpp GPU.cu kernel.cu cluster.cpp qsort.cpp
# 	$(CC) $(FLAGS) -c $(NTHREADS) $(SEARCHMODE) $(PARAMS) main.cu 
# 	$(CC) $(FLAGS) $(OBJECTS) $(CUDAOBJECTS) -o main 
