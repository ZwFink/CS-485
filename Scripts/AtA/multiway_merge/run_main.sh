#!/bin/bash
trials=6

n=1000000000
echo N="$n" >> outfile_2.tsv
for((i=0; i<trials; i++)); do
	./main 42 "$n" 312500 8 >> outfile_2.tsv
done

echo \n\n >> outfile_2.tsv
n=2000000000
echo N="$n" >> outfile_2.tsv
for((i=0; i<trials; i++)); do
	./main 42 "$n" 625000 8 >> outfile_2.tsv
done

echo \n\n >> outfile_2.tsv

n=3000000000
echo N="$n" >> outfile_2.tsv
for((i=0; i<trials; i++)); do
	./main 42 "$n" 937500 8 >> outfile_2.tsv
done

echo \n\n >> outfile_2.tsv
n=4000000000
echo N="$n" >> outfile_2.tsv
for((i=0; i<trials; i++)); do
	./main 42 "$n" 1250000 8 >> outfile_2.tsv
done

echo \n\n >> outfile_2.tsv
n=5000000000
echo N="$n" >> outfile_2.tsv
for((i=0; i<trials; i++)); do
	./main 42 "$n" 1562500 8 >> outfile_2.tsv
done

