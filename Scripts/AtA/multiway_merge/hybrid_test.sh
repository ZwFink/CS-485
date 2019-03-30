#!/bin/bash
outfile_hybrid="hybrid_out.txt"
outfile_cpu="cpu_only_out.txt"
outfile_gpu="gpu_only_out.txt"
trials=5
k=8
n_list_index=(1 2 3 4 5 6 7 8 9)
n_list=(1000000000 2000000000 3000000000 4000000000 5000000000 6000000000 7000000000 8000000000 9000000000)

cpu_only_frac=1.0
gpu_only_frac=0.0
hybrid_frac=0.47

# cpu-only

for ((i=0; i<trials; i++)); do
	for j in "${n_list_index[@]}"; do
		bs=$((312500*"$j"))
		m=$((j-1))
		echo N="${n_list[m]}" >> "$outfile_cpu"
		./main 42 "${n_list[m]}" "$bs" "$k" "$cpu_only_frac" >> "$outfile_cpu"

	done
done

# gpu-only
for ((i=0; i<trials; i++)); do
	for j in "${n_list_index[@]}"; do
		bs=$((312500*$j))
		m=$((j-1))
		echo N="${n_list[m]}" >> "$outfile_gpu"
		./main 42 "${n_list[m]}" "$bs" "$k" "$gpu_only_frac" >> "$outfile_gpu"
	done
done

# hybrid
for ((i=0; i<trials; i++)); do
	for j in "${n_list_index[@]}"; do
		bs=$((312500*"$j"))
		m=$((j-1))
		echo N="${n_list[m]}" >> "$outfile_hybrid"
		./main 42 "${n_list[m]}" "$bs" "$k" "$hybrid" >> "$outfile_hybrid"
	done
done
