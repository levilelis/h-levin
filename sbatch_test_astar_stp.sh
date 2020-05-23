#!/bin/bash

output="output_test_stp/"
domain_name="4x4-stp-"

heuristic_scheme=("--learned-heuristic --default-heuristic" "--learned-heuristic") 
algorithm="AStar"
problems_dir="problems/stp/puzzles_4x4_test_1000"

for iter in {1..1}; do
	for file in "$problems_dir"/*.pro; do
		for scheme in "${heuristic_scheme[@]}"; do
			lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
			name_scheme=${scheme// /}
			name_scheme=${name_scheme//-heuristic/}
			name_scheme=${name_scheme//--/-}
			output_exp="${output}${lower_algorithm}${name_scheme}-v${iter}"
			model=${domain_name}${lower_algorithm}${name_scheme}-v${iter}
			
			mkdir -p logs_search/${model}
			
			num_jobs=`squeue -u lelis | wc -l`
			echo ${num_jobs}
			
			if [ ${num_jobs} -gt 500 ]; then
			        sleep 10
			        num_jobs=`squeue -u lelis | wc -l`
			        echo ${num_jobs}
			fi
				
			sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},model=${model},problem=${file} run_bootstrap_test.sh
		done
	done
done