#!/bin/bash

output="output_test_stp_fixed_time/"
domain_name="5x5-stp-"

heuristic_scheme=("--learned-heuristic")
algorithm="AStar"
problems_dir="problems/stp/puzzles_5x5_test/"

scheduler="online"

for iter in {1..5}; do
	for weight in 1.0 1.5 2.0 2.5 3.0; do
		for scheme in "${heuristic_scheme[@]}"; do
			lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
			name_scheme=${scheme// /}
			name_scheme=${name_scheme//-heuristic/}
			name_scheme=${name_scheme//--/}
			output_exp="${output}${lower_algorithm}-${name_scheme}-${scheduler}-w${weight//./}-v${iter}"
			model=${domain_name}${lower_algorithm}-${name_scheme}-${scheduler}-w${weight//./}-v${iter}
			
			#echo ${output_exp}
			#echo ${model}	
			sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},model=${model},problem=${problems_dir},weight=${weight} run_bootstrap_test_astar_stp.sh
		done
	done
done
