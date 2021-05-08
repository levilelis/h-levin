#!/bin/bash

output="output_test_sokoban_fixed_time/"
domain_name="10x10-sokoban-"

heuristic_scheme=("--learned-heuristic")
#heuristic_scheme=("--default-heuristic" "--learned-heuristic")
#heuristic_scheme=("--learned-heuristic --default-heuristic" "--learned-heuristic") 
algorithm="AStar"
problems_dir="problems/sokoban/test/000.txt"

scheduler="online"

for iter in {1..5}; do
	for scheme in "${heuristic_scheme[@]}"; do
		for weight in 1.0 1.5 2.0 2.5; do
			lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
			name_scheme=${scheme// /}
			name_scheme=${name_scheme//-heuristic/}
			name_scheme=${name_scheme//--/}
			output_exp="${output}${lower_algorithm}-${name_scheme}-${scheduler}-w${weight//./}-v${iter}"
			model=${domain_name}${lower_algorithm}-${name_scheme}-${scheduler}-w${weight//./}-v${iter}
			
			#echo ${output_exp}
			#echo ${model}	
			sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},model=${model},problem=${problems_dir},weight=${weight} run_bootstrap_test_astar_sokoban.sh
		done
	done
done
