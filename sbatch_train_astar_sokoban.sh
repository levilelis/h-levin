#!/bin/bash

output="output_train_sokoban/"
domain_name="10x10-sokoban-"
algorithm="AStar"

heuristic_scheme=("--learned-heuristic")
#heuristic_scheme=("--learned-heuristic --default-heuristic" "--default-heuristic" "--learned-heuristic")

for iter in {1..1}; do
	for scheme in "${heuristic_scheme[@]}"; do
		lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
		name_scheme=${scheme// /}
		name_scheme=${name_scheme//-heuristic/}
		name_scheme=${name_scheme//--/-}
		output_exp="${output}${lower_algorithm}-${name_scheme}-v${iter}"
		model=${domain_name}${lower_algorithm}-${name_scheme}-v${iter}

		sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},model=${model} run_bootstrap_train_sokoban.sh
	done
done
