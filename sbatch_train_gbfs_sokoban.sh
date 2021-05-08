#!/bin/bash

output="output_train_sokoban/"
domain_name="10x10-sokoban-"
algorithm="GBFS"
loss="MSELoss"
heuristic_scheme=("--learned-heuristic")
#heuristic_scheme=("--learned-heuristic --default-heuristic" "--default-heuristic" "--learned-heuristic")

scheduler="online"

for iter in {2..5}; do
	for scheme in "${heuristic_scheme[@]}"; do
		lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
		name_scheme=${scheme// /}
		name_scheme=${name_scheme//-heuristic/}
		name_scheme=${name_scheme//--/-}

		output_exp="${output}${lower_algorithm}-${name_scheme}-${scheduler}-v${iter}"
		model=${domain_name}${lower_algorithm}-${name_scheme}-${scheduler}-v${iter}

		sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},loss=${loss},model=${model} run_bootstrap_train_sokoban.sh
	done
done
