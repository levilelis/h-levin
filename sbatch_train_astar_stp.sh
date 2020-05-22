#!/bin/bash

declare -a losses=("CrossEntropyLoss" "ImprovedLevinLoss" "LevinLoss")
output="output_train_stp/"
domain_name="4x4-stp-"

heuristic_scheme=("--learned-heuristic --default-heuristic" "--learned-heuristic") 
algorithm="AStar"

for iter in {1..1}; do
	for scheme in "${heuristic_scheme[@]}"; do
		lower_loss=$(echo ${loss} | tr "A-Z" "a-z")
		lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
		name_scheme=${scheme// /}
		name_scheme=${name_scheme//-heuristic/}
		name_scheme=${name_scheme//--/-}
		output_exp="${output}${lower_algorithm}${name_scheme}-v${iter}"
		model=${domain_name}${lower_algorithm}${name_scheme}-v${iter}

		sbatch --output=${output_exp} --export=scheme="${scheme}",loss=mseloss,algorithm=${algorithm},model=${model} run_bootstrap_train.sh
	done
done