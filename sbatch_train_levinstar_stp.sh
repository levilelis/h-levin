#!/bin/bash

#declare -a losses=("RegLevinLoss")
declare -a losses=("CrossEntropyLoss" "ImprovedLevinLoss" "LevinLoss")
output="output_train_stp/"
domain_name="6x6-stp-"
algorithm="LevinStar"

#heuristic_scheme=("--learned-heuristic --default-heuristic" "--default-heuristic" "--learned-heuristic")
heuristic_scheme=("--learned-heuristic")

scheduler="online"

for iter in {1..1}; do
	for scheme in "${heuristic_scheme[@]}"; do
		for loss in ${losses[@]}; do
			lower_loss=$(echo ${loss} | tr "A-Z" "a-z")
			lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
			name_scheme=${scheme// /}
			name_scheme=${name_scheme//-heuristic/}
			name_scheme=${name_scheme//--/-}
			output_exp="${output}${lower_algorithm}-${lower_loss}${name_scheme}-${scheduler}-v${iter}"
			model=${domain_name}${lower_algorithm}-${lower_loss}${name_scheme}-${scheduler}-v${iter}

			sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},loss=${loss},model=${model},scheduler=${scheduler} run_bootstrap_train_stp.sh
		done
	done
done
