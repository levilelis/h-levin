#!/bin/bash

output="output_train_stp/"
domain_name="5x5-stp-"
algorithm="PUCT"
declare -a losses=("CrossEntropyLoss")
constants=("0.5" "1.0" "1.5" "2.0") 
heuristic_scheme=("--learned-heuristic")

scheduler="online" 

for iter in {1..5}; do
	for scheme in "${heuristic_scheme[@]}"; do
		for c in "${constants[@]}"; do
			for loss in ${losses[@]}; do
				lower_loss=$(echo ${loss} | tr "A-Z" "a-z")
				lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
				name_scheme=${scheme// /}
				name_scheme=${name_scheme//-heuristic/}
				name_scheme=${name_scheme//--/-}
				c_name=${c//./}
				name_scheme=${name_scheme//--/-}
				output_exp="${output}${lower_algorithm}-${lower_loss}${name_scheme}-${scheduler}-c${c_name}-normalized-v${iter}"
				model=${domain_name}${lower_algorithm}-${lower_loss}${name_scheme}-${scheduler}-c${c_name}-normalized-v${iter}
		
				sbatch --output=${output_exp} --export=scheme="${scheme}",constant=${c},algorithm=${algorithm},loss=${loss},model=${model} run_bootstrap_train_puct_stp.sh
			done
		done
	done
done