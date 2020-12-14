#!/bin/bash

declare -a losses=("CrossEntropyLoss")
output="output_test_stp_fixed_time/"
domain_name="5x5-stp-"
problems_dir="problems/stp/puzzles_5x5_test/"

heuristic_scheme=("--learned-heuristic")
constants=("1.0" "1.5" "2.0") 
algorithm="PUCT"

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
				output_exp="${output}${lower_algorithm}-${lower_loss}${name_scheme}-${scheduler}-c${c_name}-v${iter}"
				model=${domain_name}${lower_algorithm}-${lower_loss}${name_scheme}-${scheduler}-c${c_name}-v${iter}
				#echo ${output_exp}
				#echo ${model}				
				sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},constant=${c},model=${model},problem=${problems_dir} run_bootstrap_test_puct_stp.sh
			done
		done
	done
done
