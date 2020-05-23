#!/bin/bash

declare -a losses=("CrossEntropyLoss" "ImprovedLevinLoss" "LevinLoss")
output="output_train_stp/"
domain_name="4x4-stp-"
problems_dir="problems/stp/puzzles_4x4_test_100"

heuristic_scheme=("--learned-heuristic --default-heuristic" "--default-heuristic" "--learned-heuristic" "") 
algorithm="Levin"

for iter in {1..1}; do
	for scheme in "${heuristic_scheme[@]}"; do
		for loss in ${losses[@]}; do
			for file in "$problems_dir"/*.pro; do
				lower_loss=$(echo ${loss} | tr "A-Z" "a-z")
				lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
				name_scheme=${scheme// /}
				name_scheme=${name_scheme//-heuristic/}
				name_scheme=${name_scheme//--/-}
				output_exp="${output}${lower_algorithm}-${lower_loss}${name_scheme}-v${iter}"
				model=${domain_name}${lower_algorithm}-${lower_loss}${name_scheme}-v${iter}
				
				mkdir -p logs_search/${model}
				
				num_jobs=`squeue -u lelis | wc -l`
				echo ${num_jobs}
				
				while [ ${num_jobs} -gt 500 ]; do
					sleep 60
				        num_jobs=`squeue -u lelis | wc -l`
				        #echo ${num_jobs}
				done
	
				sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},model=${model},problem=${file} run_bootstrap_test.sh
			done
		done
	done
done