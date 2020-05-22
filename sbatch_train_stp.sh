#!/bin/bash

instances="problems/stp/puzzles_4x4_train/"
training_params="--learn -d SlidingTile -b 10000"
declare -a losses=("CrossEntropyLoss" "ImprovedLevinLoss" "LevinLoss")
output="output_train_stp/"
domain_name="4x4-stp-"

heuristic_scheme=("--learned-heuristic --default-heuristic" "--default-heuristic" "--learned-heuristic" "") 
algorithm="Levin"

for iter in {1..1}; do
	for scheme in "${heuristic_scheme[@]}"; do
		for loss in ${losses[@]}; do
			lower_loss=$(echo ${loss} | tr "A-Z" "a-z")
			lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
			name_scheme=${scheme// /}
			name_scheme=${name_scheme//-heuristic/}
			name_scheme=${name_scheme//--/-}
			output_exp="${output}${lower_algorithm}-${lower_loss}${name_scheme}-v${iter}"
			params_exp="\"${scheme} -a ${algorithm} -l ${loss} -p ${instances} -m ${domain_name}${lower_algorithm}-${lower_loss}${name_scheme}-v${iter} ${training_params}\""

			sbatch --output=${output_exp} --export=params=${params_exp} run_bootstrap_train.sh
		done
	done
done

heuristic_scheme=("--learned-heuristic --default-heuristic" "--default-heuristic" "--learned-heuristic") 
algorithm="LevinStar"

for iter in {1..1}; do
	for scheme in "${heuristic_scheme[@]}"; do
		for loss in ${losses[@]}; do
			lower_loss=$(echo ${loss} | tr "A-Z" "a-z")
			lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
			name_scheme=${scheme// /}
			name_scheme=${name_scheme//-heuristic/}
			name_scheme=${name_scheme//--/-}
			output_exp="${output}${lower_algorithm}-${lower_loss}${name_scheme}-v${iter}" 
			params_exp="\"${scheme} -a ${algorithm} -l ${loss} -p ${instances} -m ${domain_name}${lower_algorithm}-${lower_loss}${name_scheme}-v${iter} ${training_params}\""
			
			sbatch --output=${output_exp} --export=params=${params_exp} run_bootstrap_train.sh
		done
	done
done

heuristic_scheme=("--learned-heuristic --default-heuristic" "--learned-heuristic") 
algorithm="GBFS"

for iter in {1..1}; do
	for scheme in "${heuristic_scheme[@]}"; do
		lower_loss=$(echo ${loss} | tr "A-Z" "a-z")
		lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
		name_scheme=${scheme// /}
		name_scheme=${name_scheme//-heuristic/}
		name_scheme=${name_scheme//--/-}
		output_exp="${output}${lower_algorithm}${name_scheme}-v${iter}"
		params_exp="\"${scheme} -a ${algorithm} -p ${instances} -m ${domain_name}${lower_algorithm}${name_scheme}-v${iter} ${training_params}\""
		
		sbatch --output=${output_exp} --export=params=${params_exp} run_bootstrap_train.sh
	done
done

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
		params_exp="\"${scheme} -a ${algorithm} -p ${instances} -m ${domain_name}${lower_algorithm}${name_scheme}-v${iter} ${training_params}\""
		
		sbatch --output=${output_exp} --export=params=${params_exp} run_bootstrap_train.sh
	done
done