#!/bin/bash

loss="CrossEntropyLoss"
algorithm="Levin"
domain_name="Witness"
problems_dir="problems/witness/puzzles_4x4/"
size_puzzle="4x4"
output="output_test_witness_4x4/"


output_exp="${output}${algorithm}-${loss}"
model=${size_puzzle}-${domain_name}-${loss}
sbatch --output="${output_exp}" --export=algorithm=${algorithm},model=${model},problem=${problems_dir},loss=${loss} FD_submit_experiment.sh
echo "finished calling sbatch"

