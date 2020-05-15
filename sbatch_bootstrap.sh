#!/bin/bash

sbatch --export=problems=puzzles_4x4,loss=LevinLoss,model_name=4x4_levin_learned_h,use_h=y,use_learned_h=y run_bootstrap.sh
sbatch --export=problems=puzzles_4x4,loss=CrossEntropyLoss,model_name=4x4_cross_learned_h,use_h=y,use_learned_h=y run_bootstrap.sh

sbatch --export=problems=puzzles_4x4,loss=LevinLoss,model_name=4x4_levin_h,use_h=y,use_learned_h=f run_bootstrap.sh
sbatch --export=problems=puzzles_4x4,loss=CrossEntropyLoss,model_name=4x4_cross_h,use_h=y,use_learned_h=f run_bootstrap.sh
