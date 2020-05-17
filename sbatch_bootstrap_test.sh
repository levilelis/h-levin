#!/bin/bash

sbatch --output=4x4_cross_h --export=params="-p problems/witness/puzzles_4x4_test/ -l CrossEntropyLoss -m 4x4_cross_h --default-heuristic" run_bootstrap_test.sh
sbatch --output=4x4_cross_learned_h --export=params="-p problems/witness/puzzles_4x4_test/ -l CrossEntropyLoss -m 4x4_cross_learned_h --learned-heuristic" run_bootstrap_test.sh
sbatch --output=4x4_levin_h --export=params="-p problems/witness/puzzles_4x4_test/ -l LevinLoss -m 4x4_levin_h --default-heuristic" run_bootstrap_test.sh
sbatch --output=4x4_levin_learned_h --export=params="-p problems/witness/puzzles_4x4_test/ -l LevinLoss -m 4x4_levin_learned_h --learned-heuristic" run_bootstrap_test.sh
sbatch --output=4x4_blind_search --export=params="-p problems/witness/puzzles_4x4_test/ --blind-search" run_bootstrap_test.sh
