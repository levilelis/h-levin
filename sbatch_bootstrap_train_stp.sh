#!/bin/bash

#Levin Search Training MULT
#Cross Entropy Loss
sbatch --output=output_train_stp/4x4-stp-crossloss-levinmult-default-h --export=params="-a LevinMult -p problems/stp/puzzles_4x4_train/ -l CrossEntropyLoss -m 4x4-stp-crossloss-levinmult-default-h --default-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh
sbatch --output=output_train_stp/4x4-stp-crossloss-levinmult-learned-h --export=params="-a LevinMult -p problems/stp/puzzles_4x4_train/ -l CrossEntropyLoss -m 4x4-stp-crossloss-levinmult-learned-h --learned-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh
sbatch --output=output_train_stp/4x4-stp-crossloss-levinmult-learned-default-h --export=params="-a LevinMult -p problems/stp/puzzles_4x4_train/ -l CrossEntropyLoss -m 4x4-stp-crossloss-levinmult-learned-default-h --learned-heuristic --default-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh

#Levin Loss
sbatch --output=output_train_stp/4x4-stp-levinloss-levinmult-default-h --export=params="-a LevinMult -p problems/stp/puzzles_4x4_train/ -l LevinLoss -m 4x4-stp-levinloss-levinmult-default-h --default-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh
sbatch --output=output_train_stp/4x4-stp-levinloss-levinmult-learned-h --export=params="-a LevinMult -p problems/stp/puzzles_4x4_train/ -l LevinLoss -m 4x4-stp-levinloss-levinmult-learned-h --learned-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh
sbatch --output=output_train_stp/4x4-stp-levinloss-levinmult-learned-default-h --export=params="-a LevinMult -p problems/stp/puzzles_4x4_train/ -l LevinLoss -m 4x4-stp-levinloss-levinmult-learned-default-h --learned-heuristic --default-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh

#Improved Levin Loss
sbatch --output=output_train_stp/4x4-stp-improvedlevinloss-levinmult-default-h --export=params="-a LevinMult -p problems/stp/puzzles_4x4_train/ -l ImprovedLevinLoss -m 4x4-stp-improvedlevinloss-levinmult-default-h --default-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh

#Levin Search Training
#Cross Entropy Loss
sbatch --output=output_train_stp/4x4-stp-crossloss-levin-learned-default-h --export=params="-a Levin -p problems/stp/puzzles_4x4_train/ -l CrossEntropyLoss -m 4x4-stp-crossloss-levin-learned-default-h --learned-heuristic --default-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh
sbatch --output=output_train_stp/4x4-stp-crossloss-levin-default-h --export=params="-a Levin -p problems/stp/puzzles_4x4_train/ -l CrossEntropyLoss -m 4x4-stp-crossloss-levin-default-h --default-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh
sbatch --output=output_train_stp/4x4-stp-crossloss-levin-learned-h --export=params="-a Levin -p problems/stp/puzzles_4x4_train/ -l CrossEntropyLoss -m 4x4-stp-crossloss-levin-learned-h --learned-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh
sbatch --output=output_train_stp/4x4-stp-crossloss-levin --export=params="-a Levin -p problems/stp/puzzles_4x4_train/ -l CrossEntropyLoss -m 4x4-stp-crossloss-levin --learn -d SlidingTile -b 10000" run_bootstrap_train.sh

#Levin Loss
sbatch --output=output_train_stp/4x4-stp-levinloss-levin-learned-default-h --export=params="-a Levin -p problems/stp/puzzles_4x4_train/ -l LevinLoss -m 4x4-stp-levinloss-levin-learned-default-h --default-heuristic --learned-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh
sbatch --output=output_train_stp/4x4-stp-levinloss-levin-default-h --export=params="-a Levin -p problems/stp/puzzles_4x4_train/ -l LevinLoss -m 4x4-stp-levinloss-levin-default-h --default-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh
sbatch --output=output_train_stp/4x4-stp-levinloss-levin-learned-h --export=params="-a Levin -p problems/stp/puzzles_4x4_train/ -l LevinLoss -m 4x4-stp-levinloss-levin-learned-h --learned-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh
sbatch --output=output_train_stp/4x4-stp-levinloss-levin --export=params="-a Levin -p problems/stp/puzzles_4x4_train/ -l LevinLoss -m 4x4-stp-levinloss-levin --learn -d SlidingTile -b 10000" run_bootstrap_train.sh

#Improved Levin Loss
sbatch --output=output_train_stp/4x4-stp-improvedlevinloss-levin-default-h --export=params="-a Levin -p problems/stp/puzzles_4x4_train/ -l ImprovedLevinLoss -m 4x4-stp-improvedlevinloss-levin-default-h --default-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh

#A* Search
sbatch --output=output_train_stp/4x4-stp-mseloss-astar-learned-h --export=params="-a AStar -p problems/stp/puzzles_4x4_train/ -m 4x4-stp-mseloss-astar-learned-h --learned-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh
sbatch --output=output_train_stp/4x4-stp-mseloss-astar-default-learned-h --export=params="-a AStar -p problems/stp/puzzles_4x4_train/ -m 4x4-stp-mseloss-astar-default-learned-h --learned-heuristic --default-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh

#GBFS
sbatch --output=output_train_stp/4x4-stp-mseloss-gbfs-learned-h --export=params="-a GBFS -p problems/stp/puzzles_4x4_train/ -m 4x4-stp-mseloss-gbfs-learned-h --learned-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh
sbatch --output=output_train_stp/4x4-stp-mseloss-gbfs-default-learned-h --export=params="-a GBFS -p problems/stp/puzzles_4x4_train/ -m 4x4-stp-mseloss-gbfs-default-learned-h --learned-heuristic --default-heuristic --learn -d SlidingTile -b 10000" run_bootstrap_train.sh
