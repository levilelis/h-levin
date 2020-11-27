#!/bin/bash

#Levin Search Training MULT

#Improved Levin Loss
sbatch --output=output_train_witness/5x5-improvedlevinloss-levinmult-default-h --export=params="-a LevinMult -p problems/witness/puzzles_5x5_train/ -l ImprovedLevinLoss -m 5x5-improvedlevinloss-levinmult-default-h --default-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
sbatch --output=output_train_witness/5x5-improvedlevinloss-levin --export=params="-a Levin -p problems/witness/puzzles_5x5_train/ -l ImprovedLevinLoss -m 5x5-improvedlevinloss-levin --learn -d Witness -b 1000" run_bootstrap_train.sh

#Cross Entropy Loss
#sbatch --output=output_train_witness/5x5-crossloss-levinmult-default-h --export=params="-a LevinMult -p problems/witness/puzzles_5x5_train/ -l CrossEntropyLoss -m 5x5-crossloss-levinmult-default-h --default-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
#sbatch --output=output_train_witness/5x5-crossloss-levinmult-learned-h --export=params="-a LevinMult -p problems/witness/puzzles_5x5_train/ -l CrossEntropyLoss -m 5x5-crossloss-levinmult-learned-h --learned-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
#sbatch --output=output_train_witness/5x5-crossloss-levinmult-learned-default-h --export=params="-a LevinMult -p problems/witness/puzzles_5x5_train/ -l CrossEntropyLoss -m 5x5-crossloss-levinmult-learned-default-h --learned-heuristic --default-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh

#Levin Loss
#sbatch --output=output_train_witness/5x5-levinloss-levinmult-default-h --export=params="-a LevinMult -p problems/witness/puzzles_5x5_train/ -l LevinLoss -m 5x5-levinloss-levinmult-default-h --default-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
#sbatch --output=output_train_witness/5x5-levinloss-levinmult-learned-h --export=params="-a LevinMult -p problems/witness/puzzles_5x5_train/ -l LevinLoss -m 5x5-levinloss-levinmult-learned-h --learned-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
#sbatch --output=output_train_witness/5x5-levinloss-levinmult-learned-default-h --export=params="-a LevinMult -p problems/witness/puzzles_5x5_train/ -l LevinLoss -m 5x5-levinloss-levinmult-learned-default-h --learned-heuristic --default-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh

#Levin Search Training
#Cross Entropy Loss
#sbatch --output=output_train_witness/5x5-crossloss-levin-learned-default-h --export=params="-a Levin -p problems/witness/puzzles_5x5_train/ -l CrossEntropyLoss -m 5x5-crossloss-levin-learned-default-h --learned-heuristic --default-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
#sbatch --output=output_train_witness/5x5-crossloss-levin-default-h --export=params="-a Levin -p problems/witness/puzzles_5x5_train/ -l CrossEntropyLoss -m 5x5-crossloss-levin-default-h --default-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
#sbatch --output=output_train_witness/5x5-crossloss-levin-learned-h --export=params="-a Levin -p problems/witness/puzzles_5x5_train/ -l CrossEntropyLoss -m 5x5-crossloss-levin-learned-h --learned-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
#sbatch --output=output_train_witness/5x5-crossloss-levin --export=params="-a Levin -p problems/witness/puzzles_5x5_train/ -l CrossEntropyLoss -m 5x5-crossloss-levin --learn -d Witness -b 1000" run_bootstrap_train.sh

#Levin Loss
#sbatch --output=output_train_witness/5x5-levinloss-levin-learned-default-h --export=params="-a Levin -p problems/witness/puzzles_5x5_train/ -l LevinLoss -m 5x5-levinloss-levin-learned-default-h --default-heuristic --learned-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
#sbatch --output=output_train_witness/5x5-levinloss-levin-default-h --export=params="-a Levin -p problems/witness/puzzles_5x5_train/ -l LevinLoss -m 5x5-levinloss-levin-default-h --default-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
#sbatch --output=output_train_witness/5x5-levinloss-levin-learned-h --export=params="-a Levin -p problems/witness/puzzles_5x5_train/ -l LevinLoss -m 5x5-levinloss-levin-learned-h --learned-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
#sbatch --output=output_train_witness/5x5-levinloss-levin --export=params="-a Levin -p problems/witness/puzzles_5x5_train/ -l LevinLoss -m 5x5-levinloss-levin --learn -d Witness -b 1000" run_bootstrap_train.sh


#A* Search
#sbatch --output=output_train_witness/5x5-mseloss-astar-learned-h --export=params="-a AStar -p problems/witness/puzzles_5x5_train/ -m 5x5-mseloss-astar-learned-h --learned-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
#sbatch --output=output_train_witness/5x5-mseloss-astar-default-learned-h --export=params="-a AStar -p problems/witness/puzzles_5x5_train/ -m 5x5-mseloss-astar-default-learned-h --learned-heuristic --default-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh

#GBFS
#sbatch --output=output_train_witness/5x5-mseloss-gbfs-learned-h --export=params="-a GBFS -p problems/witness/puzzles_5x5_train/ -m 5x5-mseloss-gbfs-learned-h --learned-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
#sbatch --output=output_train_witness/5x5-mseloss-gbfs-default-learned-h --export=params="-a GBFS -p problems/witness/puzzles_5x5_train/ -m 5x5-mseloss-gbfs-default-learned-h --learned-heuristic --default-heuristic --learn -d Witness -b 1000" run_bootstrap_train.sh
