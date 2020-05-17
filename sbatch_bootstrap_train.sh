#!/bin/bash

#Levin Search Training

#Cross Entropy Loss
sbatch --output=output_train/4x4-crossloss-levin-default-h --export=params="-a Levin -p problems/witness/puzzles_4x4/ -l CrossEntropyLoss -m 4x4-crossloss-levin-default-h --default-heuristic --learn" run_bootstrap_train.sh
sbatch --output=output_train/4x4-crossloss-levin-learned-h --export=params="-a Levin -p problems/witness/puzzles_4x4/ -l CrossEntropyLoss -m 4x4-crossloss-levin-learned-h --learned-heuristic --learn" run_bootstrap_train.sh
sbatch --output=output_train/4x4-crossloss-levin --export=params="-a Levin -p problems/witness/puzzles_4x4/ -l CrossEntropyLoss -m 4x4-crossloss-levin --learn" run_bootstrap_train.sh

#Levin Loss
sbatch --output=output_train/4x4-levinloss-default-h --export=params="-a Levin -p problems/witness/puzzles_4x4/ -l LevinLoss -m 4x4-levinloss-levin-default-h --default-heuristic --learn" run_bootstrap_train.sh
sbatch --output=output_train/4x4-levinloss-learned-h --export=params="-a Levin -p problems/witness/puzzles_4x4/ -l LevinLoss -m 4x4-levinloss-levin-learned-h --learned-heuristic --learn" run_bootstrap_train.sh
sbatch --output=output_train/4x4-levinloss --export=params="-a Levin -p problems/witness/puzzles_4x4/ -l LevinLoss -m 4x4-levinloss-levin --learn" run_bootstrap_train.sh

#A* Search
sbatch --output=output_train/4x4-mseloss-astar-learned-h --export=params="-a AStar -p problems/witness/puzzles_4x4/ -m 4x4-mseloss-astar-learned-h --learned-heuristic --learn" run_bootstrap_train.sh

#GBFS
sbatch --output=output_train/4x4-mseloss-astar-learned-h --export=params="-a GBFS -p problems/witness/puzzles_4x4/ -m 4x4-mseloss-gbfs-learned-h --learned-heuristic --learn" run_bootstrap_train.sh

