#!/bin/bash

#Levin Search Training MULT

#Improved Levin Loss
#sbatch --output=output_test_witness/6x6-improvedlevinloss-levinmult-default-h --export=params="-a LevinMult -p problems/witness/puzzles_6x6_test/ -l ImprovedLevinLoss -m 6x6-improvedlevinloss-levinmult-default-h --default-heuristic -d Witness -b 1000" run_bootstrap_test.sh
#sbatch --output=output_test_witness/6x6-improvedlevinloss-levin --export=params="-a Levin -p problems/witness/puzzles_6x6_test/ -l ImprovedLevinLoss -m 6x6-improvedlevinloss-levin -d Witness -b 1000" run_bootstrap_test.sh

#Cross Entropy Loss
#sbatch --output=output_test_witness/6x6-crossloss-levinmult-default-h --export=params="-a LevinMult -p problems/witness/puzzles_6x6_test/ -l CrossEntropyLoss -m 6x6-crossloss-levinmult-default-h --default-heuristic -d Witness -b 1000" run_bootstrap_test.sh
#sbatch --output=output_test_witness/6x6-crossloss-levinmult-learned-h --export=params="-a LevinMult -p problems/witness/puzzles_6x6_test/ -l CrossEntropyLoss -m 6x6-crossloss-levinmult-learned-h --learned-heuristic -d Witness -b 1000" run_bootstrap_test.sh
#sbatch --output=output_test_witness/6x6-crossloss-levinmult-learned-default-h --export=params="-a LevinMult -p problems/witness/puzzles_6x6_test/ -l CrossEntropyLoss -m 6x6-crossloss-levinmult-learned-default-h --learned-heuristic --default-heuristic -d Witness -b 1000" run_bootstrap_test.sh

#Levin Loss
#sbatch --output=output_test_witness/6x6-levinloss-levinmult-default-h --export=params="-a LevinMult -p problems/witness/puzzles_6x6_test/ -l LevinLoss -m 6x6-levinloss-levinmult-default-h --default-heuristic -d Witness -b 1000" run_bootstrap_test.sh
#sbatch --output=output_test_witness/6x6-levinloss-levinmult-learned-h --export=params="-a LevinMult -p problems/witness/puzzles_6x6_test/ -l LevinLoss -m 6x6-levinloss-levinmult-learned-h --learned-heuristic -d Witness -b 1000" run_bootstrap_test.sh
#sbatch --output=output_test_witness/6x6-levinloss-levinmult-learned-default-h --export=params="-a LevinMult -p problems/witness/puzzles_6x6_test/ -l LevinLoss -m 6x6-levinloss-levinmult-learned-default-h --learned-heuristic --default-heuristic -d Witness -b 1000" run_bootstrap_test.sh

#Levin Search Training
#Cross Entropy Loss
#sbatch --output=output_test_witness/6x6-crossloss-levin-learned-default-h --export=params="-a Levin -p problems/witness/puzzles_6x6_test/ -l CrossEntropyLoss -m 6x6-crossloss-levin-learned-default-h --learned-heuristic --default-heuristic -d Witness -b 1000" run_bootstrap_test.sh
#sbatch --output=output_test_witness/6x6-crossloss-levin-default-h --export=params="-a Levin -p problems/witness/puzzles_6x6_test/ -l CrossEntropyLoss -m 6x6-crossloss-levin-default-h --default-heuristic -d Witness -b 1000" run_bootstrap_test.sh
#sbatch --output=output_test_witness/6x6-crossloss-levin-learned-h --export=params="-a Levin -p problems/witness/puzzles_6x6_test/ -l CrossEntropyLoss -m 6x6-crossloss-levin-learned-h --learned-heuristic -d Witness -b 1000" run_bootstrap_test.sh
#sbatch --output=output_test_witness/6x6-crossloss-levin --export=params="-a Levin -p problems/witness/puzzles_6x6_test/ -l CrossEntropyLoss -m 6x6-crossloss-levin -d Witness -b 1000" run_bootstrap_test.sh

#Levin Loss
#sbatch --output=output_test_witness/6x6-levinloss-levin-learned-default-h --export=params="-a Levin -p problems/witness/puzzles_6x6_test/ -l LevinLoss -m 6x6-levinloss-levin-learned-default-h --default-heuristic --learned-heuristic -d Witness -b 1000" run_bootstrap_test.sh
#sbatch --output=output_test_witness/6x6-levinloss-levin-default-h --export=params="-a Levin -p problems/witness/puzzles_6x6_test/ -l LevinLoss -m 6x6-levinloss-levin-default-h --default-heuristic -d Witness -b 1000" run_bootstrap_test.sh
#sbatch --output=output_test_witness/6x6-levinloss-levin-learned-h --export=params="-a Levin -p problems/witness/puzzles_6x6_test/ -l LevinLoss -m 6x6-levinloss-levin-learned-h --learned-heuristic -d Witness -b 1000" run_bootstrap_test.sh
#sbatch --output=output_test_witness/6x6-levinloss-levin --export=params="-a Levin -p problems/witness/puzzles_6x6_test/ -l LevinLoss -m 6x6-levinloss-levin -d Witness -b 1000" run_bootstrap_test.sh


#A* Search
#sbatch --output=output_test_witness/6x6-mseloss-astar-learned-h --export=params="-a AStar -p problems/witness/puzzles_6x6_test/ -m 6x6-mseloss-astar-learned-h --learned-heuristic -d Witness -b 1000" run_bootstrap_test.sh
#sbatch --output=output_test_witness/6x6-mseloss-astar-default-learned-h --export=params="-a AStar -p problems/witness/puzzles_6x6_test/ -m 6x6-mseloss-astar-default-learned-h --learned-heuristic --default-heuristic -d Witness -b 1000" run_bootstrap_test.sh
sbatch --output=output_test_witness/6x6-astar-default-h --export=params="-a AStar -p problems/witness/puzzles_6x6_test/ --default-heuristic -d Witness -b 1000" run_bootstrap_test.sh


#GBFS
#sbatch --output=output_test_witness/6x6-mseloss-gbfs-learned-h --export=params="-a GBFS -p problems/witness/puzzles_6x6_test/ -m 6x6-mseloss-gbfs-learned-h --learned-heuristic -d Witness -b 1000" run_bootstrap_test.sh
#sbatch --output=output_test_witness/6x6-mseloss-gbfs-default-learned-h --export=params="-a GBFS -p problems/witness/puzzles_6x6_test/ -m 6x6-mseloss-gbfs-default-learned-h --learned-heuristic --default-heuristic -d Witness -b 1000" run_bootstrap_test.sh
sbatch --output=output_test_witness/6x6-gbfs-default-h --export=params="-a GBFS -p problems/witness/puzzles_6x6_test/ --default-heuristic -d Witness -b 1000" run_bootstrap_test.sh
