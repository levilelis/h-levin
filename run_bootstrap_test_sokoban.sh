#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=0
#SBATCH --time=2-00:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis

module load python/3.6
source tensorflow/bin/activate
#python src/main.py ${scheme} -a ${algorithm} -m ${model} -p ${problem} -b 2000 -time 172200 -d Sokoban -mix ${mix_epsilon}
python src/main.py ${scheme} -a ${algorithm} -m ${model} -p ${problem} -b 2000 -d Sokoban --fixed-time -number-test-instances 100 -time 3600
