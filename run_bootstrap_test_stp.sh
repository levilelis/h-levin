#!/bin/bash
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M        # memory per node
#SBATCH --time=5-0:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis

module load python/3.6
source tensorflow/bin/activate
python src/main.py ${scheme} -a ${algorithm} -m ${model} -p ${problem} -b 7000 -d SlidingTile --fixed-time -number-test-instances 100 -time 1800
#python src/main.py ${scheme} -a ${algorithm} -m ${model} -p ${problem} -b 7000 -time 172000 -d SlidingTile -mix ${mix_epsilon}

