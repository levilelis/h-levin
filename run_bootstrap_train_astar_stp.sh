#!/bin/bash
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M        # memory per node
#SBATCH --time=7-00:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis

module load python/3.6
source tensorflow/bin/activate
<<<<<<< HEAD
python src/main.py ${scheme} -a ${algorithm} -l ${loss} -w ${weight} -m ${model} -p problems/stp/puzzles_5x5_train/ --learn -d SlidingTile -b 7000 -g 10 -scheduler ${scheduler}
=======
python src/main.py ${scheme} -a ${algorithm} -l ${loss} -w ${weight} -m ${model} -p problems/stp/puzzles_5x5_train/ --learn -d SlidingTile -b 10000 -g 10
>>>>>>> 22f83f353f06264cdf6b0444ac1db4b61aefee27

