#!/bin/bash
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=4000M        # memory per node
#SBATCH --time=0-23:59      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

#module load cuda cudnn python/3.6
module load python/3.6
source tensorflow/bin/activate
python src/main.py problems/witness/${problems} ${loss} ${model_name} ${use_h} ${use_learned_h}
#python src/main.py problems/witness/puzzles_4x4 LevinLoss 4x4_levin_learned_h y y

