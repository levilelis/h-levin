#!/bin/bash
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham, 10 Beluga, 8 on Niagara.
#SBATCH --mem=16G       # memory per node
#SBATCH --time=3-23:59      # time (DD-HH:MM) # maximum allocation time: 7 days on Beluga, 24 hrs on Niagara.
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-mbowling

module load python/3.6
module load cuda cudnn
source ~/tensorflow/bin/activate
python src/main.py -a ${algorithm} -l ${loss} -m ${model} -p ${problem} -d Witness -b 2000 -g 10 -scheduler online --learn