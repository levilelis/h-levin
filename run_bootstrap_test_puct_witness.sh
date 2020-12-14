#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=0
#SBATCH --time=2-00:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis

module load python/3.6
source tensorflow/bin/activate
 
#python src/main.py ${scheme} -cpuct ${constant} -a ${algorithm} -m ${model} -p ${problem} -b 2000 -time 172000 -d Witness
python src/main.py ${scheme} -cpuct ${constant} -a ${algorithm} -m ${model} -p ${problem} -b 2000 -d Witness -time 5400 --fixed-time

#BBB SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#BBB SBATCH --mem=8000M        # memory per node
