#!/bin/bash

#SBATCH --job-name=CAE_train
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

module purge
module load cuda/8.0.44
module load python3/intel/3.6.3
source $HOME/pyenv/py3.6.3/bin/activate

python CAE_train.py
