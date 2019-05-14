#!/bin/bash
#SBATCH --job-name=Teacher_train
#SBATCH --nodes=1
#SBATCH --mem=30GB
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=results_%j.out

module purge
module load cuda/8.0.44
module load python3/intel/3.6.3
source $HOME/pyenv/py3.6.3/bin/activate

python main.py --labels data-local/labels/ilsvrc2012/64000_balanced_labels/output_train.txt  --epochs 60  --lr  0.1 --consistency 10 --consistency-type 'kl' --weight-decay 5e-5 -b 192 --labeled-batch-size 64 --ema-decay .9997 --initial-lr 0.1 --lr-rampdown-epochs 75 --lr-rampup 2 --logit-distance-cost 0.01
