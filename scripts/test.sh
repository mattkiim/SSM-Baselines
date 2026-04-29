#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH -e test.err
#SBATCH -o test.out

nvidia-smi

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

python jaxrl5/agents/rac/smoke_test_rac_learner.py