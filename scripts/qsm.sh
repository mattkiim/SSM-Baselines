#!/bin/bash -l
#SBATCH --job-name=qsm
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH -e qsm.err
#SBATCH -o qsm.out

export HOME=/scratch_root/wy524
mkdir -p $HOME/.config/wandb

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

python examples/states/train_score_matching_online.py \
  --wandb True \
  --project_name safety_qsm \
  --run_name carbutton_qsm \
  --seed 0 \
  --env_name SafetyCarButton1-v0 \
  --max_steps 1000000 \
  --epoch_length 2000 \
  --start_training 10000 \
  --eval_interval 2000 \
  --log_interval 1000

