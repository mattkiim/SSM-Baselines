#!/bin/bash -l
#SBATCH --job-name=saccbf
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH -e saccbf.err
#SBATCH -o saccbf.out

export HOME=/scratch_root/wy524
mkdir -p $HOME/.config/wandb

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

python examples/states/train_sac_cbf_online.py \
  --wandb True \
  --project_name gymnasium_long \
  --run_name cargoal1_sac_cbf \
  --seed 0 \
  --env_name SafetyCarGoal1-v0 \
  --max_steps 1000000 \
  --epoch_length 2000 \
  --start_training 10000 \
  --eval_interval 2000 \
  --log_interval 1000

