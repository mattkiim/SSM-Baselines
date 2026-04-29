#!/bin/bash -l
#SBATCH --job-name=quadrotor
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -e quadrotor_ssm1.err
#SBATCH -o quadrotor_ssm1.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

# cd examples

python examples/quadrotor/train_ssm_quad2d.py \
  --mode training \
  --env_name QuadrotorTracking2D-v0 \
  --seed 2 \
  --max_steps 2010000 \
  --start_training 10000 \
  --eval_interval 50000 \
  --eval_episodes 4 \
  --batch_size 512 \
  --utd_ratio 1 \
  --save_interval 50000 \
  --wandb False

# python examples/quadrotor/train_sac_lag_quad2d.py \
#   --env_name QuadrotorTracking2D-v0 \
#   --seed 1 \
#   --max_steps 2010000 \
#   --start_training 10000 \
#   --eval_interval 50000 \
#   --eval_episodes 4 \
#   --save_interval 500000 \
#   --batch_size 512 \
#   --utd_ratio 1 \
#   --wandb False \
#   --config examples/quadrotor/configs/sac_lag_quad2d_config.py


# python examples/quadrotor/train_cal_quad2d.py \
#   --env_name QuadrotorTracking2D-v0 \
#   --seed 1 \
#   --max_steps 2010000 \
#   --start_training 10000 \
#   --eval_interval 50000 \
#   --eval_episodes 4 \
#   --save_interval 500000 \
#   --batch_size 512 \
#   --utd_ratio 1 \
#   --wandb False \
#   --config examples/quadrotor/configs/cal_quad2d_config.py


# python examples/quadrotor/train_sac_cbf_quad2d.py \
#   --env_name QuadrotorTracking2D-v0 \
#   --seed 1 \
#   --max_steps 2010000 \
#   --start_training 10000 \
#   --eval_interval 50000 \
#   --eval_episodes 4 \
#   --save_interval 500000 \
#   --batch_size 512 \
#   --utd_ratio 1 \
#   --wandb False \
#   --config examples/quadrotor/configs/sac_cbf_quad2d_config.py

# python examples/quadrotor/train_rac_quad2d.py \
#   --env_name QuadrotorTracking2D-v0 \
#   --seed 1 \
#   --max_steps 2010000 \
#   --start_training 10000 \
#   --eval_interval 50000 \
#   --eval_episodes 4 \
#   --save_interval 500000 \
#   --batch_size 512 \
#   --utd_ratio 1 \
#   --wandb False \
#   --config examples/quadrotor/configs/rac_quad2d_config.py