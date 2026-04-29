#!/bin/bash -l
#SBATCH --job-name=q1
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH -e q1_visual.err
#SBATCH -o q1_visual.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

python examples/quadrotor/q1_visual.py \
  --ckpt_ssm results/QuadrotorTracking2D-v0/jaxrl5_quad2d_ssm_baseline/2026-01-08_seed0000 --step_ssm 1300000 \
  --ckpt_rac results/QuadrotorTracking2D-v0/jaxrl5_quad2d_rac/2026-01-10_seed0000 --step_rac 2000000 \
  --ckpt_sac_lag results/QuadrotorTracking2D-v0/jaxrl5_quad2d_sac_lag/2026-01-10_seed0000 --step_sac_lag 2000000 \
  --grid_n 41 --mc_short 10 --horizon_short 40 --mc_tts 5 --horizon_long 360 \
  --x_min -1.5 --x_max 1.5 --z_min 0 --z_max 2 \
  --cache_path examples/quadrotor/figures/q1_cache.npz