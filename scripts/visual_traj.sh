#!/bin/bash -l
#SBATCH --job-name=visual_quadrotor
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH -e visual_quadrotor.err
#SBATCH -o visual_quadrotor.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

# python examples/quadrotor/visualize_policy_trajectory.py \
#   --agent ssm \
#   --checkpoint_dir results/QuadrotorTracking2D-v0/jaxrl5_quad2d_ssm_baseline/2026-01-08_seed0000 \
#   --checkpoint_step 1300000 \
#   --episodes 1 \
#   --deterministic \
#   --out_dir results/visualizations/td3 \

python -m examples.quadrotor.rollout_utils \
  --ckpt_ssm results/QuadrotorTracking2D-v0/jaxrl5_quad2d_ssm_baseline/2026-01-08_seed0000 \
  --ckpt_rac results/QuadrotorTracking2D-v0/jaxrl5_quad2d_rac/2026-01-10_seed0000 \
  --ckpt_cal results/QuadrotorTracking2D-v0/jaxrl5_quad2d_cal/2026-01-10_seed0000 \
  --ckpt_sac_cbf results/QuadrotorTracking2D-v0/jaxrl5_quad2d_sac_cbf/2026-01-10_seed0000 \
  --ckpt_sac_lag results/QuadrotorTracking2D-v0/jaxrl5_quad2d_sac_lag/2026-01-10_seed0000 \
  --step 1500000 --horizon 100