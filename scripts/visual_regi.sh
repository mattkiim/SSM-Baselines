#!/bin/bash -l
#SBATCH --job-name=visual
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH -e visual.err
#SBATCH -o visual.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

python examples/quadrotor/visual_region.py \
  --env_name QuadrotorTracking2D-v0 \
  --algo ssm \
  --ckpt_path results/QuadrotorTracking2D-v0/jaxrl5_quad2d_ssm_baseline/2026-01-08_seed0000 \
  --step 2000000 \
  --deterministic \
  --nx 41 --nz 41 \
  --out_png results/visualizations/ssm_region.png