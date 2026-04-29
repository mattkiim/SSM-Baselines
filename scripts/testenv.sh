#!/bin/bash -l
#SBATCH --job-name=td3_ckpt_debug
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o td3_ckpt_debug.out
#SBATCH -e td3_ckpt_debug.err

set -euo pipefail

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate jaxrl

# ---- 强制用 CPU，避免 JAX 初始化 CUDA 报错 ----
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

# ---- 避免 matplotlib 权限 warning（虽然不画图，但有些模块可能 import matplotlib）----
export MPLCONFIGDIR=/tmp/matplotlib_config_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}"

cd /scratch_root/wy524/SafeScoreMatching

python - <<'PY'
from flax import serialization
p = "results/QuadrotorTracking2D-v0/jaxrl5_quad2d_td3_baseline/2026-01-04_seed0000/checkpoints/ckpt_50000.msgpack"
with open(p, "rb") as f:
    obj = serialization.msgpack_restore(f.read())
print("restored type:", type(obj))
PY
