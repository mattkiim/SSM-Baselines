#!/usr/bin/env bash
set -euo pipefail

VARIANT="${VARIANT:-warm_anchor_fork20_mirror_qx2}"

if [[ "${VARIANT}" != "warm_anchor_fork20_mirror_qx2" ]]; then
  echo "Unsupported VARIANT=${VARIANT}. Supported: warm_anchor_fork20_mirror_qx2" >&2
  exit 2
fi

python examples/quadrotor/train_rac_quad2d_stab.py \
  --env_name QuadrotorStabilization2D-v0 \
  --config examples/quadrotor/configs/rac_quad2d_stab_config.py \
  --run_name "${VARIANT}" \
  --layout_name corridor_v2 \
  --reset_mode simple_under3 \
  --obs_feature_mode state \
  --q_x 2.0 \
  --q_z 10.0 \
  --simple_bar 0.24 \
  --simple_left 0.08 \
  --simple_right 0.08 \
  --simple_gap 0.08 \
  --simple_low 0.20 \
  --simple_near 0.12 \
  --simple_fork 0.20 \
  "$@"
