#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

case "${1:-}" in
  ant) env_name=SafetyAntVelocity-v1 ;;
  humanoid) env_name=SafetyHumanoidVelocity-v1 ;;
  *) echo "Usage: bash scripts/train_rcrl_velocity.sh {ant|humanoid} [training flags...]" >&2; exit 2 ;;
esac
robot="$1"
shift

python_bin="${RCRL_PYTHON:-python}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1

exec "$python_bin" -m examples.states.train_rac_online \
  --config=examples/states/configs/rac_config.py \
  --env_name="$env_name" \
  --run_name="${robot}_rcrl_velocity" \
  --seed=0 \
  --max_steps=1000000 \
  --start_training=10000 \
  --batch_size=256 \
  --eval_interval=10000 \
  --eval_episodes=5 \
  --save_interval=50000 \
  --nowandb \
  "$@"
